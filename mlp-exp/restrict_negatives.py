import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class TotalSuppressionSteering(nn.Module):
    def __init__(self, original_mlp, sae_path, dataset, ablate_ids=None, steer_ids=None, steer_multiplier=5.0, device="cuda"):
        super().__init__()
        self.mlp = original_mlp.to(device).eval()
        self.device = device
        self.ablate_ids = ablate_ids if ablate_ids else []
        self.steer_ids = steer_ids if steer_ids else []
        self.steer_multiplier = steer_multiplier
        
        ckpt = torch.load(sae_path, weights_only=False)
        self.encoder_w = ckpt['model_state_dict']['encoder.weight'].to(device)
        self.encoder_b = ckpt['model_state_dict']['encoder.bias'].to(device)
        self.decoder_w = ckpt['model_state_dict']['decoder.weight'].to(device)
        self.decoder_b = ckpt['model_state_dict']['decoder.bias'].to(device)
        self.mean_act = ckpt['acts_mean'].to(device)
        self.std_act = ckpt['acts_std'].to(device)
        self.feature_means = self._compute_feature_means(dataset)

    def _compute_feature_means(self, dataset):
        print("Calculating stable feature means...")
        loader = DataLoader(dataset, batch_size=128)
        all_f_acts = []
        with torch.no_grad():
            for grid, coords, _ in loader:
                grid, coords = grid.to(self.device), coords.to(self.device)
                acts = self.mlp.grid_net(grid) * self.mlp.coord_net(coords)
                norm_acts = (acts - self.mean_act) / (self.std_act + 1e-6)
                f_acts = torch.relu(torch.nn.functional.linear(norm_acts - self.decoder_b, self.encoder_w, self.encoder_b))
                all_f_acts.append(f_acts.mean(dim=0))
        return torch.stack(all_f_acts).mean(dim=0)
        
    ## Iterative Precision Steering
    def forward(self, grid, coords_one_hot):

        with torch.no_grad():
            # 1. Base Latent Pass
            acts = self.mlp.grid_net(grid) * self.mlp.coord_net(coords_one_hot)
            norm_acts = (acts - self.mean_act) / (self.std_act + 1e-6)
            f_acts_orig = torch.relu(torch.nn.functional.linear(norm_acts - self.decoder_b, self.encoder_w, self.encoder_b))
            
            # 2. Apply Ablation (Neutralize Negatives)
            f_acts = f_acts_orig.clone()
            for idx in self.ablate_ids:
                f_acts[:, idx] = self.feature_means[idx]
            
            # 3. Iterative Correction (Max 3 pulses to hit 100%)
            # This ensures we don't 'over-steer' but guarantees we 'cross zero'
            for _ in range(3):
                recon = torch.nn.functional.linear(f_acts, self.decoder_w, self.decoder_b)
                current_pred = self.mlp.output_head((recon * self.std_act) + self.mean_act)
                
                # Identify samples still below our 'Safe Zone' of 0.5
                mask = (current_pred < 0.5).float()
                if mask.sum() == 0: break # Everyone is safe!
                
                # Apply a surgical pulse to the positive drivers for ONLY the failing samples
                for idx in self.steer_ids:
                    # We add a small proportional boost (0.5 + a fraction of the debt)
                    f_acts[:, idx] += mask.squeeze() * (torch.abs(current_pred.squeeze()) / len(self.steer_ids) + 0.5)

            # 4. Final Output
            final_recon = torch.nn.functional.linear(f_acts, self.decoder_w, self.decoder_b)
            return self.mlp.output_head((final_recon * self.std_act) + self.mean_act)

def get_causal_drivers(mlp, sae_path, k=10, find_positive=True, device="cuda"):
    ckpt = torch.load(sae_path, weights_only=False)
    decoder_w = ckpt['model_state_dict']['decoder.weight'].to(device)
    target_layer = next(l for l in mlp.output_head if isinstance(l, torch.nn.Linear))
    output_w = target_layer.weight.to(device)
    with torch.no_grad():
        net_contributions = torch.matmul(decoder_w.t(), output_w.t()).sum(dim=1)
    vals, indices = torch.topk(net_contributions, k=k, largest=find_positive)
    return indices.cpu().tolist()

def comparative_inference(original_mlp, steered_model, dataset, num_samples=10):
    """Prints a side-by-side comparison of original vs steered predictions."""
    print(f"\n{'#'*20} COMPARATIVE INFERENCE {'#'*20}")
    print(f"{'Sample':<8} | {'Ground Truth':<15} | {'Original Pred':<15} | {'Steered Pred':<15} | {'Status'}")
    print("-" * 85)
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    device = steered_model.device
    count = 0
    
    with torch.no_grad():
        for grid, coords, target in loader:
            if count >= num_samples: break
            
            grid, coords = grid.to(device), coords.to(device)
            
            # Run Original MLP
            orig_pred = original_mlp(grid, coords).item()
            # Run Steered Model
            steered_pred = steered_model(grid, coords).item()
            
            target_val = target.item()
            
            # Determine if the intervention fixed a negative
            status = "FIXED ✅" if orig_pred < 0 and steered_pred >= 0 else "---"
            if orig_pred >= 0 and steered_pred >= 0: status = "STABLE 🆗"
            if steered_pred < 0: status = "FAILED ❌"

            print(f"{count:<8} | {target_val:<15.4f} | {orig_pred:<15.4f} | {steered_pred:<15.4f} | {status}")
            count += 1

def run_test(model, dataset):
    loader = DataLoader(dataset, batch_size=64)
    target_neg, pred_neg = 0, 0
    with torch.no_grad():
        for grid, coords, target in loader:
            grid, coords = grid.to(model.device), coords.to(model.device)
            pred = model(grid, coords)
            target_neg += (target < 0).sum().item()
            pred_neg += (pred < 0).sum().item()
            
    reduction = (1 - (pred_neg / target_neg)) * 100 if target_neg > 0 else 0
    print(f"\nFinal Statistics:")
    print(f"Target Negatives: {target_neg}")
    print(f"Steered Negatives: {pred_neg}")
    print(f"Suppression Rate: {reduction:.2f}%")

def export_steering_configs(mlp, sae_path, dataset, output_path="steering_config.pt"):
    """
    Analyzes the SAE and MLP to find the best drivers for both 
    Positive-Only and Negative-Only modes, then saves them.
    """
    ckpt = torch.load(sae_path, weights_only=False)
    decoder_w = ckpt['model_state_dict']['decoder.weight'].cuda()
    
    # Get the MLP output head weights to find directional drivers
    target_layer = next(l for l in mlp.output_head if isinstance(l, torch.nn.Linear))
    output_w = target_layer.weight.cuda()
    
    with torch.no_grad():
        # net_contributions: positive values = push up, negative = push down
        net_contributions = torch.matmul(decoder_w.t(), output_w.t()).squeeze()
    
    # Selection: 20 ablate, 10 steer (as per your successful tests)
    config = {
        "positive_only": {
            "ablate_ids": torch.topk(net_contributions, k=20, largest=False)[1].tolist(),
            "steer_ids": torch.topk(net_contributions, k=10, largest=True)[1].tolist(),
            "target_threshold": 0.5
        },
        "negative_only": {
            "ablate_ids": torch.topk(net_contributions, k=20, largest=True)[1].tolist(),
            "steer_ids": torch.topk(net_contributions, k=10, largest=False)[1].tolist(),
            "target_threshold": -0.5
        },
        "sae_stats": {
            "mean": ckpt['acts_mean'],
            "std": ckpt['acts_std']
        }
    }
    
    torch.save(config, output_path)
    print(f"Steering configuration exported to {output_path}")

if __name__ == "__main__":

    set_seed(42)

    print(f"\n--- ABLATING NEGATIVES ---")    
    from mlp.mlp_definition import FinalSpatialMLP
    from dataset.data_generator import OneHotSpatialDataset
    
    mlp = FinalSpatialMLP()
    mlp.load_state_dict(torch.load("mlp/final_spatial_model.pth"))
    mlp.eval()
    
    sae_path = "sae/sae_weights.pth"
    dataset = OneHotSpatialDataset(pt_path="dataset/train_data.pt")

    neg_ids = get_causal_drivers(mlp, sae_path, k=10, find_positive=False)
    pos_ids = get_causal_drivers(mlp, sae_path, k=5, find_positive=True)
    
    print(f"\n--- CAUSAL DRIVERS FOUND ---")
    print(f"Ablating (Negatives): {neg_ids}")
    print(f"Steering (Positives): {pos_ids}")

    surgical_model = TotalSuppressionSteering(mlp, sae_path, dataset, 
                                            ablate_ids=neg_ids, 
                                            steer_ids=pos_ids, 
                                            steer_multiplier=5.0)
    # 1. Run full dataset validation
    run_test(surgical_model, dataset)

    # 2. Run transparent side-by-side inference
    comparative_inference(mlp, surgical_model, dataset, num_samples=30)

    export_steering_configs(mlp, "sae/sae_weights.pth", dataset, output_path="steering_config_pos.pt")