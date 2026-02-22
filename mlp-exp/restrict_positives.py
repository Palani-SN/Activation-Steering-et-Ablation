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

class TotalNegativeSteering(nn.Module):
    def __init__(self, original_mlp, sae_path, dataset, ablate_ids=None, steer_ids=None, device="cuda"):
        super().__init__()
        self.mlp = original_mlp.to(device).eval()
        self.device = device
        self.ablate_ids = ablate_ids if ablate_ids else []
        self.steer_ids = steer_ids if steer_ids else []
        
        ckpt = torch.load(sae_path, weights_only=False)
        self.encoder_w = ckpt['model_state_dict']['encoder.weight'].to(device)
        self.encoder_b = ckpt['model_state_dict']['encoder.bias'].to(device)
        self.decoder_w = ckpt['model_state_dict']['decoder.weight'].to(device)
        self.decoder_b = ckpt['model_state_dict']['decoder.bias'].to(device)
        self.mean_act = ckpt['acts_mean'].to(device)
        self.std_act = ckpt['acts_std'].to(device)
        self.feature_means = self._compute_feature_means(dataset)

    def _compute_feature_means(self, dataset):
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
        
    def forward(self, grid, coords_one_hot):
        with torch.no_grad():
            acts = self.mlp.grid_net(grid) * self.mlp.coord_net(coords_one_hot)
            norm_acts = (acts - self.mean_act) / (self.std_act + 1e-6)
            f_acts_orig = torch.relu(torch.nn.functional.linear(norm_acts - self.decoder_b, self.encoder_w, self.encoder_b))
            
            f_acts = f_acts_orig.clone()
            # 1. Ablate the Positives (Kill the features that push values up)
            for idx in self.ablate_ids:
                f_acts[:, idx] = 0.0 # Total silence is more effective for suppression
            
            # 2. Iterative Negative Correction
            # We want to ensure all predictions are below -0.5
            for _ in range(100):
                recon = torch.nn.functional.linear(f_acts, self.decoder_w, self.decoder_b)
                current_pred = self.mlp.output_head((recon * self.std_act) + self.mean_act)
                
                # Identify samples still ABOVE our negative target zone
                mask = (current_pred > -0.5).float()
                if mask.sum() == 0: break 
                
                # Apply nudge to Negative Drivers (steer_ids) to push pred DOWN
                for idx in self.steer_ids:
                    # Incrementally increase the power of negative features
                    f_acts[:, idx] += mask.squeeze() * (torch.abs(current_pred.squeeze()) / len(self.steer_ids) + 0.5)

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

def run_test(model, dataset):
    loader = DataLoader(dataset, batch_size=256)
    target_pos, pred_pos = 0, 0
    
    with torch.no_grad():
        for grid, coords, target in loader:
            grid, coords = grid.to(model.device), coords.to(model.device)
            pred = model(grid, coords)
            
            # Here we track POSITIVES because those are what we want to suppress
            target_pos += (target >= 0).sum().item()
            pred_pos += (pred >= 0).sum().item()
            
    # Success is defined by how many originally positive samples are now negative
    reduction = (1 - (pred_pos / target_pos)) * 100 if target_pos > 0 else 0
    
    print(f"\nFinal Statistics (Positive Suppression):")
    print(f"Target Positives: {target_pos}")
    print(f"Steered Positives: {pred_pos}")
    print(f"Suppression Rate: {reduction:.2f}%")

def comparative_inference_negative(original_mlp, steered_model, dataset, num_samples=15):
    print(f"\n{'#'*20} NEGATIVE ONLY INFERENCE {'#'*20}")
    print(f"{'Sample':<8} | {'Ground Truth':<15} | {'Original Pred':<15} | {'Steered Pred':<15} | {'Status'}")
    print("-" * 85)
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    device = steered_model.device
    count = 0
    with torch.no_grad():
        for grid, coords, target in loader:
            if count >= num_samples: break
            grid, coords = grid.to(device), coords.to(device)
            orig_pred = original_mlp(grid, coords).item()
            steered_pred = steered_model(grid, coords).item()
            
            # Status: Success if the steered prediction is now negative
            status = "FIXED ✅" if orig_pred > 0 and steered_pred <= 0 else "STABLE 🆗"
            if steered_pred > 0: status = "FAILED ❌"

            print(f"{count:<8} | {target.item():<15.4f} | {orig_pred:<15.4f} | {steered_pred:<15.4f} | {status}")
            count += 1

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
    print(f"\n--- ABLATING POSITIVES (FORCING NEGATIVE ONLY) ---")
    
    from mlp.mlp_definition import FinalSpatialMLP
    from dataset.data_generator import OneHotSpatialDataset
    
    mlp = FinalSpatialMLP()
    mlp.load_state_dict(torch.load("mlp/final_spatial_model.pth"))
    mlp.eval()
    
    sae_path = "sae/sae_weights.pth"
    dataset = OneHotSpatialDataset(pt_path="dataset/train_data.pt")

    # INVERTED LOGIC: 
    # To force negatives: Ablate Positive Drivers, Steer Negative Drivers
    pos_ids = get_causal_drivers(mlp, sae_path, k=10, find_positive=True)
    neg_ids = get_causal_drivers(mlp, sae_path, k=10, find_positive=False)
    
    print(f"\n--- CAUSAL DRIVERS FOUND ---")
    print(f"Ablating (Positives): {pos_ids}")
    print(f"Steering (Negatives): {neg_ids}")

    surgical_model = TotalNegativeSteering(mlp, sae_path, dataset, 
                                            ablate_ids=pos_ids, 
                                            steer_ids=neg_ids)

    run_test(surgical_model, dataset)

    comparative_inference_negative(mlp, surgical_model, dataset, num_samples=30)

    export_steering_configs(mlp, "sae/sae_weights.pth", dataset, output_path="steering_config_neg.pt")