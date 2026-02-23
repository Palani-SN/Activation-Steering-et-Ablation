import torch
from sae.sae_definition import SparseAutoencoder


def get_universal_vectors(sae, harvested_activations, tags):
    sae.eval()
    with torch.no_grad():
        # Get the sparse features (hidden layer of SAE)
        _, latent_features = sae(harvested_activations.cuda())
        latent_features = latent_features.cpu()

    # Mapping based on CONCEPT_MAP: pos_odd=0, pos_even=1, neg_odd=2, neg_even=3
    pos_mask = (tags == 0) | (tags == 1)
    neg_mask = (tags == 2) | (tags == 3)
    odd_mask = (tags == 0) | (tags == 2)
    even_mask = (tags == 1) | (tags == 3)

    # Calculate Basis Vectors in Feature Space
    v_sign = latent_features[pos_mask].mean(
        0) - latent_features[neg_mask].mean(0)
    v_parity = latent_features[odd_mask].mean(
        0) - latent_features[even_mask].mean(0)

    return v_sign, v_parity


if __name__ == "__main__":
    # 1. Load Data
    payload = torch.load("harvested_data.pt")
    activations = payload["activations"]
    tags = payload["labels"]

    # 2. Load SAE (Ensure parameters match your training config)
    sae = SparseAutoencoder(input_dim=512, dict_size=2048, k=20).cuda()
    sae.load_state_dict(torch.load("sae/universal_sae.pth"))

    # 3. Extract Vectors
    v_sign, v_parity = get_universal_vectors(sae, activations, tags)

    # 4. Researcher-Tier Metric: Orthogonality Check
    cosine_sim = torch.nn.functional.cosine_similarity(
        v_sign.unsqueeze(0), v_parity.unsqueeze(0))
    print(f"Sign-Parity Cosine Similarity: {cosine_sim.item():.4f}")
    print("Interpretation: Near 0.0 means the concepts are perfectly disentangled.")

    torch.save({"v_sign": v_sign, "v_parity": v_parity}, "steering_basis.pt")
