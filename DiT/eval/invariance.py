import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import random
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dynamic_cache import DiT_models
from diffusion import create_diffusion
from download import find_model
from sample_InvarDiff_dit import register_hooks, FeatureChangeAnalyzer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def compute_rate_matrix_for_class(
    model, 
    diffusion, 
    class_label, 
    input_size, 
    z=None, 
    device="cuda"
):
    """
    Compute rate matrix for a single class.
    
    Args:
        model: DiT model
        diffusion: Diffusion model
        class_label: Class label
        input_size: Input size
        z: Initial noise (random if None)
        device: Device
    
    Returns:
        msa_diffs: MSA module rate matrix [num_timesteps, num_layers]
        mlp_diffs: MLP module rate matrix [num_timesteps, num_layers]
    """
    num_timesteps = diffusion.num_timesteps
    num_layers = len(model.blocks)
    
    if z is None:
        z = torch.randn(1, 4, input_size, input_size, device=device)
    
    y = torch.tensor([class_label], device=device)
    
    msa_features_dict, mlp_features_dict, _, hooks = register_hooks(model)
    analyzer = FeatureChangeAnalyzer(num_layers)
    
    run_msa_diffs = torch.ones(num_timesteps, num_layers, device='cpu')
    run_mlp_diffs = torch.ones(num_timesteps, num_layers, device='cpu')
    
    try:
        sampler = diffusion.ddim_sample_loop_progressive(
            model, z.shape, z,
            clip_denoised=False, 
            model_kwargs=dict(y=y),
            progress=False, 
            device=device
        )
        
        for timestep_idx, _ in enumerate(sampler):
            msa_diffs, mlp_diffs = analyzer.step(msa_features_dict, mlp_features_dict)
            
            if msa_diffs is not None:
                run_msa_diffs[timestep_idx - 1] = msa_diffs.cpu()
                run_mlp_diffs[timestep_idx - 1] = mlp_diffs.cpu()
    
    finally:
        for hook in hooks:
            hook.remove()
    
    return run_msa_diffs, run_mlp_diffs


def analyze_rate_matrix_invariance(
    model_name="DiT-XL/2",
    num_timesteps=50,
    num_reference_classes=100,
    num_test_classes=900,
    seed=42,
    save_dir="./visualizations"
):
    """
    Analyze rate matrix invariance.
    
    Args:
        model_name: Model name
        num_timesteps: Number of timesteps
        num_reference_classes: Number of reference classes (for averaging)
        num_test_classes: Number of test classes (for MSE computation)
        seed: Random seed
        save_dir: Save directory
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_grad_enabled(False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_size = 256 // 8
    
    print(f"Loading model {model_name} on {device}...")
    diffusion = create_diffusion(str(num_timesteps))
    model = DiT_models[model_name](
        input_size=input_size,
        num_classes=1000
    ).to(device)
    
    state_dict = find_model("../pretrained_models/DiT-XL-2-256x256.pt")
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"\nComputing reference rate matrices (classes 0-{num_reference_classes-1})...")
    reference_msa_matrices = []
    reference_mlp_matrices = []
    
    for class_idx in tqdm(range(num_reference_classes), desc="Reference classes"):
        msa_diffs, mlp_diffs = compute_rate_matrix_for_class(
            model, diffusion, class_idx, input_size, device=device
        )
        reference_msa_matrices.append(msa_diffs)
        reference_mlp_matrices.append(mlp_diffs)
    
    avg_msa_reference = torch.stack(reference_msa_matrices).mean(dim=0)
    avg_mlp_reference = torch.stack(reference_mlp_matrices).mean(dim=0)
    
    print(f"Reference MSA matrix shape: {avg_msa_reference.shape}")
    print(f"Reference MLP matrix shape: {avg_mlp_reference.shape}")
    
    print(f"\nComputing MSE for test classes ({num_reference_classes}-999)...")
    test_class_labels = list(range(num_reference_classes, 1000))
    
    msa_mse_values = []
    mlp_mse_values = []
    
    for class_idx in tqdm(test_class_labels, desc="Test classes"):
        msa_diffs, mlp_diffs = compute_rate_matrix_for_class(
            model, diffusion, class_idx, input_size, device=device
        )
        
        msa_mse = torch.mean((msa_diffs - avg_msa_reference) ** 2).item()
        mlp_mse = torch.mean((mlp_diffs - avg_mlp_reference) ** 2).item()
        
        msa_mse_values.append(msa_mse)
        mlp_mse_values.append(mlp_mse)
    
    print("\nVisualizing invariance analysis...")
    visualize_invariance_curves(
        test_class_labels,
        msa_mse_values,
        mlp_mse_values,
        num_reference_classes,
        num_timesteps,
        seed,
        save_dir
    )
    
    print("\n" + "="*60)
    print("Invariance Analysis Summary")
    print("="*60)
    print(f"Reference classes: 0-{num_reference_classes-1}")
    print(f"Test classes: {num_reference_classes}-999")
    print(f"\nMSA MSE Statistics:")
    print(f"  Mean: {np.mean(msa_mse_values):.6f}")
    print(f"  Std:  {np.std(msa_mse_values):.6f}")
    print(f"  Min:  {np.min(msa_mse_values):.6f}")
    print(f"  Max:  {np.max(msa_mse_values):.6f}")
    print(f"\nMLP MSE Statistics:")
    print(f"  Mean: {np.mean(mlp_mse_values):.6f}")
    print(f"  Std:  {np.std(mlp_mse_values):.6f}")
    print(f"  Min:  {np.min(mlp_mse_values):.6f}")
    print(f"  Max:  {np.max(mlp_mse_values):.6f}")
    print("="*60)
    
    return test_class_labels, msa_mse_values, mlp_mse_values


def visualize_invariance_curves(
    class_labels,
    msa_mse_values,
    mlp_mse_values,
    num_reference_classes,
    num_timesteps,
    seed,
    save_dir="./visualizations"
):
    """
    Visualize invariance analysis curves.
    
    Args:
        class_labels: Test class labels list
        msa_mse_values: MSA module MSE values list
        mlp_mse_values: MLP module MSE values list
        num_reference_classes: Number of reference classes
        num_timesteps: Number of timesteps
        seed: Random seed
        save_dir: Save directory
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    title = (f'Rate Matrix Invariance Analysis\n'
             f'Reference: Classes 0-{num_reference_classes-1} (Average) | '
             f'Test: Classes {num_reference_classes}-999')
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    ax1.plot(class_labels, msa_mse_values, 
             color='#1f77b4', linewidth=1.5, alpha=0.7)
    ax1.fill_between(class_labels, msa_mse_values, 
                      alpha=0.3, color='#1f77b4')
    ax1.axhline(y=np.mean(msa_mse_values), 
                color='red', linestyle='--', linewidth=2, 
                label=f'Mean MSE: {np.mean(msa_mse_values):.6f}')
    
    ax1.set_title('MSA Module - Rate Matrix MSE', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Class Label', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MSE', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=11)
    
    if len(class_labels) > 100:
        step = len(class_labels) // 10
        xticks = class_labels[::step]
        ax1.set_xticks(xticks)
    
    ax2.plot(class_labels, mlp_mse_values, 
             color='#ff7f0e', linewidth=1.5, alpha=0.7)
    ax2.fill_between(class_labels, mlp_mse_values, 
                      alpha=0.3, color='#ff7f0e')
    ax2.axhline(y=np.mean(mlp_mse_values), 
                color='red', linestyle='--', linewidth=2, 
                label=f'Mean MSE: {np.mean(mlp_mse_values):.6f}')
    
    ax2.set_title('MLP Module - Rate Matrix MSE', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Class Label', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MSE', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=11)
    
    if len(class_labels) > 100:
        ax2.set_xticks(xticks)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir, 
        f"invariance_analysis_ref{num_reference_classes}_NFE{num_timesteps}_seed{seed}.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nInvariance analysis visualization saved to: {save_path}")


def main():
    analyze_rate_matrix_invariance(
        model_name="DiT-XL/2",
        num_timesteps=50,
        num_reference_classes=100,
        num_test_classes=900,
        seed=42,
        save_dir="./visualizations"
    )

if __name__ == "__main__":
    main()