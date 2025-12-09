import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from tqdm import tqdm
import random
import time
from download import find_model

from models.dynamic_cache import DiT_models
from diffusion import create_diffusion
from sample_InvarDiff_dit import register_hooks, FeatureChangeAnalyzer, threshold_ananlyse

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def register_block_hooks(model):
    layer_features_step = {}
    def create_hook(layer_idx):
        def hook(module, input, output):
            layer_features_step[layer_idx] = output.clone().detach()
        return hook
    hooks = []
    for i, block in enumerate(model.blocks):
        hook = block.register_forward_hook(create_hook(i))
        hooks.append(hook)
    
    return layer_features_step, hooks

def visualize_msa_mlp_comparison(
    avg_msa_diffs, 
    avg_mlp_diffs, 
    save_dir="./visualizations", 
    filename_prefix="msa_mlp_comparison",
    num_analysis=10,
    class_label=None
):

    msa_matrix = avg_msa_diffs.numpy()
    mlp_matrix = avg_mlp_diffs.numpy()
    
    num_timesteps, num_layers = msa_matrix.shape
    
    # epsilon = 1e-9
    
    # log_msa_matrix = np.log2(msa_matrix + epsilon)
    # log_mlp_matrix = np.log2(mlp_matrix + epsilon)
    
    # msa_valid_data = log_msa_matrix[msa_matrix > 0]
    # mlp_valid_data = log_mlp_matrix[mlp_matrix > 0]
    
    # msa_vmin = np.percentile(msa_valid_data, 5) if len(msa_valid_data) > 0 else log_msa_matrix.min()
    # msa_vmax = np.percentile(msa_valid_data, 95) if len(msa_valid_data) > 0 else log_msa_matrix.max()
    
    # mlp_vmin = np.percentile(mlp_valid_data, 5) if len(mlp_valid_data) > 0 else log_mlp_matrix.min()
    # mlp_vmax = np.percentile(mlp_valid_data, 95) if len(mlp_valid_data) > 0 else log_mlp_matrix.max()

    msa_vmin = np.percentile(msa_matrix, 5)
    msa_vmax = np.percentile(msa_matrix, 95)
    mlp_vmin = np.percentile(mlp_matrix, 5)
    mlp_vmax = np.percentile(mlp_matrix, 95)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    if class_label is not None:
        title = f'Feature rates - Single Run (Class {class_label})'
    else:
        title = f'Average Feature rates  - {num_analysis} runs'
    # fig.suptitle(title, fontsize=16, fontweight='bold')

    X, Y = np.meshgrid(np.arange(num_timesteps), np.arange(num_layers))

    im1 = ax1.imshow(msa_matrix.T, cmap='viridis', aspect='auto', origin='lower', vmin=msa_vmin, vmax=msa_vmax)
    # contours1 = ax1.contour(X, Y, log_msa_matrix.T, levels=np.linspace(msa_vmin, msa_vmax, 15), 
    #                         colors='white', linewidths=0.7, alpha=0.4)
    # ax1.clabel(contours1, inline=True, fontsize=7, fmt='%.1f')
    ax1.set_title('MHSA Module rates', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Timestep', fontweight='bold')
    ax1.set_ylabel('Layer', fontweight='bold')
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('MHSA rate', rotation=270, labelpad=15)

    im2 = ax2.imshow(mlp_matrix.T, cmap='viridis', aspect='auto', origin='lower', vmin=mlp_vmin, vmax=mlp_vmax)
    # contours2 = ax2.contour(X, Y, log_mlp_matrix.T, levels=np.linspace(mlp_vmin, mlp_vmax, 15), 
    #                         colors='white', linewidths=0.7, alpha=0.4)
    # ax2.clabel(contours2, inline=True, fontsize=7, fmt='%.1f')
    ax2.set_title('FFN Module rates', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Timestep', fontweight='bold')
    ax2.set_ylabel('Layer', fontweight='bold')
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('FFN rate', rotation=270, labelpad=15)

    if num_timesteps <= 50:
        step = max(1, num_timesteps // 10)
        ticks = range(0, num_timesteps, step)
        ax1.set_xticks(ticks)
        ax2.set_xticks(ticks)
    
    if num_layers <= 30:
        step = max(1, num_layers // 8)
        ticks = range(0, num_layers, step)
        ax1.set_yticks(ticks)
        ax2.set_yticks(ticks)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{filename_prefix}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"MHSA/FFN comparison visualization saved to: {save_path}")

def run_adjacent_diff_analysis_and_visualize(
    model_name="DiT-XL/2",
    num_timesteps=50,
    num_analysis=10,
    seed=42
):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_size = 32  # 256x256 -> 32x32
    
    print(f"Loading model {model_name} on {device}...")
    model = DiT_models[model_name](input_size=input_size, num_classes=1000).to(device)

    checkpoint_path_model = "./pretrained_models/DiT-XL-2-256x256.pt"
    if os.path.exists(checkpoint_path_model):
        checkpoint = torch.load(checkpoint_path_model, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print(f"Warning: Pretrained model not found at {checkpoint_path_model}. Using random weights.")
    
    diffusion = create_diffusion(str(num_timesteps))
    model.eval()

    num_layers = len(model.blocks)
    all_classes = list(range(1000))
    class_labels = random.sample(all_classes, min(num_analysis, len(all_classes)))
    n = len(class_labels)
    z = torch.randn(n, 4, input_size, input_size, device=device)
    y = torch.tensor(class_labels, device=device)
    
    all_msa_diffs, all_mlp_diffs = [], []

    print("\nStarting adjacent feature difference analysis...")
    with torch.no_grad():
        for run_idx in tqdm(range(num_analysis), desc="Adjacent Diff Analysis"):
            class_idx = run_idx % n
            current_z, current_y = z[class_idx:class_idx+1], y[class_idx:class_idx+1]
            
            msa_features_dict, mlp_features_dict, _, hooks = register_hooks(model)
            analyzer = FeatureChangeAnalyzer(num_layers)
            run_msa_diffs = torch.ones(num_timesteps, num_layers, device='cpu')
            run_mlp_diffs = torch.ones(num_timesteps, num_layers, device='cpu')

            try:
                sampler = diffusion.ddim_sample_loop_progressive(
                    model, current_z.shape, current_z,
                    clip_denoised=False, model_kwargs=dict(y=current_y),
                    progress=False, device=device
                )
                for timestep_idx, _ in enumerate(sampler):
                    msa_diffs, mlp_diffs = analyzer.step(msa_features_dict, mlp_features_dict)
                    if msa_diffs is not None:
                        run_msa_diffs[timestep_idx-1] = msa_diffs.cpu()
                        run_mlp_diffs[timestep_idx-1] = mlp_diffs.cpu()
            finally:
                for hook in hooks: hook.remove()
            
            all_msa_diffs.append(run_msa_diffs)
            all_mlp_diffs.append(run_mlp_diffs)

    avg_msa_differences = torch.stack(all_msa_diffs).mean(dim=0)
    avg_mlp_differences = torch.stack(all_mlp_diffs).mean(dim=0)

    print("\nVisualizing MSA/MLP feature differences...")
    visualize_msa_mlp_comparison(
        avg_msa_diffs=avg_msa_differences,
        avg_mlp_diffs=avg_mlp_differences,
        save_dir="./visualizations",
        filename_prefix=f"adjacent_diff_NFE{num_timesteps}_seed{seed}",
        num_analysis=num_analysis
    )
    print("Analysis and visualization complete.")

def run_single_class_analysis_and_visualize(
    model_name="DiT-XL/2",
    num_timesteps=50,
    class_label=207,
    seed=42
):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_size = 32
    
    print(f"Loading model {model_name} on {device} for single class analysis...")
    model = DiT_models[model_name](input_size=input_size, num_classes=1000).to(device)
    
    checkpoint_path_model = "./pretrained_models/DiT-XL-2-256x256.pt"
    if os.path.exists(checkpoint_path_model):
        checkpoint = torch.load(checkpoint_path_model, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print(f"Warning: Pretrained model not found at {checkpoint_path_model}.")
    
    diffusion = create_diffusion(str(num_timesteps))
    model.eval()

    num_layers = len(model.blocks)
    z = torch.randn(1, 4, input_size, input_size, device=device)
    y = torch.tensor([class_label], device=device)

    print(f"\nStarting adjacent feature difference analysis for class {class_label}...")
    with torch.no_grad():
        msa_features_dict, mlp_features_dict, _, hooks = register_hooks(model)
        analyzer = FeatureChangeAnalyzer(num_layers)
        run_msa_diffs = torch.ones(num_timesteps, num_layers, device='cpu')
        run_mlp_diffs = torch.ones(num_timesteps, num_layers, device='cpu')

        try:
            sampler = diffusion.ddim_sample_loop_progressive(
                model, z.shape, z,
                clip_denoised=False, model_kwargs=dict(y=y),
                progress=True,
                device=device
            )
            for timestep_idx, _ in enumerate(sampler):
                msa_diffs, mlp_diffs = analyzer.step(msa_features_dict, mlp_features_dict)
                if msa_diffs is not None:
                    run_msa_diffs[timestep_idx-1] = msa_diffs.cpu()
                    run_mlp_diffs[timestep_idx-1] = mlp_diffs.cpu()
        finally:
            for hook in hooks: hook.remove()

    print("\nVisualizing MSA/MLP feature differences for single class...")
    visualize_msa_mlp_comparison(
        avg_msa_diffs=run_msa_diffs,
        avg_mlp_diffs=run_mlp_diffs,
        save_dir="./visualizations",
        filename_prefix=f"single_class_rate_cls{class_label}_NFE{num_timesteps}_seed{seed}",
        num_analysis=1,
        class_label=class_label
    )
    print("Single class analysis and visualization complete.")

def visualize_cache_analysis(
    step_cache_book, 
    msa_cache_book,
    mlp_cache_book,
    avg_msa_diffs,
    avg_mlp_diffs,
    step_thres, 
    msa_thres,
    mlp_thres,
    save_dir="./visualizations",
    filename_prefix="cache_analysis"
):
    """
    Visualizes the cache plan on top of the sensitive rate heatmap.

    Args:
        step_cache_book (torch.Tensor or list): Boolean tensor/list indicating cross-timestep cache.
        msa_cache_book (list): List of sets containing MSA layer indices to cache.
        mlp_cache_book (list): List of sets containing MLP layer indices to cache.
        avg_msa_diffs (torch.Tensor): 2D tensor of average MSA sensitivities.
        avg_mlp_diffs (torch.Tensor): 2D tensor of average MLP sensitivities.
        step_thres (float): Threshold value for step caching.
        msa_thres (float): Threshold value for MSA caching.
        mlp_thres (float): Threshold value for MLP caching.
        save_dir (str): Directory to save the visualization.
        filename_prefix (str): Prefix for the saved file.
    """
    print("\nVisualizing final cache plan and Sensitive rates analysis...")

    msa_matrix = avg_msa_diffs.numpy()
    mlp_matrix = avg_mlp_diffs.numpy()

    num_timesteps, num_layers = msa_matrix.shape

    if isinstance(step_cache_book, torch.Tensor):
        step_cache_array = step_cache_book.cpu().numpy()
    elif isinstance(step_cache_book, list):
        step_cache_array = np.array(step_cache_book)
    else:
        raise TypeError(f"Unsupported type for step_cache_book: {type(step_cache_book)}")

    if isinstance(msa_cache_book, torch.Tensor):
        msa_cache_grid = msa_cache_book.cpu().numpy().astype(int)
    elif isinstance(msa_cache_book, list):
        msa_cache_grid = np.zeros_like(msa_matrix)
        for t, layers in enumerate(msa_cache_book):
            for l in layers:
                msa_cache_grid[t, l] = 1
    else:
        raise TypeError(f"Unsupported type for msa_cache_book: {type(msa_cache_book)}")

    if isinstance(mlp_cache_book, torch.Tensor):
        mlp_cache_grid = mlp_cache_book.cpu().numpy().astype(int)
    elif isinstance(mlp_cache_book, list):
        mlp_cache_grid = np.zeros_like(mlp_matrix)
        for t, layers in enumerate(mlp_cache_book):
            for l in layers:
                mlp_cache_grid[t, l] = 1
    else:
        raise TypeError(f"Unsupported type for mlp_cache_book: {type(mlp_cache_book)}")

    epsilon = 1e-9
    log_msa_matrix = np.log2(msa_matrix + epsilon)
    log_mlp_matrix = np.log2(mlp_matrix + epsilon)

    msa_valid_data = log_msa_matrix[msa_matrix > 0]
    mlp_valid_data = log_mlp_matrix[mlp_matrix > 0]

    msa_vmin = np.percentile(msa_valid_data, 5) if len(msa_valid_data) > 0 else log_msa_matrix.min()
    msa_vmax = np.percentile(msa_valid_data, 95) if len(msa_valid_data) > 0 else log_msa_matrix.max()

    mlp_vmin = np.percentile(mlp_valid_data, 5) if len(mlp_valid_data) > 0 else log_mlp_matrix.min()
    mlp_vmax = np.percentile(mlp_valid_data, 95) if len(mlp_valid_data) > 0 else log_mlp_matrix.max()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
 
    title = (f'Cache Plan vs. Module rates\n'
             f'Step Threshold: {step_thres:.2f}, MHSA Threshold: {msa_thres:.2f}, FFN Threshold: {mlp_thres:.2f}')
    # fig.suptitle(title, fontsize=18, fontweight='bold')

    im1 = ax1.imshow(log_msa_matrix.T, cmap='viridis', aspect='auto', origin='lower', vmin=msa_vmin, vmax=msa_vmax)
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02)
    cbar1.set_label('MHSA rates (log₂)', rotation=270, labelpad=15)
    ax1.set_title('MHSA Cache Plan', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Timestep', fontweight='bold')
    ax1.set_ylabel('Layer', fontweight='bold')

    normal_cache_msa = []
    cross_step_cache_msa = []
    
    for t in range(num_timesteps):
        for l in range(num_layers):
            if msa_cache_grid[t, l] == 1:
                if step_cache_array[t]:
                    cross_step_cache_msa.append((t, l))
                else:
                    normal_cache_msa.append((t, l))
    
    if normal_cache_msa:
        normal_t, normal_l = zip(*normal_cache_msa)
        ax1.scatter(normal_t, normal_l, s=25, facecolors='none', edgecolors='white', 
                   linewidth=1.2, label='Module Cache', alpha=0.6)
    
    if cross_step_cache_msa:
        cross_t, cross_l = zip(*cross_step_cache_msa)
        ax1.scatter(cross_t, cross_l, s=25, facecolors='none', edgecolors='darkorange', 
                   linewidth=1.5, label='Cross-Step Cache', alpha=0.8)
    
    ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., 
              facecolor="#57595C", framealpha=0.9)

    im2 = ax2.imshow(log_mlp_matrix.T, cmap='viridis', aspect='auto', origin='lower', vmin=mlp_vmin, vmax=mlp_vmax)
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02)
    cbar2.set_label('FFN rates (log₂)', rotation=270, labelpad=15)
    ax2.set_title('FFN Cache Plan', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Timestep', fontweight='bold')
    ax2.set_ylabel('Layer', fontweight='bold')

    normal_cache_mlp = []
    cross_step_cache_mlp = []
    
    for t in range(num_timesteps):
        for l in range(num_layers):
            if mlp_cache_grid[t, l] == 1:
                if step_cache_array[t]:
                    cross_step_cache_mlp.append((t, l))
                else:
                    normal_cache_mlp.append((t, l))
    
    if normal_cache_mlp:
        normal_t, normal_l = zip(*normal_cache_mlp)
        ax2.scatter(normal_t, normal_l, s=25, facecolors='none', edgecolors='white', 
                   linewidth=1.2, label='Module Cache', alpha=0.6)
    
    if cross_step_cache_mlp:
        cross_t, cross_l = zip(*cross_step_cache_mlp)
        ax2.scatter(cross_t, cross_l, s=25, facecolors='none', edgecolors='darkorange', 
                   linewidth=1.5, label='Cross-Step Cache', alpha=0.8)
    
    ax2.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., 
              facecolor="#57595C", framealpha=0.9)

    for ax in [ax1, ax2]:
        if num_timesteps <= 50:
            step = max(1, num_timesteps // 10)
            ax.set_xticks(np.arange(0, num_timesteps, step))
        if num_layers <= 30:
            step = max(1, num_layers // 8)
            ax.set_yticks(np.arange(0, num_layers, step))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{filename_prefix}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Cache analysis visualization saved to: {save_path}")

def main():
    
    seed=42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_size = 256 // 8
    num_timesteps = 50

    print("Loading model...")
    diffusion = create_diffusion(str(num_timesteps))
    model = DiT_models['DiT-XL/2'](
        input_size=input_size,
        num_classes=1000
    ).to(device)

    state_dict = find_model("./pretrained_models/DiT-XL-2-256x256.pt")
    model.load_state_dict(state_dict)
    model.eval()

    num_analysis = 12
    all_classes = list(range(1000))
    measure_labels = random.sample(all_classes, num_analysis)
    print(f"Starting analysis with {num_analysis} random classes...")

    step_thres = 0.61
    msa_thres = 0.2
    mlp_thres = 0.2
    step_cache_book, msa_cache_book, mlp_cache_book, avg_msa_diffs, avg_mlp_diffs = threshold_ananlyse(
        model, diffusion, measure_labels, input_size,
        step_thres=step_thres,
        msa_thres=msa_thres,
        mlp_thres=mlp_thres,
        num_analysis=num_analysis,
    )

    msa_cache_book = [
    [layer_idx for layer_idx in range(msa_cache_book.shape[1]) if msa_cache_book[t, layer_idx]]
    for t in range(msa_cache_book.shape[0])
    ]
    mlp_cache_book = [
        [layer_idx for layer_idx in range(mlp_cache_book.shape[1]) if mlp_cache_book[t, layer_idx]]
        for t in range(mlp_cache_book.shape[0])
    ]
    timestamp = time.strftime("%m%d_%H%M%S")
    visualize_cache_analysis(
        step_cache_book=step_cache_book,
        msa_cache_book=msa_cache_book,
        mlp_cache_book=mlp_cache_book,
        avg_msa_diffs=avg_msa_diffs,
        avg_mlp_diffs=avg_mlp_diffs,
        step_thres=step_thres,
        msa_thres=msa_thres,
        mlp_thres=mlp_thres,
        save_dir="visualizations",
        filename_prefix=f"cache_books_NFE{num_timesteps}_seed{seed}_{timestamp}"
    )

    #################################################################################### msa and mlp visualize
    
    # run_adjacent_diff_analysis_and_visualize(
    #     model_name="DiT-XL/2",
    #     num_timesteps=50,
    #     num_analysis=10,
    #     seed=42
    # )
    # run_single_class_analysis_and_visualize(
    #     model_name="DiT-XL/2",
    #     num_timesteps=50,
    #     class_label=945,
    #     seed=42
    # )

if __name__ == "__main__":
    main()