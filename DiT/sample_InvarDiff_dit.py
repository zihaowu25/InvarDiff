import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import os
import random
import numpy as np
import time
import argparse
from diffusion import create_diffusion
from download import find_model
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import json

from models.dynamic_cache import DiT_models, DynamicDiT, SimilarityAnalyzer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def register_hooks(model):
    msa_features_step = {}
    mlp_features_step = {}
    hidden_state = [None]
    def create_hook(layer_idx):
        def hook(module, input, output):
            x, block_out = output
            msa_features_step[layer_idx] = block_out[0].detach()
            mlp_features_step[layer_idx] = block_out[1].detach()
            hidden_state[0] = x
        return hook
    hooks = []
    for i, block in enumerate(model.blocks):
        hook = block.register_forward_hook(create_hook(i))
        hooks.append(hook)
    
    return msa_features_step, mlp_features_step, hidden_state, hooks

class FeatureChangeAnalyzer:
    def __init__(self, num_layers):
        self.num_layers = num_layers
        self.prev_msa_features = [None] * num_layers
        self.prev_mlp_features = [None] * num_layers
        self.curr_msa_features = [None] * num_layers
        self.curr_mlp_features = [None] * num_layers
        self.prev_x = None
        self.curr_x = None

        self.start_up = [0, 0]
        self.x_state = False
        self.msa_state = [False] * num_layers # cache state
        self.mlp_state = [False] * num_layers

    def step_forward(self, x):
        if self.start_up[0] == 0:
            self.prev_x = x
            self.start_up[0] += 1
            return None
        elif self.start_up[0] == 1:
            self.curr_x = x
            self.start_up[0] += 1
            return None
        
        step_score = SimilarityAnalyzer.compute_rate(
            self.prev_x,
            self.curr_x,
            x
        )
        self.prev_x = self.curr_x
        self.curr_x = x
        return step_score

    def step_forward_correct(self, x, x_state, timestep_idx):
        if timestep_idx == 0:
            self.prev_x = x
            return None
        elif timestep_idx == 1:
            self.curr_x = x
            return None
        
        step_score = SimilarityAnalyzer.compute_rate(
            self.prev_x,
            self.curr_x,
            x
        )
        if not x_state[timestep_idx-1]:
            self.prev_x = self.curr_x
            self.curr_x = x
            self.x_state = False
        elif not self.x_state:
            self.prev_x = self.curr_x
            self.curr_x = x
            self.x_state = True
        return step_score
    
    def step(self, msa_features_dict, mlp_features_dict):
        if self.start_up[1] == 0:
            for i in range(self.num_layers):
                self.prev_msa_features[i] = msa_features_dict[i]
                self.prev_mlp_features[i] = mlp_features_dict[i]
            self.start_up[1] += 1
            return None, None
        elif self.start_up[1] == 1:
            for i in range(self.num_layers):
                self.curr_msa_features[i] = msa_features_dict[i]
                self.curr_mlp_features[i] = mlp_features_dict[i]
            self.start_up[1] += 1
            return None, None
        
        msa_diffs_list = []
        mlp_diffs_list = []

        for layer_idx in range(self.num_layers):
            msa_score = SimilarityAnalyzer.compute_rate( 
                self.prev_msa_features[layer_idx],
                self.curr_msa_features[layer_idx],
                msa_features_dict[layer_idx]
            )
            mlp_score = SimilarityAnalyzer.compute_rate(
                self.prev_mlp_features[layer_idx],
                self.curr_mlp_features[layer_idx],
                mlp_features_dict[layer_idx]
            )
            msa_diffs_list.append(msa_score)
            mlp_diffs_list.append(mlp_score)

            self.prev_msa_features[layer_idx] = self.curr_msa_features[layer_idx]
            self.prev_mlp_features[layer_idx] = self.curr_mlp_features[layer_idx]
            self.curr_msa_features[layer_idx] = msa_features_dict[layer_idx]
            self.curr_mlp_features[layer_idx] = mlp_features_dict[layer_idx]
            
        msa_diffs = torch.stack(msa_diffs_list)
        mlp_diffs = torch.stack(mlp_diffs_list)

        return msa_diffs, mlp_diffs
    
    def step_correct(
            self, 
            msa_features_dict, 
            mlp_features_dict, 
            msa_cache_state, 
            mlp_cache_state,
            timestep_idx # 0, 1, 2, 3...
        ):
        if timestep_idx == 0:
            for i in range(self.num_layers):
                self.prev_msa_features[i] = msa_features_dict[i]
                self.prev_mlp_features[i] = mlp_features_dict[i]
            return None, None
        elif timestep_idx == 1:
            for i in range(self.num_layers):
                self.curr_msa_features[i] = msa_features_dict[i]
                self.curr_mlp_features[i] = mlp_features_dict[i]
            return None, None

        msa_diffs_list = []
        mlp_diffs_list = []

        for layer_idx in range(self.num_layers):
            msa_vary_rate = SimilarityAnalyzer.compute_rate( 
                self.prev_msa_features[layer_idx],
                self.curr_msa_features[layer_idx],
                msa_features_dict[layer_idx]
            )
            mlp_vary_rate = SimilarityAnalyzer.compute_rate(
                self.prev_mlp_features[layer_idx],
                self.curr_mlp_features[layer_idx],
                mlp_features_dict[layer_idx]
            )
            msa_score = msa_vary_rate
            mlp_score = mlp_vary_rate
            msa_diffs_list.append(msa_score)
            mlp_diffs_list.append(mlp_score)

            if not msa_cache_state[timestep_idx-1][layer_idx]:
                self.prev_msa_features[layer_idx] = self.curr_msa_features[layer_idx]
                self.curr_msa_features[layer_idx] = msa_features_dict[layer_idx]
                self.msa_state[layer_idx] = False
            elif not self.msa_state[layer_idx]:
                self.prev_msa_features[layer_idx] = self.curr_msa_features[layer_idx]
                self.curr_msa_features[layer_idx] = msa_features_dict[layer_idx]
                self.msa_state[layer_idx] = True

            if not mlp_cache_state[timestep_idx-1][layer_idx]:
                self.prev_mlp_features[layer_idx] = self.curr_mlp_features[layer_idx]
                self.curr_mlp_features[layer_idx] = mlp_features_dict[layer_idx]
                self.mlp_state[layer_idx] = False
            elif not self.mlp_state[layer_idx]:
                self.prev_mlp_features[layer_idx] = self.curr_mlp_features[layer_idx]
                self.curr_mlp_features[layer_idx] = mlp_features_dict[layer_idx]
                self.mlp_state[layer_idx] = True

        msa_diffs = torch.stack(msa_diffs_list)
        mlp_diffs = torch.stack(mlp_diffs_list)

        return msa_diffs, mlp_diffs
    
    def clean(self):
        self.prev_msa_features = [None] * self.num_layers
        self.prev_mlp_features = [None] * self.num_layers
        self.curr_msa_features = [None] * self.num_layers
        self.curr_mlp_features = [None] * self.num_layers
        
        self.msa_state = [False] * self.num_layers
        self.mlp_state = [False] * self.num_layers
        self.prev_x = None
        self.curr_x = None
        self.start_up = [0, 0]

def threshold_ananlyse(
    model, diffusion, class_labels, input_size,
    step_thres=0.2,
    nonskip_rate=0,
    msa_thres=0.1, 
    mlp_thres=0.1, 
    num_analysis=10, # len(class_labels) normally not smaller than num_analysis
    ): 

    device = next(model.parameters()).device
    n = len(class_labels)
    z = torch.randn(n, 4, input_size, input_size, device=device)
    y = torch.tensor(class_labels, device=device)

    num_timesteps = diffusion.num_timesteps
    num_layers = len(model.blocks)
    print(f"Block numbers: {num_layers}")
    all_msa_diffs = []
    all_mlp_diffs = []
    step_rates = []

    model.eval()
    msa_features_dict, mlp_features_dict, hidden_state, hooks = register_hooks(model)

    with torch.no_grad():
        for run_idx in tqdm(range(num_analysis), desc="Threshold analysis"):
            class_idx = run_idx % n
            current_z = z[class_idx:class_idx+1]
            current_y = y[class_idx:class_idx+1]

            analyzer = FeatureChangeAnalyzer(num_layers)

            run_msa_diffs = torch.ones(num_timesteps, num_layers, device='cpu')
            run_mlp_diffs = torch.ones(num_timesteps, num_layers, device='cpu')
            run_step_scores = torch.ones(num_timesteps, device='cpu')

            sampler = diffusion.ddim_sample_loop_progressive(
                    model, current_z.shape, current_z,
                    clip_denoised=False, model_kwargs=dict(y=current_y),
                    progress=False, device=device
            )
            for timestep_idx, _ in enumerate(sampler):

                msa_diffs, mlp_diffs = analyzer.step(msa_features_dict, mlp_features_dict)
                step_score = analyzer.step_forward(hidden_state[0])
                if step_score is not None:
                    run_step_scores[timestep_idx-1] = step_score.cpu()
                
                if msa_diffs is not None:
                    run_msa_diffs[timestep_idx-1] = msa_diffs.cpu()
                    run_mlp_diffs[timestep_idx-1] = mlp_diffs.cpu()

            step_rates.append(run_step_scores)
            all_msa_diffs.append(run_msa_diffs)
            all_mlp_diffs.append(run_mlp_diffs)

    avg_step_rates = torch.stack(step_rates).mean(dim=0)
    avg_msa_diffs = torch.stack(all_msa_diffs).mean(dim=0)
    avg_mlp_diffs = torch.stack(all_mlp_diffs).mean(dim=0)

    num_nonskip = int(nonskip_rate * num_timesteps)
    if num_nonskip < 1:
        num_nonskip = 1
    
    step_threshold_value = torch.quantile(avg_step_rates[1:-1], step_thres)
    msa_threshold_value = torch.quantile(avg_msa_diffs[1:-1], msa_thres)
    mlp_threshold_value = torch.quantile(avg_mlp_diffs[1:-1], mlp_thres)

    step_cache_bool = avg_step_rates < step_threshold_value
    step_cache_bool[0:num_nonskip] = False
    step_cache_bool[-1] = False

    msa_cache_bool = avg_msa_diffs < msa_threshold_value
    mlp_cache_bool = avg_mlp_diffs < mlp_threshold_value

    msa_cache_bool[0:num_nonskip, :] = False
    mlp_cache_bool[0:num_nonskip, :] = False
    msa_cache_bool[-1, :] = False
    mlp_cache_bool[-1, :] = False

    with torch.no_grad():
        # Prevent the accumulation of consecutive errors
        all_msa_diffs = []
        all_mlp_diffs = []
        step_rates = []
        for run_idx in tqdm(range(num_analysis), desc=f"Cache correction"):
            class_idx = run_idx % n
            current_z = z[class_idx:class_idx+1]
            current_y = y[class_idx:class_idx+1]

            analyzer = FeatureChangeAnalyzer(num_layers)

            run_msa_diffs = torch.ones(num_timesteps, num_layers, device='cpu')
            run_mlp_diffs = torch.ones(num_timesteps, num_layers, device='cpu')
            run_step_scores = torch.ones(num_timesteps, device='cpu')

            sampler = diffusion.ddim_sample_loop_progressive(
                    model, current_z.shape, current_z,
                    clip_denoised=False, model_kwargs=dict(y=current_y),
                    progress=False, device=device
            )
            for timestep_idx, _ in enumerate(sampler):

                msa_diffs, mlp_diffs = analyzer.step_correct(
                    msa_features_dict, 
                    mlp_features_dict, 
                    msa_cache_bool,
                    mlp_cache_bool,
                    timestep_idx
                    )
                step_score = analyzer.step_forward_correct(hidden_state[0], step_cache_bool, timestep_idx)
                if step_score is not None:
                    run_step_scores[timestep_idx-1] = step_score.cpu()

                if msa_diffs is not None:
                    run_msa_diffs[timestep_idx-1] = msa_diffs.cpu()
                    run_mlp_diffs[timestep_idx-1] = mlp_diffs.cpu()
            
            step_rates.append(run_step_scores)
            all_msa_diffs.append(run_msa_diffs)
            all_mlp_diffs.append(run_mlp_diffs)

        avg_step_rates = torch.stack(step_rates).mean(dim=0)
        avg_msa_diffs = torch.stack(all_msa_diffs).mean(dim=0)
        avg_mlp_diffs = torch.stack(all_mlp_diffs).mean(dim=0)

        step_threshold_value = torch.quantile(avg_step_rates[1:-1], step_thres)
        msa_threshold_value = torch.quantile(avg_msa_diffs[1:-1], msa_thres)
        mlp_threshold_value = torch.quantile(avg_mlp_diffs[1:-1], mlp_thres)

        step_cache_bool = avg_step_rates < step_threshold_value
        msa_cache_bool = avg_msa_diffs < msa_threshold_value
        mlp_cache_bool = avg_mlp_diffs < mlp_threshold_value

    for hook in hooks:
        hook.remove()

    step_cache_bool[0:num_nonskip] = False
    step_cache_bool[-1] = False
    msa_cache_bool[0:num_nonskip, :] = False
    mlp_cache_bool[0:num_nonskip, :] = False
    msa_cache_bool[-1, :] = False
    mlp_cache_bool[-1, :] = False

    for t in range(num_timesteps):
        if step_cache_bool[t]:
            msa_cache_bool[t, :] = True
            mlp_cache_bool[t, :] = True
            if t+1 < num_timesteps:
                msa_cache_bool[t + 1, :] = False
                mlp_cache_bool[t + 1, :] = False

    compute_skip_ratio(msa_cache_bool, mlp_cache_bool, num_layers)

    return step_cache_bool, msa_cache_bool, mlp_cache_bool, avg_msa_diffs, avg_mlp_diffs

def compute_skip_ratio(msa_cache_bool, mlp_cache_bool, num_layers):
    num_timesteps = len(msa_cache_bool)
    total_modules = num_timesteps * num_layers * 2
    skipped_modules = 0
    msa_modules = 0
    mlp_modules = 0

    for step in range(num_timesteps):

        msa_indices = msa_cache_bool[step].nonzero(as_tuple=False).squeeze(-1)
        mlp_indices = mlp_cache_bool[step].nonzero(as_tuple=False).squeeze(-1)

        msa_modules += len(msa_indices)
        mlp_modules += len(mlp_indices)
    
    skipped_modules = msa_modules + mlp_modules
    
    skip_ratio = (skipped_modules / total_modules) * 100
    msa_skip_ratio = (2 * msa_modules / total_modules) * 100
    mlp_skip_ratio = (2 * mlp_modules / total_modules) * 100
    
    print(f"Total skip ratio: {skip_ratio:.2f}%\n"
          f"MSA skip ratio: {msa_skip_ratio:.2f}%\n"
          f"MLP skip ratio: {mlp_skip_ratio:.2f}%")

def save_cache_books(
    step_cache_book, 
    msa_cache_book, 
    mlp_cache_book,
    num_timesteps,
    nonskip_rate,
    step_thres,
    msa_thres,
    mlp_thres,
    cache_book_path="./cache_books"
):
    cache_books = {
        "step_cache_book": step_cache_book.tolist() if torch.is_tensor(step_cache_book) else step_cache_book,
        "msa_cache_book": msa_cache_book.tolist() if torch.is_tensor(msa_cache_book) else msa_cache_book,
        "mlp_cache_book": mlp_cache_book.tolist() if torch.is_tensor(mlp_cache_book) else mlp_cache_book
    }
    
    os.makedirs(cache_book_path, exist_ok=True)
    thres_str = (
        f"stp{num_timesteps}_n{nonskip_rate}_sth{step_thres}"
        f"_msa{msa_thres}_mlp{mlp_thres}"
    )
    cache_book_file = f"{cache_book_path}/cache_books_{thres_str}.json"
    
    with open(cache_book_file, "w") as f:
        json.dump(cache_books, f, indent=2)
    
    print(f"\nCache books saved: {cache_book_file}")
    return cache_book_file

def load_cache_books(cache_book_path, cache_book_file):
    with open(os.path.join(cache_book_path, cache_book_file), "r") as f:
        cache_books = json.load(f)
    
    step_cache_book = torch.tensor(cache_books["step_cache_book"], dtype=torch.bool)
    msa_cache_book = torch.tensor(cache_books["msa_cache_book"], dtype=torch.bool)
    mlp_cache_book = torch.tensor(cache_books["mlp_cache_book"], dtype=torch.bool)
    
    return step_cache_book, msa_cache_book, mlp_cache_book

def main(args):
    
    set_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_size = args.image_size//8
    num_timesteps = args.num_timesteps

    diffusion = create_diffusion(str(num_timesteps))
    base_DiT = DiT_models[args.model](
        input_size = input_size,
        num_classes = args.num_classes
    ).to(device)

    print("Loading model...")
    DiT_state_dict = find_model(args.dit_ckpt)
    base_DiT.load_state_dict(DiT_state_dict)
    base_DiT.eval()
    all_classes = list(range(args.num_classes))
    # class_labels = random.sample(all_classes, args.num_sample_classes)
    class_labels = [207, 992, 387, 37, 142, 979, 417, 279]

    # class_labels = [11, 96, 130, 285, 208, 388, 323, 14]
    # class_labels = [970, 972, 975, 977, 980, 937, 947, 919]
    # class_labels = [402, 579, 760, 541, 504, 850, 892, 522]
    # class_labels = [9, 84, 292, 291, 355, 105, 88, 309]
    measure_labels = random.sample(all_classes, args.num_analysis)

    if args.generate_cache_books:
        step_cache_book, msa_cache_book, mlp_cache_book, avg_msa_differences, avg_mlp_differences = threshold_ananlyse(
            base_DiT, diffusion, measure_labels, input_size,
            step_thres=args.step_thres,
            nonskip_rate=args.nonskip_rate,
            msa_thres=args.msa_thres, 
            mlp_thres=args.mlp_thres, 
            num_analysis=args.num_analysis
        )

        cache_book_file = save_cache_books(
            step_cache_book,
            msa_cache_book,
            mlp_cache_book,
            num_timesteps=num_timesteps,
            nonskip_rate=args.nonskip_rate,
            step_thres=args.step_thres,
            msa_thres=args.msa_thres,
            mlp_thres=args.mlp_thres,
            cache_book_path=args.cache_book_path
        )
    else:
        thres_str = (
            f"stp{num_timesteps}_n{args.nonskip_rate}_sth{args.step_thres}"
            f"_msa{args.msa_thres}_mlp{args.mlp_thres}"
        )
        cache_book_file = f"cache_books_{thres_str}.json"
        
        step_cache_book, msa_cache_book, mlp_cache_book = load_cache_books(
            cache_book_path=args.cache_book_path,
            cache_book_file=cache_book_file
        )
        print(f"Cache books loaded from: {os.path.join(args.cache_book_path, cache_book_file)}")

    Dynamic_DiT = DynamicDiT(base_DiT, msa_cache_book, mlp_cache_book, step_cache_book)
    Dynamic_DiT.eval()
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    n = len(class_labels)
    z = torch.randn(n, 4, input_size, input_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([args.num_classes]*n, device=device)
    y = torch.cat([y, y_null], 0)
    # If the cfg_scale parameter is included in model_kwargs, 
    # the cfg operation will be performed.
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # times = []
    # for _ in range(args.sample_times):
    #     start_time = time.time()
    #     samples = diffusion.ddim_sample_loop(
    #         base_DiT.forward_with_cfg, z.shape, z, 
    #         clip_denoised=False, 
    #         model_kwargs=model_kwargs,
    #         progress=True,
    #         device=device
    #     )
    #     times.append(time.time() - start_time)
    #     base_DiT.reset() #pass

    # if len(times) > 0:
    #     times = np.array(times[:])
    #     avg_ddim_time = np.mean(times)
    #     print("DDIM sampling time: {:.3f}±{:.3f} s".format(avg_ddim_time, np.std(times)))
    
    times = []
    for _ in range(args.sample_times):
        start_time = time.time()
        samples = diffusion.ddim_sample_loop(
            Dynamic_DiT.forward_with_cfg, z.shape, z, 
            clip_denoised=False, 
            model_kwargs=model_kwargs,
            progress=True,
            device=device
        )
        times.append(time.time() - start_time)
        Dynamic_DiT.reset_inference()
    
    if len(times) > 1:
        times = np.array(times[1:])
        avg_accel_time = np.mean(times)
        print("Accelerated sampling time: {:.3f}±{:.3f} s".format(avg_accel_time, np.std(times)))

        # speedup_ratio = avg_ddim_time / avg_accel_time
        # print(f"Speedup ratio: {speedup_ratio:.3f}x")
    
    samples, _ = samples.chunk(2, dim=0)
    samples = vae.decode(samples / 0.18215).sample

    os.makedirs("images", exist_ok=True)
    timestamp = time.strftime("%m%d_%H%M%S")
    save_name = f"images/NFE{num_timesteps}_CFG{args.cfg_scale}_th{args.step_thres:.2f}_{args.msa_thres:.2f}_{args.mlp_thres:.2f}_seed{args.seed}_{timestamp}.png"
    save_image(samples, save_name, nrow=8, normalize=True, value_range=(-1, 1))
    print(f"Samples save to {save_name}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample images using DynamicDiT')

    parser.add_argument("--model", type=str, default="DiT-XL/2", 
                        help="Name of the DiT model.")
    parser.add_argument("--image-size", type=int, default=256, 
                        help="Image size for the model.")
    parser.add_argument("--num-classes", type=int, default=1000, 
                        help="Number of classes in the dataset.")
    parser.add_argument("--num-timesteps", type=int, default=250, 
                        help="Number of diffusion timesteps for sampling.")
    parser.add_argument("--dit-ckpt", type=str, default="./pretrained_models/DiT-XL-2-256x256.pt", 
                        help="Path to the pre-trained DiT model checkpoint.")

    parser.add_argument('--num-sample-classes', type=int, default=10, 
                        help='Number of classes to sample, which also determines the number of sampled images.')
    parser.add_argument('--cfg-scale', type=float, default=4.0, 
                        help='Classifier-free guidance scale.')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility.')
    parser.add_argument('--sample-times', type=int, default=6, 
                        help='Number of times to run sampling for timing.')
    parser.add_argument('--vae', type=str, default='ema', choices=['mse', 'ema'], 
                        help='VAE model variant to use for decoding.')
    parser.add_argument('--generate-cache-books', action='store_true',
                    help='Whether to perform threshold analysis and save cache books.')
    parser.add_argument('--cache-book-path', type=str, default='./cache_books',
                    help='Path to save/load cache books.')
    
    parser.add_argument('--nonskip-rate', type=float, default=0, # little effect on DiT
                        help="Proportion of initial timesteps that are forced not to be skipped")
    parser.add_argument('--step-thres', type=float, default=0.61,
                        help="Quantile threshold for step skipping.")
    parser.add_argument('--msa-thres', type=float, default=0.2, 
                        help='Quantile threshold for MSA module skipping.')
    parser.add_argument('--mlp-thres', type=float, default=0.2, 
                        help='Quantile threshold for MLP module skipping.')
    parser.add_argument('--num-analysis', type=int, default=16, 
                        help='Number of sampling runs for stable feature analysis.')
    
    # fast(2.8x): nonskip-rate=0, step-thres=0.63, msa-thres = 0.22, mlp-thres = 0.22
    # slow(2.5x): nonskip-rate=0, step-thres=0.61, msa-thres = 0.2, mlp-thres = 0.2
    debug_args = [
        '--model', 'DiT-XL/2',
        '--image-size', '256',
        '--num-classes', '1000',
        '--num-timesteps', '50',
        '--dit-ckpt', './pretrained_models/DiT-XL-2-256x256.pt',
        '--num-sample-classes', '8',
        '--cfg-scale', '4.0',
        '--seed', '0',
        '--sample-times', '6',
        '--nonskip-rate', '0',
        '--step-thres', '0.61',
        '--msa-thres', '0.2',
        '--mlp-thres', '0.2',
        '--num-analysis', '16',
        # '--generate-cache-books', 
    ]
    
    args = parser.parse_args()
    main(args)
