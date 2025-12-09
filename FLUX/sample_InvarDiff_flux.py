from typing import Any, Dict, Optional, Tuple, Union
from diffusers import DiffusionPipeline, FluxPipeline
from dynamic_flux import DynamicFluxTransformer2DModel, flux_sample_loop_progressive
import torch
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
import numpy as np
import os
from tqdm import tqdm
import json
import time
from PIL import Image
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def compute_rate(x_prev:torch.Tensor, x:torch.Tensor, x_post:torch.Tensor) -> torch.Tensor:
    diff_prev = x - x_prev + 1e-8
    diff_post = x_post - x + 1e-8
    slope = diff_post.norm(p=1)/diff_prev.norm(p=1)
    return slope

def compute_norm(x_prev:torch.Tensor, x:torch.Tensor) -> torch.Tensor:
    diff = x - x_prev + 1e-8
    return diff.norm(p=1)

def register_hooks(model):
    Transformer_blocks = {"attn":{}, "context_attn":{}, "ip_attn":{}, "ff":{}, "context_ff":{}}
    SingleTransformer_blocks = {"attn":{}, "mlp":{}}

    def create_hook_Tsfm(index_block):
        def hook(module, input, output):
            encoder_hidden_states, hidden_states, block_outputs = output
            Transformer_blocks["attn"][index_block] = block_outputs["attn"]
            Transformer_blocks["context_attn"][index_block] = block_outputs["context_attn"]
            Transformer_blocks["ip_attn"][index_block] = block_outputs["ip_attn"]
            Transformer_blocks["ff"][index_block] = block_outputs["ff"]
            Transformer_blocks["context_ff"][index_block] = block_outputs["context_ff"]
        return hook
        
    def create_hook_SgTsfm(index_block):
        def hook(module, input, output):
            hidden_states, block_outputs = output
            SingleTransformer_blocks["attn"][index_block] = block_outputs["attn"]
            SingleTransformer_blocks["mlp"][index_block] = block_outputs["mlp"]
        return hook
    
    hooks = []
    for i, block in enumerate(model.transformer_blocks):
        hook = block.register_forward_hook(create_hook_Tsfm(i))
        hooks.append(hook)
    for i, block in enumerate(model.single_transformer_blocks):
        hook = block.register_forward_hook(create_hook_SgTsfm(i))
        hooks.append(hook)
    return Transformer_blocks, SingleTransformer_blocks, hooks

class FeatureChangeAnalyzer:
    def __init__(self, num_transformer_layers, num_single_layers):
        self.num_transformer_layers = num_transformer_layers
        self.num_single_layers = num_single_layers
        self.Transformer_state = {
            "attn":[False]*self.num_transformer_layers,
            "context_attn":[False]*self.num_transformer_layers,
            "ip_attn":[False]*self.num_transformer_layers,
            "ff":[False]*self.num_transformer_layers,
            "context_ff":[False]*self.num_transformer_layers
        }
        self.SingleTransformer_state = {
            "attn":[False]*self.num_single_layers,
            "mlp":[False]*self.num_single_layers
        }
        self.start_cnt = [0, 0]
    
    def step_forward(self, hidden_states):
        if self.start_cnt[0] == 0:
            self.prev_hidden_states = hidden_states
            self.start_cnt[0] += 1
            return None
        elif self.start_cnt[0] == 1:
            self.current_hidden_states = hidden_states
            self.start_cnt[0] += 1
            return None
        
        step_score = compute_rate(
            self.prev_hidden_states, 
            self.current_hidden_states, 
            hidden_states
        )
        self.prev_hidden_states = self.current_hidden_states
        self.current_hidden_states = hidden_states
        return step_score

    def step(self, transformer_features_dict, single_features_dict):
        transformer_features = {}
        for key in ["attn", "context_attn", "ip_attn", "ff", "context_ff"]:
            feature_list = []
            for i in range(self.num_transformer_layers):
                x = transformer_features_dict[key][i]
                feature_list.append(x)
                transformer_features[key] = feature_list

        singleTransformer_features = {
            key: [single_features_dict[key][i] for i in range(self.num_single_layers)]
            for key in ["attn", "mlp"]
        }
        if self.start_cnt[1] == 0:
            self.current_Transformer = transformer_features
            self.current_SingleTransformer = singleTransformer_features
            self.start_cnt[1] += 1
            return None, None
        elif self.start_cnt[1] == 1:
            self.prev_Transformer_norm = {}
            for key in transformer_features.keys():
                norms = []
                for block_idx in range(self.num_transformer_layers):
                    if transformer_features[key][block_idx] is not None:
                        norm = compute_norm(
                            self.current_Transformer[key][block_idx],
                            transformer_features[key][block_idx]
                        )
                        norms.append(norm.item())
                    else: # ip-adapter = None
                        norms.append(float('nan'))

                self.prev_Transformer_norm[key] = torch.tensor(norms)
            
            self.prev_SingleTransformer_norm = {}
            for key in singleTransformer_features.keys():
                norms = []
                for block_idx in range(self.num_single_layers):
                    norm = compute_norm(
                        self.current_SingleTransformer[key][block_idx],
                        singleTransformer_features[key][block_idx]
                    )
                    norms.append(norm.item())
                self.prev_SingleTransformer_norm[key] = torch.tensor(norms)

            self.current_Transformer = transformer_features
            self.current_SingleTransformer = singleTransformer_features
            self.start_cnt[1] += 1

            return None, None
        
        transformer_rates = {}
        for key in transformer_features.keys():
            norms = []
            rates = []
            for block_idx in range(self.num_transformer_layers):
                if transformer_features[key][block_idx] is not None:
                    curr_norm = compute_norm(
                        self.current_Transformer[key][block_idx],
                        transformer_features[key][block_idx]
                    )
                    rate = curr_norm / self.prev_Transformer_norm[key][block_idx]
                    rates.append(rate.item())
                    norms.append(curr_norm.item())
                else: # ip-adapter = None
                    rates.append(float('nan'))
                    norms.append(float('nan'))

            transformer_rates[key] = torch.tensor(rates)
            self.prev_Transformer_norm[key] = torch.tensor(norms)
        
        singleTransformer_rates = {}
        for key in singleTransformer_features.keys():
            norms = []
            rates = []
            for block_idx in range(self.num_single_layers):
                curr_norm = compute_norm(
                    self.current_SingleTransformer[key][block_idx],
                    singleTransformer_features[key][block_idx]
                ) 
                rate = curr_norm / self.prev_SingleTransformer_norm[key][block_idx]
                rates.append(rate.item())
                norms.append(curr_norm.item())

            singleTransformer_rates[key] = torch.tensor(rates)
            self.prev_SingleTransformer_norm[key] = torch.tensor(norms)
        
        self.current_Transformer = transformer_features
        self.current_SingleTransformer = singleTransformer_features

        return transformer_rates, singleTransformer_rates

    def step_correct(self,
        transformer_features_dict,
        single_features_dict,
        transformer_cache_state,
        single_cache_state,
        timestep_idx
    ):
        transformer_features = {}
        for key in ["attn", "context_attn", "ip_attn", "ff", "context_ff"]:
            feature_list = []
            for i in range(self.num_transformer_layers):
                x = transformer_features_dict[key][i]
                feature_list.append(x)
                transformer_features[key] = feature_list

        singleTransformer_features = {
            key: [single_features_dict[key][i] for i in range(self.num_single_layers)]
            for key in ["attn", "mlp"]
        }
        if timestep_idx == 0:
            self.current_Transformer = transformer_features
            self.current_SingleTransformer = singleTransformer_features
            return None, None
        elif timestep_idx == 1:
            self.prev_Transformer_norm = {}
            for key in transformer_features.keys():
                norms = []
                for block_idx in range(self.num_transformer_layers):
                    if transformer_features[key][block_idx] is not None:
                        norm = compute_norm(
                            self.current_Transformer[key][block_idx],
                            transformer_features[key][block_idx]
                        )
                        norms.append(norm.item())
                    else: # ip-adapter = None
                        norms.append(float('nan'))

                self.prev_Transformer_norm[key] = torch.tensor(norms)
            
            self.prev_SingleTransformer_norm = {}
            for key in singleTransformer_features.keys():
                norms = []
                for block_idx in range(self.num_single_layers):
                    norm = compute_norm(
                        self.current_SingleTransformer[key][block_idx],
                        singleTransformer_features[key][block_idx]
                    )
                    norms.append(norm.item())
                self.prev_SingleTransformer_norm[key] = torch.tensor(norms)

            self.current_Transformer = transformer_features
            self.current_SingleTransformer = singleTransformer_features

            return None, None
        
        transformer_rates = {}
        for key in transformer_features.keys():
            rates = []
            for block_idx in range(self.num_transformer_layers):
                if transformer_features[key][block_idx] is not None:
                    curr_norm = compute_norm(
                        self.current_Transformer[key][block_idx],
                        transformer_features[key][block_idx]
                    )
                    rate = curr_norm / self.prev_Transformer_norm[key][block_idx]
                    rates.append(rate.item())
                else: # ip-adapter = None
                    rates.append(float('nan'))
                    curr_norm = float('nan')

                if not transformer_cache_state[key][timestep_idx-1][block_idx]:
                    self.prev_Transformer_norm[key][block_idx] = curr_norm
                    self.current_Transformer[key][block_idx] = transformer_features[key][block_idx]
                    self.Transformer_state[key][block_idx] = False
                elif not self.Transformer_state[key][block_idx]:
                    self.prev_Transformer_norm[key][block_idx] = curr_norm
                    self.current_Transformer[key][block_idx] = transformer_features[key][block_idx]
                    self.Transformer_state[key][block_idx] = True

            transformer_rates[key] = torch.tensor(rates)
        
        singleTransformer_rates = {}
        for key in singleTransformer_features.keys():
            rates = []
            for block_idx in range(self.num_single_layers):
                curr_norm = compute_norm(
                    self.current_SingleTransformer[key][block_idx],
                    singleTransformer_features[key][block_idx]
                )
                rate = curr_norm / self.prev_SingleTransformer_norm[key][block_idx]
                rates.append(rate.item())

                if not single_cache_state[key][timestep_idx-1][block_idx]:
                    self.prev_SingleTransformer_norm[key][block_idx] = curr_norm
                    self.current_SingleTransformer[key][block_idx] = singleTransformer_features[key][block_idx]
                    self.SingleTransformer_state[key][block_idx] = False
                elif not self.SingleTransformer_state[key][block_idx]:
                    self.prev_SingleTransformer_norm[key][block_idx] = curr_norm
                    self.current_SingleTransformer[key][block_idx] = singleTransformer_features[key][block_idx]
                    self.SingleTransformer_state[key][block_idx] = True

            singleTransformer_rates[key] = torch.tensor(rates)

        return transformer_rates, singleTransformer_rates

    def reset(self):
        pass

def threshold_analyse(
    model, pipe, measure_prompts,
    nonskip_rate = 0.1,
    step_thres = 0.5,
    attn_thres = 0.5, # = context_attn_thres, ip_attn_thres
    ff_thres = 0.5,
    context_ff_thres = 0.5,
    Single_attn_thres = 0.5, # FluxSingleTransformerBlock
    Single_mlp_thres = 0.5,
    seed = 42,
):
    device = next(model.parameters()).device
    num_prompts = len(measure_prompts)
    num_timesteps = model.num_timesteps
    num_transformer_layers = model.num_layers
    num_single_layers = model.num_single_layers
    num_nonskip = int(nonskip_rate * num_timesteps)
    if num_nonskip < 1:
        num_nonskip = 1
    print(f"Transformer blocks numbers: {num_transformer_layers}\n"
          f"Single Transformer blocks numbers: {num_single_layers}")
    
    all_transformer_rates = {
        "attn": [],
        "context_attn": [],
        "ip_attn": [],
        "ff": [],
        "context_ff": []
    }
    all_single_transformer_rates = {
        "attn": [],
        "mlp": []
    }
    step_rates = []
    model.eval()
    with torch.no_grad():
        for prompt_idx, prompt in enumerate(tqdm(measure_prompts, desc="Threshold analysis")):

            transformer_features, single_features, hooks = register_hooks(model)
            analyzer = FeatureChangeAnalyzer(num_transformer_layers, num_single_layers)
            run_transformer_rates = {
                key: torch.ones(num_timesteps, num_transformer_layers, device='cpu')
                for key in all_transformer_rates.keys()
            }
            run_single_rates = {
                key: torch.ones(num_timesteps, num_single_layers, device='cpu')
                for key in all_single_transformer_rates.keys()
            }
            run_step_scores = torch.ones(num_timesteps, device='cpu')
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)

            sampler = flux_sample_loop_progressive(
                pipe,
                prompt=prompt,
                num_inference_steps=num_timesteps,
                generator=gen,
                output_type="latent",
                max_sequence_length=512,
            )
            
            for step_data in sampler:
                if 'final_image' in step_data:
                    continue
                timestep_idx = step_data['timestep_idx']
                transformer_rates, single_rates = analyzer.step(
                    transformer_features,
                    single_features
                )
                step_score = analyzer.step_forward(model.hidden_states_cache)
                if step_score is not None:
                    run_step_scores[timestep_idx-1] = step_score.cpu()

                if transformer_rates is not None and single_rates is not None:
                    for key in run_transformer_rates.keys():
                        run_transformer_rates[key][timestep_idx-1] = transformer_rates[key].cpu()
                    for key in run_single_rates.keys():
                        run_single_rates[key][timestep_idx-1] = single_rates[key].cpu()

            step_rates.append(run_step_scores)
            for key in all_transformer_rates.keys():
                all_transformer_rates[key].append(run_transformer_rates[key])
            for key in all_single_transformer_rates.keys():
                all_single_transformer_rates[key].append(run_single_rates[key])
            
            del transformer_features, single_features, transformer_rates, single_rates
            torch.cuda.empty_cache()
            for hook in hooks:
                hook.remove()
    
    avg_step_rates = torch.stack(step_rates).mean(dim=0)
    avg_transformer_rates = {
        key: torch.stack(all_transformer_rates[key]).mean(dim=0)
        for key in all_transformer_rates.keys()
    }
    avg_single_transformer_rates = {
        key: torch.stack(all_single_transformer_rates[key]).mean(dim=0)
        for key in all_single_transformer_rates.keys()
    }

    step_threshold_value = torch.quantile(avg_step_rates[1:-1], step_thres)
    step_cache_bool = avg_step_rates < step_threshold_value
    step_cache_bool[0:num_nonskip] = False
    step_cache_bool[-1] = False

    transformer_cache_book = {}
    for module_name, threshold in [
        ("attn", attn_thres),
        ("context_attn",attn_thres),
        ("ff", ff_thres),
        ("context_ff", context_ff_thres)
    ]:
        threshold_value = torch.quantile(avg_transformer_rates[module_name][1:-1], threshold)
        cache_bool = avg_transformer_rates[module_name] < threshold_value
        cache_bool[0, :] = False
        cache_bool[-1, :] = False
        transformer_cache_book[module_name] = cache_bool

    transformer_cache_book["ip_attn"] = transformer_cache_book["attn"]
    single_transformer_cache_book = {}
    for module_name, threshold in [
        ("attn", Single_attn_thres),
        ("mlp", Single_mlp_thres)
    ]:
        threshold_value = torch.quantile(avg_single_transformer_rates[module_name][1:-1], threshold)
        cache_bool = avg_single_transformer_rates[module_name] < threshold_value
        cache_bool[0, :] = False
        cache_bool[-1, :] = False
        single_transformer_cache_book[module_name] = cache_bool

    # cache correction stage
    all_transformer_rates = {
        "attn": [],
        "context_attn": [],
        "ip_attn": [],
        "ff": [],
        "context_ff": []
    }
    all_single_transformer_rates = {
        "attn": [],
        "mlp": []
    }
    with torch.no_grad():
        for prompt_idx, prompt in enumerate(tqdm(measure_prompts, desc="cache correction")):

            transformer_features, single_features, hooks = register_hooks(model)
            analyzer = FeatureChangeAnalyzer(num_transformer_layers, num_single_layers)
            run_transformer_rates = {
                key: torch.ones(num_timesteps, num_transformer_layers, device='cpu')
                for key in all_transformer_rates.keys()
            }
            run_single_rates = {
                key: torch.ones(num_timesteps, num_single_layers, device='cpu')
                for key in all_single_transformer_rates.keys()
            }
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)

            sampler = flux_sample_loop_progressive(
                pipe,
                prompt=prompt,
                num_inference_steps=num_timesteps,
                generator=gen,
                output_type="latent",
                max_sequence_length=512,
            )
            
            for step_data in sampler:
                if 'final_image' in step_data:
                    continue
                timestep_idx = step_data['timestep_idx']
                transformer_rates, single_rates = analyzer.step_correct(
                    transformer_features,
                    single_features,
                    transformer_cache_book,
                    single_transformer_cache_book,
                    timestep_idx
                )
                if transformer_rates is not None and single_rates is not None:
                    for key in run_transformer_rates.keys():
                        run_transformer_rates[key][timestep_idx-1] = transformer_rates[key].cpu()
                    for key in run_single_rates.keys():
                        run_single_rates[key][timestep_idx-1] = single_rates[key].cpu()

            for key in all_transformer_rates.keys():
                all_transformer_rates[key].append(run_transformer_rates[key])
            for key in all_single_transformer_rates.keys():
                all_single_transformer_rates[key].append(run_single_rates[key])
            
            del transformer_features, single_features, transformer_rates, single_rates
            torch.cuda.empty_cache()
            for hook in hooks:
                hook.remove()
    
    avg_transformer_rates = {
        key: torch.stack(all_transformer_rates[key]).mean(dim=0)
        for key in all_transformer_rates.keys()
    }
    avg_single_transformer_rates = {
        key: torch.stack(all_single_transformer_rates[key]).mean(dim=0)
        for key in all_single_transformer_rates.keys()
    }
    transformer_cache_book = {}
    for module_name, threshold in [
        ("attn", attn_thres),
        ("context_attn",attn_thres),
        ("ff", ff_thres),
        ("context_ff", context_ff_thres)
    ]:
        threshold_value = torch.quantile(avg_transformer_rates[module_name][1:-1], threshold)
        cache_bool = avg_transformer_rates[module_name] < threshold_value
        cache_bool[0:num_nonskip, :] = False
        cache_bool[-1, :] = False
        transformer_cache_book[module_name] = cache_bool
    
    transformer_cache_book["ip_attn"] = transformer_cache_book["attn"]
    single_transformer_cache_book = {}
    for module_name, threshold in [
        ("attn", Single_attn_thres),
        ("mlp", Single_mlp_thres)
    ]:
        threshold_value = torch.quantile(avg_single_transformer_rates[module_name][1:-1], threshold)
        cache_bool = avg_single_transformer_rates[module_name] < threshold_value
        cache_bool[0:num_nonskip, :] = False
        cache_bool[-1, :] = False
        single_transformer_cache_book[module_name] = cache_bool
    
    for t in range(num_timesteps):
        if step_cache_bool[t]:
            for key in transformer_cache_book.keys():
                transformer_cache_book[key][t, :] = True
                if t+1 < num_timesteps:
                    transformer_cache_book[key][t+1, :] = False
            for key in single_transformer_cache_book.keys():
                single_transformer_cache_book[key][t, :] = True
                if t+1 < num_timesteps:
                    single_transformer_cache_book[key][t+1, :] = False

    step_cache_bool = step_cache_bool.tolist()
    for key in transformer_cache_book.keys():
        transformer_cache_book[key] = transformer_cache_book[key].tolist()
    for key in single_transformer_cache_book.keys():
        single_transformer_cache_book[key] = single_transformer_cache_book[key].tolist()
    
    print_skip_ratio(
        transformer_cache_book, 
        single_transformer_cache_book, 
        num_transformer_layers, 
        num_single_layers
    )
    return step_cache_bool, transformer_cache_book, single_transformer_cache_book, avg_transformer_rates, avg_single_transformer_rates

def print_skip_ratio(
    transformer_cache_book, 
    single_transformer_cache_book, 
    num_transformer_layers, 
    num_single_layers
    ):
    num_timesteps = len(transformer_cache_book["attn"])

    total_transformer_modules = num_timesteps * num_transformer_layers * 5
    skipped_transformer = 0
    module_skip_counts = {key: 0 for key in transformer_cache_book.keys()}
    
    for step in range(num_timesteps):
        for block_idx in range(num_transformer_layers):
            if (
                transformer_cache_book["attn"][step][block_idx]
                and transformer_cache_book["context_attn"][step][block_idx]
            ):
                module_skip_counts["attn"] += 1
                module_skip_counts["context_attn"] += 1
                module_skip_counts["ip_attn"] += 1
                skipped_transformer += 3
            if transformer_cache_book["ff"][step][block_idx]:
                module_skip_counts["ff"] += 1
                skipped_transformer += 1
            if transformer_cache_book["context_ff"][step][block_idx]:
                module_skip_counts["context_ff"] += 1
                skipped_transformer += 1

    total_single_transformer_modules = num_timesteps * num_single_layers * 2
    skipped_single = 0
    single_module_skip_counts = {key: 0 for key in single_transformer_cache_book.keys()}
    
    for module_name in single_transformer_cache_book.keys():
        for step in range(num_timesteps):
            skipped_count = sum(single_transformer_cache_book[module_name][step])
            single_module_skip_counts[module_name] += skipped_count
            skipped_single += skipped_count
    
    total_modules = total_transformer_modules + total_single_transformer_modules
    total_skipped = skipped_transformer + skipped_single
    transformer_skip_ratio = (skipped_transformer / total_transformer_modules) * 100
    single_transformer_skip_ratio = (skipped_single / total_single_transformer_modules) * 100
    skip_ratio = (total_skipped / total_modules) * 100
    
    print(f"Total skip ratio: {skip_ratio:.2f}%")
    print(f"\nTotal transformer blocks skip ratio:{transformer_skip_ratio:.2f}%")
    print(f"Transformer blocks skip details:")
    for module_name, count in module_skip_counts.items():
        ratio = (count / (num_timesteps * num_transformer_layers)) * 100
        print(f"  {module_name:15s}: {ratio:5.2f}%")
    
    print(f"\nTotal single transformer blocks skip ratio:{single_transformer_skip_ratio:.2f}%")
    print(f"Single Transformer blocks skip details:")
    for module_name, count in single_module_skip_counts.items():
        ratio = (count / (num_timesteps * num_single_layers)) * 100
        print(f"  {module_name:15s}: {ratio:5.2f}%")

def load_cache_books(cache_book_path, cache_book_file):
    with open(os.path.join(cache_book_path, cache_book_file), "r") as f:
        cache_books = json.load(f)
    return (
        cache_books["step_cache_book"],
        cache_books["transformer_cache_book"],
        cache_books["single_transformer_cache_book"]
    )

def main():
    seed=42
    torch.set_grad_enabled(False)

    print("Loading FLUX pipeline...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16, 
        cache_dir="/root/autodl-tmp/InvarDiff/FLUX"
    )
    original_transformer = pipe.transformer

    ## default measure 5 prompts, the number of measure prompts will have a slight impact on the speedup ratio.
    ## fast(3.3x): nonskip_rate=0.1, step_thres = 0.7, attn_thres=ff_thres=context_ff_thres=Single_attn_thres=Single_mlp_thres= 0.68 (measure 5 prompts)
    ## medium-1(2.9x): nonskip_rate=0.1, step_thres = 0.7, attn_thres=0.68, ff_thres=0, context_ff_thres=0, Single_attn_thres=0.68, Single_mlp_thres=0
    ## medium-2(2.6x): nonskip_rate=0.15, step_thres=0.7, attn_thres=0.68, ff_thres=0, context_ff_thres=0, Single_attn_thres=0.7, Single_mlp_thres=0
    ## slow(2.5x): nonskip_rate=0.22, step_thres=0.72, attn_thres=0.68, ff_thres=0.66, context_ff_thres=0, Single_attn_thres=0.68, Single_mlp_thres=0.62
    num_inference_steps = 28
    nonskip_rate = 0.1
    step_thres = 0.7 # Clearly affects the acceleration ratio and reduce the proportion of finegrained cache.

    attn_thres=0.68
    ff_thres= 0.68 # This threshold sometimes cause blemishes on the image.
    context_ff_thres=0.68

    Single_attn_thres=0.68
    Single_mlp_thres=0.68 # Lowering this threshold can reduce "moiré patterns".

    dynamic_model = DynamicFluxTransformer2DModel(
        original_transformer,
        num_inference_steps,
    )
    dynamic_model.eval()
    pipe.transformer = dynamic_model

    if 0: # Set True when calibration
        # pipe.enable_model_cpu_offload() ## when GPU RAM not enough
        pipe.to("cuda")
        ## There is no necessary correlation between the measure prompts and the test prompts.
        measure_prompts = [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe.",
            "A futuristic cityscape with flying cars and neon lights.",
            "An astronaut riding a horse on the moon.",
            "A bouquet of wildflowers in a glass vase, watercolor style.",
            "A majestic lion sitting on a rock, golden mane, sunset.",
            # "A serene landscape with mountains and a lake at sunset.",
            # "A cute cat playing with a ball of yarn.",
            # "A portrait of a woman in Renaissance style, oil painting.",
            # "A dragon flying over a burning village, epic fantasy.",
            # "A steaming cup of coffee on a wooden table, cozy morning.",
            # "A cyberpunk street with rain and neon signs, night.",
            # "A tropical beach at sunrise, palm trees, golden hour.",
            # "A magical library with floating books and glowing runes.",
            # "A vintage camera surrounded by old photographs, nostalgic mood.",
            # "A colorful parrot flying in a rainforest, vibrant colors.",
            # "A medieval castle on a hill, surrounded by fog.",
            # "A young woman in a traditional Japanese kimono, cherry blossoms.",
            # "A futuristic female cyborg with glowing blue eyes, silver armor.",
            # "A steaming bowl of ramen on a wooden counter, food photography.",
            # "A snowy forest with sunlight filtering through the trees.",
        ]
        step_cache_book, transformer_cache_book, single_transformer_cache_book, \
        avg_transformer_rates, avg_single_transformer_rates = threshold_analyse(
            model=dynamic_model,
            pipe=pipe,
            measure_prompts=measure_prompts,
            nonskip_rate=nonskip_rate,
            step_thres=step_thres,
            attn_thres=attn_thres,
            ff_thres=ff_thres,
            context_ff_thres=context_ff_thres,
            Single_attn_thres=Single_attn_thres,
            Single_mlp_thres=Single_mlp_thres,
            seed=seed
        )
        cache_books = {
            "step_cache_book": step_cache_book,
            "transformer_cache_book": transformer_cache_book,
            "single_transformer_cache_book": single_transformer_cache_book
        }
        cache_book_path = "./cache_books"
        os.makedirs(cache_book_path, exist_ok=True)
        
        thres_str = (
            f"stp{num_inference_steps}_n{nonskip_rate}_th{step_thres}"
            f"_attn{attn_thres}_ff{ff_thres}_ctxff{context_ff_thres}"
            f"_sattn{Single_attn_thres}_smlp{Single_mlp_thres}"
        )
        cache_book_file = f"{cache_book_path}/cache_books_{thres_str}.json"
        with open(cache_book_file, "w") as f:
            json.dump(cache_books, f, indent=2)

        print(f"\nCache books saved: {cache_book_file}")

    else:
        cache_book_file = (
            f"cache_books_stp{num_inference_steps}_n{nonskip_rate}_th{step_thres}"
            f"_attn{attn_thres}_ff{ff_thres}_ctxff{context_ff_thres}"
            f"_sattn{Single_attn_thres}_smlp{Single_mlp_thres}.json"
        )
        step_cache_book, transformer_cache_book, single_transformer_cache_book = load_cache_books(
            cache_book_path="./cache_books",
            cache_book_file=cache_book_file
        )
        pipe.to("cuda")
        dynamic_model.init_cache_book(transformer_cache_book, single_transformer_cache_book, step_cache_book)
        pipe.transformer = dynamic_model
        
        prompts = [
            "a photo of a broccoli",
            "A snowy mountain village at dusk, glowing windows and smoke rising.",
            "A golden retriever puppy jumping through autumn leaves.",
            "A surreal underwater city with glowing jellyfish and crystal towers.",
            "A group of astronauts planting a flag on Mars, red rocky landscape.",
            "A vintage sports car speeding down a coastal highway at sunset.",
            "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.",
            # "The grand, opulent lobby of an Art Deco skyscraper. Polished brass, geometric patterns on the marble floor, and a massive, intricate chandelier casting warm light. Cinematic, symmetrical, elegant, 1920s style.",
            # "A detailed portrait of a vibrant, colorful toucan perched on a mossy branch. The background is a lush, out-of-focus jungle with soft morning light filtering through the leaves. Photorealistic, natural style, high detail, shallow depth of field.",
            # "A steampunk robot reading a book in a Victorian library.",
            # "Macro portrait of a silver tabby cat with vivid green eyes, crisp whiskers, rich fur texture, shallow depth of field,\
            #     cyberpunk megacity at night, rain-slick streets reflecting vivid neon signs and holograms. Flying cars, towering glass and steel facades, cinematic wide angle, high contrast.",
            # "a photo of a white sandwich",
            # "a photo of a person"
        ]
        images = []
        times = []

        for prompt in prompts:
            start_time = time.time()
            image = pipe(
                prompt, 
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(device="cpu").manual_seed(seed)
                ).images[0]
            times.append(time.time() - start_time)
            images.append(image)

        if len(prompts)>1:
            times = np.array(times[1:])
            avg_time = np.mean(times)
            std_time = np.std(times)
            print(f"Sampling time: {avg_time:.4f}±{std_time:.4f} s")
        else:
            print(f"Sampling time: {times[0]:.4f} s")

        width, height = images[0].size
        combined = Image.new("RGB", (width * len(images), height))
        for idx, img in enumerate(images):
            combined.paste(img, (idx * width, 0))

        os.makedirs("images", exist_ok=True)
        timestamp = time.strftime("%m%d_%H%M%S")
        save_name = (f"images/imgs_stp{num_inference_steps}_n{nonskip_rate}_th{step_thres}"
                     f"_attn{attn_thres}_ff{ff_thres}_ctxff{context_ff_thres}"
                     f"_sattn{Single_attn_thres}_smlp{Single_mlp_thres}_{timestamp}.png")
        combined.save(save_name)
        print(f"Images saved to {save_name}")

if __name__ == "__main__":
    main()