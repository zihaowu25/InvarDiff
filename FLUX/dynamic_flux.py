import torch
import torch.nn as nn
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock, FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, scale_lora_layers, unscale_lora_layers
from diffusers.utils import logging
import numpy as np
from typing import Any, Dict, Optional, Union, List, Callable
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.get_logger(__name__)

class DynamicFluxSingleTransformerBlock(FluxSingleTransformerBlock):

    def forward(
        self,
        hidden_states,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
        attn_cache=None,
        mlp_cache=None
    ):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        if mlp_cache is not None:
            mlp_hidden_states = mlp_cache
        else:
            mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        
        if attn_cache is not None:
            attn_output = attn_cache
        else:
            joint_attention_kwargs = joint_attention_kwargs or {}
            attn_output = self.attn(
                hidden_states=norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **joint_attention_kwargs,
            )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        block_outputs = {"attn": attn_output, "mlp": mlp_hidden_states}

        return hidden_states, block_outputs

class DynamicFluxTransformerBlock(FluxTransformerBlock):

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
        attn_cache=None,
        context_attn_cache=None,
        ip_attn_cache=None,
        ff_cache=None,
        context_ff_cache=None
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        
        # Attention.
        if attn_cache is not None and context_attn_cache is not None:
            attn_output = attn_cache
            context_attn_output = context_attn_cache
            ip_attn_output = ip_attn_cache
        else:
            joint_attention_kwargs = joint_attention_kwargs or {}
            attention_outputs = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **joint_attention_kwargs,
            )
            attn_output, context_attn_output = attention_outputs[0], attention_outputs[1]
            ip_attn_output = attention_outputs[2] if len(attention_outputs) == 3 else None

        # Process attention outputs for the `hidden_states`.
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

        if ff_cache is not None:
            ff_output = ff_cache
        else:
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
        if ip_attn_output is not None:
            hidden_states = hidden_states + ip_attn_output

        # Process attention outputs for the `encoder_hidden_states`.
        encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output

        if context_ff_cache is not None:
            context_ff_output = context_ff_cache
        else:
            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

            context_ff_output = self.ff_context(norm_encoder_hidden_states)

        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        block_outputs = {
            "attn":attn_output, "context_attn":context_attn_output, "ip_attn":ip_attn_output,
            "ff":ff_output, "context_ff":context_ff_output
        }
        return encoder_hidden_states, hidden_states, block_outputs

class DynamicFluxTransformer2DModel(nn.Module):
    def __init__(self, base_model: FluxTransformer2DModel, num_timesteps):
        super().__init__()

        self.config = base_model.config
        self.base_model = base_model
        self.device = next(base_model.x_embedder.parameters()).device
        self.dtype = getattr(base_model, "dtype", torch.float32)
        self.num_timesteps = num_timesteps
        self.transformer_blocks = base_model.transformer_blocks
        self.single_transformer_blocks = base_model.single_transformer_blocks
        for block in self.transformer_blocks:
            block.forward = DynamicFluxTransformerBlock.forward.__get__(block, block.__class__)
        for block in self.single_transformer_blocks:
            block.forward = DynamicFluxSingleTransformerBlock.forward.__get__(block, block.__class__)
        self.num_layers = self.config.num_layers
        self.num_single_layers = self.config.num_single_layers
        self.hidden_states_cache = None
        self.Transformer_cache = {
            "attn":[None]*self.num_layers,
            "context_attn":[None]*self.num_layers,
            "ip_attn":[None]*self.num_layers,
            "ff":[None]*self.num_layers,
            "context_ff":[None]*self.num_layers
        }
        self.SingleTransformer_cache = {
            "attn":[None]*self.num_single_layers,
            "mlp":[None]*self.num_single_layers
        }
        self.step_cache_bool = [False] * self.num_timesteps
        self.Transformer_cache_book = {
            "attn": [[False for _ in range(self.num_layers)] for _ in range(self.num_timesteps)],
            "context_attn": [[False for _ in range(self.num_layers)] for _ in range(self.num_timesteps)],
            "ip_attn": [[False for _ in range(self.num_layers)] for _ in range(self.num_timesteps)],
            "ff": [[False for _ in range(self.num_layers)] for _ in range(self.num_timesteps)],
            "context_ff": [[False for _ in range(self.num_layers)] for _ in range(self.num_timesteps)],
        }
        self.SingleTransformer_cache_book = {
            "attn": [[False for _ in range(self.num_single_layers)] for _ in range(self.num_timesteps)],
            "mlp": [[False for _ in range(self.num_single_layers)] for _ in range(self.num_timesteps)],
        }
        self.step_cnt = 0
        self.block_cache_enable = False

    def init_cache_book(self, Transformer_cache_book, SingleTransformer_cache_book, step_cache_book):
        self.Transformer_cache_book: Dict[str, List[List[bool]]] = Transformer_cache_book
        self.SingleTransformer_cache_book: Dict[str, List[List[bool]]] = SingleTransformer_cache_book
        self.step_cache_bool = step_cache_book
        self.block_cache_enable = True
        
    def reset(self):
        self.Transformer_cache = {
            "attn":[None]*self.num_layers,
            "context_attn":[None]*self.num_layers,
            "ip_attn":[None]*self.num_layers,
            "ff":[None]*self.num_layers,
            "context_ff":[None]*self.num_layers
        }
        self.SingleTransformer_cache = {
            "attn":[None]*self.num_single_layers,
            "mlp":[None]*self.num_single_layers
        }
        self.step_cnt = 0

    def cache_context(self, *args, **kwargs):
        return self.base_model.cache_context(*args, **kwargs)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
        device = next(self.base_model.parameters()).device
        hidden_states = hidden_states.to(device)
        if pooled_projections is not None:
            pooled_projections = pooled_projections.to(device)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(device)
        if timestep is not None:
            timestep = timestep.to(device)
        if guidance is not None:
            guidance = guidance.to(device)
        if txt_ids is not None:
            txt_ids = txt_ids.to(device)
        if img_ids is not None:
            img_ids = img_ids.to(device)
        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            joint_attention_kwargs["ip_adapter_image_embeds"] = joint_attention_kwargs["ip_adapter_image_embeds"].to(device)

        if self.step_cache_bool[self.step_cnt] and self.step_cnt>0:
            hidden_states = self.hidden_states_cache
            timestep = timestep.to(hidden_states.dtype) * 1000
            if guidance is not None:
                guidance = guidance.to(hidden_states.dtype) * 1000
            else:
                guidance = None
            
            temb = (
                self.base_model.time_text_embed(timestep, pooled_projections)
                if guidance is None
                else self.base_model.time_text_embed(timestep, guidance, pooled_projections)
            )
        else:
            hidden_states = self.base_model.x_embedder(hidden_states)
            timestep = timestep.to(hidden_states.dtype) * 1000
            if guidance is not None:
                guidance = guidance.to(hidden_states.dtype) * 1000
            else:
                guidance = None
            
            temb = (
                self.base_model.time_text_embed(timestep, pooled_projections)
                if guidance is None
                else self.base_model.time_text_embed(timestep, guidance, pooled_projections)
            )
            encoder_hidden_states = self.base_model.context_embedder(encoder_hidden_states)

            if txt_ids.ndim == 3:
                logger.warning(
                    "Passing `txt_ids` 3d torch.Tensor is deprecated."
                    "Please remove the batch dimension and pass it as a 2d torch Tensor"
                )
                txt_ids = txt_ids[0]
            if img_ids.ndim == 3:
                logger.warning(
                    "Passing `img_ids` 3d torch.Tensor is deprecated."
                    "Please remove the batch dimension and pass it as a 2d torch Tensor"
                )
                img_ids = img_ids[0]

            ids = torch.cat((txt_ids, img_ids), dim=0)
            image_rotary_emb = self.base_model.pos_embed(ids)

            if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
                ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
                ip_hidden_states = self.base_model.encoder_hid_proj(ip_adapter_image_embeds)
                joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

            for index_block, block in enumerate(self.transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )

                else:
                    attn_cache = self.Transformer_cache["attn"][index_block] \
                        if self.Transformer_cache_book["attn"][self.step_cnt][index_block] else None
                    context_attn_cache = self.Transformer_cache["context_attn"][index_block] \
                        if self.Transformer_cache_book["context_attn"][self.step_cnt][index_block] else None
                    ip_attn_cache = self.Transformer_cache["ip_attn"][index_block] \
                        if self.Transformer_cache_book["ip_attn"][self.step_cnt][index_block] else None
                    ff_cache = self.Transformer_cache["ff"][index_block] \
                        if self.Transformer_cache_book["ff"][self.step_cnt][index_block] else None
                    context_ff_cache = self.Transformer_cache["context_ff"][index_block] \
                        if self.Transformer_cache_book["context_ff"][self.step_cnt][index_block] else None
                    
                    encoder_hidden_states, hidden_states, block_outputs = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                        attn_cache=attn_cache,
                        context_attn_cache=context_attn_cache,
                        ip_attn_cache=ip_attn_cache,
                        ff_cache=ff_cache,
                        context_ff_cache=context_ff_cache
                    )
                    if self.block_cache_enable:
                        self.Transformer_cache["attn"][index_block] = block_outputs["attn"]
                        self.Transformer_cache["context_attn"][index_block] = block_outputs["context_attn"]
                        self.Transformer_cache["ip_attn"][index_block] = block_outputs["ip_attn"]
                        self.Transformer_cache["ff"][index_block] = block_outputs["ff"]
                        self.Transformer_cache["context_ff"][index_block] = block_outputs["context_ff"]

                # controlnet residual
                if controlnet_block_samples is not None:
                    interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    # For Xlabs ControlNet.
                    if controlnet_blocks_repeat:
                        hidden_states = (
                            hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                        )
                    else:
                        hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

            for index_block, block in enumerate(self.single_transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        temb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )

                else:
                    single_attn_cache = self.SingleTransformer_cache["attn"][index_block] \
                        if self.SingleTransformer_cache_book["attn"][self.step_cnt][index_block] else None
                    single_mlp_cache = self.SingleTransformer_cache["mlp"][index_block] \
                        if self.SingleTransformer_cache_book["mlp"][self.step_cnt][index_block] else None
                    hidden_states, block_outputs = block(
                        hidden_states=hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                        attn_cache=single_attn_cache,
                        mlp_cache=single_mlp_cache
                    )
                    if self.block_cache_enable:
                        self.SingleTransformer_cache["attn"][index_block] = block_outputs["attn"]
                        self.SingleTransformer_cache["mlp"][index_block] = block_outputs["mlp"]

                # controlnet residual
                if controlnet_single_block_samples is not None:
                    interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                        hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                        + controlnet_single_block_samples[index_block // interval_control]
                    )

            hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
            self.hidden_states_cache = hidden_states

        hidden_states = self.base_model.norm_out(hidden_states, temb)
        output = self.base_model.proj_out(hidden_states)

        self.step_cnt += 1 # update step count
        if self.step_cnt >= self.num_timesteps:
            self.reset()
        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

def flux_sample_loop_progressive(
    pipe,
    prompt: str,
    prompt_2: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    negative_prompt_2: Optional[str] = None,
    true_cfg_scale: float = 1.0,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    num_images_per_prompt: int = 1,
    generator: Optional[torch.Generator] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    ip_adapter_image: Optional[Any] = None,
    ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    negative_ip_adapter_image: Optional[Any] = None,
    negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    output_type: str = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    sigmas: Optional[List[float]] = None,
):
    """
    Progressive sampling generator that fully mimics FluxPipeline's __call__ method.
    Yields after each timestep so external code can process features collected by hooks.

    Args:
        pipe: FluxPipeline instance.
        prompt: Text prompt.
        prompt_2: Optional second text prompt.
        negative_prompt: Negative text prompt for true-CFG.
        negative_prompt_2: Optional second negative text prompt.
        true_cfg_scale: True classifier-free guidance scale (default: 1.0, disabled).
        height: Output image height.
        width: Output image width.
        num_inference_steps: Number of inference (denoising) steps.
        guidance_scale: Classifier-free guidance scale.
        num_images_per_prompt: Number of images to generate per prompt.
        generator: torch.Generator for reproducible randomness.
        latents: Optional precomputed latents.
        prompt_embeds: Optional precomputed prompt embeddings.
        pooled_prompt_embeds: Optional precomputed pooled prompt embeddings.
        negative_prompt_embeds: Optional precomputed negative prompt embeddings.
        negative_pooled_prompt_embeds: Optional precomputed negative pooled prompt embeddings.
        ip_adapter_image: Optional image input for IP-Adapter.
        ip_adapter_image_embeds: Optional precomputed IP-Adapter image embeddings.
        negative_ip_adapter_image: Optional negative image input for IP-Adapter.
        negative_ip_adapter_image_embeds: Optional negative IP-Adapter image embeddings.
        output_type: "pil" or "latent".
        return_dict: Whether to yield dicts or raw final output.
        joint_attention_kwargs: Additional kwargs for attention processors.
        callback_on_step_end: Callback function called at the end of each step.
        callback_on_step_end_tensor_inputs: List of tensor names to pass to callback.
        max_sequence_length: Maximum token sequence length for encoding.
        sigmas: Optional custom sigmas for scheduler.

    Yields:
        dict with keys:
            - 'timestep_idx': current timestep index
            - 'timestep': current scheduler timestep value
            - 'latents': current latents tensor
            - 'model_output': model's predicted noise/output for this step
        After completion yields a dict containing 'final_image' if return_dict is True.
    """
    device = pipe._execution_device
    
    if joint_attention_kwargs is None:
        joint_attention_kwargs = {}
    else:
        joint_attention_kwargs = joint_attention_kwargs.copy()
    
    lora_scale = joint_attention_kwargs.get("scale", None)
    
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    
    if prompt_embeds is None or pooled_prompt_embeds is None:
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
    else:
        dtype = pipe.text_encoder.dtype if pipe.text_encoder is not None else pipe.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
    
    do_true_cfg = true_cfg_scale > 1.0 and negative_prompt is not None
    if do_true_cfg:
        (
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            _,
        ) = pipe.encode_prompt(
            prompt=negative_prompt,
            prompt_2=negative_prompt_2,
            prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
    
    num_channels_latents = pipe.transformer.config.in_channels // 4
    if latents is None:
        latents, latent_image_ids = pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            None,
        )
    else:
        latent_height = latents.shape[1]
        latent_width = latents.shape[2] if latents.ndim > 2 else latents.shape[1]
        latent_image_ids = pipe._prepare_latent_image_ids(
            latents.shape[0],
            latent_height,
            latent_width,
            device,
            prompt_embeds.dtype,
        )
    
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    image_seq_len = latents.shape[1]
    
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.base_image_seq_len,
        pipe.scheduler.config.max_image_seq_len,
        pipe.scheduler.config.base_shift,
        pipe.scheduler.config.max_shift,
    )
    
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    
    # Prepare guidance
    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None
    
    # Handle IP-Adapter
    # Ensure negative IP-Adapter images exist if positive ones do (and vice versa)
    if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and \
       (negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None):
        negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
    elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and \
         (negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None):
        ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
    
    image_embeds = None
    negative_image_embeds = None
    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
        )
    if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
        negative_image_embeds = pipe.prepare_ip_adapter_image_embeds(
            negative_ip_adapter_image,
            negative_ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
        )
    
    # Progressive denoising loop
    for timestep_idx, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="Denoising steps", leave=False):
        # Add IP-Adapter embeddings to joint_attention_kwargs if available
        if image_embeds is not None:
            joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
        
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        
        model_device = next(pipe.transformer.parameters()).device
        latents = latents.to(model_device)
        timestep = timestep.to(model_device)
        guidance = guidance.to(model_device) if guidance is not None else None
        pooled_prompt_embeds = pooled_prompt_embeds.to(model_device) if pooled_prompt_embeds is not None else None
        prompt_embeds = prompt_embeds.to(model_device) if prompt_embeds is not None else None
        text_ids = text_ids.to(model_device) if text_ids is not None else None
        latent_image_ids = latent_image_ids.to(model_device) if latent_image_ids is not None else None
        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            joint_attention_kwargs["ip_adapter_image_embeds"] = joint_attention_kwargs["ip_adapter_image_embeds"].to(model_device)
        
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )[0]
        
        # Handle true-CFG (negative prompt forward pass)
        if do_true_cfg:
            if negative_image_embeds is not None:
                joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
            
            neg_noise_pred = pipe.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=negative_pooled_prompt_embeds,
                encoder_hidden_states=negative_prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=False,
            )[0]
            
            noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
        
        device = pipe._execution_device
        latents = latents.to(device)
        noise_pred = noise_pred.to(device)

        latents_dtype = latents.dtype
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        if latents.dtype != latents_dtype:
            if torch.backends.mps.is_available():
                latents = latents.to(latents_dtype)
        
        if callback_on_step_end is not None:
            callback_kwargs = {}
            for k in callback_on_step_end_tensor_inputs:
                callback_kwargs[k] = locals()[k]
            callback_outputs = callback_on_step_end(pipe, timestep_idx, t, callback_kwargs)
            
            latents = callback_outputs.pop("latents", latents)
            prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
        
        # Yield current state (key feature: allows external feature collection)
        yield {
            'timestep_idx': timestep_idx,
            'timestep': t,
            'latents': latents,
            'model_output': noise_pred,
        }
    
    if output_type == "latent":
        image = latents
    else:
        latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type=output_type)
    
    pipe.maybe_free_model_hooks()
    
    if return_dict:
        yield {'final_image': image}
    else:
        yield image
