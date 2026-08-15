# Copyright 2024 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
import os
import gc
import copy
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch
from PIL import Image

from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, calculate_shift, retrieve_timesteps
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor

# L-DINO-CoT: Localized VQA Scoring imports
from lvqa_dinot import (
    LDINOOptimizer, 
    EntityInfo, 
    create_entities_from_simple_format,
    GroundedSAMSegmenter,
    DependencyGraphEvaluator
)
from lvqa_dinot.differentiable_blur import apply_blur_mask

logger = logging.get_logger(__name__)

class FluxDiNOTPipeline(FluxPipeline):
    
    def init_vqa_model(self, vqa_model, device):
        self.vqa_model = vqa_model
        self.vqa_model_device = device
        self.ldino_optimizer = None

    @staticmethod
    def directional_gaussian_torch(grad, alpha, beta, generator=None):
        """
        Samples a directional Gaussian noise vector in PyTorch.
        """
        n = grad.numel()
        device = grad.device
        dtype = grad.dtype
        g = grad.view(-1) / (grad.view(-1).norm() + 1e-9)
        lam1 = beta + alpha * (1 - 1/n)
        lam2 = beta - alpha / n

        # Sample standard normals
        w = torch.randn(n, device=device, dtype=dtype, generator=generator)
        c = torch.dot(g, w)
        w_perp = w - c * g

        noise = (lam1**0.5) * c * g + (lam2**0.5) * w_perp
        return noise.view(grad.shape)

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        optimization_epoch: int = 5, # default less for Flux due to scale
        dependency_graph: Optional[Dict] = None,
        use_localized_vqa: bool = True,
    ):
        # === Setup phase (no gradients needed) ===
        with torch.no_grad():
            height = height or self.default_sample_size * self.vae_scale_factor
            width = width or self.default_sample_size * self.vae_scale_factor

            self.check_inputs(
                prompt, prompt_2, height, width,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
            )

            self._guidance_scale = guidance_scale
            self._joint_attention_kwargs = joint_attention_kwargs
            self._current_timestep = None
            self._interrupt = False

            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self._execution_device

            lora_scale = (
                self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
            )
            has_neg_prompt = negative_prompt is not None or (
                negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
            )
            do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
            
            (
                prompt_embeds,
                pooled_prompt_embeds,
                text_ids,
            ) = self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )
            
            if do_true_cfg:
                (
                    negative_prompt_embeds,
                    negative_pooled_prompt_embeds,
                    negative_text_ids,
                ) = self.encode_prompt(
                    prompt=negative_prompt,
                    prompt_2=negative_prompt_2,
                    prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                    lora_scale=lora_scale,
                )

            num_channels_latents = self.transformer.config.in_channels // 4
            latents, latent_image_ids = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
            if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
                sigmas = None
            image_seq_len = latents.shape[1]
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )

            timestep_device = device
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                timestep_device,
                sigmas=sigmas,
                mu=mu,
            )
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
            self._num_timesteps = len(timesteps)

            if self.transformer.config.guidance_embeds:
                guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
                guidance = guidance.expand(latents.shape[0])
            else:
                guidance = None

            if self.joint_attention_kwargs is None:
                self._joint_attention_kwargs = {}
        # === End of setup (no_grad block closes here) ===
            
        def denoise(lat_in, max_steps=None, return_latents=False, return_tweedie=False, start_step=0):
            max_steps_local = max_steps if max_steps is not None else num_inference_steps
            noise_list_local = {}
            tweedie_est_local = None
            
            lat_curr = lat_in
            for i, t in enumerate(timesteps):
                if i < start_step:
                    continue
                if i >= max_steps_local:
                    break
                    
                timestep_expanded = t.expand(lat_curr.shape[0]).to(lat_curr.dtype)
                
                # Flux passes guidance and uses single forward pass for embedded guidance
                noise_pred = self.transformer(
                    hidden_states=lat_curr,
                    timestep=timestep_expanded / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                
                if do_true_cfg:
                    neg_noise_pred = self.transformer(
                        hidden_states=lat_curr,
                        timestep=timestep_expanded / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                    
                noise_list_local[t] = noise_pred.detach()
                
                if return_tweedie and (i == max_steps_local - 1 or i == len(timesteps) - 1):
                    sigma_val = t / 1000.0  
                    tweedie_est_local = lat_curr - sigma_val * noise_pred
                    
                lat_curr = self.scheduler.step(noise_pred, t, lat_curr, return_dict=False)[0]
                
            if return_tweedie and return_latents:
                return noise_list_local, tweedie_est_local, lat_curr
            if return_tweedie:
                return noise_list_local, tweedie_est_local
            if return_latents:
                return noise_list_local, lat_curr
            return noise_list_local

        def reverse(lat_start, noise_list_cached):
            lat_curr = lat_start
            for t in noise_list_cached:
                noise_pred = noise_list_cached[t]
                lat_curr = self.scheduler.step(noise_pred, t, lat_curr, return_dict=False)[0]
            return lat_curr

        latents_init = latents.clone().detach()
        with torch.no_grad():
            noise_list = denoise(latents_init)
            self.scheduler.set_timesteps(num_inference_steps)

        latents_init.requires_grad_(True)
        # Note: packing and unpacking are fully differentiable reshapes/perms
        latents_final = reverse(latents_init, noise_list)
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Decode helper
        def decode_latents(packed_lats):
            lat_unpacked = self._unpack_latents(packed_lats, height, width, self.vae_scale_factor)
            lat_dec = (lat_unpacked / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            img_dec = self.vae.decode(lat_dec.to(self.vae.dtype), return_dict=False)[0]
            return img_dec
            
        curr_image_raw = decode_latents(latents_final)
        
        prompts_str = prompt[0] if isinstance(prompt, list) else prompt
        dir_name = f"flux_dinot_details/{prompts_str[:20].replace(' ', '_')}_{generator.initial_seed() if hasattr(generator, 'initial_seed') else 0}"
        os.makedirs(dir_name, exist_ok=True)
        
        with torch.no_grad():
            init_pil = self.image_processor.postprocess(curr_image_raw.detach().cpu(), output_type="pil")[0]
            init_pil.save(f"{dir_name}/0.png")
            print(f"[Flux-DINO] Saved initial image to {dir_name}/0.png")

        curr_image = (curr_image_raw.float() / 2 + 0.5).clamp(0, 1)
        curr_image = curr_image.detach().cpu().to(self.vqa_model_device).requires_grad_(True)
        
        if dependency_graph is None:
            dependency_graph = {
                "nodes": [{"id": "q0", "type": "Entity", "concept": prompts_str, "question": prompts_str, "parent_id": None}]
            }
            
        graph_evaluator = DependencyGraphEvaluator(dependency_graph)
        use_ldino = use_localized_vqa
        entities = None
        initial_masks = {}
        
        if use_ldino:
            if self.ldino_optimizer is None:
                self.ldino_optimizer = LDINOOptimizer(
                    vqa_model=self.vqa_model,
                    device=self.vqa_model_device,
                    save_visualizations=True
                )
            
            auto_entity_attributes = {
                n["concept"]: [] for n in graph_evaluator.nodes.values() if n.get("type") == "Entity"
            }
            
            if auto_entity_attributes:
                entities = self.ldino_optimizer.setup_entities(prompts_str, auto_entity_attributes, output_dir=dir_name)
                with torch.no_grad():
                    image_pil = self.image_processor.postprocess(curr_image_raw.detach().cpu(), output_type="pil")[0]
                    all_entity_names = [e.name for e in entities]
                    print(f"[L-DINO] Segmenting entities: {all_entity_names}")
                    initial_masks = self.ldino_optimizer.segmenter.segment_multiple(image_pil, all_entity_names)
                    
                    mask_dir = f"{dir_name}/ldino_debug/initial"
                    os.makedirs(mask_dir, exist_ok=True)
                    for e_name, m_np in initial_masks.items():
                        if m_np is not None:
                            m_pil = Image.fromarray((m_np * 255).astype(np.uint8))
                            m_pil.save(f"{mask_dir}/mask_{e_name}.png")
                    print(f"[L-DINO] Saved initial masks to {mask_dir}/")

        gradients_cpu = []
        individual_scores = []
        
        # === VQA scoring + gradient computation (needs gradients!) ===
        with torch.enable_grad():
            for n_id, q_text in graph_evaluator.questions.items():
                if latents_init.grad is not None:
                    latents_init.grad.zero_()
                    
                lat_for_node = reverse(latents_init, noise_list)
                self.scheduler.set_timesteps(num_inference_steps)
                img_raw_node = decode_latents(lat_for_node)
                img_scaled_node = (img_raw_node.float() / 2 + 0.5).clamp(0, 1)
                img_node = img_scaled_node.detach().cpu().to(self.vqa_model_device).requires_grad_(True)
                
                target_image = img_node
                matched_entity = None
                
                if use_ldino:
                    concept = graph_evaluator.nodes[n_id].get("concept", q_text)
                    for ent in entities:
                        if ent.name in concept:
                            matched_entity = ent.name
                            break
                    
                    if matched_entity and matched_entity in initial_masks:
                        mask_np = initial_masks[matched_entity]
                        if mask_np is not None:
                            mask_tensor = torch.from_numpy(mask_np).to(self.vqa_model_device).float().unsqueeze(0).unsqueeze(0)
                            target_image = apply_blur_mask(img_node, mask_tensor)
                
                score = self.vqa_model(target_image, [q_text])
                score_val = score.item()
                if math.isnan(score_val):
                    score_val = 0.0
                individual_scores.append((n_id, score_val))
                
                score.backward()
                
                grad_on_sd = img_node.grad.cpu().to(img_raw_node.device, dtype=torch.float32) if img_node.grad is not None else torch.zeros_like(img_raw_node).float()
                grad_on_sd = torch.nan_to_num(grad_on_sd, nan=0.0, posinf=0.0, neginf=0.0)
                img_raw_node.backward(grad_on_sd / 2.0)
                
                if latents_init.grad is not None:
                    grad_cpu = latents_init.grad.detach().cpu().clone()
                    grad_cpu = torch.nan_to_num(grad_cpu, nan=0.0, posinf=0.0, neginf=0.0)
                    # Flux Packed space shape is (1, seq_len, 64)
                    if use_ldino and matched_entity and matched_entity in initial_masks:
                        mask_np_loc = initial_masks[matched_entity]
                        mask_pil = Image.fromarray((mask_np_loc * 255).astype(np.uint8))
                        
                        # Compute expected unpacked spatial dims
                        latent_h = 2 * (int(height) // (self.vae_scale_factor * 2))
                        latent_w = 2 * (int(width) // (self.vae_scale_factor * 2))
                        
                        mask_latent = mask_pil.resize((latent_w, latent_h), Image.Resampling.BILINEAR)
                        mask_tensor_latent = torch.from_numpy(np.array(mask_latent)/255.0).float().unsqueeze(0).unsqueeze(0).to(device)
                        mask_tensor_latent = mask_tensor_latent.expand(1, num_channels_latents, latent_h, latent_w)
                        mask_packed = self._pack_latents(mask_tensor_latent, 1, num_channels_latents, latent_h, latent_w)
                        mask_packed = mask_packed.detach().cpu()
                        grad_cpu = grad_cpu * mask_packed
                    
                    gradients_cpu.append((n_id, grad_cpu))
        
        raw_scores_map = dict(individual_scores)
        masked_scores, root_scores, avg_vqa_score = graph_evaluator.evaluate(raw_scores_map)
        
        root_gradients = []
        for root_id, _ in root_scores.items():
            tree_nodes = set()
            def dfs(nid):
                if nid in tree_nodes: return
                tree_nodes.add(nid)
                for child_id, child in graph_evaluator.nodes.items():
                    parents = child.get("parent_id")
                    if (isinstance(parents, list) and nid in parents) or parents == nid:
                        dfs(child_id)
            dfs(root_id)
            
            tree_grad = torch.zeros_like(gradients_cpu[0][1])
            for n_id, g in gradients_cpu:
                if n_id in tree_nodes:
                    coeff = 1.0
                    for on_id in tree_nodes:
                        if on_id != n_id:
                            coeff *= max(raw_scores_map.get(on_id, 1e-9), 1e-9)
                    tree_grad += coeff * g
            root_gradients.append(tree_grad)
            
        final_grad = torch.mean(torch.stack(root_gradients), dim=0) if root_gradients else torch.zeros_like(latents_init.detach().cpu())
        stored_grad = final_grad.to(latents_init.device)
        stored_grad = torch.nan_to_num(stored_grad, nan=0.0, posinf=0.0, neginf=0.0)
        grad_norm = stored_grad.float().norm().item()
        MAX_GRAD_NORM = 100.0
        if grad_norm > MAX_GRAD_NORM:
            stored_grad = stored_grad * (MAX_GRAD_NORM / grad_norm)
            print(f"[Gradient] Clipped grad norm: {grad_norm:.2f} -> {MAX_GRAD_NORM}")
        print(f"[Gradient] stored_grad norm: {stored_grad.float().norm().item():.6f}")

        raw_avg = sum(v for _, v in individual_scores) / max(len(individual_scores), 1)
        max_vqa = raw_avg
        target_latents = latents_init.detach().clone()
        vqa_score_val = max(raw_avg, 1e-6)
        print(f"\n[Flux-DINO] Initial raw avg VQA: {raw_avg:.4f}, hierarchical: {avg_vqa_score:.4f}")
        print(f"[Flux-DINO] Starting {optimization_epoch} optimization epochs...")

        for ep in range(optimization_epoch):
            if math.isnan(vqa_score_val):
                vqa_score_val = max(max_vqa, 1e-6)
            step_lr = max(min(1 - vqa_score_val ** 0.5, 0.3), 0.01)
            grad = stored_grad
            
            pool_size = 5
            QUICK_STEPS = 20
            if QUICK_STEPS > num_inference_steps:
               QUICK_STEPS = num_inference_steps
            noise_pool = []
            
            grad_flat = grad.view(-1)
            n = grad_flat.numel()
            alpha = math.sqrt(n)
            beta = 1.0
            
            for _ in range(pool_size):
                noise_sample = self.directional_gaussian_torch(grad_flat, alpha=alpha, beta=beta, generator=generator)
                noise_pool.append(noise_sample.view(grad.shape))
            
            candidate_scores = []
            candidate_caches = []
            
            with torch.no_grad():
                for noise_cand in noise_pool:
                    lat_std = latents_init.float().std().item()
                    latents_tmp = (latents_init.detach() + step_lr * lat_std * noise_cand).to(latents_init.dtype)
                    
                    scheduler_checkpoint = copy.deepcopy(self.scheduler)
                    noise_list_tmp, tweedie_est, latents_preview = denoise(
                        latents_tmp, max_steps=QUICK_STEPS, return_tweedie=True, return_latents=True
                    )
                    
                    img_preview_raw = decode_latents(tweedie_est)
                    img_preview = (img_preview_raw.float() / 2 + 0.5).clamp(0, 1)
                    img_preview = img_preview.float().cpu().to(self.vqa_model_device)
                    
                    s_preview = self.vqa_model(img_preview, [prompts_str]).item()
                    candidate_scores.append(s_preview)
                    candidate_caches.append({
                        "latents": latents_preview.detach(),
                        "noise_list": {k: v.detach() for k, v in noise_list_tmp.items()},
                        "scheduler": copy.deepcopy(self.scheduler)
                    })
                    
                    self.scheduler = scheduler_checkpoint
            
            best_idx = int(np.argmax(candidate_scores))
            best_noise = noise_pool[best_idx]
            
            lat_std = latents_init.float().std().item()
            latents_init = (latents_init.detach() + step_lr * lat_std * best_noise).to(latents_init.dtype).detach().requires_grad_(True)
            
            with torch.no_grad():
                best_cache = candidate_caches[best_idx]
                self.scheduler = best_cache["scheduler"]
                latents_remaining = best_cache["latents"]
                noise_list_first = best_cache["noise_list"]
                
                noise_list_second = denoise(
                    latents_remaining, start_step=QUICK_STEPS, max_steps=num_inference_steps
                )
                
                noise_list_curr = {**noise_list_first, **noise_list_second}
                self.scheduler.set_timesteps(num_inference_steps)
                
            with torch.no_grad():
                curr_latents_save = reverse(latents_init.detach(), noise_list_curr)
                self.scheduler.set_timesteps(num_inference_steps)
                opt_image_save = decode_latents(curr_latents_save)
                ep_pil = self.image_processor.postprocess(opt_image_save.detach().cpu(), output_type="pil")[0]
                ep_pil.save(f"{dir_name}/epoch_{ep}.png")
            
            new_individual_scores = []
            new_gradients_cpu = []
            
            if use_ldino and self.ldino_optimizer and hasattr(self.ldino_optimizer, 'segmenter'):
                with torch.no_grad():
                    all_entity_names = [e.name for e in entities]
                    current_masks = self.ldino_optimizer.segmenter.segment_multiple(ep_pil, all_entity_names)
                    
                    mask_dir = f"{dir_name}/ldino_debug/epoch_{ep}"
                    os.makedirs(mask_dir, exist_ok=True)
                    for e_name, m_np in current_masks.items():
                        if m_np is not None:
                            m_pil = Image.fromarray((m_np * 255).astype(np.uint8))
                            m_pil.save(f"{mask_dir}/mask_{e_name}.png")
            else:
                current_masks = initial_masks
                
            # === Per-epoch VQA scoring + gradient computation (needs gradients!) ===
            with torch.enable_grad():
                for n_id, q_text in graph_evaluator.questions.items():
                    if latents_init.grad is not None:
                        latents_init.grad.zero_()
                    
                    lat_for_node = reverse(latents_init, noise_list_curr)
                    self.scheduler.set_timesteps(num_inference_steps)
                    img_for_node_raw = decode_latents(lat_for_node)
                    img_for_node_scaled = (img_for_node_raw.float() / 2 + 0.5).clamp(0, 1)
                    img_for_node = img_for_node_scaled.detach().cpu().to(self.vqa_model_device).requires_grad_(True)
                    
                    target_img_node = img_for_node
                    matched_ent_node = None
                    if use_ldino:
                        concept_node = graph_evaluator.nodes[n_id].get("concept", q_text)
                        for ent in entities:
                            if ent.name in concept_node:
                                matched_ent_node = ent.name
                                break
                        if matched_ent_node and matched_ent_node in current_masks:
                            mk_np = current_masks[matched_ent_node]
                            if mk_np is not None:
                                mk_ts = torch.from_numpy(mk_np).to(self.vqa_model_device).float().unsqueeze(0).unsqueeze(0)
                                target_img_node = apply_blur_mask(img_for_node, mk_ts)
                    
                    node_score = self.vqa_model(target_img_node, [q_text])
                    s_val = node_score.item()
                    if math.isnan(s_val):
                        s_val = 0.0
                    new_individual_scores.append((n_id, s_val))
                    node_score.backward()
                    
                    grad_relay = img_for_node.grad.cpu().to(img_for_node_raw.device, dtype=torch.float32) if img_for_node.grad is not None else torch.zeros_like(img_for_node_raw).float()
                    grad_relay = torch.nan_to_num(grad_relay, nan=0.0, posinf=0.0, neginf=0.0)
                    img_for_node_raw.backward(grad_relay / 2.0)
                    
                    if latents_init.grad is not None:
                        g_cpu = latents_init.grad.detach().cpu().clone()
                        g_cpu = torch.nan_to_num(g_cpu, nan=0.0, posinf=0.0, neginf=0.0)
                        if use_ldino and matched_ent_node and matched_ent_node in current_masks:
                            mk_np_loc = current_masks[matched_ent_node]
                            mk_pil = Image.fromarray((mk_np_loc * 255).astype(np.uint8))
                            # Compute expected unpacked spatial dims
                            latent_h = 2 * (int(height) // (self.vae_scale_factor * 2))
                            latent_w = 2 * (int(width) // (self.vae_scale_factor * 2))
                            mk_lat = mk_pil.resize((latent_w, latent_h), Image.Resampling.BILINEAR)
                            mk_ts_lat = torch.from_numpy(np.array(mk_lat)/255.0).float().unsqueeze(0).unsqueeze(0).to(device)
                            mk_ts_lat = mk_ts_lat.expand(1, num_channels_latents, latent_h, latent_w)
                            mk_packed = self._pack_latents(mk_ts_lat, 1, num_channels_latents, latent_h, latent_w)
                            mk_packed = mk_packed.detach().cpu()
                            g_cpu = g_cpu * mk_packed
                        new_gradients_cpu.append((n_id, g_cpu))
            
            raw_map_ep = dict(new_individual_scores)
            masked_ep, roots_ep, _ = graph_evaluator.evaluate(raw_map_ep)
            raw_avg_ep = sum(v for _, v in new_individual_scores) / max(len(new_individual_scores), 1)
            vqa_score_val = max(raw_avg_ep, 1e-6)
            
            root_gradients_ep = []
            for root_id, _ in roots_ep.items():
                tree_nodes = set()
                def dfs(nid):
                    if nid in tree_nodes: return
                    tree_nodes.add(nid)
                    for child_id, child in graph_evaluator.nodes.items():
                        parents = child.get("parent_id")
                        if (isinstance(parents, list) and nid in parents) or parents == nid:
                            dfs(child_id)
                dfs(root_id)
                
                tree_grad = torch.zeros_like(new_gradients_cpu[0][1])
                for n_id, g in new_gradients_cpu:
                    if n_id in tree_nodes:
                        coeff = 1.0
                        for on_id in tree_nodes:
                            if on_id != n_id:
                                coeff *= max(raw_map_ep.get(on_id, 1e-9), 1e-9)
                        tree_grad += coeff * g
                root_gradients_ep.append(tree_grad)
                
            stored_grad = torch.mean(torch.stack(root_gradients_ep), dim=0).to(latents_init.device) if root_gradients_ep else torch.zeros_like(latents_init.detach().cpu()).to(latents_init.device)
            stored_grad = torch.nan_to_num(stored_grad, nan=0.0, posinf=0.0, neginf=0.0)
            g_norm_ep = stored_grad.float().norm().item()
            if g_norm_ep > MAX_GRAD_NORM:
                stored_grad = stored_grad * (MAX_GRAD_NORM / g_norm_ep)
            
            if vqa_score_val > max_vqa:
                max_vqa = vqa_score_val
                target_latents = latents_init.detach().clone()
            
            print(f"  [Epoch {ep}] VQA={vqa_score_val:.4f} (best={max_vqa:.4f}), lr={step_lr:.4f}")
            gc.collect()
            torch.cuda.empty_cache()
            
        print(f"\n[Flux-DINO] Optimization complete! Best VQA: {max_vqa:.4f}")
        with torch.no_grad():
            noise_list = denoise(target_latents)
            self.scheduler.set_timesteps(num_inference_steps)
            final_latents = reverse(target_latents, noise_list)
            self.scheduler.set_timesteps(num_inference_steps)
            final_image = decode_latents(final_latents)
            
        images = self.image_processor.postprocess(final_image.detach().cpu(), output_type=output_type)
        
        self.maybe_free_model_hooks()
        if not return_dict: return (images,)
        return FluxPipelineOutput(images=images)
