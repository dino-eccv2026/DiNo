"""
L-DINO-CoT Flux Image Generation Script

Generates images using L-VQA optimized diffusion with FLUX.1.
Loads prompts from a JSON file (e.g., sample_data/single_prompt.json).

Usage:
    python run_dinot_flux.py --json sample_data/single_prompt.json --output results_flux
"""

import torch
import t2v_metrics
import os
import json
import argparse
import sys
import traceback
import gc

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

from pipelines.pipeline_flux_dinot import FluxDiNOTPipeline

# Supported VQA models
ALL_VQA_MODELS = [
    "clip-flant5-xl",
    "llava-v1.5-13b",
    "sharegpt4v-13b",
    "instructblip-flant5-xl",
]

torch.cuda.empty_cache()

# Parse arguments
parser = argparse.ArgumentParser(description="Generate L-VQA optimized images with Flux")
parser.add_argument("--json", "-j", type=str, required=True,
                    help="JSON file with decomposed prompts (e.g., sample_data/single_prompt.json)")
parser.add_argument("--output", "-o", type=str, default="results_flux",
                    help="Output directory for generated images")
parser.add_argument("--num_per_prompt", "-n", type=int, default=1,
                    help="Number of images to generate per prompt")
parser.add_argument("--seed", "-s", type=int, default=33,
                    help="Base random seed")
parser.add_argument("--start_idx", type=int, default=0,
                    help="Start index in prompts list")
parser.add_argument("--max_prompts", type=int, default=None,
                    help="Maximum number of prompts to process")
parser.add_argument("--epochs", "-e", type=int, default=5,
                    help="Number of optimization epochs (default: 5)")
parser.add_argument("--inference_steps", type=int, default=28,
                    help="Number of inference steps (default: 28)")
parser.add_argument("--flux_model", type=str, default="black-forest-labs/FLUX.1-schnell",
                    help="Which Flux variant to use")
parser.add_argument("--guidance_scale", type=float, default=3.5,
                    help="Guidance scale (used directly as guidance tensor)")
parser.add_argument("--disable_cpu_offload", action="store_true",
                    help="Disable Sequence Model CPU Offload (offload is highly recommended for VRAM savings)")
parser.add_argument("--sd_device", type=str, default="cuda:0",
                    help="Device for Flux pipeline")
parser.add_argument("--vqa_device", type=str, default="cuda:1",
                    help="Device for VQA model")
parser.add_argument("--vqa_model", type=str, default="clip-flant5-xl", choices=ALL_VQA_MODELS,
                    help="VQA model to use for scoring")
parser.add_argument("--no_ldino", action="store_true",
                    help="Disable L-DINO-CoT (simpler generation, less memory)")

args = parser.parse_args()

# Setup devices
sd_device = args.sd_device
vqa_model_device = args.vqa_device

print(f"[Setup] Loading Flux Pipeline {args.flux_model}")
pipe = FluxDiNOTPipeline.from_pretrained(
    args.flux_model,
    torch_dtype=torch.bfloat16
)

if not args.disable_cpu_offload:
    print("[Setup] Enabling Sequence Model CPU Offload")
    pipe.enable_model_cpu_offload(device=torch.device(sd_device))
else:
    print(f"[Setup] Moving pipeline to {sd_device}")
    pipe.to(sd_device)

print(f"[Setup] Loading VQA model ({args.vqa_model}) on {vqa_model_device}")
vqa_model = t2v_metrics.VQAScore(model=args.vqa_model, device=vqa_model_device)
pipe.init_vqa_model(vqa_model, vqa_model_device)

# Generator
generator = torch.Generator(device=sd_device).manual_seed(args.seed)

# Load prompts from JSON
if not os.path.exists(args.json):
    print(f"[Error] JSON file not found: {args.json}")
    sys.exit(1)

print(f"[Data] Loading prompts from {args.json}")
with open(args.json, 'r') as f:
    raw = json.load(f)

prompts = raw.get("prompts", raw) if isinstance(raw, dict) else raw
print(f"[Data] Loaded {len(prompts)} prompts")

# Determine range
start_idx = args.start_idx
end_idx = len(prompts)
if args.max_prompts is not None:
    end_idx = min(start_idx + args.max_prompts, len(prompts))

print(f"[Run] Processing prompts {start_idx} to {end_idx}")
print(f"[Run] Generating {args.num_per_prompt} images per prompt")
print(f"[Run] Output directory: {args.output}")
print(f"[Run] Optimization epochs: {args.epochs}")
print(f"[Run] L-DINO-CoT: {'Disabled' if args.no_ldino else 'Enabled'}")

os.makedirs(args.output, exist_ok=True)

results = {
    "config": {
        "model": args.flux_model,
        "seed": args.seed,
        "num_per_prompt": args.num_per_prompt,
        "optimization_epochs": args.epochs,
        "inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "vqa_model": args.vqa_model,
        "ldino_enabled": not args.no_ldino
    },
    "images": []
}

for idx in range(start_idx, end_idx):
    data = prompts[idx]
    prompt = data['prompt']

    print(f"\n{'='*60}")
    print(f"[{idx+1}/{end_idx}] Prompt: {prompt}")
    print(f"{'='*60}")

    safe_prompt = prompt.replace(" ", "_").replace("/", "_")[:50]
    prompt_dir = os.path.join(args.output, f"{idx:03d}_{safe_prompt}")
    os.makedirs(prompt_dir, exist_ok=True)

    for i in range(args.num_per_prompt):
        current_seed = args.seed + idx * args.num_per_prompt + i
        generator.manual_seed(current_seed)

        print(f"\n  [Image {i+1}/{args.num_per_prompt}] seed={current_seed}")

        pipeline_args = {
            "prompt": prompt,
            "dependency_graph": data.get('dependency_graph', None),
            "num_inference_steps": args.inference_steps,
            "height": 1024,
            "width": 1024,
            "guidance_scale": args.guidance_scale,
            "generator": generator,
            "optimization_epoch": args.epochs,
        }

        if not args.no_ldino:
            pipeline_args["use_localized_vqa"] = True

        try:
            output = pipe(**pipeline_args)
            images = output.images
            image = images[0] if isinstance(images, list) else images

            out_path = os.path.join(prompt_dir, f"seed_{current_seed}.png")
            image.save(out_path)
            print(f"  ✓ Saved: {out_path}")

            results["images"].append({
                "prompt": prompt, "seed": current_seed,
                "path": out_path, "prompt_idx": idx, "image_idx": i
            })

        except Exception as e:
            print(f"  ✗ Error: {e}")
            traceback.print_exc()
            results["images"].append({
                "prompt": prompt, "seed": current_seed,
                "path": None, "error": str(e),
                "prompt_idx": idx, "image_idx": i
            })

        gc.collect()
        torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()

manifest_path = os.path.join(args.output, "manifest.json")
with open(manifest_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n[Done] Saved manifest to {manifest_path}")

print(f"\n{'='*60}")
print(f"Generation complete!")
print(f"  Total prompts: {end_idx - start_idx}")
print(f"  Total images: {len([r for r in results['images'] if r.get('path')])}")
print(f"  Output directory: {args.output}")
print(f"{'='*60}")
