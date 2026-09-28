import argparse
import os

import torch
from diffusers.utils import load_image

from inference_stage1 import predict_coc_and_disparity
from inference_stage2 import render_bokeh
from utils import add_bokeh_args, get_bokeh_scale, get_stage_dir, load_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run stage 1 and stage 2 end to end on a single image.")
    parser.add_argument("--image_path", type=str, required=True)
    add_bokeh_args(parser)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.1-Fill-dev")
    parser.add_argument(
        "--anybokeh_model_name_or_path",
        type=str,
        default="itsmag11/AnyBokeh",
        help="Hub repo id or local folder with the stage1/ and stage2/ checkpoints.",
    )
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--stage1_resolution", type=int, default=512)
    parser.add_argument("--stage2_resolution", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.bokeh_scale = get_bokeh_scale(parser, args)
    return args


def main(args):
    image = load_image(args.image_path)
    w, h = image.size
    if not (0 <= args.focus_x < w and 0 <= args.focus_y < h):
        raise ValueError(f"Focus point ({args.focus_x}, {args.focus_y}) is outside the {w}x{h} image.")

    stage1_dir = get_stage_dir(args.anybokeh_model_name_or_path, "stage1")
    stage2_dir = get_stage_dir(args.anybokeh_model_name_or_path, "stage2")

    # Both stages share the same FLUX.1-Fill-dev backbone, so only the LoRA is swapped in between.
    pipe = load_pipeline(args.pretrained_model_name_or_path)

    pipe.load_lora_weights(stage1_dir, weight_name="pytorch_lora_weights.safetensors")
    pipe.vae.enable_tiling()
    coc, disp = predict_coc_and_disparity(
        pipe,
        stage1_dir,
        args.image_path,
        args.stage1_resolution,
        args.num_inference_steps,
        args.seed,
    )

    pipe.unload_lora_weights()
    pipe.vae.disable_tiling()
    torch.cuda.empty_cache()

    pipe.load_lora_weights(stage2_dir, weight_name="pytorch_lora_weights.safetensors")
    result = render_bokeh(
        pipe,
        stage2_dir,
        image,
        coc,
        disp,
        (args.focus_x, args.focus_y),
        args.bokeh_scale,
        args.stage2_resolution,
        args.num_inference_steps,
        args.seed,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    result.save(args.output_path, quality=95)


if __name__ == "__main__":
    args = parse_args()
    main(args)
