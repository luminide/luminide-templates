import os
import sys
import yaml
import math
import torch
import numpy as np
from PIL import Image

sys.path.append("../film_net")
from eval import interpolator
from eval import util

from config import Config
from diffusion import StableDiffusionEngine


def interpolate(conf, imgs, engine):
    # insert intermediate frames using pretrained film_net model
    film_net = interpolator.Interpolator(
        "../film_net_models/film_net/Style/saved_model", None)
    imgs = [np.array(img, dtype=np.float32) / 255. for img in imgs]
    imgs = util.interpolate_recursively_from_memory(
        imgs, conf.interp_recursion, film_net)
    imgs = [engine.pipe.numpy_to_pil(img.clip(0, 1))[0] for img in imgs]
    return imgs

def main():
    conf = Config()
    print(f"Configuration: \n{conf}")
    engine = StableDiffusionEngine(conf)

    imgs = []
    for idx in range(conf.num_frames + 1):
        print(f"Generating frame {idx}...")
        # we linearly interpolate text embeddings from the two given
        # prompts before generating a frame
        lerp_weight = 1.0 * idx / conf.num_frames
        image = engine.generate_frame(lerp_weight)
        imgs.append(image)
        image.save(f'result{idx:03d}.png')

    print("Using film_net to interpolate frames...")
    imgs = interpolate(conf, imgs, engine)
    append_images = 20*[imgs[0]] + imgs + 20*[imgs[-1]]
    append_images += append_images[::-1]
    output_filename = "animation.gif"
    imgs[0].save(
        output_filename, format="GIF", append_images=append_images,
        save_all=True, duration=conf.frame_duration, loop=0)
    print(f"Saved result to {output_filename}")

if __name__ == "__main__":
    main()
