import os
import sys
import yaml
import math
import av
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

def save_animation(conf, imgs):
    # animated GIF
    imgs[0].save(
        "animation.gif", format="GIF", append_images=imgs,
        save_all=True, duration=1000 // conf.frame_rate, loop=0)

    # H.264 encoded mp4
    container = av.open("animation.mp4", "w")
    stream = container.add_stream(
        codec_name="h264", rate=conf.frame_rate,
        options={"preset": "slow", "crf": "15"} )
    stream.width = conf.frame_width
    stream.height = conf.frame_height
    stream.pix_fmt = "yuv420p"

    for img in imgs:
        frame = av.VideoFrame.from_image(img)
        packet = stream.encode(frame)
        container.mux(packet)

    # flush stream
    for packet in stream.encode():
        container.mux(packet)
    container.close()

def animate(conf):
    imgs = []
    engine = StableDiffusionEngine(conf)
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
    print(f"Interpolated to {len(imgs)} frames")

    # add a few freeze frames
    imgs = 20*[imgs[0]] + imgs + 20*[imgs[-1]]

    # append frames in reverse
    imgs += imgs[::-1]
    return imgs

def main():
    conf = Config()
    print(f"Configuration: \n{conf}")

    imgs = animate(conf)
    save_animation(conf, imgs)
    print(f"Saved output to animation.gif")

if __name__ == "__main__":
    main()
