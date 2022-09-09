# adapted from https://github.com/huggingface/diffusers

import torch
import inspect
from typing import Optional
from torch import autocast
from diffusers import (
    StableDiffusionPipeline,
    PNDMScheduler,
)

class AnimationPipeline(StableDiffusionPipeline):

    @torch.no_grad()
    def create_text_embeddings(self, prompts):
        self.text_embeddings_list = []
        for i, prompt in enumerate(prompts):
            # get prompt text embeddings
            text_input = self.tokenizer(
                prompt, padding="max_length",
                max_length=self.tokenizer.model_max_length, truncation=True,
                return_tensors="pt")
            text_embeddings = self.text_encoder(
                text_input.input_ids.to(self.device))[0]

            # get unconditional embeddings for classifier free guidance
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""], padding="max_length",
                max_length=max_length, return_tensors="pt")
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device))[0]
            self.text_embeddings_list.append(
                torch.cat([uncond_embeddings, text_embeddings]))

    @torch.no_grad()
    def __call__(
        self,
        lerp_weight: float,
        num_inference_steps: int,
        guidance_scale: float,
        height: Optional[int] = 576,
        width: Optional[int] = 704,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
    ):
        # set timesteps
        extra_set_kwargs = {}
        offset = 1
        extra_set_kwargs["offset"] = 1
        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # get the original timestep using init_timestep
        init_timestep = num_inference_steps + offset
        init_timestep = min(init_timestep, num_inference_steps)
        timesteps = self.scheduler.timesteps[-init_timestep]
        #import ipdb;ipdb.set_trace()
        timesteps = torch.tensor(
            [timesteps], dtype=torch.long, device=self.device)

        # get weighted average of text embeddings
        text_embeddings = (
            (1 - lerp_weight) * self.text_embeddings_list[0] +
            lerp_weight * self.text_embeddings_list[1])

        latents_shape = (1, self.unet.in_channels, height // 8, width // 8)
        latents = torch.randn(
            latents_shape, generator=generator, device=self.device)
        t_start = max(num_inference_steps - init_timestep + offset, 0)
        for i, t in enumerate(self.scheduler.timesteps[t_start:]):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t,
                encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]

        # scale and decode the image latents with vae
        image = self.vae.decode(1 / 0.18215 * latents)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)[0]
        return image


class StableDiffusionEngine:
    def __init__(
            self, conf, beta_start=0.00085, beta_end=0.012,
            beta_schedule="scaled_linear"):
        self.conf = conf
        self.device = "cuda"

        scheduler = PNDMScheduler(
            beta_start=beta_start, beta_end=beta_end,
            beta_schedule=beta_schedule, skip_prk_steps=True, tensor_format="np")
        model_id_or_path = "CompVis/stable-diffusion-v1-4"
        pipe = AnimationPipeline.from_pretrained(
            model_id_or_path, scheduler=scheduler, revision="fp16",
            torch_dtype=torch.float16, use_auth_token=True)
        self.pipe = pipe.to(self.device)
        with autocast("cuda"):
            pipe.create_text_embeddings([conf.begin_prompt, conf.end_prompt])

    def generate_frame(self, lerp_weight):
        generator = torch.Generator(device=self.device).manual_seed(self.conf.seed)
        with autocast("cuda"):
            image = self.pipe(
                lerp_weight=lerp_weight,
                num_inference_steps=self.conf.num_inference_steps,
                guidance_scale=self.conf.guidance_scale,
                height=self.conf.frame_height, width=self.conf.frame_width,
                generator=generator)
        return image
