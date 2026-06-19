# =============================================================================
#
# Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# =============================================================================

from PIL import Image
import os
import numpy as np
import torch
import install
from transformers import CLIPTokenizer
from modules import paths
from qai_appbuilder import (
    QNNContext,
    Runtime,
    LogLevel,
    ProfilingLevel,
    PerfProfile,
    QNNConfig,
    timer,
)
import qairt_constants as consts
import common_utils as utils
from pipeline_utils import StableDiffusionInput, UpscalerPipeline, QPipeline, set_scheduler, ensure_qnn_config

model_path_1_5 = os.path.abspath(
    os.path.join(paths.models_path, "Stable-diffusion", "qcom-Stable-Diffusion-v1.5")
)
model_path_2_1 = os.path.abspath(
    os.path.join(paths.models_path, "Stable-diffusion", "qcom-Stable-Diffusion-v2.1")
)


user_alpha = torch.tensor([128],dtype=torch.float32).numpy()

class TextEncoder(QNNContext):
    # @timer
    def Inference(self, input_data, sd_version):
        input_datas = [input_data]

        # Output of Text encoder should be of shape (1, 77, 768)
        if sd_version == "1.5":
            output_data = super().Inference(input_datas)[0]
            output_data = output_data.reshape((1, 77, 768))
        elif sd_version == "2.1":
            output_data = super().Inference(input_datas)[0]
            output_data = output_data.reshape((1, 77, 1024))
        return output_data

class TextEncoder_2(QNNContext):
    def Inference(self, input_data):
        input_datas=[input_data]
        output_data = super().Inference(input_datas)

        # Output of Text encoder should be of shape (1, 77, 1280)
        output_data_1 = output_data[0].reshape((1, 77, 1280))
        output_data_2 = output_data[1].reshape((1, 1280))
        return output_data_1, output_data_2 



class Unet(QNNContext):
    # @timer
    def Inference(self, version, input_data_1, input_data_2, input_data_3, user_alpha =None, input_data_4 = None):
        input_data_1 = input_data_1.reshape(input_data_1.size)
        input_data_3 = input_data_3.reshape(input_data_3.size)
        if version=="1.5" or version=="2.1":

            input_datas = [input_data_1, input_data_2, input_data_3]
            output_data = super().Inference(input_datas)[0]
            output_data = output_data.reshape(1, 64, 64, 4)
        else:
            user_alpha = user_alpha.reshape(user_alpha.shape)
            input_datas=[input_data_1, input_data_2, input_data_3, user_alpha, input_data_4]
            output_data = super().Inference(input_datas)[0]
            output_data = output_data.reshape((1, 128, 128, 4))


        return output_data


class VaeDecoder(QNNContext):
    # @timer
    def Inference(self, input_data, version):
        input_data = input_data.reshape(input_data.size)
        input_datas = [input_data]
        if version == "1.5" or version == "2.1":
            output_data = super().Inference(input_datas)[0]
        else:
            output_data = super().Inference(input_datas)[0]
            # Denormalize and convert floating point output into 8 bits RGB image
            output_data = np.clip((output_data / 2 + 0.5)*255.0, 0.0, 255.0).astype(np.uint8)
            # Output of VAE should be of shape (1024, 1024, 3)
            output_data = output_data.reshape((1024, 1024, 3)) 

        return output_data

class QnnStableDiffusionPipeline(QPipeline):
    TOKENIZER_MAX_LENGTH = 77  # Define Tokenizer output max length (must be 77)
    scheduler = None
    tokenizer = None
    tokenizer_2 = None
    text_encoder = None
    text_encoder_2 = None
    unet = None
    vae_decoder = None

    sd_version = None

    def __init__(self, model_name):
        super().__init__(model_name)
        self.set_model_version(model_name)
        self.load_model()

        torch.from_numpy(
            np.array([1])
        )  # Let LazyImport to import the torch & numpy lib here.

    def is_sd_1_5(self):
        return self.sd_version == "1.5"
    def is_sd_2_1(self):
        return self.sd_version == "2.1"

    def run_tokenizer(self, prompt):
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.TOKENIZER_MAX_LENGTH,
            truncation=True,
        )
        text_input = np.array(text_input.input_ids, dtype=np.int32)
        return text_input
    
    def run_tokenizer_2(self, prompt):
        text_input = self.tokenizer_2(prompt, padding="max_length", max_length=self.TOKENIZER_MAX_LENGTH, truncation=True)
        text_input = np.array(text_input.input_ids, dtype=np.float32)
        return text_input

    def run_scheduler(
        self,
        sd_input,
        noise_pred_uncond,
        noise_pred_text,
        latent_in,
        timestep,
    ):
        # Convert all inputs from NHWC to NCHW
        noise_pred_uncond = np.transpose(noise_pred_uncond, (0, 3, 1, 2)).copy()
        noise_pred_text = np.transpose(noise_pred_text, (0, 3, 1, 2)).copy()
        latent_in = np.transpose(latent_in, (0, 3, 1, 2)).copy()

        # Convert all inputs to torch tensors
        noise_pred_uncond = torch.from_numpy(noise_pred_uncond)
        noise_pred_text = torch.from_numpy(noise_pred_text)
        latent_in = torch.from_numpy(latent_in)

        # Merge noise_pred_uncond and noise_pred_text based on user_text_guidance
        noise_pred = noise_pred_uncond + sd_input.user_text_guidance * (
            noise_pred_text - noise_pred_uncond
        )

        # Run Scheduler step
        latent_out = self.scheduler.step(
            noise_pred, timestep, latent_in
        ).prev_sample.numpy()

        # Convert latent_out from NCHW to NHWC
        latent_out = np.transpose(latent_out, (0, 2, 3, 1)).copy()

        return latent_out

    # Function to get timesteps
    def get_timestep(self, step):
        return np.int32(self.scheduler.timesteps.numpy()[step])

    def set_model_version(self, model_name):
        if model_name == "Stable-Diffusion-1.5":
            self.sd_version = "1.5"
        elif model_name == "Stable-Diffusion-2.1":
            self.sd_version = "2.1"

    def load_model(self):
        ensure_qnn_config()
        # model names
        model_text_encoder = "text_encoder"
        model_unet = "model_unet"
        model_vae_decoder = "vae_decoder"

        model_path = None
        if self.is_sd_1_5() or self.is_sd_2_1():
            model_binaries = [
                "text_encoder.serialized.bin",
                "unet.serialized.bin",
                "vae_decoder.serialized.bin"
            ]
            if self.is_sd_1_5():
                model_path = model_path_1_5
                self.tokenizer = CLIPTokenizer.from_pretrained(
                    "stable-diffusion-v1-5/stable-diffusion-v1-5",
                    subfolder="tokenizer",
                    revision="main",
                    cache_dir=consts.CACHE_DIR
                )
            else:
                model_path = model_path_2_1
                self.tokenizer = CLIPTokenizer.from_pretrained(
                    "sd2-community/stable-diffusion-2-1",
                    subfolder="tokenizer",
                    revision="main",
                    cache_dir=consts.CACHE_DIR,
                )
            text_encoder_model = "{}\\{}".format(model_path, model_binaries[0])
            unet_model = "{}\\{}".format(model_path, model_binaries[1])
            vae_decoder_model = "{}\\{}".format(model_path, model_binaries[2])
            print(f"Loading models from {model_path}")
            # Instance for Unet
            self.unet = Unet(model_unet, unet_model)

            # Instance for TextEncoder
            self.text_encoder = TextEncoder(model_text_encoder, text_encoder_model)

            # Instance for VaeDecoder
            self.vae_decoder = VaeDecoder(model_vae_decoder, vae_decoder_model)

    def reload_model(self, model_name):
        print(model_name)
        self.set_model_version(model_name)
        del self.tokenizer
        del self.text_encoder
        del self.text_encoder_2
        del self.unet
        del self.vae_decoder
        self.load_model()


    # Execute the Stable Diffusion pipeline
    def model_execute(
        self,
        sd_input: StableDiffusionInput,
        callback,
        # upscaler_pipeline: UpscalerPipeline,
    ) -> Image:
        image = None

        PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)
        if self.sd_version == "2.1":
            self.scheduler = set_scheduler("sd2-community/stable-diffusion-2-1", sd_input.sampler_name)
        elif self.sd_version == "1.5":
            self.scheduler = set_scheduler("stable-diffusion-v1-5/stable-diffusion-v1-5", sd_input.sampler_name)

        self.scheduler.set_timesteps(
            sd_input.user_step
        )  # Setting up user provided time steps for Scheduler

        # Run Tokenizer
        cond_tokens = self.run_tokenizer(sd_input.user_prompt)
        uncond_tokens = self.run_tokenizer(sd_input.uncond_prompt)

        if not (self.is_sd_1_5() or self.is_sd_2_1()):
            cond_tokens_2 = self.run_tokenizer_2(sd_input.user_prompt)
            uncond_tokens_2 = self.run_tokenizer_2(sd_input.uncond_prompt)
        


        # Run Text Encoder on Tokens
        user_text_embedding = self.text_encoder.Inference(cond_tokens, self.sd_version)
        uncond_text_embedding = self.text_encoder.Inference(uncond_tokens, self.sd_version )
        if not (self.is_sd_1_5() or self.is_sd_2_1()):
            user_text_embedding_2, user_pooled_embedding = self.text_encoder_2.Inference(cond_tokens_2)
            uncond_text_embedding_2, uncond_pooled_embedding = self.text_encoder_2.Inference(uncond_tokens_2)
            user_text_embedding = np.concatenate((user_text_embedding, user_text_embedding_2), axis=2)
            uncond_text_embedding = np.concatenate((uncond_text_embedding, uncond_text_embedding_2), axis=2) 

        if not (self.is_sd_1_5() or self.is_sd_2_1()):
            random_init_latent = torch.randn((1, 4, 128, 128), generator=torch.manual_seed(sd_input.user_seed)).numpy()
            latent_in = random_init_latent.transpose(0, 2, 3, 1).copy()
        # Initialize the latent input with random initial latent
        else:
            random_init_latent = torch.randn((1, 4, 64, 64), generator=torch.manual_seed(sd_input.user_seed))
            random_init_latent = random_init_latent * self.scheduler.init_noise_sigma
            latent_in = random_init_latent.numpy().transpose(0, 2, 3, 1)

        for step, timestep in enumerate(self.scheduler.timesteps):
            # Scale latent for this timestep before feeding into UNet
            latent_input = self.scheduler.scale_model_input(
                torch.as_tensor(latent_in).contiguous(),
                timestep,
            ).numpy()

            time_embedding = np.array([[timestep.item()]], dtype=np.float32)

            if not (self.is_sd_1_5() or self.is_sd_2_1()):
                unconditional_noise_pred = self.unet.Inference(self.sd_version, uncond_pooled_embedding, latent_in, uncond_text_embedding, user_alpha, time_embedding)
                conditional_noise_pred = self.unet.Inference(self.sd_version, user_pooled_embedding, latent_in, user_text_embedding, user_alpha, time_embedding)
            else:
                unconditional_noise_pred = self.unet.Inference(
                    self.sd_version, time_embedding, latent_input, uncond_text_embedding
                )
                conditional_noise_pred = self.unet.Inference(
                    self.sd_version, time_embedding, latent_input, user_text_embedding
                )

            latent_in = self.run_scheduler(
                sd_input,
                unconditional_noise_pred,
                conditional_noise_pred,
                latent_in,
                timestep,
            )

            callback(step)

        # Run VAE
        import datetime

        now = datetime.datetime.now()
        if not (self.is_sd_1_5() or self.is_sd_2_1()):
            output_image = self.vae_decoder.Inference(latent_in/0.13025, self.sd_version)
        else:
            output_image = self.vae_decoder.Inference(latent_in, self.sd_version)
        formatted_time = now.strftime("%Y_%m_%d_%H_%M_%S")

        if len(output_image) == 0:
            callback(None)
        else:
            if not (self.is_sd_1_5() or self.is_sd_2_1()):
                image_size = 1024
            else:
                image_size = 512

            if (self.is_sd_1_5() or self.is_sd_2_1()):
                output_image = np.clip(output_image * 255.0, 0.0, 255.0).astype(np.uint8)
                output_image = output_image.reshape(image_size, image_size, -1)
            image = Image.fromarray(output_image, mode="RGB")  # .save(image_path)

            callback(image)

        PerfProfile.RelPerfProfileGlobal()
        return image

    # Release all the models.
    def __del__(self):
        del self.text_encoder
        del self.unet
        del self.vae_decoder
        if self.scheduler:
            del self.scheduler
        del self.tokenizer

    def is_model_loaded(self):
        return self.unet != None
