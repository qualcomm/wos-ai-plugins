# =============================================================================
#
# Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# =============================================================================

import time
from PIL import Image
import os
import cv2
import numpy as np
import torch
from transformers import CLIPTokenizer
from diffusers import DPMSolverMultistepScheduler
from diffusers.utils import load_image
import sys
import json
import qairt_constants as consts
import traceback

from qai_appbuilder import (
    QNNContext,
    Runtime,
    LogLevel,
    ProfilingLevel,
    PerfProfile,
    QNNConfig,
    timer,
)

OUT_H, OUT_W = 512, 512

tokenizer = None
scheduler = None
tokenizer_max_length = 77

text_encoder = None
unet = None
vae_decoder = None
controlnet = None

user_prompt = ""
uncond_prompt = ""
user_seed = np.int64(1)
user_step = 20
user_text_guidance = 9.0
input_image_path = None
output_image_path = None


def reset():
    global scheduler, tokenizer, text_encoder, unet, vae_decoder, controlnet
    global user_prompt, uncond_prompt, user_seed, user_step, user_text_guidance
    global input_image_path, output_image_path

    tokenizer = None
    scheduler = None
    text_encoder = None
    unet = None
    vae_decoder = None
    controlnet = None

    user_prompt = ""
    uncond_prompt = ""
    user_seed = np.int64(1)
    user_step = 20
    user_text_guidance = 9.0
    input_image_path = None
    output_image_path = None


class TextEncoder(QNNContext):
    def Inference(self, input_data):
        output_data = super().Inference([input_data])[0]
        return output_data.reshape((1, 77, 768))


class Unet(QNNContext):
    def Inference(
        self,
        input_data_1,
        input_data_2,
        input_data_3,
        input_data_4,
        input_data_5,
        input_data_6,
        input_data_7,
        input_data_8,
        input_data_9,
        input_data_10,
        input_data_11,
        input_data_12,
        input_data_13,
        input_data_14,
        input_data_15,
        input_data_16,
    ):
        input_data_1 = input_data_1.reshape(input_data_1.size)
        input_data_3 = input_data_3.reshape(input_data_3.size)
        input_data_4 = input_data_4.reshape(input_data_4.size)
        input_data_5 = input_data_5.reshape(input_data_5.size)
        input_data_6 = input_data_6.reshape(input_data_6.size)
        input_data_7 = input_data_7.reshape(input_data_7.size)
        input_data_8 = input_data_8.reshape(input_data_8.size)
        input_data_9 = input_data_9.reshape(input_data_9.size)
        input_data_10 = input_data_10.reshape(input_data_10.size)
        input_data_11 = input_data_11.reshape(input_data_11.size)
        input_data_12 = input_data_12.reshape(input_data_12.size)
        input_data_13 = input_data_13.reshape(input_data_13.size)
        input_data_14 = input_data_14.reshape(input_data_14.size)
        input_data_15 = input_data_15.reshape(input_data_15.size)
        input_data_16 = input_data_16.reshape(input_data_16.size)

        input_datas = [
            input_data_1, input_data_2, input_data_3, input_data_4,
            input_data_5, input_data_6, input_data_7, input_data_8,
            input_data_9, input_data_10, input_data_11, input_data_12,
            input_data_13, input_data_14, input_data_15, input_data_16,
        ]

        output_data = super().Inference(input_datas)[0]
        return output_data.reshape(1, 64, 64, 4)


class VaeDecoder(QNNContext):
    def Inference(self, input_data):
        output_data = super().Inference([input_data.reshape(input_data.size)])[0]
        return output_data


class ControlNet(QNNContext):
    def Inference(self, input_data_1, input_data_2, input_data_3, input_data_4):
        input_data_1 = input_data_1.reshape(input_data_1.size)
        input_data_3 = input_data_3.reshape(input_data_3.size)
        input_data_4 = input_data_4.reshape(input_data_4.size)
        return super().Inference([input_data_1, input_data_2, input_data_3, input_data_4])


def model_initialize():
    global scheduler, tokenizer, text_encoder, unet, vae_decoder, controlnet

    text_encoder_model = "{}\\{}.bin".format(consts.CONTROLNET_DIR, "text_encoder")
    unet_model = "{}\\{}.bin".format(consts.CONTROLNET_DIR, "unet")
    vae_decoder_model = "{}\\{}.bin".format(consts.CONTROLNET_DIR, "vae")
    controlnet_model = "{}\\{}.bin".format(consts.CONTROLNET_DIR, "controlnet")

    text_encoder = TextEncoder("text_encoder", text_encoder_model)
    print("Model initialization complete for 1 model(s).")

    vae_decoder = VaeDecoder("vae_decoder", vae_decoder_model)
    print("Model initialization complete for 2 model(s).")

    controlnet = ControlNet("controlnet", controlnet_model)
    print("Model initialization complete for 3 model(s).")

    unet = Unet("model_unet", unet_model)
    print("Model initialization complete for 4 model(s).")

    tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32", 
            cache_dir=consts.CACHE_DIR
    )

    # Scheduler - initializing the Scheduler.
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
    )

    torch.from_numpy(np.array([1]))
    return True


def run_tokenizer(prompt):
    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer_max_length, truncation=True
    )
    return np.array([text_input.input_ids], dtype=np.int32)


def setup_parameters(input_img_path, output_img_path, prompt, un_prompt, seed, step, text_guidance):
    global user_prompt, uncond_prompt, user_seed, user_step, user_text_guidance
    global input_image_path, output_image_path

    user_prompt = prompt
    uncond_prompt = un_prompt
    user_seed = seed
    user_step = step
    user_text_guidance = text_guidance
    input_image_path = input_img_path
    output_image_path = output_img_path

    assert isinstance(user_seed, np.int64), "user_seed should be of type int64"
    assert isinstance(user_step, int), "user_step should be of type int"
    assert isinstance(user_text_guidance, float), "user_text_guidance should be of type float"
    assert 5.0 <= user_text_guidance <= 15.0, "user_text_guidance should be a float from [5.0, 15.0]"


def run_scheduler(noise_pred_uncond, noise_pred_text, latent_in, timestep):
    noise_pred_uncond = torch.from_numpy(np.transpose(noise_pred_uncond, (0, 3, 1, 2)).copy())
    noise_pred_text = torch.from_numpy(np.transpose(noise_pred_text, (0, 3, 1, 2)).copy())
    latent_in = torch.from_numpy(np.transpose(latent_in, (0, 3, 1, 2)).copy())

    noise_pred = noise_pred_uncond + user_text_guidance * (noise_pred_text - noise_pred_uncond)
    latent_out = scheduler.step(noise_pred, timestep, latent_in).prev_sample.numpy()
    return np.transpose(latent_out, (0, 2, 3, 1)).copy()


def make_canny_image(input_image: Image):
    image = np.asarray(input_image)
    image = cv2.Canny(image, 100, 200)
    cv2.imwrite(os.path.join(consts.OUTPUTS_DIR, "canny.png"), image)
    image = np.concatenate([image[:, :, None]] * 3, axis=2)
    image = np.array(Image.fromarray(image))[None, :].astype(np.float32) / 255.0
    return image


def model_execute(callback):
    PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)
    print("Model Execution Start")
    scheduler.set_timesteps(user_step)

    cond_tokens = run_tokenizer(user_prompt)
    uncond_tokens = run_tokenizer(uncond_prompt)

    uncond_text_embedding = text_encoder.Inference(uncond_tokens)
    user_text_embedding = text_encoder.Inference(cond_tokens)

    latents_shape = (1, 4, OUT_H // 8, OUT_W // 8)
    random_init_latent = torch.randn(latents_shape, generator=torch.manual_seed(user_seed))
    latent_in = (random_init_latent * scheduler.init_noise_sigma).numpy().transpose(0, 2, 3, 1)

    canny_image = make_canny_image(load_image(input_image_path))

    for step, timestep in enumerate(scheduler.timesteps):
        print(f"Step {step} Running...")
        sys.stdout.flush()

        latent_input = scheduler.scale_model_input(
            torch.as_tensor(latent_in).contiguous(), timestep
        ).numpy()
        time_embedding = np.array([[timestep.item()]], dtype=np.float32)

        controlnet_out = controlnet.Inference(user_text_embedding, canny_image, latent_input, time_embedding)
        conditional_noise_pred = unet.Inference(time_embedding, latent_input, user_text_embedding, *controlnet_out)

        controlnet_out = controlnet.Inference(uncond_text_embedding, canny_image, latent_input, time_embedding)
        unconditional_noise_pred = unet.Inference(time_embedding, latent_input, uncond_text_embedding, *controlnet_out)

        latent_in = run_scheduler(unconditional_noise_pred, conditional_noise_pred, latent_in, timestep)
        callback(step)

    output_image = vae_decoder.Inference(latent_in).reshape(OUT_H, OUT_W, -1)
    PerfProfile.RelPerfProfileGlobal()

    if len(output_image) == 0:
        callback(None)
        return False

    output_image = np.clip(output_image * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(output_image, mode="RGB").save(output_image_path)
    callback(output_image_path)
    return True


def model_destroy():
    global text_encoder, unet, vae_decoder, controlnet
    del text_encoder
    del unet
    del vae_decoder
    del controlnet


def SetQNNConfig():
    QNNConfig.Config(consts.QNN_LIBS_DIR, Runtime.HTP, LogLevel.ERROR, ProfilingLevel.BASIC)


def modelExecuteCallback(result):
    if result is None or isinstance(result, str):
        print("modelExecuteCallback result: " + (result or "None"))
    else:
        print("modelExecuteCallback result: " + str(int((result + 1) * 100 / user_step)))


def load_model():
    reset()
    SetQNNConfig()
    model_initialize()
    return True


def unload_model():
    model_destroy()
    return True


def run_model(input_img_path, output_img_path, user_prompt, uncond_prompt, user_seed, user_step, user_text_guidance):
    setup_parameters(
        input_img_path,
        output_img_path,
        user_prompt,
        uncond_prompt,
        np.int64(int(user_seed)),
        int(user_step),
        float(user_text_guidance),
    )
    return model_execute(modelExecuteCallback)


def handle_command(command):
    action = command.get("action")
    params = command.get("params", {})
    start_time = time.time()

    if action == "load_model":
        result = load_model()
        result_str = f"LOAD_MODEL_RESULT_START\n{result}\nLOAD_MODEL_RESULT_END"
    elif action == "run_model":
        result = run_model(*list(params.values()))
        result_str = f"RUN_MODEL_RESULT_START\n{result}\nRUN_MODEL_RESULT_END"
    elif action == "unload_model":
        result = unload_model()
        result_str = f"UNLOAD_MODEL_RESULT_START\n{result}\nUNLOAD_MODEL_RESULT_END"

    print(f"Time taken for {action}: {time.time() - start_time}\n")
    return result_str


if __name__ == "__main__":
    try:
        while True:
            command_line = sys.stdin.readline().strip()
            if not command_line:
                break
            command = json.loads(command_line)
            response = handle_command(command)
            print(response)
            print("\nEND_DATA\n")
            sys.stdout.flush()
    except Exception as e:
        print(f"\n{str(e)}\n{traceback.format_exc()}")
        print("\nEND_DATA\n")
        sys.stdout.flush()
