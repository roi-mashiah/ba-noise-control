import os
import torch
import torchvision.transforms.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # Use GPU 2 and 3
import requests
import sys
from PIL import Image
import numpy as np
from io import BytesIO
import json

sys.path.append("/home/dcor/omerdh/DLproject/ReNoise-Inversion")
sys.path.append("/home/dcor/omerdh/DLproject/ReNoise-Inversion/src")
sys.path.append("/home/dcor/omerdh/DLproject/ReNoise-Inversion/src/utils")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "pipes"))
)
from eunms import Model_Type, Scheduler_Type
from enums_utils import get_pipes
from config import RunConfig
from main import run as invert


from diffusers import DDIMScheduler
from pipeline_stable_diffusion_opt import StableDiffusionPipeline
from pytorch_lightning import seed_everything
from scipy.fft import idct

from injection_utils import register_attention_editor_diffusers
from bounded_attention import BoundedAttention
import utils


def dct_based_image(max_freq_ind, num_active_bins=70, grid_size=(512, 512, 3)):
    img = np.random.randn(*grid_size) * 10 + 50
    random_ind = np.random.randint(1, max_freq_ind + 1, size=(2, num_active_bins))

    for r in random_ind.T:
        img[r[0] - 1, r[1] - 1, :] = np.random.rand(1, 3) * 1000 + 3000

    # Apply 2D IDCT
    temp = idct(img, axis=0, norm="ortho")
    inv_img = idct(temp, axis=1, norm="ortho")
    inv_img[inv_img < 64] = 0.0
    inv_img[inv_img > 232] = 255.0
    inv_img = np.clip(inv_img, 0, 255).astype(np.uint8)
    return inv_img


def load_model(device):
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    model = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        scheduler=scheduler,
        cross_attention_kwargs={"scale": 0.5},
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)
    model.enable_xformers_memory_efficient_attention()
    model.enable_sequential_cpu_offload()
    return model


def run(
    boxes,
    prompt,
    subject_token_indices,
    out_dir="out",
    seed=161,
    batch_size=1,
    filter_token_indices=None,
    eos_token_index=None,
    init_step_size=8,
    final_step_size=2,
    first_refinement_step=15,
    num_clusters_per_subject=3,
    cross_loss_scale=1.5,
    self_loss_scale=0.5,
    classifier_free_guidance_scale=7.5,
    num_gd_iterations=5,
    loss_threshold=0.2,
    num_guidance_steps=15,
    start_code=None,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model(device)

    seed_everything(seed)
    prompts = [prompt] * batch_size
    if start_code == None:
        start_code = torch.randn([len(prompts), 4, 64, 64], device=device)

    editor = BoundedAttention(
        boxes,
        prompts,
        subject_token_indices,
        list(range(12, 20)),
        list(range(12, 20)),
        cross_mask_layers=list(range(14, 20)),
        self_mask_layers=list(range(14, 20)),
        filter_token_indices=filter_token_indices,
        eos_token_index=eos_token_index,
        cross_loss_coef=cross_loss_scale,
        self_loss_coef=self_loss_scale,
        max_guidance_iter=num_guidance_steps,
        max_guidance_iter_per_step=num_gd_iterations,
        start_step_size=init_step_size,
        end_step_size=final_step_size,
        loss_stopping_value=loss_threshold,
        min_clustering_step=first_refinement_step,
        num_clusters_per_box=num_clusters_per_subject,
    )

    register_attention_editor_diffusers(model, editor)
    images = model(
        prompts, latents=start_code, guidance_scale=classifier_free_guidance_scale
    )
    return images


def save_images(images, boxes, out_dir, source):
    for image in images:
        image = F.to_pil_image(image)
        utils.draw_box(image, boxes).save(
            os.path.join(out_dir, f"{source}_w_boxes.png")
        )

    print("Syntheiszed images are saved in", out_dir)


def load_json(path=r"./db_inputs.json"):
    try:
        with open(path, "r") as reader:
            raw_json = json.load(reader)
        return raw_json
    except Exception as ex:
        print(f"Problem when opening the json!!!\n{ex}")


#def get_inputs(input_obj, preconstructed_img=False):
#    prompt, boxes, subject_token_indices = (
#        input_obj["prompt"],
#        input_obj["boxes"],
#        input_obj["references"],
#    )
#    if preconstructed_img:
#        img_path = os.path.join(
#            "/home/dcor/omerdh/DLproject/omri_files/bounded-attention/inputs/",
#            input_obj["img_path"],
#        )
#        input_image = Image.open(img_path).convert("RGB").resize((512, 512))
#    else:
#        input_image = None
#    return prompt, boxes, subject_token_indices, input_image


def main():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_type = Model_Type.SD15
        scheduler_type = Scheduler_Type.DDIM

#    inputs_json = load_json()
#    for input_key, input_obj in inputs_json.items():
#        if input_key != "45":
#           continue
#        prompt, boxes, subject_token_indices, input_image = get_inputs(input_obj, True)

        input_image = Image.open("carrot_brocc.jpeg").convert("RGB").resize((512, 512))

        boxes = [
            [0.0195, 0.2929, 0.2929, 0.5468],
            [0.3906, 0.2246, 0.6933, 0.5371]
        ]
        prompt = "A carrot on the left of a broccoli"
        subject_token_indices = [[2], [8]]

        pipe_inversion, pipe_inference = get_pipes(
            model_type, scheduler_type, device=device
        )
        config = RunConfig(
            model_type=model_type,
            num_inference_steps=50,
            num_inversion_steps=50,
            num_renoise_steps=1,
            scheduler_type=scheduler_type,
            perform_noise_correction=False,
            seed=7865,
        )

        _, inv_latent, _, all_latents = invert(
            input_image,
            prompt,
            config,
            pipe_inversion=pipe_inversion,
            pipe_inference=pipe_inference,
            do_reconstruction=False,
        )
        

        gaussian_noise = torch.randn(1, 4, 64, 64).cuda()

        # Get the dimensions (height and width) of the relevant 2D part of my_tensor
        height, width = inv_latent.shape[2], inv_latent.shape[3]

        # Create a mask with zeros and apply Gaussian noise outside the boxes
        mask = torch.zeros(1, 1, height, width).cuda()

        for box in boxes:
            # Convert relative coordinates to absolute pixel coordinates
            x_min = int(box[0] * width)
            y_min = int(box[1] * height)
            x_max = int(box[2] * width)
            y_max = int(box[3] * height)
    
        # Set the mask for the area inside the box to 1
        mask[:, :, y_min:y_max, x_min:x_max] = 1

        # Expand mask to cover all channels (4 channels in this case)
        mask = mask.expand_as(inv_latent)

        # Apply the mask: keep original values inside boxes and apply Gaussian noise outside
        inv_latent = inv_latent * mask + gaussian_noise * (1 - mask)

        ba_images = run(
            boxes, prompt, subject_token_indices, init_step_size=8, final_step_size=2
        )
        nc_images = run(
            boxes,
            prompt,
            subject_token_indices,
            init_step_size=8,
            final_step_size=2,
            start_code=inv_latent,
        )

        sample_count = len(os.listdir("out"))
        out_dir = os.path.join("out", f"omrirandbackgroundoriginalboxes4")
        os.makedirs(out_dir)

        save_images(ba_images, boxes, out_dir, source="ba")
        save_images(nc_images, boxes, out_dir, source="noise_control")


if __name__ == "__main__":
    main()
