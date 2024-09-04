import numpy as np
import torch
# fix imports
from eunms import Model_Type, Scheduler_Type
from utils.enums_utils import get_pipes
from config import RunConfig

from main import run as invert
from run_sd import run

def draw_boxes_on_grid(boxes, grid_size=(64, 64)):
    grid = np.zeros(grid_size)
    
    for box in boxes:
        x_min = int(box[0] * grid_size[1])
        y_min = int(box[1] * grid_size[0])
        x_max = int(box[2] * grid_size[1])
        y_max = int(box[3] * grid_size[0])
        
        grid[y_min:y_max, x_min:x_max] = 1
    
    return grid

def get_latents_from_renoise(input_image, prompt):
    model_type = Model_Type.SD15
    scheduler_type = Scheduler_Type.DDIM
    pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device)
    config = RunConfig(model_type = model_type,
                        num_inference_steps = 50,
                        num_inversion_steps = 50,
                        num_renoise_steps = 1,
                        scheduler_type = scheduler_type,
                        perform_noise_correction = False,
                        seed = 7865)
    _, inv_latent, _, all_latents = invert(input_image,
                                        prompt,
                                        config,
                                        pipe_inversion=pipe_inversion,
                                        pipe_inference=pipe_inference,
                                        do_reconstruction=False)

    return inv_latent

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    boxes = [
        [0.05, 0.2, 0.45, 0.8],
        [0.55, 0.2, 0.95, 0.8],
    ]
    prompt = ""
    input_image = draw_boxes_on_grid(boxes)

    # reshape to size of ReNoise input
    latents = get_latents_from_renoise(input_image, prompt)

    subject_token_indices = [[2, 3], [6, 7]]

    # call bounded attention
    run(boxes, prompt, subject_token_indices, init_step_size=8, final_step_size=2, start_code=latents)
