import numpy as np
import torch, sys, os
from scipy.ndimage import gaussian_filter
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from PIL import Image

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../ReNoise-Inversion"))
)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../ReNoise-Inversion/src"))
)
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../ReNoise-Inversion/src/utils")
    )
)
from eunms import Model_Type, Scheduler_Type
from enums_utils import get_pipes
from config import RunConfig
from main import run as invert
from run_sd import run as run_sd
from scipy.fft import idct
import matplotlib.pyplot as plt


class ImageHelpers:
    start_color = lambda: np.random.randint(0, 255, (3,))
    end_color = lambda: np.random.randint(0, 255, (3,))

    @staticmethod
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

    @staticmethod
    def create_color_gradient(image, start_color, end_color, direction="horizontal"):
        gradient = np.zeros_like(image, dtype=float)
        height, width, _ = image.shape
        norm_image = image / 255.0

        if direction == "horizontal":
            for i in range(width):
                alpha = i / width
                gradient[:, i, :] = (1 - alpha) * np.array(
                    start_color
                ) + alpha * np.array(end_color)
        elif direction == "vertical":
            for i in range(height):
                alpha = i / height
                gradient[i, :, :] = (1 - alpha) * np.array(
                    start_color
                ) + alpha * np.array(end_color)

        blended_image = norm_image * gradient

        blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)

        return blended_image

    @staticmethod
    def add_texture(image, texture_intensity=0.5, sigma=1):
        texture = np.random.randn(*image.shape) * 255
        texture = gaussian_filter(texture, sigma=sigma)
        texture_image = (1 - texture_intensity) * image + texture_intensity * texture
        return np.clip(texture_image, 0, 255).astype(np.uint8)

    @staticmethod
    def draw_boxes_on_grid(boxes, start_color, end_color, grid_size=(512, 512, 3)):
        grid = np.random.randn(*grid_size) * 50 + 128
        gradient_image = ImageHelpers.create_color_gradient(
            grid, start_color, end_color, direction="horizontal"
        )

        for box in boxes:
            x_min = int(box[0] * grid_size[1])
            y_min = int(box[1] * grid_size[0])
            x_max = int(box[2] * grid_size[1])
            y_max = int(box[3] * grid_size[0])

            gradient_image[y_min:y_max, x_min:x_max] = (
                ImageHelpers.create_color_gradient(
                    grid[y_min:y_max, x_min:x_max],
                    start_color,
                    end_color,
                    direction="horizontal",
                )
            )

        input_image = ImageHelpers.add_texture(
            gradient_image, texture_intensity=0.3, sigma=5
        )

        return input_image

    @staticmethod
    def load_image(path):
        im = Image.open(path)
        im = im.convert("RGB")
        return np.array(im, dtype=np.uint8)

    @staticmethod
    def block_based_adaptive_blur(image, center, max_sigma, min_sigma, block_size=32):
        _, channels, height, width = image.shape
        blurred_image = np.zeros_like(image, dtype=float)

        # Calculate distance map for the entire image
        y, x = np.ogrid[:height, :width]
        distance_map = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        max_distance = np.sqrt(center[0] ** 2 + center[1] ** 2)
        sigma_map = min_sigma + (max_sigma - min_sigma) * (distance_map / max_distance)

        # Iterate over the image in blocks
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                # Determine the sigma for this block (use the center pixel of the block)
                block_sigma = sigma_map[i : i + block_size, j : j + block_size].mean()

                # Apply Gaussian blur to the block
                for c in range(channels):
                    blurred_block = gaussian_filter(
                        image[i : i + block_size, j : j + block_size, c],
                        sigma=block_sigma,
                    )
                    blurred_image[i : i + block_size, j : j + block_size, c] = (
                        blurred_block
                    )

        return np.clip(blurred_image, 0, 255).astype(np.uint8)


def get_latents_from_renoise(input_image, prompt):
    model_type = Model_Type.SD15
    scheduler_type = Scheduler_Type.DDIM
    print("Scheduler_Type:")
    print(Scheduler_Type)
    pipe_inversion, pipe_inference = get_pipes(
        model_type, scheduler_type, device=device
    )
    config = RunConfig(
        model_type=model_type,
        num_inversion_steps=50,
        num_renoise_steps=0,
        scheduler_type=scheduler_type,
        perform_noise_correction=False,
        seed=7865,
    )
    rec_img, inv_latent, noise, all_latents = invert(
        input_image,
        prompt,
        config,
        pipe_inversion=pipe_inversion,
        pipe_inference=pipe_inference,
        do_reconstruction=True,
    )
    transform = transforms.Compose([transforms.PILToTensor()])
    return transform(rec_img).to(device), inv_latent


def rearrange_latent(latents, boxes, bypass=False):
    if bypass:
        return latents.to(device)
    start_code = np.random.randn(*latents.shape)
    _, _, w, h = start_code.shape
    for box in boxes:
        x_min = int(box[0] * w)
        y_min = int(box[1] * h)
        x_max = int(box[2] * w)
        y_max = int(box[3] * h)
        sub_grid = np.random.randn(1, 4, y_max - y_min + 8, x_max - x_min + 8)
        sub_grid[:, :, 4:-4, 4:-4] = latents[:, :, y_min:y_max, x_min:x_max]
        center = (sub_grid.shape[1] // 2, sub_grid.shape[0] // 2)
        smoothed_grid = ImageHelpers.block_based_adaptive_blur(
            sub_grid, center, max_sigma=3, min_sigma=0.2, block_size=8
        )
        start_code[:, :, y_min:y_max, x_min:x_max] = smoothed_grid[:, :, 4:-4, 4:-4]

    return torch.from_numpy(start_code).to(device)


def get_pixel_indices_from_boxes(boxes, img_dims=(512, 512, 3)):
    resized_boxes = []
    for box in boxes:
        temp_box = [
            int(img_dims[0] * box[0]),
            int(img_dims[1] * box[1]),
            int(img_dims[0] * box[2]),
            int(img_dims[1] * box[3]),
        ]
        resized_boxes.append(temp_box)
    return resized_boxes


def get_box_with_margins(box, img_dims, width=20):
    half_width = int(width / 2)
    col_0_0 = 0 if box[0] < half_width else box[0] - half_width
    col_0_1 = width if box[0] < half_width else box[0] + half_width
    col_1_0 = img_dims[0] - width if img_dims[0] - box[2] < half_width else box[2] - half_width
    col_1_1 = img_dims[0] if img_dims[0] - box[2] < half_width else box[2] + half_width
    row_0_0 = 0 if box[1] < half_width else box[1] - half_width
    row_0_1 = width if box[1] < half_width else box[1] + half_width
    row_1_0 = img_dims[1] - width if img_dims[1] - box[3] < half_width else box[3] - half_width
    row_1_1 = img_dims[0] if img_dims[0] - box[3] < half_width else box[3] + half_width
    return col_0_0, col_0_1, col_1_0, col_1_1, row_0_0, row_0_1, row_1_0, row_1_1


def create_noise_wiith_deterministic_bbs(boxes, img_dims=(512, 512, 3), width=20):
    frame = np.random.randn(*img_dims)
    #normalized_noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
    #frame = (normalized_noise * 255).astype(np.uint8)
    resized_boxes = get_pixel_indices_from_boxes(boxes, img_dims)
    for box in resized_boxes:
        col_0_0, col_0_1, col_1_0, col_1_1, row_0_0, row_0_1, row_1_0, row_1_1 = get_box_with_margins(box, img_dims, width)
        print("row_0_0, row_0_1, row_1_0, row_1_1, col_0_0, col_0_1, col_1_0, col_1_1 = {}, {}, {}, {}, {}, {}, {}, {}".format(row_0_0, row_0_1, row_1_0, row_1_1, col_0_0, col_0_1, col_1_0, col_1_1))
        frame[row_0_0:row_1_1, col_0_0:col_0_1] = np.array([100, 100, 100])
        frame[row_0_0:row_1_1, col_1_0:col_1_1] = np.array([100, 100, 100])
        frame[row_0_0:row_0_1, col_0_0:col_1_1] = np.array([100, 100, 100])
        frame[row_1_0:row_1_1, col_0_0:col_1_1] = np.array([100, 100, 100])
    return frame.astype(np.uint8)
    
def create_noise_wiith_filled_bbs(boxes, img_dims=(512, 512, 3), width=20):
    noise = np.random.randn(*img_dims)
    normalized_noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
    frame = (normalized_noise * 255).astype(np.uint8)
    #resized_boxes = get_pixel_indices_from_boxes(boxes, img_dims)
    #for box in resized_boxes:
    #    col_0_0, col_0_1, col_1_0, col_1_1, row_0_0, row_0_1, row_1_0, row_1_1 = get_box_with_margins(box, img_dims, width)
    #    print("row_0_0, row_0_1, row_1_0, row_1_1, col_0_0, col_0_1, col_1_0, col_1_1 = {}, {}, {}, {}, {}, {}, {}, {}".format(row_0_0, row_0_1, row_1_0, row_1_1, col_0_0, col_0_1, col_1_0, col_1_1))
    #    #frame[row_0_0:row_1_1, col_0_0:col_0_1] = np.array([0, 0, 0])
    #    #frame[row_0_0:row_1_1, col_1_0:col_1_1] = np.array([0, 0, 0])
    #    #frame[row_0_0:row_0_1, col_0_0:col_1_1] = np.array([0, 0, 0])
    #    #frame[row_1_0:row_1_1, col_0_0:col_1_1] = np.array([0, 0, 0])
    #    alpha = 0.1
    #    frame[row_0_0:row_1_1, col_0_0:col_1_1] = (frame[row_0_0:row_1_1, col_0_0:col_1_1] * (1 - alpha) + 255 * alpha).astype(np.uint8)
    return frame


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    '''
    boxes = [
        [0.05, 0.1, 0.35, 0.4],  # Object 1
        [0.45, 0.15, 0.75, 0.45],  # Object 2
        [0.6, 0.6, 0.9, 0.9],  # Object 3
    ]
    '''
    boxes = [
        [0.05, 0.5, 0.35, 0.7],  # Object 1
        [0.45, 0.6, 0.75, 0.8]  # Object 2
        ]

    # input_image = ImageHelpers.draw_boxes_on_grid(
    #     boxes,
    #     ImageHelpers.start_color(),
    #     ImageHelpers.end_color(),
    #     grid_size=(512, 512, 3),
    # )

    # input_image = create_noise_wiith_deterministic_bbs(boxes)
    # plt.savefig("/home/dcor/omerdh/DLproject/out/noise_with_boxes.png")
    # #return
    max_f = 130
    active_bins = 100
    # input_image = ImageHelpers.dct_based_image(max_f, active_bins)
    #input_image = ImageHelpers.load_image("edited.jpg")
    #input_image = create_noise_wiith_deterministic_bbs(boxes, img_dims=(512, 512, 3), width=2)
    input_image = create_noise_wiith_filled_bbs(boxes, img_dims=(512, 512, 3), width=2)
    plt.imshow(input_image)
    plt.savefig('/home/dcor/omerdh/DLproject/bounded-attention/out/noise_with_bbs_and_margins.png')
    recon_img, latents = get_latents_from_renoise(input_image, prompt="")
    #start_code = rearrange_latent(latents.cpu(), boxes, False)
    

    prompt = "A balck crow and a red cat"
    subject_token_indices = [[2, 3], [6, 7]]
    run_description = {
        "prompt": prompt,
        "boxes": boxes,
        "token_ind": subject_token_indices,
        "free_text": f"start code is inverted image of prompt, BB of the inversion is Zt with smoothing. Gaussian noise with full BBs, filling disabled. Normalized noise.",
    }
    
    #latents = None

    run_sd(
        boxes,
        prompt,
        subject_token_indices,
        init_step_size=25,
        final_step_size=10,
        cross_loss_scale=1.5,
        start_code=latents,
        input_image=input_image,
        recon_input=recon_img,
        description=run_description,
    )
