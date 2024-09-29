from scipy.ndimage import gaussian_filter
from scipy.stats import cauchy
import torchvision.transforms.functional as F
from PIL import Image
from scipy.fft import idct
import matplotlib.pyplot as plt
import numpy as np


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
    def draw_boxes_on_grid_monotone(boxes, grid_size=(512, 512, 3)):
        grid = np.random.randn(*grid_size) * 50 + 128
        for box in boxes:
            x_min = int(box[0] * grid_size[1])
            y_min = int(box[1] * grid_size[0])
            x_max = int(box[2] * grid_size[1])
            y_max = int(box[3] * grid_size[0])
            grid[y_min:y_max, x_min:x_max] = 255.0

        return np.clip(grid, 0, 255).astype(np.uint8)

    @staticmethod
    def plant_patches_in_latent(boxes, grid_size=(1, 4, 64, 64)):
        grid = np.random.randn(*grid_size)
        for box in boxes:
            x_min = int(box[0] * grid_size[-1])
            y_min = int(box[1] * grid_size[-1])
            x_max = int(box[2] * grid_size[-1])
            y_max = int(box[3] * grid_size[-1])
            grid[:, :, y_min:y_max, x_min:x_max] = 1.1 * np.random.randn(
                1, 4, y_max - y_min, x_max - x_min
            )

        return grid

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
        im = im.convert("RGB").resize((512, 512))
        return im

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


if __name__ == "__main__":
    pass
