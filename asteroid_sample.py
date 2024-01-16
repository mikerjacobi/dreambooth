from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch
import numpy as np
from PIL import Image
import cv2
import numpy as np
import rembg

# quality thresholds
vertical_symmetry_threshold = 0.0
nonzero_alpha_pixel_threshold = 150000

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")
pipe.load_lora_weights("mikerjacobi/asteroid-lora-model")

# rembg_models: u2net, isnet-anime
session = rembg.new_session(model_name="u2net")

def rmbg_and_center(image, size=512, border_ratio=0.2):
    # remove background
    carved_image = rembg.remove(image, session=session)  # [H, W, 4]
    mask = carved_image[..., -1] > 0

    # recenter
    final_rgba = np.zeros((size, size, 4), dtype=np.uint8)
    coords = np.nonzero(mask)
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    h = x_max - x_min
    w = y_max - y_min
    desired_size = int(size * (1 - border_ratio))
    scale = desired_size / max(h, w)
    h2 = int(h * scale)
    w2 = int(w * scale)
    x2_min = (size - h2) // 2
    x2_max = x2_min + h2
    y2_min = (size - w2) // 2
    y2_max = y2_min + w2
    final_rgba[x2_min:x2_max, y2_min:y2_max] = cv2.resize(
        carved_image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA
    )

    return final_rgba

import numpy as np

def symmetry_score(image):
    """
    Calculate the vertical symmetry score of an image.

    Args:
    - image (numpy.ndarray): A 512x512 numpy array representing an image.

    Returns:
    - float: Vertical symmetry score between 0 and 1.
    """
    if image.shape != (512, 512, 4):
        raise ValueError(f"Image must be 512x512 with 3 color channels. got {image.shape}")

     # Extract the alpha channel and create a mask for non-zero alpha pixels
    alpha_mask = image[:, :, 3] != 0

    # Process for Vertical Symmetry
    left_half_alpha = alpha_mask[:, :256]
    right_half_alpha = alpha_mask[:, 256:]
    right_half_flipped_alpha = np.flip(right_half_alpha, axis=1)

    # Mask where both corresponding pixels have non-zero alpha
    valid_vertical_mask = left_half_alpha & right_half_flipped_alpha

    # Apply mask to RGB channels
    left_half = image[:, :256, :3]
    right_half = image[:, 256:, :3]
    right_half_flipped = np.flip(right_half, axis=1)

    # Calculate vertical symmetry score only on valid pixels
    vertical_similarity = np.sum((left_half == right_half_flipped) & valid_vertical_mask[..., None]) / np.sum(valid_vertical_mask)
    return vertical_similarity

def count_non_zero_alpha_pixels(image):
    """
    Count the number of pixels in an RGBA image where the alpha value is not 0.

    Args:
    - image (numpy.ndarray): A numpy array representing an RGBA image.

    Returns:
    - int: Number of pixels with non-zero alpha values.
    """
    if image.shape[2] != 4:
        raise ValueError("Image must have 4 channels (RGBA).")

    # Extract the alpha channel
    alpha_channel = image[:, :, 3]

    # Count non-zero alpha pixels
    non_zero_alpha_count = np.count_nonzero(alpha_channel)

    return non_zero_alpha_count

keep, keep_goal = 0, 100
while True:
  image = pipe("A picture of a sks asteroid", num_inference_steps=25).images[0]
  image_npy = np.array(image)
  try:
    image_npy = rmbg_and_center(image_npy, border_ratio=0.1)
  except ValueError:
    continue
  vert_sym = symmetry_score(image_npy)
  #print(f"sym {vert_sym}")
  non_zero_alpha_pixels = count_non_zero_alpha_pixels(image_npy)
  #print(f"Number of non-zero alpha pixels: {non_zero_alpha_pixels}")
  if vert_sym > vertical_symmetry_threshold and non_zero_alpha_pixels > nonzero_alpha_pixel_threshold: 
    keep += 1
    Image.fromarray(image_npy).save(f"projects/asteroids/inferences/a_asteroid_{keep}.png")
    print(f"keeper {keep}, nzap:{non_zero_alpha_pixels} vertsym:{vert_sym}")
    if keep >= keep_goal:
      break
  else:
    print(f"bad, nzap:{non_zero_alpha_pixels} vertsym:{vert_sym}")
    
