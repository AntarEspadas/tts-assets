import cv2
import os
import numpy as np
from glob import glob

IN_DIR = "./paren-la-nave/tmp/raw-img/"
OUT_DIR = "./paren-la-nave/tmp/blank-img/"

def remove_answer():
    paths = glob(IN_DIR + "*.png")

    start = (50,385)
    end = (290, 540)

    color = (255, 255, 255)

    for i, path in enumerate(paths):
        image = cv2.imread(path)
        image = cv2.rectangle(image, start, end, color, -1)

        cv2.imwrite(OUT_DIR + f"img_{i:03d}.png", image)

def make_grid():
    ...

# --- CONFIGURATION ---
image_folder = OUT_DIR  # Folder with 70 images
output_file = "image_grid_cropped.png"
grid_cols = 10
grid_rows = 7
target_size = (360, 550)  # (width, height)
pad_color = (255, 255, 255)  # white
# ----------------------

def center_crop_or_pad(img, size, pad_color=(255, 255, 255)):
    target_w, target_h = size
    h, w = img.shape[:2]

    # If image is large enough, crop
    if w >= target_w and h >= target_h:
        x_start = (w - target_w) // 2
        y_start = (h - target_h) // 2
        cropped = img[y_start:y_start + target_h, x_start:x_start + target_w]
        return cropped

    # Else, pad the image
    scale_w = target_w / w
    scale_h = target_h / h
    scale = min(scale_w, scale_h)

    resized = cv2.resize(img, (int(w * scale), int(h * scale)))

    # Create white canvas
    canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)

    # Center the resized image
    y_offset = (target_h - resized.shape[0]) // 2
    x_offset = (target_w - resized.shape[1]) // 2

    canvas[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized
    return canvas

def load_images(folder, target_size, max_images):
    images = []
    files = sorted(os.listdir(folder))[140:]
    for filename in files:
        filepath = os.path.join(folder, filename)
        img = cv2.imread(filepath)
        if img is None:
            print(f"Warning: Could not load image {filepath}")
            continue
        img = center_crop_or_pad(img, target_size)
        images.append(img)
    return images

def create_image_grid(images, rows, cols, fill_color=(255, 255, 255)):
    cell_w, cell_h = target_size
    total_cells = rows * cols
    padded_images = images + [np.full((cell_h, cell_w, 3), fill_color, dtype=np.uint8)] * (total_cells - len(images))

    img_rows = []
    for i in range(rows):
        row_imgs = padded_images[i * cols : (i + 1) * cols]
        img_row = np.hstack(row_imgs)
        img_rows.append(img_row)

    grid_img = np.vstack(img_rows)
    return grid_img

def main():
    total_cells = grid_cols * grid_rows
    images = load_images(image_folder, target_size, total_cells)

    print(f"Loaded {len(images)} image(s).")
    grid_image = create_image_grid(images, grid_rows, grid_cols)
    cv2.imwrite(output_file, grid_image)
    print(f"Saved grid image to {output_file}")

if __name__ == "__main__":
    main()