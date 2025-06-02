import os
import cv2
import rasterio
import numpy as np
from PIL import Image

def convert_tiff_to_png(tiff_path, output_path, resize_max_dim=None):
    with rasterio.open(tiff_path) as src:
        image = src.read()  # (bands, H, W)
        image = np.transpose(image, (1, 2, 0))  # (H, W, bands)

        if image.shape[2] > 3:
            image = image[:, :, :3]  # Use RGB only

        # Normalize if values > 255
        if image.max() > 255:
            image = (255 * (image / image.max())).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        # Resize if needed
        if resize_max_dim:
            h, w = image.shape[:2]
            scale = resize_max_dim / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        Image.fromarray(image).save(output_path)
        print(f"[✓] Saved: {output_path}")

# === Example Usage ===
if __name__ == "__main__":
    convert_tiff_to_png("data/PREFF95000.tif", "data/pre_event.png")
    convert_tiff_to_png("data/POSTCB600.tif", "data/post_event.png")
