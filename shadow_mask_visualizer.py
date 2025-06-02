import numpy as np
import matplotlib.pyplot as plt
import os

# Crop size (e.g., 1000x1000 pixels from top-left corner)
CROP_SIZE = 1000

def visualize_shadow_mask(npy_path, title=None):
    if not os.path.exists(npy_path):
        print(f"[ERROR] File not found: {npy_path}")
        return

    mask = np.load(npy_path)

    # Crop the top-left corner of the mask
    cropped_mask = mask[:CROP_SIZE, :CROP_SIZE]

    plt.figure(figsize=(8, 8))
    plt.imshow(cropped_mask, cmap="gray")
    plt.title(title or f"Shadow Mask (cropped): {os.path.basename(npy_path)}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Visualize both pre and post shadow masks (cropped)
    visualize_shadow_mask("output/pre_shadow_mask.npy", title="Pre-Event Shadow Mask (Cropped)")
    visualize_shadow_mask("output/post_shadow_mask.npy", title="Post-Event Shadow Mask (Cropped)")