import os
import cv2
import numpy as np
import rasterio
import json
from shapely.geometry import Polygon, mapping
from rasterio.transform import xy as pixel_to_geo
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte

# -------------------------
# Config
# -------------------------
THRESHOLD_VALUE = 20
MIN_SHADOW_AREA = 100
MAX_ENTROPY = 5.3  # Maximum entropy allowed for a region to be considered a true shadow

# -------------------------
# Preprocess and Shadow Detection
# -------------------------
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None, None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    return image, gray, enhanced

def detect_shadows(enhanced, threshold_value):
    _, shadow_mask = cv2.threshold(enhanced, threshold_value, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    shadow_cleaned = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
    shadow_cleaned = cv2.dilate(shadow_cleaned, kernel, iterations=1)
    return shadow_cleaned

def compute_entropy_mask(gray_image):
    gray_u8 = img_as_ubyte(gray_image / 255.0)
    print("[INFO] Computing entropy map...")
    return entropy(gray_u8, disk(5))

# -------------------------
# Pixel to Geo conversion
# -------------------------
def pixel_coords_to_geo(coords, transform):
    return [pixel_to_geo(transform, y, x) for x, y in coords]

# -------------------------
# Save shadows to GeoJSON + Binary Mask
# -------------------------
def save_shadow_polygons(image_path, geotiff_path, output_geojson_path, output_mask_path):
    img, gray, enhanced = preprocess_image(image_path)
    if img is None:
        return

    shadow_mask = detect_shadows(enhanced, THRESHOLD_VALUE)
    entropy_map = compute_entropy_mask(gray)

    # Save binary mask for pixel-level analysis
    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    np.save(output_mask_path, shadow_mask)
    print(f"[✓] Saved binary shadow mask to {output_mask_path}.npy")

    with rasterio.open(geotiff_path) as src:
        transform = src.transform
        crs = src.crs

    contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features = []

    print(f"[INFO] Filtering {len(contours)} shadow regions...")
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        bbox_entropy = entropy_map[y:y+h, x:x+w].mean()
        if bbox_entropy > MAX_ENTROPY:
            continue  # Too textured to be a real shadow (early filter)
        if i % 100 == 0:
            print(f"  ...processing region {i+1}/{len(contours)}")

        area = cv2.contourArea(cnt)
        if area < MIN_SHADOW_AREA:
            continue

        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        pixel_coords = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]  # Ensure JSON serializable
        geo_coords = pixel_coords_to_geo(pixel_coords, transform)

        feature = {
            "type": "Feature",
            "geometry": mapping(Polygon(geo_coords)),
            "properties": {
                "shadow_area_px": float(area),
                "pixel_coords": pixel_coords,
                "avg_entropy": float(bbox_entropy)
            }
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": str(crs)}},
        "features": features
    }

    os.makedirs(os.path.dirname(output_geojson_path), exist_ok=True)
    with open(output_geojson_path, "w") as f:
        json.dump(geojson, f, indent=4)

    print(f"[✓] Saved {len(features)} shadows to {output_geojson_path}")

# -------------------------
# Run Both Pre and Post
# -------------------------
def run_shadow_detection():
    CONFIG = {
        "pre": {
            "image": "data/pre_event.png",
            "tif": "data/PREFF95000.tif",
            "out_geojson": "output/pre_shadows.geojson",
            "out_mask": "output/pre_shadow_mask"
        },
        "post": {
            "image": "data/post_event.png",
            "tif": "data/POSTCB600.tif",
            "out_geojson": "output/post_shadows.geojson",
            "out_mask": "output/post_shadow_mask"
        }
    }

    for label, paths in CONFIG.items():
        print(f"\n[INFO] Processing {label.upper()} image...")
        save_shadow_polygons(paths["image"], paths["tif"], paths["out_geojson"], paths["out_mask"])

# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    run_shadow_detection()
#[✓] Saved binary shadow mask to output/pre_shadow_mask.npy
#[✓] Saved 60949 shadows to output/pre_shadows.geojson

#[INFO] Processing POST image...
#[✓] Saved binary shadow mask to output/post_shadow_mask.npy
#[✓] Saved 52665 shadows to output/post_shadows.geojson
