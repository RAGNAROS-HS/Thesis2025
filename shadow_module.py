import os
import json
import numpy as np
import geopandas as gpd
import cv2
from shapely.geometry import shape, Polygon
from shapely.affinity import translate
from PIL import Image
import rasterio

# -------------------------
# Config
# -------------------------
DEFAULT_HEIGHT_M = 10.0
MIN_BUILDING_AREA_M2 = 20.0
MAX_SUN_ELEVATION = 70.0
UNCERTAIN_PROB = 0.5
MIN_PRE_SHADOW_PIXELS = 10

# -------------------------
# Utility Functions
# -------------------------
def load_metadata(path):
    with open(path) as f:
        return json.load(f)

def compute_shadow_length(height_m, sun_elevation_deg):
    if sun_elevation_deg <= 0:
        return np.inf
    return height_m / np.tan(np.radians(sun_elevation_deg))

def project_shadow(building_geom, shadow_length_m, sun_azimuth_deg, gsd):
    shadow_length_px = shadow_length_m / gsd
    azimuth_rad = np.radians(sun_azimuth_deg)
    dx = -shadow_length_px * np.sin(azimuth_rad)
    dy = -shadow_length_px * np.cos(azimuth_rad)
    return translate(building_geom, xoff=dx, yoff=dy)

def world_to_pixel_coords(geom, transform):
    def to_pixel(x, y):
        col, row = ~transform * (x, y)
        return [int(col), int(row)]

    if geom.geom_type == 'Polygon':
        return [np.array([to_pixel(x, y) for x, y in geom.exterior.coords], dtype=np.int32)]
    elif geom.geom_type == 'MultiPolygon':
        return [
            np.array([to_pixel(x, y) for x, y in part.exterior.coords], dtype=np.int32)
            for part in geom.geoms
        ]
    else:
        return []

def count_shadow_pixels(mask, polygon, transform):
    if not polygon.is_valid or polygon.is_empty:
        return 0
    h, w = mask.shape
    mask_region = np.zeros((h, w), dtype=np.uint8)
    coords = world_to_pixel_coords(polygon, transform)
    for coord in coords:
        cv2.fillPoly(mask_region, [coord], 1)
    return int(np.sum(mask_region & (mask > 0)))

# -------------------------
# Main Pipeline
# -------------------------
def shadow_collapse_pipeline():
    print("[INFO] Loading metadata and data...")
    meta = load_metadata("metadata/sun_angles.json")
    pre = meta["pre_event"]
    post = meta["post_event"]

    with rasterio.open("data/PREFF95000.tif") as src:
        transform = src.transform
        crs = src.crs

    gdf = gpd.read_file("data/buildings.geojson").to_crs(crs)
    pre_mask = np.load("output/pre_shadow_mask.npy")
    post_mask = np.load("output/post_shadow_mask.npy")

    h, w = pre_mask.shape
    prob_map = np.zeros((h, w), dtype=np.float32)

    # Compute true post-event shadow length
    shadow_len_post = compute_shadow_length(DEFAULT_HEIGHT_M, post["sun_elevation"])

    # Compute normalized pre shadow length using tangent ratio
    tan_pre = np.tan(np.radians(pre["sun_elevation"]))
    tan_post = np.tan(np.radians(post["sun_elevation"]))
    if tan_post == 0:
        scaling_ratio = 1
    else:
        scaling_ratio = tan_pre / tan_post
    normalized_pre_len = shadow_len_post * scaling_ratio

    print("[INFO] Processing buildings...")
    for idx, row in gdf.iterrows():
        try:
            geom = row.geometry
            area = geom.area
            if area < MIN_BUILDING_AREA_M2:
                continue

            slr = None
            if pre["sun_elevation"] > MAX_SUN_ELEVATION or post["sun_elevation"] > MAX_SUN_ELEVATION:
                prob = UNCERTAIN_PROB
            else:
                pre_proj = project_shadow(geom, normalized_pre_len, pre["sun_azimuth"], pre["gsd"])
                post_proj = project_shadow(geom, shadow_len_post, post["sun_azimuth"], post["gsd"])

                pre_count = count_shadow_pixels(pre_mask, pre_proj, transform)
                post_count = count_shadow_pixels(post_mask, post_proj, transform)

                if pre_count < MIN_PRE_SHADOW_PIXELS:
                    prob = UNCERTAIN_PROB
                else:
                    slr = (pre_count - post_count) / (pre_count + 1e-5)
                    prob = np.clip(slr, 0, 1)

                if idx % 10 == 0:
                    print(f"[BUILDING {idx}] Pre: {pre_count}, Post: {post_count}, SLR: {slr if slr is not None else 'n/a'}, Prob: {prob:.2f}")

            mask_building = np.zeros((h, w), dtype=np.uint8)
            all_coords = world_to_pixel_coords(geom, transform)
            for coords in all_coords:
                cv2.fillPoly(mask_building, [coords], 1)
            prob_map[mask_building > 0] = prob

        except Exception as e:
            print(f"[ERROR] Skipping building {idx} due to: {e}")
            continue

    return prob_map

# -------------------------
# Save Output
# -------------------------
def save_probability_map(prob_map, out_path="output/shadow_probability_map.png", tif_path="output/shadow_probability_map.tif"):
    img = (prob_map * 255).astype(np.uint8)
    Image.fromarray(img).save(out_path)

    with rasterio.open("data/PREFF95000.tif") as ref:
        transform = ref.transform
        crs = ref.crs

    with rasterio.open(
        tif_path, "w",
        driver="GTiff",
        height=img.shape[0],
        width=img.shape[1],
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(img, 1)
    print(f"[✓] Saved GeoTIFF probability map to {tif_path}")
    print(f"[✓] Saved probability map to {out_path}")

# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    print("[START] Shadow-based collapse probability analysis")
    prob_map = shadow_collapse_pipeline()
    save_probability_map(prob_map)
    print("[DONE] Analysis complete")
