import os
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import rasterize

# === CONFIGURATION ===
PRE_TIF = "PREFF95000 (1).tif"
POST_TIF = "POSTCB600 (1).tif"
GEOJSON = "hotosm_tur_destroyed_buildings_polygons_geojson (1).geojson"

OUTPUT_DIR = "val"
SCENE_ID = "scene123"  # used in filenames

# === CREATE DIRECTORIES ===
image_dir = os.path.join(OUTPUT_DIR, "images")
mask_dir = os.path.join(OUTPUT_DIR, "masks")
os.makedirs(image_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

# === FUNCTION TO CONVERT TIFF TO .NPY ===
def convert_tif_to_npy(tif_path, out_path, bands=3):
    with rasterio.open(tif_path) as src:
        array = src.read(out_dtype=np.uint8)[:bands]
        np.save(out_path, array)

# === SAVE IMAGE ARRAYS ===
pre_npy_path = os.path.join(image_dir, f"{SCENE_ID}_pre_disaster.npy")
post_npy_path = os.path.join(image_dir, f"{SCENE_ID}_post_disaster.npy")
convert_tif_to_npy(PRE_TIF, pre_npy_path)
convert_tif_to_npy(POST_TIF, post_npy_path)

# === CREATE MASK FROM GEOJSON ===
with rasterio.open(PRE_TIF) as src:
    transform = src.transform
    shape = (src.height, src.width)
    crs = src.crs

    # Load and project GeoJSON to match the raster
    gdf = gpd.read_file(GEOJSON).to_crs(crs)

    # Rasterize geometries
    shapes = [(geom, 1) for geom in gdf.geometry]
    mask = rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype=np.uint8)

# === SAVE MASK ===
mask_path = os.path.join(mask_dir, f"{SCENE_ID}.npy")
np.save(mask_path, mask)

print(f"Saved:\n- {pre_npy_path}\n- {post_npy_path}\n- {mask_path}")
