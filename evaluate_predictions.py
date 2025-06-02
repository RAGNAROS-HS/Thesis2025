import geopandas as gpd
import rasterio
import numpy as np
from shapely.geometry import shape
from rasterio.features import rasterize
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# -------------------------
# Config
# -------------------------
PREDICTION_TIF = "output/shadow_probability_map.tif"
BUILDINGS_GEOJSON = "data/buildings.geojson"
GROUND_TRUTH_GEOJSON = "data/ground_truth.geojson"
THRESHOLD = 0.75

# -------------------------
# Load prediction raster
# -------------------------
def load_prediction_mask(threshold):
    with rasterio.open(PREDICTION_TIF) as src:
        prob = src.read(1) / 255.0
        transform = src.transform
        crs = src.crs
    mask = prob > threshold
    return mask, transform, crs

# -------------------------
# Main Evaluation Logic
# -------------------------
def evaluate():
    from rasterio import open as rio_open
    print("[INFO] Loading data...")
    pred_mask, transform, crs = load_prediction_mask(THRESHOLD)

    gdf_all = gpd.read_file(BUILDINGS_GEOJSON).to_crs(crs)
    gdf_truth = gpd.read_file(GROUND_TRUTH_GEOJSON).to_crs(crs)

    print("[INFO] Rasterizing buildings...")
    gdf_all["building_id"] = range(1, len(gdf_all)+1)
    building_raster = rasterize(
        ((geom, bid) for geom, bid in zip(gdf_all.geometry, gdf_all["building_id"])),
        out_shape=pred_mask.shape,
        transform=transform,
        fill=0,
        dtype="int32"
    )

    print("[INFO] Evaluating predictions...")
    y_true = []
    y_pred = []

    for idx, row in gdf_all.iterrows():
        bid = row["building_id"]
        mask_pixels = (building_raster == bid)
        if not np.any(mask_pixels):
            continue

        predicted_collapse = pred_mask[mask_pixels].mean() > 0.5
        is_destroyed = gdf_truth.intersects(row.geometry).any()

        y_true.append(1 if is_destroyed else 0)
        y_pred.append(1 if predicted_collapse else 0)

        if idx % 10 == 0:
            status = "Correct" if (is_destroyed == predicted_collapse) else "Incorrect"
            print(f"[INFO] Evaluating building {idx}/{len(gdf_all)} - Predicted: {'Destroyed' if predicted_collapse else 'Intact'}, Truth: {'Destroyed' if is_destroyed else 'Intact'} ({status})")

    # Save undetected collapses (False Negatives) as GeoTIFF
    print("[INFO] Saving False Negative mask...")
    fn_mask = np.zeros(pred_mask.shape, dtype=np.uint8)
    for idx, (yt, yp) in enumerate(zip(y_true, y_pred)):
        if yt == 1 and yp == 0:
            bid = gdf_all.iloc[idx]["building_id"]
            fn_mask[building_raster == bid] = 255

    with rio_open(PREDICTION_TIF) as src:
        profile = src.profile
        profile.update(dtype="uint8", count=1)

        with rio_open("output/false_negatives.tif", "w", **profile) as dst:
            dst.write(fn_mask, 1)

    print("[✓] Saved undetected collapses mask to output/false_negatives.tif")

    print("\n[RESULTS]")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.3f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.3f}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")

if __name__ == "__main__":
    evaluate()
