import cv2
import numpy as np
import rasterio
import json
from rasterio.transform import rowcol
from rasterio.warp import transform_geom
from shapely.geometry import Polygon, mapping, shape
from shapely.strtree import STRtree
import fiona
from fiona.crs import from_string
import sys
import contextlib

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None, None, None
    print(f"Image loaded with shape: {image.shape}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    return image, gray, enhanced

def detect_shadows(enhanced, threshold_value):
    _, mask = cv2.threshold(enhanced, threshold_value, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    print("Shadow mask computed.")
    return mask

def pixel_to_geo(coords, transform):
    return [rasterio.transform.xy(transform, y, x) for x, y in coords]

def calculate_shadow_length_direction(polygon, sun_azimuth_deg):
    rad = np.deg2rad(sun_azimuth_deg + 180)
    shadow_vector = np.array([np.cos(rad), np.sin(rad)])
    coords = np.array(polygon.exterior.coords)
    projections = coords.dot(shadow_vector)
    length = projections.max() - projections.min()
    return length

def load_buildings(path):
    with fiona.open(path, 'r') as src:
        buildings = [feature for feature in src]
        crs = src.crs
        schema = src.schema.copy()
    for feature in buildings:
        if 'properties' not in feature or feature['properties'] is None:
            feature['properties'] = {}
        feature['properties']['height_m'] = None
    schema['properties']['height_m'] = 'float'
    print(f"Loaded {len(buildings)} building footprints from {path}")
    return buildings, crs, schema

def estimate_building_heights(image_path, geotiff_path, threshold, output_path, buildings_path,
                              sun_elevation_deg=35.0, sun_azimuth_deg=180.0, gsd=0.5, min_area=100):
    image, gray, enhanced = preprocess_image(image_path)
    if image is None:
        return

    mask = detect_shadows(enhanced, threshold)

    with rasterio.open(geotiff_path) as src:
        transform = src.transform
        geotiff_crs = src.crs

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Detected {len(contours)} shadow contours")

    sun_elevation_rad = np.deg2rad(sun_elevation_deg)
    shadow_geometries = []
    shadow_heights = []

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            print(f"Contour {i} skipped due to small area: {area:.2f}")
            continue

        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        pixel_coords = [(pt[0][0], pt[0][1]) for pt in approx]
        geo_coords = pixel_to_geo(pixel_coords, transform)
        try:
            shadow_poly = Polygon(geo_coords)
            if not shadow_poly.is_valid:
                print(f"Contour {i} produced an invalid polygon.")
                continue
        except Exception as e:
            print(f"Error creating polygon for contour {i}: {e}")
            continue

        shadow_length_pixels = calculate_shadow_length_direction(shadow_poly, sun_azimuth_deg)
        height = shadow_length_pixels * gsd * np.tan(sun_elevation_rad)
        print(f"Contour {i}: Shadow length = {shadow_length_pixels:.2f}, Estimated height = {height:.2f}")

        shadow_geometries.append(shadow_poly)
        shadow_heights.append(height)

    print(f"Processed {len(shadow_geometries)} valid shadow geometries.")

    buildings, bldg_crs, bldg_schema = load_buildings(buildings_path)

    if bldg_crs != geotiff_crs:
        print("Reprojecting building geometries to geotiff CRS for spatial matching...")
    else:
        print("Building CRS and geotiff CRS match.")

    shadow_index = STRtree(shadow_geometries)

    for idx, bldg in enumerate(buildings):
        if bldg_crs != geotiff_crs:
            bldg_geom = shape(transform_geom(bldg_crs, geotiff_crs, bldg['geometry']))
        else:
            bldg_geom = shape(bldg['geometry'])

        query_results = shadow_index.query(bldg_geom.buffer(50))
        nearest_height = 0
        min_dist = float('inf')

        if len(query_results) > 0:
            for result in query_results:
                if isinstance(result, (int, np.integer)):
                    i_match = int(result)
                    sg = shadow_geometries[i_match]
                else:
                    try:
                        i_match = shadow_geometries.index(result)
                        sg = result
                    except ValueError:
                        for j, candidate in enumerate(shadow_geometries):
                            if candidate.equals_exact(result, tolerance=1e-6):
                                i_match = j
                                sg = candidate
                                break
                        else:
                            continue
                dist = bldg_geom.distance(sg)
                if dist < min_dist:
                    min_dist = dist
                    nearest_height = shadow_heights[i_match]
                    print(f"Building {idx}: Nearest shadow distance = {dist:.2f}, Height = {nearest_height:.2f}")
        else:
            print(f"Building {idx}: No shadow match found within buffer.")

        bldg['properties']['height_m'] = round(nearest_height, 2)
        print(f"Building {idx} assigned height: {bldg['properties']['height_m']} m")

    non_zero = sum(1 for b in buildings if b['properties'].get('height_m', 0) > 0)
    print(f"Total buildings with non-zero height: {non_zero}/{len(buildings)}")

    with fiona.open(output_path, 'w', driver='GeoJSON', crs=from_string(str(geotiff_crs)), schema=bldg_schema) as out:
        for bldg in buildings:
            out.write({
                'type': 'Feature',
                'geometry': bldg['geometry'],
                'properties': bldg['properties']
            })

    print(f"GeoJSON output saved at: {output_path}")

def main():
    # === CONFIGURATION ===
    tiff_image_path = r'C:\Users\Hugo\Desktop\heightCalc\POSTCB600.tif'
    shadow_image_path = tiff_image_path
    output_building_geojson_path = r'C:\Users\Hugo\Desktop\heightCalc\outputPOST_buildings_height.geojson'
    building_geojson_path = r'C:\Users\Hugo\Desktop\heightCalc\Antakya.geojson'
    threshold_value = 20
    sun_elevation_deg = 38.0
    sun_azimuth_deg = 162.1
    gsd = 0.67
    min_shadow_area = 100

    # === RUN ===
    estimate_building_heights(
        shadow_image_path, tiff_image_path, threshold_value,
        output_building_geojson_path, building_geojson_path,
        sun_elevation_deg, sun_azimuth_deg, gsd, min_shadow_area
    )

# Redirect all print output to a file
if __name__ == "__main__":
    output_file = r'C:\Users\Hugo\Desktop\heightCalc\building_height_estimation.txt'
    with open(output_file, 'w') as f:
        with contextlib.redirect_stdout(f):
            main()
