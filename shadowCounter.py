import cv2
import numpy as np
import rasterio
import json
from rasterio.transform import rowcol
from shapely.geometry import Polygon, mapping, shape
from shapely.strtree import STRtree
import fiona
from fiona.crs import from_string

# Preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None, None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    return image, gray, enhanced

# Detect shadows in the grayscale image
def detect_shadows(enhanced, threshold_value):
    _, mask = cv2.threshold(enhanced, threshold_value, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

# Convert pixel to geographic coordinates
def pixel_to_geo(coords, transform):
    return [rasterio.transform.xy(transform, y, x) for x, y in coords]

# Calculate shadow length in pixels
def calculate_shadow_length(pixels):
    return max(np.linalg.norm(np.array(p1) - np.array(p2)) for i, p1 in enumerate(pixels) for p2 in pixels[i+1:])

# Load building footprints
def load_buildings(path):
    with fiona.open(path, 'r') as src:
        return [feature for feature in src], src.crs

# Main function
def estimate_building_heights(image_path, geotiff_path, threshold, output_path, buildings_path,
                              sun_elevation_deg=35.0, gsd=0.5, min_area=100):
    image, gray, enhanced = preprocess_image(image_path)
    if image is None: return

    mask = detect_shadows(enhanced, threshold)

    with rasterio.open(geotiff_path) as src:
        transform = src.transform
        crs = src.crs

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Detected {len(contours)} shadow contours")

    sun_elevation_rad = np.deg2rad(sun_elevation_deg)
    shadow_features = []
    shadow_geometries = []
    shadow_heights = []

    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < min_area:
            continue
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        pixel_coords = [(pt[0][0], pt[0][1]) for pt in approx]
        geo_coords = pixel_to_geo(pixel_coords, transform)
        shadow_poly = Polygon(geo_coords)
        shadow_length = calculate_shadow_length(pixel_coords)
        height = shadow_length * gsd * np.tan(sun_elevation_rad)
        shadow_geometries.append(shadow_poly)
        shadow_heights.append(height)
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(contours)} contours")

    buildings, bldg_crs = load_buildings(buildings_path)
    print(f"Loaded {len(buildings)} building footprints")

    shadow_index = STRtree(shadow_geometries)

    for idx, bldg in enumerate(buildings):
        geom = shape(bldg['geometry'])
        matches = shadow_index.query(geom.buffer(50))
        max_height = 0
        for match in matches:
            i = shadow_geometries.index(match)
            dist = geom.distance(match)
            if dist < 50 and shadow_heights[i] > max_height:
                max_height = shadow_heights[i]

        if 'properties' not in bldg:
            bldg['properties'] = {}
        bldg['properties']['height_m'] = round(max_height, 2)
        

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(buildings)} buildings | Last height: {bldg['properties']['height_m']} m")

    all_props = set()
    for b in buildings:
        all_props.update(b['properties'].keys())

    schema = {
        'geometry': 'Unknown',
        'properties': {key: 'float' if key == 'height_m' else 'str' for key in all_props if key != 'label'}
    }

    with fiona.open(output_path, 'w', driver='GeoJSON', crs=from_string(str(bldg_crs)), schema=schema) as out:
        for bldg in buildings:
            out.write({
                'type': 'Feature',
                'geometry': bldg['geometry'],
                'properties': bldg['properties']
            })

    print(f"Saved: {output_path}")

# === CONFIG ===
tiff_image_path = r'C:\Users\Hugo\Documents\GitHub\Thesis2025\img\PREFF95000.tif'
shadow_image_path = tiff_image_path
output_building_geojson_path = r'C:\Users\Hugo\Documents\GitHub\Thesis2025\output\PRE_buildings_height.geojson'
building_geojson_path = r'C:\Users\Hugo\Documents\GitHub\Thesis2025\data\Antakya.geojson'
threshold_value = 20
sun_elevation_deg = 50.8
gsd = 0.57
min_shadow_area = 100

# === RUN ===
estimate_building_heights(
    shadow_image_path, tiff_image_path, threshold_value,
    output_building_geojson_path, building_geojson_path,
    sun_elevation_deg, gsd, min_shadow_area
)
