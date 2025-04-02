import cv2
import numpy as np
import rasterio
import json
from rasterio.transform import rowcol
from shapely.geometry import Polygon, mapping

# Function to preprocess the image (convert to grayscale and enhance)
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None, None, None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    
    return image, gray, enhanced  # Return original image, grayscale, and enhanced

# Function to detect shadows in the enhanced grayscale image
def detect_shadows(enhanced, threshold_value):
    _, shadow_mask = cv2.threshold(enhanced, threshold_value, 255, cv2.THRESH_BINARY_INV)  # Detect dark areas
    kernel = np.ones((5, 5), np.uint8)  
    shadow_cleaned = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
    shadow_cleaned = cv2.dilate(shadow_cleaned, kernel, iterations=1)  
    return shadow_cleaned

# Convert pixel coordinates to georeferenced (lat/lon or projected)
def pixel_to_geo(coords, transform):
    geo_coords = [rasterio.transform.xy(transform, y, x) for x, y in coords]  # Convert pixel to geo-coordinates
    return geo_coords

# Function to detect shadows and save as GeoJSON
def save_shadows_geojson(image_path, geotiff_path, threshold_value, output_geojson, min_shadow_area=100):
    original_image, gray, enhanced = preprocess_image(image_path)
    if original_image is None or gray is None or enhanced is None:
        return  

    shadow_mask = detect_shadows(enhanced, threshold_value)
    
    # Read GeoTIFF for georeferencing
    with rasterio.open(geotiff_path) as dataset:
        transform = dataset.transform  # Affine transform for pixel-to-geo conversion
        crs = dataset.crs  # Get projection info

    # Find contours of detected shadows
    contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_shadow_area:  # Filter small noise
            epsilon = 0.005 * cv2.arcLength(cnt, True)  # Approximate contour for smoother polygons
            approx = cv2.approxPolyDP(cnt, epsilon, True)  
            
            # Convert pixel coordinates to geographic coordinates
            pixel_coords = [(pt[0][0], pt[0][1]) for pt in approx]
            geo_coords = pixel_to_geo(pixel_coords, transform)

            # Create GeoJSON feature
            feature = {
                "type": "Feature",
                "geometry": mapping(Polygon(geo_coords)),
                "properties": {
                    "shadow_area_pixels": area  
                }
            }
            features.append(feature)

    # Create GeoJSON object
    geojson_data = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": str(crs)}},  # Ensure correct CRS
        "features": features
    }

    # Save to file
    with open(output_geojson, "w") as f:
        json.dump(geojson_data, f, indent=4)
    
    print(f"GeoJSON saved: {output_geojson}, Features: {len(features)}")

# === CONFIGURATION ===
tiff_image_path = r'C:\Users\Hugo\Documents\GitHub\Thesis2025\img\PREFF95000.tif'  # Input TIFF for reference
shadow_image_path = r'C:\Users\Hugo\Documents\GitHub\Thesis2025\img\PREFF95000.tif'  # Image for shadow detection
output_geojson_path = r'C:\Users\Hugo\Documents\GitHub\Thesis2025\output\PRE_shadows.geojson'  # Output GeoJSON path
threshold_value = 20  # Adjusted for better detection
min_shadow_area = 100  # Ignore very small shadows

# === RUN PROCESS ===
save_shadows_geojson(shadow_image_path, tiff_image_path, threshold_value, output_geojson_path, min_shadow_area)