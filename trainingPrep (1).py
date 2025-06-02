import os
import tarfile
import json
import numpy as np
import shapely.wkt
import cv2
from tqdm import tqdm
from rasterio.features import rasterize
import io
import matplotlib.pyplot as plt

# CONFIG
PATCH_SIZE = 256
STRIDE = 224
SHARD_SIZE = 4096
EXCLUDED_SCENE_KEYWORD = "antakya"

TRAIN_TAR = r"D:\train.tar"
VAL_TAR = r"D:\hold.tar"
OUT_DIR = r"D:\shards_challenge"
TEMP_DIR = os.path.join(OUT_DIR, "extracted")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

TRACKING_FILE = os.path.join(OUT_DIR, "shard_tracking.json")
if os.path.exists(TRACKING_FILE):
    with open(TRACKING_FILE, 'r') as f:
        SHARD_LOG = json.load(f)
else:
    SHARD_LOG = {}

# Global stats
global_tile_stats = {
    "total_tiles": 0,
    "positive_tiles": 0,
    "damage_pixels": 0,
    "total_pixels": 0,
    "class_pixel_counts": {0: 0, 1: 0, 2: 0}
}

# To visualize later
most_damage_tile = None
most_damage_mask = None
most_damage_count = 0

def normalize_img(img):
    return img / 255.0

def load_polygons_by_class(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    damage_polys = []
    nodamage_polys = []
    for feature in data['features']['xy']:
        subtype = feature['properties'].get('subtype', '').lower()
        try:
            poly = shapely.wkt.loads(feature['wkt'])
            if subtype in ['destroyed', 'major-damage']:
                damage_polys.append(poly)
            elif subtype in ['no-damage', 'minor-damage']:
                nodamage_polys.append(poly)
        except Exception as e:
            print(f"  [Warning] WKT parse failed: {e}")
    print(f"  → Loaded {len(damage_polys)} damage and {len(nodamage_polys)} intact polygons")
    return damage_polys, nodamage_polys

def rasterize_multiclass_mask(damage_polys, nodamage_polys, shape):
    H, W = shape
    transform = [1, 0, 0, 0, -1, H]
    mask = np.zeros((H, W), dtype=np.uint8)
    if nodamage_polys:
        mask = rasterize(
            [(poly, 1) for poly in nodamage_polys],
            out_shape=(H, W),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            default_value=1
        )
    if damage_polys:
        damage_mask = rasterize(
            [(poly, 2) for poly in damage_polys],
            out_shape=(H, W),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            default_value=2
        )
        mask = np.where(damage_mask == 2, 2, mask)
    print(f"  → Final mask shape: {mask.shape} with unique values: {np.unique(mask)}")
    return mask

def process_scene(scene_id, pre_path, post_path, json_path):
    global most_damage_tile, most_damage_mask, most_damage_count

    print(f"\nProcessing scene: {scene_id}")
    pre_img = cv2.imread(pre_path, cv2.IMREAD_COLOR)
    post_img = cv2.imread(post_path, cv2.IMREAD_COLOR)

    if pre_img is None or post_img is None:
        print(f"❌ Skipping {scene_id} due to unreadable image(s)")
        return []

    pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    post_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    img_6ch = np.vstack((normalize_img(pre_img), normalize_img(post_img)))
    print("  → Loaded and stacked pre/post images")
    damage_polys, nodamage_polys = load_polygons_by_class(json_path)
    mask = rasterize_multiclass_mask(damage_polys, nodamage_polys, pre_img.shape[1:])

    tiles = []
    _, H, W = img_6ch.shape
    for y in range(0, H - PATCH_SIZE + 1, STRIDE):
        for x in range(0, W - PATCH_SIZE + 1, STRIDE):
            img_tile = img_6ch[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            mask_tile = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            tiles.append((img_tile, mask_tile))

            for cls in [0, 1, 2]:
                global_tile_stats["class_pixel_counts"][cls] += int((mask_tile == cls).sum())
            damage_count = int((mask_tile == 2).sum())
            global_tile_stats["damage_pixels"] += damage_count
            global_tile_stats["total_pixels"] += mask_tile.size
            global_tile_stats["total_tiles"] += 1
            if damage_count > 0:
                global_tile_stats["positive_tiles"] += 1
            if damage_count > most_damage_count:
                most_damage_count = damage_count
                most_damage_tile = img_tile.copy()
                most_damage_mask = mask_tile.copy()

    print(f"  → Total tiles: {len(tiles)}")
    return tiles

def write_tar_shard(shard_path, samples, scene_ids):
    print(f"Writing shard: {shard_path} with {len(samples)} samples")

    with tarfile.open(shard_path, "w") as tarf:
        for i, (img, mask) in enumerate(samples):
            base = f"{i:06d}"
            img_bytes = io.BytesIO()
            mask_bytes = io.BytesIO()
            np.save(img_bytes, img.astype(np.float32))
            np.save(mask_bytes, mask.astype(np.uint8))
            img_bytes.seek(0)
            mask_bytes.seek(0)
            img_info = tarfile.TarInfo(name=f"{base}.img.npy")
            img_info.size = len(img_bytes.getbuffer())
            tarf.addfile(img_info, img_bytes)
            mask_info = tarfile.TarInfo(name=f"{base}.mask.npy")
            mask_info.size = len(mask_bytes.getbuffer())
            tarf.addfile(mask_info, mask_bytes)
    SHARD_LOG[os.path.basename(shard_path)] = scene_ids
    with open(TRACKING_FILE, 'w') as f:
        json.dump(SHARD_LOG, f, indent=2)

def extract_tar_to_folder(tar_path, output_dir):
    print(f"📂 Extracting {tar_path} to {output_dir}...")
    with tarfile.open(tar_path, 'r') as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting files"):
            if member.name.startswith("train/"):
                member.name = member.name[len("train/"):]
            elif member.name.startswith("hold/"):
                member.name = member.name[len("hold/"):]
            elif member.name.startswith("test/"):
                member.name = member.name[len("test/"):]
            tar.extract(member, path=output_dir, filter='data')

def extract_and_process(split_name, tar_path):
    extract_dir = os.path.join(TEMP_DIR, split_name)
    extract_tar_to_folder(tar_path, extract_dir)
    image_dir = os.path.join(extract_dir, "images")
    label_dir = os.path.join(extract_dir, "labels")
    scenes = {}
    for fname in os.listdir(image_dir):
        if fname.endswith("_pre_disaster.png") and EXCLUDED_SCENE_KEYWORD not in fname:
            sid = fname.replace("_pre_disaster.png", "")
            pre_path = os.path.join(image_dir, f"{sid}_pre_disaster.png")
            post_path = os.path.join(image_dir, f"{sid}_post_disaster.png")
            json_path = os.path.join(label_dir, f"{sid}_post_disaster.json")
            if os.path.exists(post_path) and os.path.exists(json_path):
                scenes[sid] = (pre_path, post_path, json_path)
    print(f"✅ Found {len(scenes)} scenes")
    samples, shard_idx = [], 0
    skipped_count = 0
    scene_log_path = os.path.join(OUT_DIR, f"{split_name}_scenes_done.txt")
    done_scenes = set()
    if os.path.exists(scene_log_path):
        with open(scene_log_path, 'r') as f:
            done_scenes = set(line.strip() for line in f)
    existing_shards = set(SHARD_LOG.keys())
    for sid, (pre, post, label) in tqdm(scenes.items(), desc=f"Processing {split_name}"):
        if sid in done_scenes:
            print(f"⏩ Skipping already-processed scene: {sid}")
            continue
        tiles = process_scene(sid, pre, post, label)
        if not tiles:
            skipped_count += 1
            continue
        samples.extend(tiles)
        pending_scenes = [sid]
        while len(samples) >= SHARD_SIZE:
            shard = samples[:SHARD_SIZE]
            samples = samples[SHARD_SIZE:]
            shard_name = f"{split_name}-{shard_idx:05d}.tar"
            shard_path = os.path.join(OUT_DIR, shard_name)
            if shard_name in existing_shards:
                print(f"🔁 Rewriting possibly incomplete shard: {shard_name}")
            write_tar_shard(shard_path, shard, pending_scenes)
            shard_idx += 1
            pending_scenes = []
        with open(scene_log_path, 'a') as f:
            f.write(sid + "\n")
    if samples:
        shard_name = f"{split_name}-{shard_idx:05d}.tar"
        shard_path = os.path.join(OUT_DIR, shard_name)
        write_tar_shard(shard_path, samples, [sid for sid in scenes if sid not in done_scenes])
    print(f"🚫 Skipped {skipped_count} scene(s) due to unreadable image files.")

if __name__ == "__main__":
    extract_and_process("train", TRAIN_TAR)
    extract_and_process("val", VAL_TAR)
    print("\n📊 Output Dataset Summary (All Tiles):")
    print(f"Total tiles: {global_tile_stats['total_tiles']}")
    print(f"Positive tiles: {global_tile_stats['positive_tiles']} "
          f"({global_tile_stats['positive_tiles'] / global_tile_stats['total_tiles']:.2%})")
    print(f"Total pixels: {global_tile_stats['total_pixels']:,}")
    for cls, label in [(0, "Background"), (1, "No Damage"), (2, "Damage")]:
        pixels = global_tile_stats["class_pixel_counts"][cls]
        pct = pixels / global_tile_stats["total_pixels"] if global_tile_stats["total_pixels"] else 0
        print(f"{label} pixels (class {cls}): {pixels:,} ({pct:.4%})")

    # ✅ Visualize the most damaged tile and its mask
    if most_damage_tile is not None:
        print("\n🖼 Visualizing most damage-heavy tile...")
        rgb = most_damage_tile[:3].transpose(1, 2, 0)
        mask = most_damage_mask
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(rgb)
        plt.title("Tile RGB")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='viridis', vmin=0, vmax=2)
        plt.title("Mask (0=bg, 1=no dmg, 2=damage)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
