import torch

def inspect_checkpoint(path):
    try:
        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location='cpu')
        print("\nTop-level keys:", list(checkpoint.keys()))

        for key, value in checkpoint.items():
            print(f"\n🔑 {key}:")
            if isinstance(value, dict):
                print(f"  ↪️ Dict with {len(value)} entries")
                example_key = list(value.keys())[0]
                if isinstance(value[example_key], torch.Tensor):
                    print(f"    Example: {example_key} -> tensor shape {value[example_key].shape}")
                else:
                    print(f"    Example: {example_key} -> {type(value[example_key])}")
            elif isinstance(value, torch.Tensor):
                print(f"  ↪️ Tensor: shape {value.shape}")
            else:
                print(f"  ↪️ Type: {type(value)} | Value: {value}")

    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")

if __name__ == "__main__":
    path = r"C:\Users\Hugo\Downloads\decent15.pth"  # Adjust path if needed
    inspect_checkpoint(path)
