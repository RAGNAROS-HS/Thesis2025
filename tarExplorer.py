import tarfile
import os
from collections import defaultdict
from pathlib import Path

def inspect_tar_extensions(tar_path, max_list=5):
    print(f"\n🔍 Inspecting TAR file: {tar_path}")
    try:
        with tarfile.open(tar_path, 'r') as tar:
            members = tar.getmembers()
            ext_to_files = defaultdict(list)
            dirs = set()

            for m in members:
                if m.isdir():
                    dirs.add(m.name)
                    continue
                ext = Path(m.name).suffix.lower()
                ext_to_files[ext].append(m.name)

            print(f"📁 Directories: {sorted(dirs)}")
            print(f"\n📊 File types found: {len(ext_to_files)}")

            for ext, files in sorted(ext_to_files.items(), key=lambda x: -len(x[1])):
                print(f"  {ext or '[no extension]'} : {len(files)} files")
                for sample in files[:max_list]:
                    print(f"    - {sample}")
                if len(files) > max_list:
                    print(f"    ... ({len(files) - max_list} more)\n")

    except Exception as e:
        print(f"❌ Error opening {tar_path}: {e}")

def main():
    tar_files = [
        r"D:\train.tar",
        r"D:\hold.tar",
        r"D:\test.tar"
    ]

    for tar_path in tar_files:
        if os.path.exists(tar_path):
            inspect_tar_extensions(tar_path)
        else:
            print(f"⚠️ File not found: {tar_path}")

if __name__ == "__main__":
    main()
