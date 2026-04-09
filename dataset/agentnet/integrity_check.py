import os
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

def verify_image_folder(folder_path):
    corrupted_files = []
    supported_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    
    # Get list of files
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_extensions)]
    
    print(f"Checking {len(files)} images in {folder_path}...")
    
    for filename in tqdm(files):
        file_path = os.path.join(folder_path, filename)
        try:
            with Image.open(file_path) as img:
                # 1. Verify checks the file header (fast)
                img.verify() 
                
            # 2. Re-open to check actual pixel data (thorough)
            # Some corruptions only appear during full decoding
            with Image.open(file_path) as img:
                img.load() 
                
        except (UnidentifiedImageError, IOError, SyntaxError, OSError) as e:
            print(f"\n[CORRUPT] {filename}: {e}")
            corrupted_files.append(file_path)
            
    return corrupted_files

# Usage
bad_images = verify_image_folder("data/win_mac_Images")

if not bad_images:
    print("All images are healthy!")
else:
    print(f"\nFound {len(bad_images)} corrupted files.")
    # Option: Move them to a 'trash' folder or delete