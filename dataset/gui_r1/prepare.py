"""
GUI-R1 Dataset Preparation (Model-Agnostic, Schema-Bypass)
Directly streams Parquet files via Hugging Face FileSystem and PyArrow to completely
bypass the repository's broken schemas, keeping RAM usage minimal.
"""

from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq
from datasets import Dataset, Features, Image, Value, Sequence
from PIL import Image as PILImage
import io
import os

def prepare_and_save(
        dataset_path="ritzzai/GUI-R1",
        split="test",
        output_dir="./data/gui_r1_raw",
        resize_to: tuple = None,
    ):
    
    print(f"Connecting to Hugging Face Hub to stream {dataset_path} ({split})...")
    fs = HfFileSystem()
    
    # Find all parquet files for the requested split
    repo_path = f"datasets/{dataset_path}"
    files = fs.glob(f"{repo_path}/*{split}*.parquet")
    
    if not files:
        raise ValueError(f"No parquet files found for split '{split}'")

    def data_generator():
        for file_path in files:
            print(f"Streaming: {file_path}")
            
            # Open the file over the network
            with fs.open(file_path, "rb") as f:
                pf = pq.ParquetFile(f)
                
                # Process in small batches of 50 rows to keep RAM usage near zero
                for batch in pf.iter_batches(batch_size=50):
                    pydict = batch.to_pydict()
                    num_rows = len(pydict[list(pydict.keys())[0]])
                    
                    for i in range(num_rows):
                        # Extract row safely
                        example = {k: v[i] for k, v in pydict.items()}
                        
                        # 1. Handle Image: Extract bytes and convert to PIL
                        img_data = example.get("image")
                        img = None
                        if img_data is not None:
                            # HF parquets store images as structs: {'bytes': b'...', 'path': ...}
                            img_bytes = img_data.get("bytes") if isinstance(img_data, dict) else img_data
                            
                            if img_bytes:
                                try:
                                    img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                                    if resize_to is not None:
                                        img = img.resize(resize_to)
                                except Exception as e:
                                    print(f"Warning: Failed to decode an image: {e}")
                        
                        # 2. Handle Bounding Box safely (force float to avoid int/float schema crashes)
                        raw_bbox = example.get("gt_bbox")
                        clean_bbox = [float(x) for x in raw_bbox] if raw_bbox is not None else []

                        # 3. Yield only the exact columns we care about
                        yield {
                            "image": img,
                            "gt_action": str(example.get("gt_action", "")),
                            "gt_bbox": clean_bbox,
                            "gt_input_text": str(example.get("gt_input_text", "")),
                            "instruction": str(example.get("instruction", "")),
                            "history": str(example.get("history", ""))
                        }

    # Lock in a strict schema so HF doesn't try to guess
    strict_features = Features({
        "image": Image(),
        "gt_action": Value("string"),
        "gt_bbox": Sequence(Value("float32")),
        "gt_input_text": Value("string"),
        "instruction": Value("string"),
        "history": Value("string"),
    })

    print(f"Saving cleaned, model-agnostic dataset to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Build dataset from generator. Because we strictly typed it and drop bad columns, 
    # the Arrow writer will never panic.
    processed_dataset = Dataset.from_generator(
        data_generator, 
        features=strict_features
    )
    processed_dataset.save_to_disk(output_dir)
    
    print(f"\nDone! Cleaned dataset safely saved to {output_dir}")

if __name__ == "__main__":
    prepare_and_save(
        split="test", 
        output_dir="./data/gui_r1_raw_test",
        # resize_to=(1024, 1024) # Uncomment to resize images and save disk space
    )