import json
import os
from PIL import Image
from datasets import Dataset

# --- Configuration ---
# Update this variable with the absolute or relative path to your dataset folder
DATASET_DIR = "AgentNet_Raw"  # Change this to your actual dataset directory

def load_unified_agentnet(data_sources):
    """
    data_sources: list of tuples (full_jsonl_path, full_image_dir_path)
    """
    dataset_records = []
    
    for jsonl_path, image_dir in data_sources:
        print(f"Loading data from: {jsonl_path}...")
        
        # Read the JSONL (JSON Lines) file and construct records
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                import pdb; pdb.set_trace()
                instruction = data.get("natural_language_task", data.get("instruction", "Complete the task."))
                trajectories = data.get("traj", [])
                
                messages = [
                    {
                        "role": "system", 
                        "content": [{"type": "text", "text": "You are a capable GUI agent. Analyze the screen and provide the exact reasoning and actions to achieve the user's goal."}]
                    }
                ]
                images_list = []
                valid_trajectory = True
                
                for step in trajectories:
                    image_file = step.get("image", step.get("screenshot", ""))
                    # image_dir is now the fully resolved path
                    image_path = os.path.join(image_dir, image_file)
                    
                    try:
                        img = Image.open(image_path).convert("RGB")
                        images_list.append(img)
                    except (FileNotFoundError, IOError):
                        valid_trajectory = False
                        break 

                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": f"Goal: {instruction}\nObservation: {step.get('observation', '')}"}
                        ]
                    })
                    
                    messages.append({
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": f"Thought: {step.get('thought', '')}\nAction: {step.get('action', '')}"}
                        ]
                    })
                
                if valid_trajectory and len(images_list) > 0:
                    dataset_records.append({
                        "messages": messages,
                        "images": images_list
                    })
                    
    return dataset_records

# --- Execution ---

# Dynamically build the paths using the DATASET_DIR variable
sources = [
    (
        os.path.join(DATASET_DIR, "agentnet_ubuntu_5k.jsonl"), 
        os.path.join(DATASET_DIR, "ubuntu_images")
    ),
    (
        os.path.join(DATASET_DIR, "agentnet_win_mac_18k.jsonl"), 
        os.path.join(DATASET_DIR, "win_mac_images")
    )
]

# 1. Load all records into a single Python list
all_records = load_unified_agentnet(sources)

# 2. Convert to a Hugging Face Dataset
hf_dataset = Dataset.from_list(all_records)

# 3. Shuffle the entire dataset
shuffled_dataset = hf_dataset.shuffle(seed=42)

print(f"Success! Created a single, shuffled dataset with {len(shuffled_dataset)} multi-turn trajectories.")