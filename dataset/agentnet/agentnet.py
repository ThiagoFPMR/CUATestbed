import json
import os
from PIL import Image
from datasets import Dataset


def filter_trajectory(traj: dict, alignment_thr: int = 7) -> bool:
    """
    Filter function to filter out low-quality trajectories.
    
    Args:
        traj (dict): A single trajectory from the dataset.
        alignment_thr (int): The threshold for the alignment score below which incomplete trajectories will be filtered out.
    Returns:
        bool: True if the trajectory is valid and should be included, False otherwise.
    """

    # Filtering out the examples without all 11 attributes
    if "task_completed" not in traj:
        return False
    # Filtering out incomplete trajectories with low alignment scores (e.g., less than 7) that are not marked as "infeasible"
    if traj["task_completed"] is False and traj.get("domain", "") != "infeasible" and traj["alignment_score"] < alignment_thr:
        return False

    return True

def filter_step(step_idx: int, trajectory: dict, min_instruction_len: int = 10) -> bool:
    """
    Filter function to filter out low-quality steps within trajectories.
    
    Args:
        step_idx (int): The index of the step within the trajectory.
        trajectory (dict): The trajectory to which the step belongs.
        min_instruction_len (int): The minimum length of the human instruction in words.
    Returns:
        bool: True if the step is valid and should be included, False otherwise.
    """
    steps = trajectory.get("traj", [])

    # Filter out redudant and incorrect steps
    if step_idx < len(steps) - 1:
        if steps[step_idx + 1]["value"]["last_step_redundant"] is True:
            return False
        if steps[step_idx + 1]["value"]["last_step_correct"] is False:
            return False

    # Filter for incomplete trajectories
    action_type = steps[step_idx]["value"]["code"].split(".")[1].split("(")[0]
    if trajectory.get("task_completed", True) is False:
        # Don't add the "terminate" action
        if action_type == "terminate":
            return False
        # Filter out low-effort human instructions (less than min_instruction_len words)
        if len(trajectory.get("instruction", "").split()) < min_instruction_len:
            return False

    return True

def get_action_from_step(step: dict) -> str:
    """Extract the action from a step's code."""
    return step["value"]["code"].split(".")[1]

def conversation_from_step(
    step_idx: int, 
    trajectory: dict,
    system_prompt: str,
    images_dir: str,
    history_length: int = 3,
    history_images: bool = True,
    min_instruction_len: int = 10
) -> list:
    """
    Construct a conversation history up to a specific step in the trajectory.
    
    Args:
        step_idx (int): The index of the step within the trajectory.
        trajectory (dict): The trajectory to which the step belongs.
        system_prompt (str): The system prompt to use.
        images_dir (str): The directory where the step images are stored.
        history_length (int): The number of previous steps to include in the conversation history.
        history_images (bool): Whether to include images in the conversation history.
        min_instruction_len (int): The minimum length of the human instruction in words.
    Returns:
        list: A list of messages representing the conversation history up to the specified step.
    """
    steps = trajectory.get("traj", [])
    instruction = trajectory["instruction"]
    if len(instruction.split()) < min_instruction_len:
        instruction = trajectory["natural_language_task"]
    
    # Start with the system prompt
    messages = [
        {
            "role": "system", 
            "content": [{"type": "text", "text": system_prompt}]
        }
    ]
    
    # Add up to `history_length` previous steps to the conversation history
    for idx in range(max(0, step_idx - history_length), step_idx):
        step = steps[idx]
        if history_images:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": os.path.join(images_dir, step["image"])},
                ]
            })

        messages.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"Step: {idx}, Action: {get_action_from_step(step)}"}
            ]
        })
    
    # Add the current step
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": os.path.join(images_dir, steps[step_idx]["image"])},
            {"type": "text", "text": f"Task instruction: {instruction}"}
        ]
    })

    return messages

def load_agentnet_records(
    system_prompt: str, 
    data_sources: list,
    history_length: int = 3,
    history_images: bool = True,
    min_instruction_len: int = 10
) -> list:
    """
    Load the AgentNet dataset from multiple sources, apply filtering, and construct conversation histories for each valid step.

    Args:
        system_prompt (str): The system prompt to use for all conversations.
        data_sources (list): A list of tuples, where each tuple contains the path to 
                             a JSONL file and the corresponding image directory (e.g., 
                             [(jsonl_path1, image_dir1), (jsonl_path2, image_dir2), ...]).    
        history_length (int): The number of previous steps to include in the conversation history.
        history_images (bool): Whether to include images in the conversation history.
        min_instruction_len (int): The minimum length of the human instruction in words.
    
    Returns:
        list: A list of conversation histories, where each conversation is a list of messages.
    """
    dataset_records = []
    for jsonl_path, image_dir in data_sources:
        print(f"Loading data from: {jsonl_path}...")

        with open(jsonl_path, 'r', encoding='utf-8') as source:
            for line in source:
                trajectory = json.loads(line)
                if not filter_trajectory(trajectory):
                    continue
                
                steps = trajectory.get("traj", [])
                for step_idx in range(len(steps)):
                    if not filter_step(step_idx, trajectory):
                        continue

                    conversation = conversation_from_step(
                        step_idx, 
                        trajectory,
                        system_prompt,
                        image_dir,
                        history_length,
                        history_images,
                        min_instruction_len
                    )
                    dataset_records.append({"messages": conversation})

    return dataset_records

def load_batch_images(batch: dict) -> dict:
    """
    Load images for a batch of conversations and replace image paths with actual image data.
    
    Args:
        batch (dict): A batch of conversations, where each conversation is a list of messages.
    Returns:
        dict: The batch with images loaded as PIL Image objects.
    """
    new_batch = []
    for conversation in batch["messages"]:
        new_conversation = []
        for message in conversation:
            new_content = []
            for item in message["content"]:
                if item["type"] == "image":
                    try:
                        image_path = item["image"]
                        image = Image.open(image_path).convert("RGB")
                        new_content.append({"type": "image", "image": image})
                    except Exception as e:
                        print(f"Error loading image {item['image']}: {e}")
                        continue
                else:
                    new_content.append(item)
            new_conversation.append({"role": message["role"], "content": new_content})
        new_batch.append(new_conversation)
    return {"messages": new_batch}

def load_agentnet(
    system_prompt: str, 
    history_length: int = 3, 
    history_images: bool = True, 
    min_instruction_len: int = 10
) -> Dataset:
    """
    Load the AgentNet dataset from multiple sources, apply filtering, construct conversation histories 
    for each valid step, and return a Hugging Face Dataset object.
    Args:
        system_prompt (str): The system prompt to use for all conversations.
        history_length (int): The number of previous steps to include in the conversation history.
        history_images (bool): Whether to include images in the conversation history.
        min_instruction_len (int): The minimum length of the human instruction in words.
    Returns:
        Dataset: A Hugging Face Dataset object containing the processed AgentNet data.
    """ 
    DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(
            f"Dataset directory '{DATASET_DIR}' not found. "
            "Please download the dataset using the following command:\n"
            "huggingface-cli download xlangai/AgentNet --repo-type dataset --local-dir ./data \n"
        )

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

    # Load all records and apply set transformations to load images in batches
    agentnet_records = load_agentnet_records(system_prompt, sources, history_length, history_images, min_instruction_len)
    agentnet_dataset = Dataset.from_list(agentnet_records)
    agentnet_dataset.set_transform(load_batch_images)

    return agentnet_dataset


if __name__ == "__main__":
    system_prompt = "You are an assistant that helps users complete tasks on their computer. You can perform various actions such as clicking, typing, and navigating the interface. Please provide step-by-step instructions to help the user achieve their goal."
    dataset = load_agentnet(system_prompt)
    import pdb; pdb.set_trace()