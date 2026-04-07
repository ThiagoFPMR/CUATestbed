import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import re
from collections import Counter
import json
import os
from tqdm import tqdm

def find_near_duplicates_f1(data_list, text_key, f1_threshold=0.85):
    """
    Directly compares every string in the 2.3k set against the 18k set
    using Token-Level F1.
    """
    
    # 1. Internal helper for the F1 math
    def get_f1_from_counters(cnt_a, cnt_b, len_a, len_b):
        common = cnt_a & cnt_b
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        
        precision = 1.0 * num_same / len_b
        recall = 1.0 * num_same / len_a
        return (2 * precision * recall) / (precision + recall)

    # 2. Separate groups based on your boolean logic
    group_2k = []
    group_18k = []
    
    for item in data_list:
        # --- IMPLEMENT YOUR BOOLEAN LOGIC HERE ---
        is_in_2k_group = not item.get("has_11", False)  # Example logic: if "has_11" is False, it's in the 2k group 
        # -----------------------------------------
        if is_in_2k_group:
            group_2k.append(item)
        else:
            group_18k.append(item)

    # 3. Pre-tokenize and Pre-count (CRITICAL for performance)
    # We store (Counter, Total_Token_Count) for every string
    processed_2k = []
    for d in group_2k:
        tokens = re.findall(r'\w+', d[text_key].lower())
        processed_2k.append((Counter(tokens), len(tokens)))

    processed_18k = []
    for d in group_18k:
        tokens = re.findall(r'\w+', d[text_key].lower())
        processed_18k.append((Counter(tokens), len(tokens)))

    print(f"Starting brute-force F1 comparison: {len(group_2k)} x {len(group_18k)}")

    results = []

    # 4. The Nested Loop (41.4 Million Comparisons)
    bar = tqdm(total=len(group_2k) * len(group_18k), desc="Comparing for Near-Duplicates")
    for i, (cnt_a, len_a) in enumerate(processed_2k):
        if len_a == 0:
            bar.update(len(group_18k))
            continue
            
        for j, (cnt_b, len_b) in enumerate(processed_18k):
            if len_b == 0: 
                bar.update(1)
                continue
            
            # Fast check: If the length difference is too massive, F1 can't possibly 
            # hit the threshold. This saves us from doing the Counter intersection.
            max_possible_f1 = (2.0 * min(len_a, len_b)) / (len_a + len_b)
            if max_possible_f1 < f1_threshold:
                bar.update(1)
                continue

            score = get_f1_from_counters(cnt_a, cnt_b, len_a, len_b)
            
            if score >= f1_threshold:
                results.append({
                    "source": group_2k[i],
                    "duplicate": group_18k[j],
                    "f1_score": score
                })
            bar.update(1)
    # import pdb; pdb.set_trace()
    return results

if __name__ == "__main__":

    DATASET_DIR = "AgentNet_Raw"
    # Dynamically build the paths using the DATASET_DIR variable
    sources = [
        os.path.join(DATASET_DIR, "agentnet_ubuntu_5k.jsonl"), 
        os.path.join(DATASET_DIR, "agentnet_win_mac_18k.jsonl"),
    ]

    instructions = []

    for jsonl_file in sources:
        print(f"Profiling data from: {jsonl_file}...")

        imgs_dir = DATASET_DIR + "/ubuntu_images"

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:  
                example = json.loads(line)
                instruct_info = {
                    "has_11": "task_completed" in example,
                    "instruction": example.get("instruction", ""),
                }
                instructions.append(instruct_info)

    duplicates = find_near_duplicates_f1(instructions, text_key="instruction", f1_threshold=0.8)

    import pdb; pdb.set_trace()