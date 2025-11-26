from datasets import load_dataset
from utils import resize_images, convert_to_rgb, make_conversation, is_numeric_answer

def load_math_vista():
    dataset = load_dataset("AI4Math/MathVista", split = "testmini")
    dataset = dataset.filter(is_numeric_answer)
    dataset = dataset.map(resize_images)
    dataset = dataset.map(convert_to_rgb)
    train_dataset = dataset.map(make_conversation)
    train_dataset = train_dataset.remove_columns("image")
    train_dataset = train_dataset.rename_column("decoded_image", "image")
    return train_dataset