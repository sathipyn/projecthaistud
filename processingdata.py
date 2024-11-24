import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical


class DataProcessor:
    def __init__(self, image_size=(64, 512)):
        self.image_size = image_size
        # Enhanced character map with Indonesian-specific characters
        self.char_map = {c: i for i, c in enumerate(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            " .,!?:()[]+-*/=<>^_%|~²³°"  # Basic symbols
            "∑∏∆∇∫∮∝∞∈∉∋∌∩∪⊂⊃⊆⊇≈≠≡≤≥"  # Mathematical symbols
            "αβγδεζηθικλμνξοπρστυφχψω"  # Greek letters
            "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
            "…""''„‚«»‹›"  # Additional punctuation
        )}
        self.idx_to_char = {v: k for k, v in self.char_map.items()}
def load_data(image_folder, label_file, image_size=(512, 512,)):
    # Load labels with explicit encoding
    try:
        labels_df = pd.read_csv(label_file, encoding='utf-8')  # Try utf-8 first
    except UnicodeDecodeError:
        # If utf-8 fails, fallback to a common encoding like ISO-8859-1
        labels_df = pd.read_csv(label_file, encoding='ISO-8859-1')
   
    file_names = labels_df['file_name']
    texts = labels_df['extracted_text']
   
    # Load images and preprocess
    images = []
    for file in file_names:
        img_path = os.path.join(image_folder, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Grayscale
        img = cv2.resize(img, image_size)  # Resize
        img = cv2.threshold(img, 0, 512, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # Better binarization
        img = img / 255.0  # Normalize
        images.append(img)
   
    images = np.array(images).reshape(-1, image_size[0], image_size[1], 1)  # Add channel dimension
    return images, texts

def encode_labels(texts, char_map):
    encoded = [[char_map[c] for c in text] for text in texts]
    return encoded

if __name__ == "__main__":
    image_folder = "data/output_images/"
    label_file = "data/labeling_template.csv"
    images, texts = load_data(image_folder, label_file)
    print("Loaded images:", images.shape)
    print("Sample text:", texts[0])
