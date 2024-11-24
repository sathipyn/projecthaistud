import tensorflow as tf
import cv2
import numpy as np
import os
from processingdata import DataProcessor
from modelocr import build_ocr_model  

class OCRPredictor:
    def __init__(self, model_path="models/ocr_model.h5", image_size=(512, 512)):
        print(f"Loading model from {model_path}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        self.model = tf.keras.models.load_model(model_path)
        self.image_size = image_size
        self.processor = DataProcessor()
        self.char_map = self.processor.char_map
        self.idx_to_char = {v: k for k, v in self.char_map.items()}
        print("Model loaded successfully")

    def preprocess_image(self, image_path):
        """Preprocess single image for prediction"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at {image_path}")
            
        # Read and preprocess image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image at {image_path}")
            
        print(f"Original image shape: {img.shape}")
        
        # Resize image
        img = cv2.resize(img, self.image_size)
        print(f"Resized image shape: {img.shape}")
        
        # Normalize image without thresholding
        img = img.astype(np.float32) / 255.0
        
        # Reshape for model input
        img = img.reshape(1, self.image_size[0], self.image_size[1], 1)
        print(f"Final preprocessed shape: {img.shape}")
        
        return img

    def decode_predictions(self, pred, top_k=3, threshold=0.2):
        """Convert model predictions to text using top-k predictions and confidence threshold."""
        try:
            # Initialize an empty text string
            text = ''

            # Loop through each timestep (e.g., each position in the sequence)
            for t in range(pred.shape[1]):
                # Get top-k predicted indices for this timestep
                top_k_indices = np.argsort(pred[0, t])[-top_k:]  # Get the top-k indices with highest probability
                top_k_probs = pred[0, t, top_k_indices]  # Corresponding top-k probabilities

                # Find the most probable index and its probability
                most_probable_idx = top_k_indices[-1]  # The index with highest probability
                most_probable_prob = top_k_probs[-1]

                # If the probability of the most likely prediction is above the threshold, include it
                if most_probable_prob >= threshold:
                    char = self.idx_to_char.get(most_probable_idx)
                    if char:
                        text += char

            return text.strip()
        
        except Exception as e:
            print(f"Error in decode_predictions: {str(e)}")
            raise


    def predict(self, image_path):
        """Predict text from image"""
        try:
            # Preprocess image
            print("\nPreprocessing image...")
            processed_img = self.preprocess_image(image_path)
            
            # Make prediction
            print("\nRunning inference...")
            predictions = self.model.predict(processed_img, verbose=1)
            
            # Decode predictions to text
            print("\nDecoding predictions...")
            predicted_text = self.decode_predictions(predictions)
            
            if not predicted_text:
                print("Warning: Empty prediction result")
            else:
                print(f"Successfully decoded text of length: {len(predicted_text)}")
                
            return predicted_text
        
        except FileNotFoundError as e:
            print(f"File error: {str(e)}")
            return None
        except ValueError as e:
            print(f"Value error: {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print("Full traceback:")
            print(traceback.format_exc())
            return None

def main():
    try:
        # Initialize predictor
        predictor = OCRPredictor()
        
        # Example usage
        image_path = "data/output_images/question_100.png"  # Ganti dengan path gambar yang ingin ditest
        
        print(f"\nStarting OCR prediction for image: {image_path}")
        predicted_text = predictor.predict(image_path)
        
        if predicted_text:
            print("\nPredicted Text:")
            print("-" * 50)
            print(predicted_text)
            print("-" * 50)
        else:
            print("\nNo text was predicted from the image.")
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
