"""
Garbage Classification Prediction Tool
Supports both webcam input and file path prediction using trained model
"""

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import argparse
import os
from pathlib import Path
import sys

class GarbageClassifier:
    def __init__(self, model_path='best_model.keras'):
        """
        Initialize the garbage classifier
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
        
        # Label mapping for 12 garbage categories
        self.label_map = {
            0: 'battery',
            1: 'biological',
            2: 'brown-glass',
            3: 'cardboard',
            4: 'clothes',
            5: 'green-glass',
            6: 'metal',
            7: 'paper',
            8: 'plastic',
            9: 'shoes',
            10: 'trash',
            11: 'white-glass'
        }
        
        # Define colors for each class (BGR format for OpenCV)
        self.colors = {
            'battery': (0, 255, 255),      # Yellow
            'biological': (0, 255, 0),     # Green
            'brown-glass': (19, 69, 139),  # Brown
            'cardboard': (203, 192, 255),  # Light pink
            'clothes': (255, 0, 255),      # Magenta
            'green-glass': (0, 128, 0),    # Dark green
            'metal': (128, 128, 128),      # Gray
            'paper': (255, 255, 255),      # White
            'plastic': (255, 0, 0),        # Blue
            'shoes': (0, 0, 255),          # Red
            'trash': (0, 0, 0),            # Black
            'white-glass': (255, 255, 240) # Light cyan
        }
    
    def load_model(self):
        """Load the trained model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            sys.exit(1)
    
    def preprocess_image(self, image):
        """
        Preprocess image for prediction
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Preprocessed image tensor
        """
        # Convert to PIL Image if it's a numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Resize to model input size
        image = image.resize((224, 224))
        
        # Convert to array and normalize
        image_array = np.array(image) / 255.0
        
        # Add batch dimension
        image_tensor = np.expand_dims(image_array, axis=0)
        
        return image_tensor
    
    def predict(self, image):
        """
        Make prediction on image
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)[0]
        predicted_class_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_class_idx])
        predicted_class = self.label_map[predicted_class_idx]
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions)[::-1][:3]
        top_predictions = [
            {
                'label': self.label_map[idx],
                'confidence': float(predictions[idx])
            }
            for idx in top_indices
        ]
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'all_scores': predictions.tolist()
        }
    
    def predict_from_file(self, file_path):
        """
        Predict from image file
        
        Args:
            file_path: Path to image file
            
        Returns:
            Prediction results
        """
        try:
            # Load image
            image = Image.open(file_path)
            
            # Make prediction
            result = self.predict(image)
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing file {file_path}: {str(e)}")
            return None
    
    def predict_from_webcam(self):
        """
        Real-time prediction from webcam
        """
        print("🎥 Starting webcam prediction...")
        print("Press 'q' to quit, 'c' to capture and save prediction")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        capture_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            # Create a copy for prediction
            prediction_frame = frame.copy()
            
            # Make prediction
            result = self.predict(prediction_frame)
            
            # Get prediction info
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            color = self.colors.get(predicted_class, (255, 255, 255))
            
            # Draw prediction on frame
            text = f"{predicted_class}: {confidence:.2%}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, color, 2, cv2.LINE_AA)
            
            # Draw confidence bar
            bar_width = int(confidence * 300)
            cv2.rectangle(frame, (10, 50), (10 + bar_width, 70), color, -1)
            cv2.rectangle(frame, (10, 50), (310, 70), (255, 255, 255), 2)
            
            # Show top 3 predictions
            y_offset = 100
            for i, pred in enumerate(result['top_predictions'][:3]):
                pred_text = f"{i+1}. {pred['label']}: {pred['confidence']:.2%}"
                pred_color = self.colors.get(pred['label'], (255, 255, 255))
                cv2.putText(frame, pred_text, (10, y_offset + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, pred_color, 2)
            
            # Show instructions
            cv2.putText(frame, "Press 'q' to quit, 'c' to capture", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Garbage Classification', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Capture and save prediction
                capture_count += 1
                filename = f"capture_{capture_count}_{predicted_class}.jpg"
                cv2.imwrite(filename, prediction_frame)
                print(f"📸 Captured: {filename} - Predicted: {predicted_class} ({confidence:.2%})")
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("🎥 Webcam prediction ended")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Garbage Classification Prediction Tool')
    parser.add_argument('--mode', choices=['webcam', 'file'], required=True,
                       help='Prediction mode: webcam or file')
    parser.add_argument('--model', default='best_model.keras',
                       help='Path to trained model file (default: best_model.keras)')
    parser.add_argument('--input', help='Input file path (required for file mode)')
    parser.add_argument('--batch', help='Directory path for batch prediction')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = GarbageClassifier(args.model)
    
    if args.mode == 'webcam':
        # Webcam prediction
        classifier.predict_from_webcam()
        
    elif args.mode == 'file':
        if args.batch:
            # Batch prediction from directory
            batch_predict(classifier, args.batch)
        elif args.input:
            # Single file prediction
            single_file_predict(classifier, args.input)
        else:
            print("❌ Error: Please provide --input for file path or --batch for directory")


def single_file_predict(classifier, file_path):
    """Predict single file"""
    print(f"🔍 Predicting: {file_path}")
    
    result = classifier.predict_from_file(file_path)
    
    if result:
        print(f"\n📊 Prediction Results:")
        print(f"🎯 Predicted Class: {result['predicted_class']}")
        print(f"🎪 Confidence: {result['confidence']:.2%}")
        print(f"\n🏆 Top 3 Predictions:")
        for i, pred in enumerate(result['top_predictions']):
            print(f"  {i+1}. {pred['label']}: {pred['confidence']:.2%}")


def batch_predict(classifier, directory_path):
    """Predict all images in a directory"""
    print(f"📁 Batch prediction from: {directory_path}")
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    directory = Path(directory_path)
    if not directory.exists():
        print(f"❌ Error: Directory not found: {directory_path}")
        return
    
    # Get all image files
    image_files = [f for f in directory.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print("❌ No image files found in directory")
        return
    
    print(f"📊 Found {len(image_files)} image files")
    results = []
    
    for file_path in image_files:
        print(f"🔍 Processing: {file_path.name}")
        result = classifier.predict_from_file(str(file_path))
        
        if result:
            results.append({
                'file': file_path.name,
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence']
            })
            print(f"  ✅ {result['predicted_class']} ({result['confidence']:.2%})")
        else:
            print(f"  ❌ Failed to process")
    
    # Summary
    print(f"\n📈 Batch Prediction Summary:")
    print(f"Total files processed: {len(results)}")
    
    # Group by predicted class
    class_counts = {}
    for result in results:
        class_name = result['predicted_class']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("\n🗂️ Classification Summary:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count} files")


def interactive_mode():
    """Interactive mode for easy usage"""
    print("🚀 Garbage Classification Tool - Interactive Mode")
    print("=" * 50)
    
    # Initialize classifier
    model_path = input("Enter model path (press Enter for 'best_model.keras'): ").strip()
    if not model_path:
        model_path = 'best_model.keras'
    
    classifier = GarbageClassifier(model_path)
    
    while True:
        print("\n📋 Choose an option:")
        print("1. 🎥 Webcam prediction")
        print("2. 📁 Single file prediction") 
        print("3. 📁 Batch directory prediction")
        print("4. ❌ Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            classifier.predict_from_webcam()
        elif choice == '2':
            file_path = input("Enter image file path: ").strip()
            if file_path:
                single_file_predict(classifier, file_path)
        elif choice == '3':
            dir_path = input("Enter directory path: ").strip()
            if dir_path:
                batch_predict(classifier, dir_path)
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        main()
    else:
        # Run in interactive mode
        interactive_mode()
