"""
Simple demo script for garbage classification prediction
"""

from predict_garbage import GarbageClassifier
import sys
import os

def main():
    print("🗑️ Garbage Classification Demo")
    print("=" * 40)
    
    # Check if model exists
    model_path = 'best_model.keras'
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please make sure you have trained the model first!")
        return
    
    # Initialize classifier
    try:
        classifier = GarbageClassifier(model_path)
        print("✅ Classifier initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing classifier: {e}")
        return
    
    # Check if test image exists
    test_image = 'metal118.jpg'
    if os.path.exists(test_image):
        print(f"\n🔍 Testing with {test_image}:")
        result = classifier.predict_from_file(test_image)
        
        if result:
            print(f"Predicted: {result['predicted_class']} ({result['confidence']:.2%})")
            print("Top 3 predictions:")
            for i, pred in enumerate(result['top_predictions']):
                print(f"  {i+1}. {pred['label']}: {pred['confidence']:.2%}")
    
    print(f"\n🎯 Available prediction modes:")
    print("1. Interactive mode: python predict_garbage.py")
    print("2. Webcam mode: python predict_garbage.py --mode webcam")
    print("3. File mode: python predict_garbage.py --mode file --input image.jpg")
    print("4. Batch mode: python predict_garbage.py --mode file --batch /path/to/images/")

if __name__ == "__main__":
    main()
