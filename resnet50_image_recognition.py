import numpy as np

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

def main():
    while True:
        image_path = input("Please enter path to image file or `quit` to exit the program: ")
        if image_path.lower() == "quit":
            break
        try: 
            if image_path.endswith(".png") or image_path.endswith(".jpg") or image_path.endswith(".jpeg"):
                image_prediction(image_path)
        except FileNotFoundError as e:
            print("File not found: ", e)
        except Exception as e:
            print("Error occured: ", e)

if __name__ == "__main__":
    main()