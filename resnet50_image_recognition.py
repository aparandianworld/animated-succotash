import numpy as np

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def image_prediction(model, image_path):
    try: 
        img = image.load_img(image_path, target_size=(224, 224))
        # convert image to numpy array, add dimension, and preprocess the image for ResNet50
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        # make prediction
        predictions = model.predict(img_array)
        # decode predictions into a list of tuples(class_id, class_name, probability)
        decoded_predictions = decode_predictions(predictions, top=5)[0]
        logger.info("Predictions: ", decoded_predictions)
        for i, (class_id, class_name, probability) in enumerate(decoded_predictions):
            logger.info(f"{i + 1}: {class_name} ({probability:.2f})")
    except Exception as e:
        logging.error("Error occured during image prediction: ", str(e))

def main():
    model = ResNet50(weights="imagenet")
    logging.info("ResNet50 model loaded successfully.")
    while True:
        image_path = input("Please enter path to image file or `quit` to exit the program: ")
        if image_path.lower() == "quit":
            print("Exiting the program...")
            break
        try: 
            if image_path.lower().endswith((".png", ".jpg", ".jpeg")):
                image_prediction(model, image_path)
            else:
                print("Invalid image file format. Please try again.")
        except FileNotFoundError as e:
            print(f"File not found: ", str(e))
        except Exception as e:
            print(f"Error occured: ", str(e))

if __name__ == "__main__":
    main()