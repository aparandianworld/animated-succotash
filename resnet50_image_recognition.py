import numpy as np

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

import logging
import os

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
        logging.info(f"Predictions: {decoded_predictions}")
        for i, (class_id, class_name, probability) in enumerate(decoded_predictions):
            logging.info(f"{i + 1}: {class_name} ({probability:.2f})")
    except Exception as e:
        logging.error("Error occured during image prediction: ", str(e))

def main():
    try:
        model = ResNet50(weights="imagenet")
        logging.info("ResNet50 model loaded successfully.")
    except Exception as e:
        logging.error("Error occured while loading the model: ", str(e))
        return

    while True:
        image_path = input("Please enter path to image file or `quit` to exit the program: ")
        if image_path.lower() == "quit":
            logging.info("Exiting the program...")
            break
        if not os.path.isfile(image_path):
            logging.error("Image file provided not found or not accessible: ", str(e))
            continue
        if not image_path.lower().endswith((".jpg", ".jpeg", ".png")):
            logging.error("Image file provided is not a JPEG, JPG, or PNG file.")
            continue

        image_prediction(model, image_path)

if __name__ == "__main__":
    main()