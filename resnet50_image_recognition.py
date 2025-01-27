import numpy as np

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

def image_prediction(image_path):
    try: 
        model = ResNet50(weights="imagenet")
        img = image.load_img(image_path, target_size=(224, 224))
        # convet image to numpy array, add dimension, and preprocess the image for ResNet50
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        # make prediction
        predictions = model.predict(img_array)
        # decode predictions into a list of tuples(class_id, class_name, probability)
        decoded_predictions = decode_predictions(predictions, top=5)[0]
        print("Predictions: ", decoded_predictions)
        for i, (class_id, class_name, probability) in enumerate(decoded_predictions):
            print(f"{i + 1}: {class_name} ({probability:.2f})")
    except Exception as e:
        raise ('Error occured: ', e)

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