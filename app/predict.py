import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('model/flower_recognition_model.h5')

# Function to predict the class of an image
def predict_flower(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    predicted_class = class_names[np.argmax(predictions)]

    return predicted_class
