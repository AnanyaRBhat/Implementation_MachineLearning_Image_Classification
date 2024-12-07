import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# The 2 Cifar-Databses - Labelled
CIFAR10_LABELS = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

CIFAR100_LABELS = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal",
    "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel",
    "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television",
    "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale",
    "willow_tree", "wolf", "woman", "worm"
]

# Combined Labels (10 CIFAR-10 classes + 100 CIFAR-100 classes)
COMBINED_LABELS = CIFAR10_LABELS + CIFAR100_LABELS


st.title("Image Classification")

# Option selection
option = st.selectbox(
    "Choose a model",
    ("Combined CIFAR-100 and CIFAR-10 Model", "VGG16 Pretrained Model", "ResNet50 Pretrained Model")
)

#Drop down options , 1-> Combined Cifar
if option == "Combined CIFAR-100 and CIFAR-10 Model":
    st.write("You selected the Combined CIFAR-100 and CIFAR-10 Model.")
    model_cifar10 = load_model('cifar10_model.h5')  # Load the model saved for CIFAR-10
    model_cifar100 = load_model('cifar100_model.h5')  # Load the model saved for CIFAR-100
    st.write("Combined CIFAR-100 and CIFAR-10 model loaded.")
    
    # Uploading image
    uploaded_file = st.file_uploader("Upload an image (32x32 resolution)", type=['jpg', 'png'])
    if uploaded_file is not None:
        # Preprocess and predict
        image = load_img(uploaded_file, target_size=(32, 32))
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Get predictions from both models
        prediction_cifar10 = model_cifar10.predict(image_array)
        prediction_cifar100 = model_cifar100.predict(image_array)

        # Extend CIFAR-10 predictions to match the 110-class space
        extended_cifar10 = np.zeros(110)
        extended_cifar10[:10] = prediction_cifar10[0]

        # Combine predictions (soft voting)
        combined_prediction = (extended_cifar10 + np.pad(prediction_cifar100[0], (10, 0))) / 2

        # Get the final class
        class_id = np.argmax(combined_prediction)
        confidence = combined_prediction[class_id] * 100
        label = COMBINED_LABELS[class_id]

        st.write(f"The image is recognized as **{label}** .")

# VGG16 Model Option
elif option == "VGG16 Pretrained Model":
    st.write("You selected the VGG16 Pretrained Model.")
    model = VGG16(weights='imagenet')  # Load VGG16 pretrained model
    st.write("VGG16 model loaded.")
    
    # Upload image
    uploaded_file = st.file_uploader("Upload an image ", type=['jpg', 'png'])
    if uploaded_file is not None:
        # Preprocess and predict
        image = load_img(uploaded_file, target_size=(224, 224))
        image_array = img_to_array(image)
        image_array = preprocess_input(image_array)
        image_array = tf.expand_dims(image_array, axis=0)
        prediction = model.predict(image_array)
        
        # Decode predictions
        decoded = decode_predictions(prediction, top=1)[0][0]  # Top-1 prediction
        label, confidence = decoded[1], decoded[2] * 100
        st.write(f"The image is recognized as **{label}**.")

# ResNet50 Model Option
elif option == "ResNet50 Pretrained Model":
    st.write("You selected the ResNet50 Pretrained Model.")
    model = ResNet50(weights='imagenet')  # Load ResNet50 pretrained model
    st.write("ResNet50 model loaded.")
    
    # Upload image
    #better if it has 224*22
    uploaded_file = st.file_uploader("Upload an image" , type=['jpg', 'png']) 
    if uploaded_file is not None:
        # Preprocess and predict
        image = load_img(uploaded_file, target_size=(224, 224))
        image_array = img_to_array(image)
        image_array = preprocess_input(image_array)
        image_array = tf.expand_dims(image_array, axis=0)
        prediction = model.predict(image_array)
        
        # Decode predictions
        decoded = decode_predictions(prediction, top=1)[0][0]  
        label, confidence = decoded[1], decoded[2] * 100
        st.write(f"The image is recognized as **{label}** .")
