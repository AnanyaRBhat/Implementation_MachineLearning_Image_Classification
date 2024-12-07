This project demonstrates image classification using both trained and pre-trained models. It incorporates CIFAR-10 and CIFAR-100 datasets, as well as VGG16 and ResNet50 pre-trained models. A Streamlit interface is provided for easy interaction with the models.

STEPS TO RUN THE PROJECT

**Step 1:** 
Train the CIFAR-10 Model

Open and run Project1.ipynb.

After successful execution, the cifar10_model.h5 file will be saved in the same directory.


**Step 2:** 
Explore Pre-trained Models

Open and run Project2.ipynb.

This notebook uses VGG16 and ResNet50 pre-trained models for image classification.


**Step 3:** 
Train the CIFAR-100 Model

Open and run Project3.ipynb.

The cifar100_model.h5 file will be generated after successful execution.


**Step 4:** 

Run the Streamlit Application

Run app2.py using the command: 
streamlit run app2.py

------------------------------------------------------------------------------------------------------------------------------------------------------------

PROJECT STRUCTURE

**Project1.ipynb:**

Contains the code for training the CIFAR-10 model. Once you run this notebook, the cifar10_model.h5 file will be generated and saved in the same directory.

**Project2.ipynb:**

Demonstrates the use of pre-trained models (VGG16 and ResNet50) for image classification.

**Project3.ipynb:**

Contains the code for training the CIFAR-100 model. After running this notebook, the cifar100_model.h5 file will be generated and saved in the same directory.

**app2.py:**

This is the Streamlit interface for the project. It allows users to upload an image and choose a model for classification.

**combine_Cifar-10_Cifar-100.ipynb:**

This notebook contains reference code for combining CIFAR-10 and CIFAR-100 models. It is provided for exploration and is not essential for running the main project.

Options include:

1 -> Combine_Cifar-10_Cifar-100

2 -> VGG16 Pre-trained Model

3 -> ResNet50 Pre-trained Model



Sample images for testing the models.
The interface will open in your browser, allowing you to test the models by uploading images and selecting the desired model.

**Datasets Used**

CIFAR-10

CIFAR-100

ImageNet (for pre-trained VGG16 and ResNet50 models)

**Pre-trained Models**

VGG16 (trained on ImageNet)

ResNet50 (trained on ImageNet)
