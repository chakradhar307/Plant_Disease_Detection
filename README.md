# 🌿 Plant Disease Detection
# **📌 Overview**

This project leverages deep learning techniques to identify and classify plant diseases from leaf images. Utilizing a Convolutional Neural Network (CNN), the model is trained to recognize various plant diseases, aiding in early detection and effective management.

# **🏗️ Model Architecture**

The model is a Convolutional Neural Network (CNN) designed to classify plant diseases from leaf images. It consists of multiple convolutional layers with ReLU activations and max-pooling, followed by dropout layers to prevent overfitting, and fully connected dense layers with a softmax output for multi-class classification. Optionally, pre-trained models like EfficientNetB0 can be used for better accuracy through transfer learning.

# **🧪 Model Creation**

The model is developed using TensorFlow/Keras, focusing on:

Data Preprocessing: Resizing images to 224x224 pixels and normalizing pixel values.

Training: Compiling the model with the Adam optimizer and categorical crossentropy loss function, and training it on the dataset.

Evaluation: Assessing model performance on a validation dataset.

# **⚙️ Requirements**

Ensure you have the following Python libraries installed:

pip install tensorflow matplotlib numpy scikit-learn streamlit

# **🧾 Usage**

Clone the Repository:

git clone https://github.com/chakradhar307/Plant_Disease_Detection.git
cd Plant_Disease_Detection


# **Prepare Your Dataset:**

Ensure your dataset is organized with subdirectories for each class (e.g., Healthy, Diseased).

# **Run the Notebook:**

Open and execute the ModelCreation.ipynb notebook in Jupyter Notebook or Google Colab.

# **📊 Results**

The model achieves an accuracy of approximately 95% on the validation dataset, demonstrating its effectiveness in plant disease classification.

# **📝 License**

This project is licensed under the MIT License.
