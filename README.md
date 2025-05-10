[![Python](https://img.shields.io/badge/Python-black?logo=python&logoColor=white)](https://python.org/)
[![Tensorflow](https://img.shields.io/badge/Tensorflow-black?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-black)](https://www.kaggle.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-black?logo=opencv&logoColor=white)](https://opencv.org/)


<div align="center">
<hr/>

# _::::[ SMOKER ]<BR>[ DETECTOR ]::::::::_

</div>

<hr/>

# **🚭SmokerDetector_TF**
**This project is a deep learning-based solution to detect smoking activity in real-time video feeds. The goal is to assist in the enforcement of public smoking bans by identifying individuals smoking in public areas using a trained image classification model.**

<hr/>

# **🧠 Overview**
A custom convolutional neural network (CNN) was implemented using TensorFlow and Keras, featuring multiple dropout layers and L2 regularization to improve generalization and prevent overfitting. The system processes input images or frames to classify whether the person is smoking or not.**

<hr/>

# **📂 Datasets**
Kaggle Smoking Classification Dataset**

**DataMendeley Smoking Images Dataset**

**Both datasets were preprocessed and filtered manually to enhance training data quality.**

<hr/>

# **⚙️ Model Architecture**
**Built with Conv2D layers for spatial feature extraction.**

**Dropout layers added at different depths to reduce overfitting.**

**L2 regularization applied to kernels for penalizing complexity.**

**Adam optimizer and binary_crossentropy loss used for training.**

<hr/>

# **🧪 Training Strategy**
**Data Augmentation for robustness.**

**Early stopping and ReduceLROnPlateau for efficient training.**

**Weighted class training to manage class imbalance.**

<hr/>

# **🧠 Prediction Workflow**
**The model takes input images or video frames and predicts the smoking activity. For deployment:**

**OpenCV is used for real-time video feed processing.**

**Optimized for systems with GPU for faster inference.**

<hr/>

# **🚀 Deployment**
**This system is intended to be deployed in public surveillance setups (e.g., metro stations, malls, public transport) to identify smokers. Cameras are already installed in many areas, reducing implementation cost — only backend integration is required.**

**🔔 Authorities can be alerted automatically based on model predictions.**

<hr/>

# **📈 Future Scope**
**Integrate with cloud-based APIs for large-scale city surveillance.**

**Improve accuracy with more diverse datasets.**

**Add feature to track individuals violating smoking rules multiple times.**

**Develop a lightweight model for deployment on edge devices (like Raspberry Pi).**

**Provide heatmaps to show high-smoking zones for strategic law enforcement.**

<hr/>

# **🧾 References**
**TensorFlow (https://www.tensorflow.org/)**

**Keras (https://keras.io/)**

**OpenCV (https://opencv.org/)**

**Kaggle Dataset - Smoking Detection (https://www.kaggle.com/datasets/sujaykapadnis/smoking)**

**DataMendeley Dataset - Smoking Images (https://data.mendeley.com/datasets/j45dj8bgfc/1)**

**All libraries used are listed in requirements.txt.**

<hr/>

# **💻 Setup Instructions**
**1. Clone the Repository**
```git clone https://github.com/yourusername/smoking-detection-system.git```

       cd smoking-detection-system
   
**3. Install Dependencies**
   
       pip install -r requirements.txt

<hr/>

# **🔗 Link To Kaggle Notebook and Custom Dataset**

**Kaggle Notebook - Smoker-Detector (https://www.kaggle.com/code/adityadiwan2005/smoker-detector)**

**Custom Dataset - Smoking Images (https://www.kaggle.com/datasets/adityadiwan2005/smoking2/data)**


<hr/>
