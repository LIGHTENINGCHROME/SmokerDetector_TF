# 🚭SmokerDetector_TF
This project is a deep learning-based solution to detect smoking activity in real-time video feeds. The goal is to assist in the enforcement of public smoking bans by identifying individuals smoking in public areas using a trained image classification model.

# 🧠 Overview
A custom convolutional neural network (CNN) was implemented using TensorFlow and Keras, featuring multiple dropout layers and L2 regularization to improve generalization and prevent overfitting. The system processes input images or frames to classify whether the person is smoking or not.

# 📂 Datasets
Kaggle Smoking Classification Dataset

DataMendeley Smoking Images Dataset

Both datasets were preprocessed and filtered manually to enhance training data quality.

# ⚙️ Model Architecture
Built with Conv2D layers for spatial feature extraction.

Dropout layers added at different depths to reduce overfitting.

L2 regularization applied to kernels for penalizing complexity.

Adam optimizer and binary_crossentropy loss used for training.

# 🧪 Training Strategy
Data Augmentation for robustness.

Early stopping and ReduceLROnPlateau for efficient training.

Weighted class training to manage class imbalance.

# 🧠 Prediction Workflow
The model takes input images or video frames and predicts the smoking activity. For deployment:

OpenCV is used for real-time video feed processing.

Optimized for systems with GPU for faster inference.

# 🚀 Deployment
This system is intended to be deployed in public surveillance setups (e.g., metro stations, malls, public transport) to identify smokers. Cameras are already installed in many areas, reducing implementation cost — only backend integration is required.

🔔 Authorities can be alerted automatically based on model predictions.
# 📈 Future Scope
Integrate with cloud-based APIs for large-scale city surveillance.

Improve accuracy with more diverse datasets.

Add feature to track individuals violating smoking rules multiple times.

Develop a lightweight model for deployment on edge devices (like Raspberry Pi).

Provide heatmaps to show high-smoking zones for strategic law enforcement.

# 🧾 References
TensorFlow

Keras

OpenCV

Kaggle Dataset - Smoking Detection

DataMendeley Dataset - Smoking Images

All libraries used are listed in requirements.txt.

# 💻 Setup Instructions
1. Clone the Repository
bash
Copy code
git clone https://github.com/yourusername/smoking-detection-system.git
cd smoking-detection-system
2. Install Dependencies
bash
Copy code
pip install -r requirements.txt
3. Run Training (if needed)
bash
Copy code
python train.py
4. Run Real-Time Detection
bash
Copy code
python detect_smoking.py
🔗 Project Link
🔗 GitHub Project Repository

Let me know if you want a version with badges, license, or contribution guidelines!
