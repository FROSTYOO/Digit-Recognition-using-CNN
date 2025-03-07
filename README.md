# 🖋️ Handwritten Digit Recognition

This project is a **Handwritten Digit Recognition** system that allows users to input digits via:

1. **Webcam** - Capture handwritten digits in real-time.
2. **Drawing Canvas** - Draw a digit using a mouse or touchscreen.
3. **Image Upload** - Upload an image containing multiple digits for recognition.

The system utilizes a **Convolutional Neural Network (CNN) model** trained on the **MNIST dataset** to accurately classify digits from 0 to 9.

## 🚀 Features

- **Live Webcam Detection** 📷
- **Canvas-Based Digit Drawing** ✍️
- **Multiple Digit Detection from Images** 🖼️
- **Interactive UI using Streamlit** 🖥️
- **Optimized CNN Model for High Accuracy** 🎯

---

## 📌 Tech Stack

- **Python** 🐍
- **TensorFlow/Keras** (Deep Learning Model)
- **OpenCV** (Image Processing)
- **NumPy** (Data Handling)
- **Pillow** (Image Manipulation)
- **Streamlit** (Web App Framework)
- **streamlit-drawable-canvas** (Canvas Input)

---

## 📂 Project Structure

```
digit_recognition_project/
│── app.py  # Main Streamlit app
│── digit_recognition_model.h5  # Trained CNN model
│── requirements.txt  # List of dependencies
│── README.md  # Project documentation
```

---

## 🎮 How to Run the Project Locally

### **1️⃣ Clone the Repository**

```bash
git clone https://github.com/YOUR_USERNAME/digit-recognition.git
cd digit-recognition
```

### **2️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Streamlit App**

```bash
streamlit run app.py
```

The app will launch in your browser at `http://localhost:8501/` 🚀

---

## 🌍 Deployment Options

### **1️⃣ Deploy on Streamlit Cloud** (Recommended)

1. Push your project to **GitHub**
2. Go to [Streamlit Cloud](https://share.streamlit.io/) and deploy

### **2️⃣ Deploy on Render**

1. Push your project to **GitHub**
2. Go to [Render](https://render.com/) and create a new Web Service
3. Set the **build command**:

```bash
pip install -r requirements.txt
streamlit run app.py
```

4. Click **Deploy**

---

## 📸 Screenshots

### **1️⃣ Webcam-Based Recognition**



### **2️⃣ Drawing Canvas Input**



### **3️⃣ Image Upload for Multi-Digit Detection**



---

## 📬 Contact & Contributions

🙋‍♂️ **Developed by**: [Sauman Sarkar](https://www.linkedin.com/in/sauman-sarkar-1a095418b)

🔗 **GitHub**: [github.com/FROSTYOO](https://github.com/FROSTYOO)

💡 **Want to improve this project?**

- Fork the repo, create a branch, and submit a PR! 🚀

---

## ⭐ Acknowledgments

- **MNIST Dataset** by Yann LeCun
- **TensorFlow & Keras** for deep learning
- **Streamlit** for an easy-to-build UI

