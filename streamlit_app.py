import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load the trained modelopen c
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_recognition_model.h5")

model = load_model()

st.title("📷 Handwritten Digit Recognition via Webcam")
st.write("Click the button to activate the webcam and detect handwritten digits.")

# Button to Start/Stop Webcam
start_webcam = st.button("📷 Start Webcam")
stop_webcam = st.button("🛑 Stop Webcam")

if start_webcam:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()  # Placeholder for video feed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("⚠️ Failed to capture frame.")
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

        # Find contours (digit detection)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 50 < w < 300 and 50 < h < 300:  # Ignore small noises
                digit = thresh[y:y + h, x:x + w]  # Extract digit ROI
                digit = cv2.resize(digit, (28, 28))  # Resize to 28x28
                digit = digit / 255.0  # Normalize
                digit = digit.reshape(1, 28, 28, 1)  # Reshape for CNN model

                # Predict the digit
                prediction = model.predict(digit)
                predicted_digit = np.argmax(prediction)

                # Draw bounding box & label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Digit: {predicted_digit}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show updated frame in Streamlit
        stframe.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()

if stop_webcam:
    st.write("🛑 Webcam Stopped.")

# ----------------------------- #
# Handwritten Digit Drawing Canvas
# ----------------------------- #
# Streamlit UI
st.title("🖌 Handwritten Digit Recognition")
st.write("Draw a digit (0-9) in the box below and the model will predict it!")

# Create a Canvas for Drawing
canvas_result = st_canvas(
    fill_color="black",  # Set the background to black
    stroke_width=10,
    stroke_color="white",  # Draw white digits
    background_color="black",  # Match MNIST dataset style
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Ensure the canvas is not empty before processing
if canvas_result is not None and canvas_result.image_data is not None:
    img = canvas_result.image_data[:, :, :3]  # Remove alpha channel
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    img = cv2.resize(img, (28, 28))  # Resize to match MNIST input

    # Normalize pixel values (MNIST expects white digits on black background)
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)  # Reshape for CNN model

    # Predict the digit
    prediction = model.predict(img)
    digit = np.argmax(prediction)

    # Display the result
    st.write(f"🎯 **Predicted Digit:** {digit}")


# ----------------------------- #
# Multiple Digit Detection from Image
# ----------------------------- #
st.title("📄 Upload an Image with Multiple Digits")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

def preprocess_digit(digit_image):
    digit_image = cv2.resize(digit_image, (28, 28))  # Resize to 28x28
    digit_image = digit_image / 255.0  # Normalize
    digit_image = digit_image.reshape(1, 28, 28, 1)  # Reshape for model
    return digit_image

def detect_digits(image):
    # Convert PIL Image to OpenCV format
    image = np.array(image.convert('L'))  # Convert to grayscale
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)  # Apply threshold

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    predictions = []
    bounding_boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Ignore small noises
        if w > 10 and h > 10:
            digit = thresh[y:y+h, x:x+w]  # Extract digit
            digit = preprocess_digit(digit)  # Preprocess digit
            prediction = np.argmax(model.predict(digit))  # Predict digit
            predictions.append((prediction, (x, y, w, h)))
            bounding_boxes.append((x, y, w, h))

    # Sort by x-coordinate (left to right)
    predictions.sort(key=lambda x: x[1][0])
    return predictions, bounding_boxes

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Detect digits
    predictions, bounding_boxes = detect_digits(image)

    # Draw bounding boxes and display results
    image = np.array(image.convert('RGB'))  # Convert to RGB
    for (digit, (x, y, w, h)) in predictions:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, str(digit), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    st.image(image, caption="Detected Digits", use_column_width=True)

    st.write("### Predicted Digits: ", "".join(str(d[0]) for d in predictions))
