import tensorflow as tf
import cv2
import numpy as np

# Load the Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
if face_cascade.empty():
    print("Error loading Haar cascade.")
    exit()

try:
    model = tf.keras.models.load_model('57.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def detect_faces(gray_image):
    return face_cascade.detectMultiScale(gray_image, 1.3, 5)

def predict_emotion(face_image):
    img = extract_features(face_image)
    pred = model.predict(img)
    return np.argmax(pred[0]), np.max(pred[0])

webcam = cv2.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    ret, im = webcam.read()
    if not ret:
        break
    
    im_resized = cv2.resize(im, (640, 480))
    gray = cv2.cvtColor(im_resized, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detect_faces(gray)
    
    for (p, q, r, s) in faces:
        image = gray[q:q+s, p:p+r]
        cv2.rectangle(im_resized, (p, q), (p+r, q+s), (255, 0, 0), 2)

        # Resize image for prediction
        image = cv2.resize(image, (48, 48))
        emotion_index, confidence = predict_emotion(image)
        prediction_label = f"{labels[emotion_index]} : {round(confidence * 100, 2)}%"

        cv2.putText(im_resized, prediction_label, (p, max(q-10, 10)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)

        print(f"Predicted Label: {prediction_label}")

    start_time = cv2.getTickCount()
    elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    fps = 1 / elapsed_time
    # cv2.putText(im_resized, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Output", im_resized)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
