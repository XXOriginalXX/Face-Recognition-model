import cv2
import numpy as np
import pickle

# Load the trained model and label encoder
with open('face_recognition_model.pkl', 'rb') as f:
    clf = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Function to detect faces, preprocess, and make a prediction
def predict_face(frame, clf, label_encoder):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract the face from the frame
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (128, 128)).flatten()  # Resize and flatten
        
        # Predict using the trained classifier
        prediction = clf.predict([face_img])
        person_name = label_encoder.inverse_transform(prediction)[0]
        
        # Display prediction on the video frame
        cv2.putText(frame, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame

# Start capturing video from webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break

    # Recognize face and update the frame
    frame_with_prediction = predict_face(frame, clf, label_encoder)

    # Display the resulting frame
    cv2.imshow('Live Face Recognition', frame_with_prediction)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture when done
cap.release()
cv2.destroyAllWindows()
