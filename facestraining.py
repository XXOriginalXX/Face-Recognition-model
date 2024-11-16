import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset (assumed to be organized into folders by person's name)
def load_images_from_folder(folder):
    images = []
    labels = []
    for person in os.listdir(folder):
        person_path = os.path.join(folder, person)
        for filename in os.listdir(person_path):
            img = cv2.imread(os.path.join(person_path, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                images.append(img.flatten())  # Flatten the image into a 1D vector
                labels.append(person)
    return np.array(images), np.array(labels)

# Load dataset
images, labels = load_images_from_folder(r"C:\Users\adith\OneDrive\Desktop\Projects\ML\faces")

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Train classifier
clf = SVC(kernel='linear', probability=True)  # Using 'linear' kernel for simplicity
clf.fit(X_train, y_train)

# Test classifier
y_pred = clf.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')

# Save the trained model and label encoder
with open('face_recognition_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
