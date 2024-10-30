import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
data_dir = r"C:\Users\AKSHAYA\Desktop\PROJ\static\dataset"

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training')

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation')


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(26, activation='softmax')  # 26 classes for A-Z
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10)


model.save('ISL_Det.h5')


import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load the trained model
model = tf.keras.models.load_model('ISL_Det.h5')

# Define a dictionary to map class indices to letters
class_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 
                11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 
                21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Initialize the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    # Convert the BGR image to RGB
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    image_for_drawing = image.copy()

    # Process the image and detect the hands
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks
            landmark_list = hand_landmarks.landmark

            # Determine bounding box coordinates
            x_min = min([landmark.x for landmark in landmark_list])
            x_max = max([landmark.x for landmark in landmark_list])
            y_min = min([landmark.y for landmark in landmark_list])
            y_max = max([landmark.y for landmark in landmark_list])
            x_min = int(x_min * image.shape[1])
            x_max = int(x_max * image.shape[1])
            y_min = int(y_min * image.shape[0])
            y_max = int(y_max * image.shape[0])

            # Extract hand region
            hand_region = image[y_min:y_max, x_min:x_max]

            # Resize hand region to 64x64 pixels
            hand_region = cv2.resize(hand_region, (64, 64))

            # Convert color to RGB as model expects
            hand_region = cv2.cvtColor(hand_region, cv2.COLOR_BGR2RGB)

            # Preprocess (normalize, etc.)
            hand_region = hand_region / 255.0  # Assuming your model expects pixel values between 0 and 1

            # Predict the gesture
            predictions = model.predict(np.expand_dims(hand_region, axis=0))
            predicted_class = np.argmax(predictions)
            predicted_letter = class_labels[predicted_class]

            # Display the predicted letter
            cv2.putText(image_for_drawing, predicted_letter, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(image_for_drawing, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the resulting frame
    cv2.imshow('MediaPipe Hands + Sign Language Recognition', cv2.cvtColor(image_for_drawing, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release the capture
hands.close()
cap.release()
cv2.destroyAllWindows()
