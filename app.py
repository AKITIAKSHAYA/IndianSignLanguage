import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import base64

app = Flask(__name__)
socketio = SocketIO(app)

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

@app.route('/')
def index():
    return render_template('recognition.html')




previous_coords = None

def smooth_coordinates(new_coords, previous_coords, alpha=0.5):
    if previous_coords is None:
        return new_coords
    return tuple(alpha * n + (1 - alpha) * p for n, p in zip(new_coords, previous_coords))

@socketio.on('frame')
def handle_frame(frame_data):
    global previous_coords
    try:
        # Convert base64 image to numpy array
        nparr = np.frombuffer(base64.b64decode(frame_data['image'].split(',')[1]), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process the image and detect the hands
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the image
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

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

                # Smooth bounding box coordinates
                new_coords = (x_min, x_max, y_min, y_max)
                smoothed_coords = smooth_coordinates(new_coords, previous_coords)
                previous_coords = smoothed_coords

                x_min, x_max, y_min, y_max = map(int, smoothed_coords)

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

                # Debug: Print the predicted letter
                print(f"Predicted letter: {predicted_letter}")

                # Convert the processed image back to base64
                _, buffer = cv2.imencode('.jpg', image)
                encoded_image = base64.b64encode(buffer).decode('utf-8')
                image_data = f"data:image/jpeg;base64,{encoded_image}"

                # Send the processed frame along with the prediction to the client
                emit('processed_frame', {'image': image_data, 'prediction': predicted_letter})
        else:
            # Debug: No hands detected
            print("No hands detected.")
            emit('processed_frame', {'image': frame_data['image'], 'prediction': 'No hands detected'})
    except Exception as e:
        # Debug: Exception occurred
        print(f"Exception: {e}")
        emit('processed_frame', {'image': frame_data['image'], 'prediction': 'Error processing frame'})

if __name__ == '__main__':
    socketio.run(app,port=5002, debug=True)