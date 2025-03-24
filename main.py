import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hand and Drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels dictionary
labels_dict = {0: 'none', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 
               6: 'G', 7: 'A', 8: 'H', 9: 'I', 10: 'K', 11: 'L', 
               12: 'O', 13: 'F**k', 14: 'OK', 15: 'Hi', 16: '+1 respect',
               17: '+1 respect'}
predicted_character = ""
word = ""  # Variable to concatenate characters
last_added_time = time.time()  # Track the time of the last addition

def process_frame():
    global predicted_character, word, last_added_time
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    if not ret or frame is None:
        print("Cannot read from camera. Check camera connection.")
        return None, None

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Check for multiple hands and select the first one
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # Only take the first detected hand

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y

            x_.append(x)
            y_.append(y)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Predict character
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]
        
        # Append character to word if different from the last character
        if (predicted_character != "none" and 
            (not word or predicted_character != word[-1]) and 
            (time.time() - last_added_time) >= 2):  # Check if 2 seconds have passed
            word += predicted_character
            last_added_time = time.time()  # Update the time

        # Draw bounding box and predicted character
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
    
    else:
        predicted_character = "none"  # Set as "none" if no hand is detected

    return frame, word

def release_resources():
    cap.release()
    cv2.destroyAllWindows()

def main():
    while True:
        frame, word = process_frame()
        if frame is None:
            break

        cv2.imshow('frame', frame)

        # Press 'q' to exit, 'c' to clear the word
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            word = ""  # Clear the word

    release_resources()

if __name__ == "__main__":
    main()
