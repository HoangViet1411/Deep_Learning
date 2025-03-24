import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

def get_labels_from_pickle():
    try:
        data_dict = pickle.load(open('./data.pickle', 'rb'))
        labels = list(set(data_dict['labels']))
        labels_dict = {i: label for i, label in enumerate(labels)}
        return labels_dict
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy tệp data.pickle. Đảm bảo bạn đã tạo tập dữ liệu.")
        return {}  # Trả về từ điển rỗng nếu không tìm thấy tệp

try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    print("Lỗi: Không tìm thấy tệp model.p. Đảm bảo bạn đã huấn luyện mô hình.")
    exit()

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = get_labels_from_pickle()
predicted_character = ""
word = ""
last_added_time = time.time()
previous_predictions = []  # Bộ lọc dự đoán

def process_frame():
    global predicted_character, word, last_added_time, previous_predictions
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    if not ret or frame is None:
        print("Không thể đọc từ camera. Kiểm tra kết nối camera.")
        return None, None

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

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

        prediction = model.predict([np.asarray(data_aux)])

        # Kiểm tra kiểu dữ liệu của prediction[0]
        if isinstance(prediction[0], str):
            predicted_character = prediction[0]
        else:
            predicted_character = labels_dict.get(int(prediction[0]), "unknown")

        previous_predictions.append(predicted_character)
        if len(previous_predictions) > 5:
            previous_predictions.pop(0)

        predicted_character = max(set(previous_predictions), key=previous_predictions.count)

        if (predicted_character != "none" and
            (not word or predicted_character != word[-1]) and
            (time.time() - last_added_time) >= 2):
            word += predicted_character
            last_added_time = time.time()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    else:
        predicted_character = "none"

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

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            word = ""

    release_resources()

if __name__ == "__main__":
    main()