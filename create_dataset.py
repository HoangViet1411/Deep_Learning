import os
import pickle
import mediapipe as mp
import cv2

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    DATA_DIR = './data'

    data = []
    labels = []
    for dir_ in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_)

        if not os.path.isdir(dir_path):
            continue

        for img_path in os.listdir(dir_path):
            try:
                img = cv2.imread(os.path.join(dir_path, img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                results = hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        x_ = [landmark.x for landmark in hand_landmarks.landmark]
                        y_ = [landmark.y for landmark in hand_landmarks.landmark]

                        data_aux = []
                        for i in range(len(hand_landmarks.landmark)):
                            data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                            data_aux.append(hand_landmarks.landmark[i].y - min(y_))

                        data.append(data_aux)
                        labels.append(dir_)
                else:
                    print(f"Không tìm thấy bàn tay trong {os.path.join(dir_path, img_path)}")
            except Exception as e:
                print(f"Lỗi khi xử lý {os.path.join(dir_path, img_path)}: {e}")

    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

if __name__ == "__main__":
    main()