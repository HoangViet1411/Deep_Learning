import os
import cv2
import time

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

def collect_images(class_index):
    class_dir = os.path.join(DATA_DIR, str(class_index))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(class_index))

    counter = 0
    while True:  # Vòng lặp liên tục để chụp ảnh khi nhấn Q hoặc thoát khi nhấn X
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        cv2.putText(frame, 'Press "Q" to capture an image!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(1)  # Thời gian chờ 1ms để nhận phím nhấn
        
        if key == ord('q'):  # Nếu nhấn Q thì chụp ảnh
            if counter < dataset_size:
                cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
                print(f"Captured image {counter}")
                counter += 1
            else:
                print("Dataset size reached, stopping image collection.")
                break  # Dừng vòng lặp nếu đủ số ảnh đã chụp
        
        elif key == ord('x'):  # Nếu nhấn X thì thoát chương trình
            print("Exiting image collection...")
            break

    print("Image collection complete.")

def release_resources():
    cap.release()
    cv2.destroyAllWindows()

# Sử dụng hàm `collect_images` với class_index = 0
collect_images(0)
release_resources()
