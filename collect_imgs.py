import os
import cv2
import time

DATA_DIR = './data'

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở camera")
    exit()

def collect_images(folder_name):  # Thêm tham số folder_name
    class_dir = os.path.join(DATA_DIR, folder_name)  # Sử dụng folder_name
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Thu thập dữ liệu cho thư mục {folder_name}')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lỗi khi lấy khung hình")
            break

        cv2.putText(frame, 'Sẵn sàng? Nhấn "Q" để bắt đầu!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < 100:  # Số lượng ảnh để chụp
        ret, frame = cap.read()
        if not ret:
            print("Lỗi khi lấy khung hình")
            break

        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1
        print(f"Thư mục {folder_name}: {counter}/100")

        if cv2.waitKey(25) == ord('q'):
            break

    print(f"Thu thập dữ liệu cho thư mục {folder_name} hoàn thành.")

def release_resources():
    cap.release()
    cv2.destroyAllWindows()