#1. Thư viện sử dụng
---
pickle, cv2, mediapipe, numpy, time, RandomForestClassifier, train_test_split, accuracy_score, os.

#2. Nguyên tắt hoạt động
---

Đầu tiên chạy file collect_img.py, file này cho phép người dùng có thể dùng trực tiếp camera để lất data trục tiếp từ bên ngoài.
  + Data được train sẵn: https://drive.google.com/drive/folders/1EPPtCBwjIyTQJXm-_KtyvtdzDsX67g8H?usp=drive_link

Tiếp theo cho chạy file create_dataset.py file này nhằm mục đích lọc ra và đưa các vectơ vào trong tay của các hình ảnh như này:

<img width="480" alt="create_img" src="https://github.com/user-attachments/assets/c21ea96f-e03d-479d-96d1-b69d5df91495">

Sau đó chạy file train_classifier.py file này sẽ đưa các hình ảnh được tạo lại ở file create_dataset.py vào file model.p:

<img width="160" alt="data_p" src="https://github.com/user-attachments/assets/9e3781b9-799c-456d-8fb5-0fd8b525cebc">

Cuối cùng chạy file inference_classifier.py để chạy camera và đọc các thao tác tay từ đó cứ mỗi 1000 mili giây sẽ tự động ghi lại cử chỉ tay người dùng:

<img width="480" alt="tscam" src="https://github.com/user-attachments/assets/5538937d-469e-4f5a-9441-f543e598e9ac">

#3. Đánh giá mô hình
---

<img width="1020" alt="Figure_1" src="https://github.com/user-attachments/assets/b9387471-d06c-4786-b95c-3ed7e3325a11">

#4. Bảng ký hiệu tay
---

<img width="480" alt="1" src="https://github.com/user-attachments/assets/7da4f464-a3c0-45ad-9411-15b1ae07428a">

