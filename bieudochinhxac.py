import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Hàm tính toán độ chính xác
def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred) * 100

# Load dữ liệu
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Tạo và huấn luyện model
model = RandomForestClassifier(
    n_estimators=5,       # Giảm số cây
    max_depth=3,          # Giới hạn độ sâu của mỗi cây
    min_samples_leaf=10,  # Tăng số lượng mẫu tối thiểu ở mỗi lá
    max_features=0.5      # Giới hạn số đặc trưng tại mỗi lần chia nhánh
)

model.fit(x_train, y_train)

# Dự đoán
y_predict = model.predict(x_test)

# Cross-validation và tính độ chính xác
scores = cross_val_score(model, data, labels, cv=5)  # 5-fold cross-validation
average_accuracy = np.mean(scores) * 100

# In độ chính xác từng fold và độ chính xác trung bình
print("Accuracy for each fold:", scores)
print("Average accuracy:", average_accuracy)

# Vẽ biểu đồ
plt.figure(figsize=(10, 5))
plt.bar(range(1, 6), scores * 100, label="Accuracy for each fold", color='skyblue')
plt.axhline(y=average_accuracy, color='orange', linestyle='--', label="Average Accuracy")
plt.xlabel("Fold")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy for each fold and Average Accuracy")
plt.legend()
plt.xticks(range(1, 6))
plt.show()

# Lưu model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
