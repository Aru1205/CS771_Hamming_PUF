import os
import time
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from submit import my_map, my_params

# Directory containing Proj.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Full paths to datasets
TRN_PATH = os.path.join(BASE_DIR, "public_trn.txt")
TST_PATH = os.path.join(BASE_DIR, "public_tst.txt")

print("Training file:", TRN_PATH)
print("Testing file :", TST_PATH)

# Load train data
trn = np.loadtxt(TRN_PATH)

X_trn = trn[:, :-1]
Y_trn = trn[:, -1]

# Load test data
tst = np.loadtxt(TST_PATH)

X_tst = tst[:, :-1]
Y_tst = tst[:, -1]

print("\nRaw Data Shapes")
print("Train shape:", trn.shape)
print("Test shape :", tst.shape)

# Feature map
X_map_trn = my_map(X_trn)
X_map_tst = my_map(X_tst)

print("\nMapped Data Shapes")
print("Mapped train shape:", X_map_trn.shape)
print("Mapped test shape :", X_map_tst.shape)

# Train model
clf = LinearSVC()
clf.set_params(**my_params())

print("\nTraining...")

start_time = time.time()

clf.fit(X_map_trn, Y_trn)

end_time = time.time()

print(f"Training time: {end_time - start_time:.4f} seconds")

# Predictions
train_pred = clf.predict(X_map_trn)
test_pred = clf.predict(X_map_tst)

# Accuracy
train_acc = accuracy_score(Y_trn, train_pred)
test_acc = accuracy_score(Y_tst, test_pred)

print("\n================ RESULTS ================\n")

print("Mapped dimension:", X_map_trn.shape[1])

print(f"Train Accuracy : {train_acc:.6f}")
print(f"Test Accuracy  : {test_acc:.6f}")

# Precision / Recall / F1
precision = precision_score(Y_tst, test_pred)
recall = recall_score(Y_tst, test_pred)
f1 = f1_score(Y_tst, test_pred)

print(f"\nPrecision      : {precision:.6f}")
print(f"Recall         : {recall:.6f}")
print(f"F1 Score       : {f1:.6f}")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(Y_tst, test_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(Y_tst, test_pred))

# Model Information
print("Weight Matrix Shape :", clf.coef_.shape)
print("Weight L2 Norm      :", np.linalg.norm(clf.coef_))
print("Intercept           :", clf.intercept_)