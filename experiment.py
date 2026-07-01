import os
import time
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from submit import my_map

# ==================================================
# Load Dataset
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRN_PATH = os.path.join(BASE_DIR, "public_trn.txt")
TST_PATH = os.path.join(BASE_DIR, "public_tst.txt")

trn = np.loadtxt(TRN_PATH)
tst = np.loadtxt(TST_PATH)

X_trn = trn[:, :-1]
y_trn = trn[:, -1]

X_tst = tst[:, :-1]
y_tst = tst[:, -1]

print("Mapping features...")

X_map_trn = my_map(X_trn)
X_map_tst = my_map(X_tst)

print("Mapped dimension:", X_map_trn.shape[1])

# ==================================================
# Experiment 1: LinearSVC Loss Function
# ==================================================

print("\n" + "=" * 70)
print("EXPERIMENT 1: LinearSVC Loss Function")
print("=" * 70)

print(f"{'Loss':15} {'Train Time(s)':15} {'Train Acc':15} {'Test Acc':15}")

for loss in ["hinge", "squared_hinge"]:

    clf = LinearSVC(
        penalty="l2",
        loss=loss,
        C=1.0,
        max_iter=50000,
        tol=1e-5,
        random_state=42
    )

    start = time.perf_counter()
    clf.fit(X_map_trn, y_trn)
    train_time = time.perf_counter() - start

    train_acc = clf.score(X_map_trn, y_trn)
    test_acc = clf.score(X_map_tst, y_tst)

    print(
        f"{loss:15}"
        f"{train_time:<15.4f}"
        f"{train_acc:<15.6f}"
        f"{test_acc:<15.6f}"
    )

# ==================================================
# Experiment 2: LinearSVC Effect of C
# ==================================================

print("\n" + "=" * 70)
print("EXPERIMENT 2: LinearSVC Effect of C")
print("=" * 70)

print(f"{'C':10} {'Train Time(s)':15} {'Train Acc':15} {'Test Acc':15}")

for C in [0.01, 0.1, 1, 10, 100]:

    clf = LinearSVC(
        penalty="l2",
        loss="squared_hinge",
        C=C,
        dual=True,
        max_iter=50000,
        tol=1e-5,
        random_state=42
    )

    start = time.perf_counter()
    clf.fit(X_map_trn, y_trn)
    train_time = time.perf_counter() - start

    train_acc = clf.score(X_map_trn, y_trn)
    test_acc = clf.score(X_map_tst, y_tst)

    print(
        f"{C:<10}"
        f"{train_time:<15.4f}"
        f"{train_acc:<15.6f}"
        f"{test_acc:<15.6f}"
    )

# ==================================================
# Experiment 3: Logistic Regression Effect of C
# ==================================================

print("\n" + "=" * 70)
print("EXPERIMENT 3: Logistic Regression Effect of C")
print("=" * 70)

print(f"{'C':10} {'Train Time(s)':15} {'Train Acc':15} {'Test Acc':15}")

for C in [0.01, 0.1, 1, 10, 100]:

    clf = LogisticRegression(
        penalty="l2",
        C=C,
        solver="lbfgs",
        max_iter=5000,
        tol=1e-5,
        random_state=42
    )

    start = time.perf_counter()
    clf.fit(X_map_trn, y_trn)
    train_time = time.perf_counter() - start

    train_acc = clf.score(X_map_trn, y_trn)
    test_acc = clf.score(X_map_tst, y_tst)

    print(
        f"{C:<10}"
        f"{train_time:<15.4f}"
        f"{train_acc:<15.6f}"
        f"{test_acc:<15.6f}"
    )