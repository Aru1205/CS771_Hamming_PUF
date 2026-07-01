import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC

from submit import my_map

# ==================================================
# Load Data
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
# Experiment: Train Size vs Test Accuracy
# ==================================================

train_sizes = [
    100, 200, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000
]

accuracies = []

rng = np.random.default_rng(42)

print("\nTrain Size vs Test Accuracy")
print("-" * 40)

for size in train_sizes:

    idx = rng.choice(
        len(X_map_trn),
        size=size,
        replace=False
    )

    X_sub = X_map_trn[idx]
    y_sub = y_trn[idx]

    clf = LinearSVC(
        penalty="l1",
        loss="squared_hinge",
        C=1.0,
        dual=False,
        max_iter=50000,
        tol=1e-5,
        random_state=42
    )

    clf.fit(X_sub, y_sub)

    acc = clf.score(X_map_tst, y_tst)

    accuracies.append(acc)

    print(
        f"Train Size = {size:5d} | "
        f"Test Accuracy = {acc:.6f}"
    )

# ==================================================
# Find Minimum Train Size Needed
# ==================================================

print("\nAccuracy Threshold Analysis")
print("-" * 40)

for target in [0.95, 0.97, 0.99]:

    found = False

    for size, acc in zip(train_sizes, accuracies):

        if acc >= target:

            print(
                f"Minimum train size for "
                f"{100*target:.0f}% accuracy = {size}"
            )

            found = True
            break

    if not found:

        print(
            f"{100*target:.0f}% accuracy not reached."
        )

# ==================================================
# Plot
# ==================================================

plt.figure(figsize=(8, 5))

plt.plot(
    train_sizes,
    accuracies,
    marker="o",
    linewidth=2
)

plt.xlabel("Training Set Size")
plt.ylabel("Test Accuracy")
plt.title("Effect of Training Set Size on Hamming PUF Attack")

plt.grid(True, alpha=0.3)

plt.tight_layout()

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "train_size_vs_accuracy.jpg"
)

plt.savefig(
    OUTPUT_FILE,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.1
)

print(f"\nPlot saved to:\n{OUTPUT_FILE}")

plt.show()