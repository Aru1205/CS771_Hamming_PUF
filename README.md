# Breaking the Hamming PUF using Linear Machine Learning Models

A machine learning approach for breaking a **32-bit Hamming Physical Unclonable Function (PUF)** by transforming the challenge-response prediction problem into a **288-dimensional linear classification task**.

This project was completed as part of **CS771: Introduction to Machine Learning** at IIT Kanpur.

---

## Overview

Physical Unclonable Functions (PUFs) are hardware security primitives designed to generate unpredictable responses to input challenges.

This project demonstrates that the proposed **Hamming PUF** can be modeled exactly by an appropriate feature transformation, allowing simple linear classifiers to predict its responses with extremely high accuracy.

The work includes:

- Mathematical derivation of an exact linear feature mapping
- Implementation of the feature transformation
- LinearSVC and Logistic Regression experiments
- Hyperparameter analysis
- Dataset size vs. prediction accuracy study

---

## Repository Structure

```
.
├── submit.py                # Assignment implementation
├── report.pdf               # Project report
├── README.md
└── figures/
    └── train_size_vs_accuracy.png
```

---

## Feature Engineering

The original 32-bit challenge is transformed into a **288-dimensional feature vector** consisting of

- 16 even-bit features
- 16 odd-bit features
- 256 even-odd interaction terms

This transformation converts the Hamming PUF decision function into a linear decision boundary that can be learned using standard linear classifiers.

---

## Models Used

- LinearSVC
- Logistic Regression

The project evaluates:

- Regularization parameter (C)
- Loss functions
- Training time
- Test accuracy

---

## Results

### Best LinearSVC Performance

| Metric | Value |
|---------|------:|
| Test Accuracy | **99.8%** |
| Training Accuracy | 99.93% |
| Feature Dimension | 288 |

### Logistic Regression

- Achieved **99.72%** test accuracy
- Faster convergence than LinearSVC
- Served as a probabilistic baseline

---

## Dataset Size Study

The project also investigates how many Challenge-Response Pairs (CRPs) are required for successful learning.

Observed thresholds:

| Test Accuracy | Approximate Training Samples |
|---------------|----------------------------:|
| 95% | 2000 |
| 97% | 2500 |
| 99% | 4000 |

The results indicate that the exact feature mapping is highly sample-efficient.

---

## Technologies

- Python
- NumPy
- scikit-learn
- Matplotlib

---

## Report

The complete mathematical derivation, experiments, and discussion are available in:

**report.pdf**

---

## Key Takeaways

- Derived an exact linear representation of the Hamming PUF.
- Reduced the learning problem to linear classification in a 288-dimensional space.
- Achieved nearly perfect prediction accuracy using standard linear models.
- Demonstrated strong sample efficiency through training size analysis.

---

## Acknowledgements

Developed as part of **CS771: Introduction to Machine Learning**, IIT Kanpur.
