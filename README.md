# classify_digits.py

# 🔢 Handwritten Digit Recognition using SVM

This project demonstrates handwritten digit classification using the **Support Vector Machine (SVM)** algorithm on the **Digits dataset** from `scikit-learn`.

---

## 🧾 Overview

- **Dataset**: `sklearn.datasets.load_digits()`  
- **Goal**: Classify 8x8 images of digits (0–9)
- **Algorithm**: Support Vector Classification (SVC) with `gamma=0.001`
- **Preprocessing**: Data scaling using `StandardScaler`

---

## 📁 Features

- Grayscale images of digits (8x8 pixels)
- Target labels: Integers from 0 to 9
- Total samples: ~1,800

---

## 🧠 Workflow

1. **Load Data**  
   Uses `load_digits()` to import image data and labels.

2. **Preprocess Data**  
   - Flatten 2D images into 1D vectors  
   - Standardize features using `StandardScaler`

3. **Train/Test Split**  
   50% of data for training, 50% for testing.

4. **Model Training**  
   Trained with `svm.SVC(gamma=0.001)`.

5. **Evaluation**  
   - Outputs a classification report with precision, recall, f1-score  
   - Visualizes predictions on sample test images

---

## 📊 Output Example

```

Classification report:
precision    recall  f1-score   support

```
       0       1.00      1.00      1.00        89
       1       0.99      0.96      0.97        88
       2       0.97      1.00      0.98        91
       ...

accuracy                           0.98       899
```

macro avg       0.98      0.98      0.98       899
weighted avg       0.98      0.98      0.98       899

````

---

## 🖼️ Visualization

Displays a few test images with their predicted labels using `matplotlib`.

---

## 🚀 How to Run

1. Install the required libraries:

```bash
pip install numpy matplotlib scikit-learn
````

2. Run the script:

```bash
python digit_recognition_svm.py
```

---

## 📌 Notes

* The SVM classifier performs very well on this dataset with minimal tuning.
* Increasing `gamma` or using a different kernel (e.g., `'rbf'`, `'poly'`) may impact performance.
