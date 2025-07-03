# Breast Cancer Classification using Support Vector Machines (SVM)

This project showcases the use of **Support Vector Machine (SVM)** classifiers to predict breast cancer diagnosis (Benign or Malignant) based on medical features from the Breast Cancer Wisconsin dataset.

---

## 📁 Dataset

* Source: `Breast_cancer_data.csv`
* Features: 30 real-valued features computed from digitized images of fine needle aspirate (FNA) of breast mass.
* Target: Diagnosis (`M` = Malignant, `B` = Benign)

---

## 🔍 Project Workflow

### ✅ Step 1: Data Preprocessing

* Dropped unused columns (e.g., ID)
* Encoded categorical target values
* Handled missing values using column-wise mean
* Standardized features using `StandardScaler`

### ✅ Step 2: Model Training

* **Linear SVM** classifier
* **RBF SVM** classifier
* **GridSearchCV** for hyperparameter tuning (C, gamma)

### ✅ Step 3: Evaluation

* Accuracy Score
* Classification Report
* Confusion Matrix (saved as PNG)

### ✅ Step 4: Visualization

* 📈 2D Decision Boundary using first two features
* 🌐 3D Scatter Plot using first 3 features

---

## 📊 Visual Outputs

| 📌 Visualization                  | Description                        |
| --------------------------------- | ---------------------------------- |
| `linear_svm_confusion_matrix.png` | Confusion matrix for linear kernel |
| `rbf_svm_confusion_matrix.png`    | Confusion matrix for RBF kernel    |
| `svm_2d_decision_boundary.png`    | Linear SVM decision boundary in 2D |
| `svm_3d_scatter_plot.png`         | Data distribution in 3D space      |

---

## 🧪 Cross-Validation

* Applied 5-fold cross-validation on the tuned RBF SVM
* Reported mean accuracy score

---

## 🛠️ Tech Stack

* Python, NumPy, Pandas, Matplotlib, Seaborn
* Scikit-learn (SVM, scaling, CV, metrics)

---

## 📌 How to Run

```bash
pip install pandas numpy seaborn sk-learn matplotlib
pip install ipykernel
python -m ipykernel install --user --
jupyter notebook
```

Run `Task_7_SVM_Classifier.ipynb` to reproduce all results.

---

## 💡 Inspiration

Early detection and classification of cancer can significantly improve treatment outcomes. This project applies classical machine learning (SVM) for life-critical predictions.

---

## 🔗 Connect

Made by [Manish Srivastav](https://www.linkedin.com/in/roxtop07/) 🚀
