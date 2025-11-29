# Machine_Learning
# Diabetes Classification using Random Forest and PCA

This repository contains my implementation of **Assignment 3** for Machine Learning, in which I build a classifier to predict whether a patient is diabetic or not using the **Pima Indians Diabetes Database**, and then apply **Principal Component Analysis (PCA)** to visualize the data in 2D.

## Dataset

The dataset used is the **Pima Indians Diabetes Database** from Kaggle:

- Kaggle link: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database  
- Each row represents one patient.
- Input features (columns) include:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
- Target column:
  - `Outcome` = 0 (not diabetic) or 1 (diabetic)

> Note: The dataset itself is not mine; it belongs to the original UCI/Kaggle source. Please download it directly from Kaggle if needed.

## Files in this Repository

- `MaheenAbdullah.ipynb`  
  Jupyter/Colab notebook containing:
  - Data loading and basic exploration
  - Train–test split and feature scaling
  - Random Forest classifier training and evaluation
  - Feature importance analysis
  - PCA (2 components) and 2D visualization of the dataset

- `README.md`  
  This file, explaining the project.

## Methods Used

### 1. Data Preprocessing

- Loaded `diabetes.csv` into a Pandas DataFrame.
- Separated features (`X`) and target (`y`).
- Split the data into training and test sets using `train_test_split` (80% train, 20% test, with stratification on the label).
- Applied `StandardScaler` to normalize the features so they have zero mean and unit variance.

### 2. Classifier: Random Forest

I used **RandomForestClassifier** from scikit-learn to build a binary classifier:

- The model consists of multiple decision trees (an ensemble).
- Each tree is trained on a bootstrap sample of the data and considers a random subset of features at each split.
- Final prediction is made by majority vote of all trees.
- Random Forest also provides **feature importances**, which I used to identify which medical features (e.g., Glucose, BMI, etc.) are most influential in predicting diabetes.

### 3. Model Evaluation

To evaluate the classifier, I used:

- **Accuracy score** on the test set
- **Confusion matrix** to see true positives, true negatives, false positives, and false negatives
- **Classification report** (precision, recall, F1-score for each class)

These metrics provide a detailed view of how well the model distinguishes between diabetic and non-diabetic patients.

### 4. PCA and Visualization

To visualize the structure of the data:

- I applied **Principal Component Analysis (PCA)** with `n_components=2` on the scaled features.
- PCA finds new axes (principal components) that capture the maximum variance in the data.
- I plotted the first two principal components in a 2D scatter plot, coloring points by the `Outcome` label (0 or 1).

This gives an intuitive view of how the patients are distributed in a reduced 2D space and whether diabetic and non-diabetic patients show any visible clustering.

## How to Run

### Option 1: Google Colab

1. Upload the notebook (`MaheenAbdullah.ipynb`) to Google Colab.
2. Download `diabetes.csv` from Kaggle and upload it to the Colab environment.
3. Run the cells sequentially from top to bottom.

### Option 2: Local Jupyter Notebook

1. Clone this repository:

   ```bash
   git clone https://github.com/<MaheenAbdullah2004>/<Machine_Learning>.git
   cd <your-repo-name>
