# Heart Disease Prediction Using Machine Learning

This project utilizes machine learning algorithms to predict the presence of heart disease based on medical parameters. It includes preprocessing, model training, evaluation, and comparison across multiple classifiers like **K-Nearest Neighbors**, **Decision Tree**, and **Random Forest**.

---

## Introduction
Heart disease is one of the leading causes of death worldwide. Early prediction can significantly improve patient outcomes and reduce medical costs. This project utilizes machine learning techniques to predict heart disease based on features such as age, cholesterol level, blood pressure, and more.

The primary goal is to build, evaluate, and compare models for accurate predictions using:
- K-Nearest Neighbors (KNN)
- Decision Trees
- Random Forests

## Dataset
The project uses the **UCI Heart Disease Dataset**, which contains 303 instances with 14 attributes such as:
- Age
- Sex
- Chest pain type
- Resting blood pressure
- Serum cholesterol
- Maximum heart rate achieved
- Presence of heart disease (target variable)

You can find more details about the dataset [here](https://www.kaggle.com/datasets/ketangangal/heart-disease-dataset-uci).

## Models Used
1. **KNeighborsClassifier**:
   - A simple and intuitive classification algorithm based on proximity.
   - Sensitive to the choice of `k` and feature scaling.

2. **DecisionTreeClassifier**:
   - A tree-based model that splits data recursively based on feature thresholds.
   - Prone to overfitting but interpretable.

3. **RandomForestClassifier**:
   - An ensemble method combining multiple decision trees.
   - Provides better generalization and reduces overfitting.

## Project Workflow
1. **Data Collection**:
   -Dataset containing heart disease-related patient data is loaded into the notebook.
   -The dataset consists of 1025 rows and 14 attributes (both numerical and categorical).

3. **Data Preprocessing**:
   -Check for missing values
   -Label Encoding: Categorical features like sex, chest_pain_type, thal, and slope are encoded     to numeric values.
   -Feature Scaling (if applicable): Not strictly necessary for Decision Trees or Random
    Forest, but important for KNN.
   -Splitting dataset into training and test sets using train_test_split.

5. **Exploratory Data Analysis (EDA)**:
   -Distribution plots for features like age, cholesterol, max heart rate, etc.
   -Correlation heatmap to identify important features.
   -Class distribution of target (presence/absence of heart disease).

7. **Model Training**:
   -Train and evaluate the following models using the training set:
   *K-Nearest Neighbors (KNN)
   *Decision Tree Classifier
   *Random Forest Classifier

9. **Model Evaluation**:
   - Evaluate each model using:
     *Accuracy score
     *Confusion Matrix
     *Classification Report (Precision, Recall, F1-score)

## Dependencies
Install the required libraries using the following command:
```bash
pip install -r requirements.txt
```
### Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/heart-disease-prediction-ml.git
   cd heart-disease-prediction-ml

   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook or Python script:
   ```bash
   jupyter notebook heart_disease_prediction.ipynb
   ```
   or
   ```bash
   jupyter notebook heart-disease-prediction-using-machine-learning.ipynb

   ```

4. View the evaluation metrics and model comparison in the output.

## Results
The models were evaluated using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **ROC-AUC**

### Performance Summary:
| Model                | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------------------|----------|-----------|--------|----------|---------|
| KNeighborsClassifier | 86.34%   | 86%       | 88%    | 87%      | 86%     |
| DecisionTree         | 98.54%   | 100%      | 97%    | 99%      | 98.5%   |
| RandomForest         | 100%     | 100%      | 100%   | 100%     | 100%    |

(RandomForest typically performs best due to its ensemble nature.)

