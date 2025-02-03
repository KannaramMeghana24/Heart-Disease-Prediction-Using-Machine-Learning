# Heart Disease Prediction Using Machine Learning

This repository contains a machine learning project for predicting the likelihood of heart disease based on clinical data. The project leverages multiple classification algorithms including **KNeighborsClassifier**, **DecisionTreeClassifier**, and **RandomForestClassifier** to compare their performance in predicting heart disease.

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
1. **Data Preprocessing**:
   - Handle missing values (if any).
   - Perform one-hot encoding for categorical features.
   - Scale numerical features for KNN.

2. **Exploratory Data Analysis (EDA)**:
   - Visualize feature distributions and correlations.
   - Identify key predictors of heart disease.

3. **Model Training and Evaluation**:
   - Split the data into training and testing sets.
   - Train KNN, Decision Tree, and Random Forest models.
   - Evaluate performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

4. **Hyperparameter Tuning**:
   - Optimize models using GridSearchCV or RandomizedSearchCV.

5. **Results Comparison**:
   - Compare models based on evaluation metrics.

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
   git clone https://github.com/yourusername/heart-disease-prediction.git
   cd heart-disease-prediction
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
   python heart_disease_prediction.py
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
| KNeighborsClassifier | xx%      | xx%       | xx%    | xx%      | xx%     |
| DecisionTree         | xx%      | xx%       | xx%    | xx%      | xx%     |
| RandomForest         | xx%      | xx%       | xx%    | xx%      | xx%     |

(RandomForest typically performs best due to its ensemble nature.)

