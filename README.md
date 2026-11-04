# ST10408316-RPDA8412-Data-Analytics

# Predictive Performance and Fairness Analysis
### Investigating Algorithmic Bias in Income Prediction Using the UCI Adult Dataset

## Project Overview

This project investigates the predictive performance and fairness of a machine learning model trained to predict whether an individual's annual income exceeds $50,000.

Using the UCI Adult Income Dataset, the study examines whether a logistic regression model can achieve strong predictive accuracy while producing unequal outcomes across demographic groups.

The project explores the relationship between machine learning accuracy, algorithmic bias and fairness, particularly across sex and race.

## Research Objectives

The main objectives of this project are to:

1. Prepare and explore the Adult Income Dataset for binary classification.
2. Develop and optimise a logistic regression model.
3. Evaluate predictive performance using standard classification metrics.
4. Identify influential predictors and potential proxy variables.
5. Investigate algorithmic fairness across sex and race.
6. Assess the reliability of fairness findings through supplementary analyses.

## Dataset

**Dataset:** UCI Adult Income Dataset

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult)

The dataset contains census-derived information, including:

- Age
- Education
- Occupation
- Marital status
- Working hours
- Sex
- Race
- Capital gains and losses

The target variable indicates whether an individual's annual income is:

- `>50K` — Income above $50,000
- `<=50K` — Income at or below $50,000

Sensitive attributes such as sex and race are excluded from model predictors but retained for fairness evaluation.

## Methodology

### 1. Data Preprocessing

The dataset undergoes several preprocessing steps:

- Identification and removal of missing values
- Removal of duplicate records
- Encoding of categorical variables
- Standardisation of numerical features
- Separation of predictor and target variables
- Stratified training and testing split (80/20)

### 2. Model Development

A **Logistic Regression** classifier is used to predict income categories.

The modelling process includes:

- Preprocessing pipelines
- Five-fold cross-validation
- Hyperparameter tuning
- Regularisation parameter optimisation
- Model training and testing

### 3. Performance Evaluation

Model performance is assessed using:

| Metric | Purpose |
|---|---|
| Accuracy | Measures overall prediction correctness |
| Precision | Measures the reliability of positive predictions |
| Recall | Measures the proportion of actual positives identified |
| F1-Score | Balances precision and recall |
| ROC-AUC | Evaluates the model's ability to distinguish classes |
| Confusion Matrix | Displays correct and incorrect classifications |

### 4. Feature Importance

The analysis investigates which variables contribute most strongly to income predictions.

Techniques include:

- Logistic regression coefficient analysis
- Permutation feature importance
- Examination of potential proxy variables

### 5. Algorithmic Fairness Analysis

Fairness is evaluated across demographic groups using:

**Selection Rate**

The proportion of individuals within each group predicted to earn more than $50,000.

**Disparate Impact Ratio**

Compares positive prediction rates between demographic groups.

A ratio below 0.80 is used as a potential indicator of adverse impact, although this threshold alone does not establish discrimination.

**True Positive Rate (TPR)**

Measures how frequently the model correctly identifies positive cases within each group.

**False Positive Rate (FPR)**

Measures how frequently the model incorrectly predicts positive outcomes within each group.

### 6. Robustness Analysis

Additional experiments assess the stability and reliability of the findings through:

- Bootstrap confidence intervals
- Analysis of excluded records
- Proxy-variable sensitivity testing
- Repeated train/test splits

## Technologies Used

- Python
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- scikit-learn
- IPython

## Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

Replace the URL with your repository's actual GitHub link.

### Step 2: Install Dependencies

```bash
pip install notebook pandas numpy matplotlib scikit-learn ipython
```

### Step 3: Download the Dataset

Download the `adult.data` file from:

https://archive.ics.uci.edu/dataset/2/adult

Place the dataset in the same directory as the Jupyter Notebook.

### Step 4: Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

`Adult_Income_Bias_Analysis_IIE_Harvard_Referenced.ipynb`

Run all cells sequentially.

## Expected Outputs

The notebook generates several outputs, including:

- Classification performance metrics
- ROC curve
- Confusion matrix
- Feature importance visualisations
- Fairness comparison tables
- Disparate impact charts
- Bootstrap confidence intervals
- Robustness and sensitivity analysis results

Some results are exported as PNG figures and CSV tables.

## Discussion

Machine learning models can achieve strong overall predictive performance while still producing unequal outcomes across demographic groups.

This project investigates how differences in selection rates and classification errors can reveal potential fairness concerns.

The analysis also considers whether removing sensitive attributes from model training is sufficient to prevent biased predictions.

## Limitations

Several limitations should be considered:

- The Adult Income Dataset is historical and may not represent current income patterns.
- Existing socioeconomic inequalities may be reflected in the training data.
- Excluding sensitive attributes does not necessarily eliminate proxy discrimination.
- Fairness metrics may produce different interpretations of model behaviour.
- Statistical disparities do not automatically establish unlawful discrimination.

The findings are intended for academic research rather than real-world high-stakes decision-making.

## References

Barocas, S. and Selbst, A.D. 2016. Big data's disparate impact. *California Law Review*, 104(3):671–732. Available at: https://doi.org/10.15779/Z38BG31 [Accessed 4 November 2026].

Becker, B. and Kohavi, R. 1996. *Adult*. UCI Machine Learning Repository. Available at: https://doi.org/10.24432/C5XW20 [Accessed 4 November 2026].

Additional academic references and in-text citations are provided in the Jupyter Notebook using IIE Harvard referencing style.

## Project Purpose

This project forms part of an academic investigation into predictive performance and algorithmic fairness in machine learning.

It aims to demonstrate how classification models can be evaluated not only for predictive accuracy but also for fairness, transparency and potential demographic disparities.
