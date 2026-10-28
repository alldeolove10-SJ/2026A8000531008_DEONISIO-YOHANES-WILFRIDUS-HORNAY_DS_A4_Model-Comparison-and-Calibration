# DS-A4: Model Comparison and Calibration

## 1. Project Overview

This project focuses on comparing machine learning models and evaluating their performance. It uses the Breast Cancer Wisconsin dataset to study classification, probability calibration, and the effect of different decision thresholds.

The main goal is to compare models using different evaluation metrics and understand how well their predicted probabilities match the actual outcomes.

## 2. Project Structure

The repository contains the following folders and files:

- `data/` - Contains the frozen dataset and data provenance information.
- `figures/` - Contains plots used to compare and evaluate the models.
- `notebooks/` - Contains the Jupyter notebooks for the analysis.
- `report/` - Contains the project report in Word and PDF formats.
- `results/` - Contains CSV files with model comparisons, predictions, and evaluation results.
- `src/` - Contains the Python script used to run the analysis.
- `AI_use_log.md` - Records the use of AI tools.
- `protocol.json` - Contains the analysis protocol.
- `README.md` - Provides information about the project.

## 3. Dataset

The dataset is stored in `data/wdbc_sklearn_frozen.csv`.

It is based on the Breast Cancer Wisconsin dataset and is used for binary classification. The data contains features calculated from breast mass measurements.

The frozen dataset is kept in the repository so that the analysis can use a consistent version of the data.

## 4. Method

The analysis follows these main steps:

1. Load and check the dataset.
2. Prepare the data for classification.
3. Train and compare machine learning models.
4. Evaluate model performance using different metrics.
5. Check probability calibration.
6. Study the effect of different classification thresholds.
7. Examine selected groups of observations.
8. Save the results and visualizations.

## 5. Figures

The `figures/` folder contains the following plots:

- `01_roc_curves.png` - Compares model performance using ROC curves.
- `02_precision_recall.png` - Shows the precision-recall performance.
- `03_calibration_curves.png` - Compares predicted probabilities with observed outcomes.
- `04_metric_comparison.png` - Compares evaluation metrics across models.
- `05_threshold_sensitivity.png` - Shows how different thresholds affect model performance.
- `06_critical_slice.png` - Examines model performance for selected groups of observations.

## 6. Results

The `results/` folder contains the output files from the analysis, including:

- `cv_model_comparison.csv` - Cross-validation model comparison.
- `environment.json` - Information about the analysis environment.
- `illustrative_feature_slice_analysis...` - Results for selected feature groups.
- `paired_bootstrap_interval.csv` - Bootstrap confidence intervals for model comparisons.
- `paired_cv_predictions.csv` - Cross-validation predictions.
- `paired_test_predictions.csv` - Test-set predictions.
- `test_model_comparison.csv` - Model comparison on the test data.
- `threshold_sensitivity_grid.csv` - Results for different classification thresholds.

These files provide more details about model performance and the comparison results.

## 7. How to Run

### Requirements

- Python 3.11 or later
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Jupyter Notebook
But in my6 case I am using Hupyter Notebook
Install the required libraries with:

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

### Run the Analysis

From the main project folder, run:

```bash
python src/run_analysis.py
```

You can also open a notebook from the `notebooks/` folder and run the cells in order.

## 8. Report

The `report/` folder contains the project report in Word and PDF formats. It explains the model comparison, evaluation metrics, calibration results, and main findings.

## 9. Conclusion

This project shows how different classification models can be compared using several evaluation methods. It also demonstrates why probability calibration and threshold selection are important when evaluating machine learning models.

The results and figures help explain the strengths and limitations of the models.
