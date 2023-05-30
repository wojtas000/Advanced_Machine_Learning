# Project 2 - feature selection
In this project, our aim was to propose methods of feature selection and classification, which allow to build good classification model on `spam` and `artificial` datasets.
We explored various feature selection methods, along with flagship classifiers.

## Feature selection methods
1. PCA (feature extraction)
2. Recursive Feature Elimination
3. Lasso, ElasticNet
4. Random Forest
5. Filter methods (correlation, mutual information)
6. ANOVA
7. chi2 
8. Boruta
9. Bonus:
    - stack (multiple feature selection methods executed one after another)
    -  ensemble (collective of couple feature selection methods)

## Classifiers
1. SVM (Rbf and linear kernel)
2. Random Forest
3. Decision Tree Classifier
4. XGBoost Classifier
5. Logistic regression with L2 penalty

## Directory structure
1. `data` - directory containing files connected with datasets. Specifically:
    - `data_{dataset_name}` - train/test datasets,
    - `results_{dataset_name}` - results for each feature selection method, saved to `.csv`s,
    - `final_results` -  best selected features and posterior probabilities for 1 class for each dataset.
2. `Feature_selection_package` - python package for functionality devoted to creating feature selector classes and evaluating their performances
3. `Visualisation_package` - python package for visualizing results
4. `{dataset_name}_data_experiments.ipynb` - files for conducting experiments
5. `visualisations.ipynb` - file for visualizing final results 
