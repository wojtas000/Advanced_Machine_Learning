Project Plan:

Understand the Aim and Goals:   DONE

The aim of the project is to compare different feature selection methods.
The goal is to propose methods of feature selection and classification that allow building a model with large predictive power using a small number of features.  
----------------------------------------------------------------------------

Familiarize with the Datasets:      DONE, Janek doczytaj tutaj -->

Dataset "artificial": an artificial dataset with relevant and irrelevant features (files: artificial train.data, artificial train.labels, artificial valid.data).
Dataset "spam": a dataset for spam message recognition (files: spam train.csv, spam test.data).
Dataset Characteristics:

Artificial dataset: 500 features, 2000 observations in the training data, and 600 observations in the validation data.
Spam dataset: 7879 features, 4572 observations in the training data, and 1000 observations in the validation data.
File Structure:

Each dataset has three files: training data, labels for training data, and validation data.

After getting the results:

    Create the following result files for each dataset:
    CODE artificial prediction.txt: posterior probabilities for the validation data of the artificial dataset.
    CODE artificial features.txt: selected features for the artificial dataset.
    CODE spam prediction.txt: posterior probabilities for the validation data of the spam dataset.
    CODE spam features.txt: selected features for the spam dataset.
    CODE denotes the code of the student (first student from the group).
    Project Group:
----------------------------------------------------------------------------

Feature Selection Methods: DONE

Implement and test at least 4 feature selection methods on both datasets.
----------------------------------------------------------------------------

Model Training and Prediction: TO DO

Use the training data to train the model and select relevant features.
Make predictions for observations in the validation data.
Assign posterior probabilities (P(y = 1|x1, ..., xp)) to each observation in the validation data.
----------------------------------------------------------------------------

Evaluation and Scoring: TO DO

Evaluate the predictive performance of the model using the Balanced Accuracy measure (BA).
Calculate the score based on BA and the number of features used in the model.
For the Artificial dataset:
Score(BA, m) = BA - 0.01(1/5 * (m - 1))+
For the Spam dataset:
Score(BA, m) = BA - 0.01(1/100 * (m - 1))+
Total points for the score: 50% of the project grade.
Presentation:

Prepare a presentation summarizing the project results.
Record the presentation (up to 5 minutes).
Presentations will be scheduled on specific dates (Group 1: 5 June 2023, Group 2: 12 June 2023).
Half of the groups who didn't present the 1st stationary will present the 2nd project.
Report:

Write a report (max 4 pages A4) describing the methods used and the results of the experiments.
Include the description of feature selection methods and their impact on predictive performance.
Reports contribute to 25% of the project grade.