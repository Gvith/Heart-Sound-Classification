# Heart-Sound-Classification

This project was developed to analyze heart sounds and identify the normal versus abnormal hear sounds. The main task of this project was to develop a prediction tool that can classify the heart beat as either normal or abnormal. Inspired from the prior works on heart sound classification, we developed a system that uses a set of features extracted from heart sounds to train a random forest classifier.

## Feature Extraction:

Based on [one of the previous works](http://ieeexplore.ieee.org/abstract/document/7868819/), we extracted the the following set of features.

### Time Domain Features
The mean and Standard deviations of the below features

- Heart cycle intervals
- S1, Systole, S2, Diastole intervals
- ratio of systolic interval to RR interval of each heart cycle
- ratio of diastolic interval to RR interval of each heart cycle 
- ratio of systolic to diastolic interval of each heart cycle.
- ratio of the mean absolute amplitude during systole to that during the S1 period in each heart cycle.
- ratio of the mean absolute amplitude during diastole to that during the S2 period in each heart cycle.
- skewness and kurtosis of the amplitude during S1, Systole, S2 and Diastole

### Frequency Domain Features
- The power density spectrum of S1, Systole, S2, Diastole across 9 frequency bands: 25-45, 45-65, 65-85, 85-105, 105-125, 125-150, 150-200, 200-300, 300-400 Hz
- 12 Mel Frequency Cepstral Coefficients for each of S1, Systole, S2 and Diastole phases of the heart sound.

All in all, we obtain 120 features. To reduce the dimensionality of this feature space, we using a feature selection method that retains the informative features, discarding the rest.


## Feature Selection

We use the Gini impurity measure to perform feature selection. This measure is obtained as a consequence of using Random Forest Classifier. Gini impurity determines the goodness of the split using a particular feature. A feature which highly distinguishes between the positive and negative class has a low gini impurity measure which is a desirable for the random forest classifier. Using an iterative process, we run the feature selection multiple times to assess the stability of the feature selection.

## Model training

We use random forest classifier to train a model on a 5 fold cross-validation set. We use grid search which exhaustively searches a predetermined search space to optimize the parameters of our classifier. The optimal parameters are then used to train the model with a 90%-10% train-test split. The trained model is then tested on a blind test set and the performace is measured in terms of precision, recall and F-score


## Execution Commands

To run the feature extraction module:
```
python feature_extraction.py --train_state=<path-to-heart-sound-state-file> --train_amps=<path-to-heart-sound-amplitude-file> --feat_names=<path-to-feature-names--file> --out=<path-to-the-output-file.csv>
```
### Parameters
**Required:**
- *--train_state*
- *--train_amps*
- *--feat_names*
- *--out*

**Example:**
```
python feature_extraction.py --train_state="Dataset/training/training_state.csv" --train_amps="Dataset/training/training_amplitudes.csv" --feat_names="Dataset/training/feature_names.csv" --out="Dataset/training/output.csv"
```

To run the prediction model

```
python ml_classifier.py --feat=<path-to-the-feature-extraction-output-file> --module=<either 'train-cross-val' or 'train-test'> --n_maj=<number/proportion of majority samples> --n_min=<number/proportion of minority samples> --test_size=<proportion of test size> --out=<path-to-the-output-directory> --yname=<name of the column to use as a predictor variable>
```
### Parameters
**Required:**
- *--feat*
- *--module*
- *--out*

**Optional:**
- *--n_maj*(Default = number of minority samples)
- *--n_min*(Default = number of minority samples)
- *--test_size*(Default=0.2)
- *--y_name*(Default="Labels")

**Example:**

```
python ml_classifier.py --feat="Dataset/training/output.csv" --module="train-test" --n_maj=700 --n_min=600 --test_size=0.1 --out="Dataset/results/output"
```

#### NOTE:
The prediction model operates in two configurations:
- *train-cross-val* - In this configuration, the random forest classifier is trained on the training data and tested using a 5-fold cross validation set.
- *train-test* - In this configuration, the random forest classifier is trained on the training data and tested using blind test set.
