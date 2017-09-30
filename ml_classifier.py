from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix
from collections import Counter
from utils import *
import pandas as pd
import numpy as np
import sys
import os
import argparse

# parse the command line arguments and store the values in its respective variables
parser = argparse.ArgumentParser()

# --feat: specify the path of the training features file (.csv) format
# --module: choose the different modules to run as per requirement. The two modules available are:
# --module='train-cross-val': train the model and tests on cross validation set
# --module='train-test': trains the model and tests it on blind test set
# --n_maj: specify the number of samples from the majority class to be used in the training process.
# --n_min: specify the number of samples from the minority class to be used in the training process.
# Default for n_maj and n_min chooses equal number of samples from both classes
# --y_name: specify the name of the column to be used as the predictor variable
# --test-size: specify the size of the test data in terms of proportion of the entire Dataset
# The default value is 20% of the entire data.
# --out: specify the path to the output folder where the graphs and plots should be stored
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--feat", type=str, help="path to the feature list file", required=True)
requiredNamed.add_argument("--module", type=str, help="specifies the module to execute. Possible values include: ['train-cross-val','train-test'] ", required=True)
parser.add_argument("--n_maj", type=float, help="number of samples from the majority class. Default=<equal number of positive and negative samples>")
parser.add_argument("--n_min", type=float, help="number of samples from the majority class. Default=<equal number of positive and negative samples>")
parser.add_argument("--y_name", type=str, help="name of the column to predict. Default='Labels'")
parser.add_argument("--test_size", type=float, help="proportion of the data to be reserved as test set. Default=0.2")
requiredNamed.add_argument("--out", type=str, help="path to the output folder", required=True)

args = parser.parse_args()

# This function is used to find the optimal set of parameters for the
# Random Forest Classifier. It uses GridSearchCV to perform an exhaustive
# search over the specified search space and return the optimal parameters
# for this dataset
def parameter_tuning(x, y):

    # specify the search space for paramters you want to optimize
    estimators = np.arange(10, 60, 20)
    m_depth = np.arange(1, 10, 2)

    # use stratified K fold to test the optimization results
    cvs = StratifiedKFold(5)
    params = [{'n_estimators': estimators, 'max_depth': m_depth}]

    # intialize a random forest classifier
    rf = RandomForestClassifier(class_weight='balanced', random_state=0)

    # search for the optimal set of parameters for Random Forest Classifier
    gsvm = GridSearchCV(estimator=rf, param_grid=params, n_jobs=-1, cv=list(cvs.split(x, y)))
    gsvm.fit(x, y)
    return gsvm.best_params_

# This function is used to perform feature selection to identify the informative
# features from the high dimensional feature space. This is useful if the
# dimensionality of the data is large and only a few features contribute to the
# prediction task. In order to overcome the randomness in feature selection process
# we perform the feature selection for multiple times and extract the union of
# top 30 features.[On trial and error, this was observed to give us the best
# performance]
def feature_selection(data_frame, y_name):

    feature_list = []
    columns = data_frame.columns.values.tolist()
    columns.remove(y_name)

    # run the feature selection for 50 times
    for i in range(50):
        # shuffle the data on each iteration
        data_frame = data_frame.sample(frac=1).reset_index(drop=True)

        training_data = data_frame[columns]
        training_label = data_frame[y_name]

        # invoke the paramter tuning function
        rf_params = parameter_tuning(training_data, training_label)

        # intialize the random forest classifier
        feat_select = RandomForestClassifier(n_estimators=rf_params['n_estimators'], max_depth=rf_params['max_depth'], class_weight='balanced', random_state=0)
        feat_select.fit(training_data, training_label)
        #print feature_subset.feature_importances_

        # extract the feature importances and sort them in the descending order of their importance score
        features = sorted(zip(map(lambda x: round(x, 4), feat_select.feature_importances_), columns),
                     reverse=True)

        sc, f_names = zip(*features)

        # extract top 30 features as informative ones
        feature_list.append(set(list(f_names[:30])))

    # take the union of the different feature selection runs
    feat_union = list(set.union(*feature_list))

    print "Number of features selected: ", len(feat_union)
    return feat_union

# This method trains a random forest classifier model to predict the heart sound
# as either normal or Abnormal
# It tests the performance of the model on a 5 fold cross validation set
def model_training(x_train, y_train, cross_val, y_name, n_maj=None, n_min=None):

    # cross_val flag is used to specify if the model is used to test on a
    # cross validation set or the blind test set
    if cross_val:
        # splits the training data to perform 5-fold cross validation
        ss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

        pr_list = []
        re_list = []
        fs_list = []

        for train_index, test_index in ss.split(x_train, y_train):
            X_train = x_train[train_index]
            Y_train = y_train[train_index]
            X_test = x_train[test_index]
            Y_test = y_train[test_index]

            #invoke the parameter tuning functiom
            rf_params = parameter_tuning(X_train, Y_train)

            # intialize the random forest classifier
            rfc = RandomForestClassifier(n_estimators=rf_params['n_estimators'], max_depth=rf_params['max_depth'], n_jobs=-1, random_state=0)
            rfc.fit(X_train, Y_train)

            # use the classifier to predict on cross val set
            y_predicted = rfc.predict(X_test)

            # compute the precision, recall and fscore
            pr, re, fs, _ = precision_recall_fscore_support(Y_test, y_predicted, pos_label=1, average='binary')
            pr_list.append(pr)
            re_list.append(re)
            fs_list.append(fs)

        return np.mean(pr_list), np.mean(re_list), np.mean(fs_list)

    else:
        # This section only trains the model which is used to test on the blind test set
        rf_params = parameter_tuning(x_train, y_train)
        rfc = RandomForestClassifier(n_estimators=rf_params['n_estimators'], max_depth=rf_params['max_depth'], n_jobs=-1, random_state=0)
        rfc.fit(x_train, y_train)

        return rfc

# This method tests the performance of the random forest classifier
# on the test data. It plots the confusion matrix and Precision-Recall
# curve plots to analyze the results
def model_testing(rf_model, x_test, y_test, output_folder):

    # test the random forest classifier
    predicted_labels = rf_model.predict(x_test)

    # gather the prediction probabilities used to plot precision recall curves
    pred_probab = rf_model.predict_proba(x_test)

    # compute precision, recall and fscore
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, predicted_labels, pos_label=1, average='binary')

    # compute the confusion matrix
    c_mat = confusion_matrix(y_test.values, predicted_labels)

    # plot the confusion matrix and precision recall curves
    plot_confusion_matrix(c_mat, ['Normal', 'Abnormal'], output_folder)
    plot_precision_recall_curve(y_test.values, pred_probab, output_folder)

    return precision, recall, fscore

if __name__=='__main__':

    if args.feat is None:
        print "Specify path to feature list. Usage: <python ml_calssifier.py -h> for more information"
        sys.exit(0)

    if args.module not in ['train-cross-val', 'train-test']:
        print "Specify the correct module name. Usage: <python ml_calssifier.py -h> for more information"
        sys.exit(0)

    if args.out is None:
        print "Output path not specified. Usage: <python ml_calssifier.py -h> for more information"
        sys.exit(0)
    elif not os.path.exists(args.out):
        print "Output path does not exists"
        sys.exit(0)

    if args.y_name is None:
        y_name = 'Labels'
    else:
        y_name = args.y_name

    if args.n_maj is not None:
        n_maj = args.n_maj
    else:
        n_maj = None
    if args.n_min is not None:
        n_min = args.n_min
    else:
        n_min = None

    try:
        df_features = pd.read_csv(args.feat)
    except FileNotFoundError:
        print "Enter the correct file path for the feature list"
        sys.exit(0)

    # shuffle the data
    df_features = df_features.sample(frac=1).reset_index(drop=True)

    # invoke the feature selection module
    feat_list = feature_selection(df_features, y_name)

    X = df_features[feat_list]
    Y = df_features[y_name]

    if args.module == 'train-cross-val':

        # Undersampling is used in situations where one of the data among the different classes is highly imbalanced.
        # invoke the undersampling module that sub-samples the majority class and returns nearly balanced data.
        X, Y = undersampling(X.values, Y.values, majority_class=-1, minority_class=1, maj_proportion=n_maj, min_proportion=n_min)

        # invoke the training module
        precision, recall, fscore = model_training(X, Y, True, y_name,  n_maj, n_min)
        print "Results on 5-fold Cross Validation Set"
        print "Precision: ", precision
        print "Recall: ", recall
        print "F-score: ", fscore

    elif args.module == 'train-test':
        if args.test_size is None:
            test_size = 0.2
        else:
            test_size = args.test_size

        # split the data into non-overlapping train and test instances
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=0)

        # invoke the undersampling module
        x_sampled, y_sampled = undersampling(x_train.values, y_train.values, majority_class=-1, minority_class=1, maj_proportion=n_maj, min_proportion=n_min)

        # invoke the training module
        rf = model_training(x_sampled, y_sampled, False, y_name,  n_maj, n_min)

        # invoke the test module
        precision, recall, fscore = model_testing(rf, x_test, y_test, args.out)
        print "Results on the Test Set"
        print "Precison: ", precision
        print "Recall: ", recall
        print "F-score: ", fscore
