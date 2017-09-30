from matplotlib import pyplot as plt
import itertools
import pylab as pl
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

# This functions is used to undersample the majority class and prevent it from
# overshadowing the minority class
def undersampling(X, Y, majority_class, minority_class, maj_proportion, min_proportion):

    # fetch the row indices matching the majority and minority class
    row_ind_maj = np.where(Y == majority_class)[0]
    row_ind_min = np.where(Y == minority_class)[0]

    if maj_proportion is None:
        maj_num = row_ind_min.shape[0]
    elif maj_proportion.is_integer():
        maj_num = int(maj_proportion)
    elif isinstance(maj_proportion, float):
        maj_num = int(maj_proportion * row_ind_maj.shape[0])

    if min_proportion is None:
        min_num = row_ind_min.shape[0]
    elif min_proportion.is_integer():
        min_num = int(min_proportion)
    elif isinstance(min_proportion, float):
        min_num = int(min_proportion * row_ind_min.shape[0])

    # sample the data based on the proportions of the majority and minority class
    sampled_maj_indices = row_ind_maj[np.arange(0, maj_num)]
    sampled_min_indices = row_ind_min[np.arange(0, min_num)]

    sample_indices = np.concatenate((sampled_maj_indices, sampled_min_indices))

    Y = Y[sample_indices]
    X = X[sample_indices]

    return X, Y

# This function plots the confusion matrix
def plot_confusion_matrix(cm, classes, output_folder, normalize=True, title='Confusion matrix', cmap=plt.cm.gist_yarg):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=3)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output_folder+'confusion_matrix.png')

# This function plots the precision-recall curve
def plot_precision_recall_curve(y_true, y_prob, output_folder):
    # Compute Precision-Recall and plot curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob[:, 1])
    area = auc(recall, precision)
    print "Area Under Curve: %0.2f" % area

    pl.clf()
    pl.plot(recall, precision, label='Precision-Recall curve')
    pl.xlabel('Recall')
    pl.ylabel('Precision')
    pl.ylim([0.0, 1.05])
    pl.xlim([0.0, 1.0])
    pl.title('Precision-Recall example: AUC=%0.2f' % area)
    pl.legend(loc="lower left")
    pl.savefig(output_folder+'precision_recall_curve.png')
