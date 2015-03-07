from sklearn import cross_validation
import numpy as np
import sys

from scipy import sparse
from itertools import combinations

def group_data(data, degree=3, locations=None):
    new_data = []
    n_samples, n_features = data.shape

    feature_numbers = locations
    if not locations:
        feature_numbers = range(n_features)

    combs = combinations(feature_numbers, degree)
    for indices in combs:
        # print indices
        feature_combinations_values = [hash(tuple(v)) % ((sys.maxsize + 1) * 2) for v in data[:, indices]]
        new_data.append(feature_combinations_values)
    return np.array(new_data).T


def OneHotEncoder(data, keymap=None):
     """
     OneHotEncoder takes data matrix with categorical columns and
     converts it to a sparse binary matrix.

     Returns sparse binary matrix and keymap mapping categories to indicies.
     If a keymap is supplied on input it will be used instead of creating one
     and any categories appearing in the data that are not in the keymap are
     ignored
     """
     if keymap is None:
          keymap = []
          for col in data.T:
               uniques = set(list(col))
               keymap.append(dict((key, i) for i, key in enumerate(uniques)))
     # print keymap
     total_pts = data.shape[0]
     # print total_pts
     outdat = []

     for feature_number, feature_values in enumerate(data.T):
          km = keymap[feature_number]
          num_labels = len(km)
          spmat = sparse.lil_matrix((total_pts, num_labels))
          # print feature_values
          for j, value in enumerate(feature_values):
               if value in km:
                    spmat[j, km[value]] = 1
          outdat.append(spmat)
     outdat = sparse.hstack(outdat).tocsr()
     return outdat, keymap

def train_test_split(train_data, target_column='count', exclude_columns=None, test_size=0.2, random_state=1):
    target_col = [target_column]
    not_features = []
    not_features.extend(target_col)
    not_features.extend(exclude_columns)

    # feature_cols = train_data.columns
    feature_cols_idx = ~train_data.columns.isin(not_features)
    # print feature_cols_idx

    features_train_data = train_data[train_data.columns[feature_cols_idx]]
    target_train_data = train_data[target_col]
    X_train, X_test, y_train, y_test =\
        cross_validation.train_test_split(features_train_data, target_train_data,
                                          test_size=test_size, random_state=random_state)
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    # print train_data[train_data.columns[feature_cols_idx]].columns
    return X_train, X_test, y_train, y_test


def train_test_split_with_columns(train_data, target_column='count', exclude_columns=None, test_size=0.2):
    target_col = [target_column]
    not_features = []
    not_features.extend(target_col)
    not_features.extend(exclude_columns)

    # feature_cols = train_data.columns
    feature_cols_idx = ~train_data.columns.isin(not_features)
    # print feature_cols_idx

    features_train_data = train_data[train_data.columns[feature_cols_idx]]
    columns = features_train_data.columns
    target_train_data = train_data[target_col]
    X_train, X_test, y_train, y_test =\
        cross_validation.train_test_split(features_train_data, target_train_data,
                                          test_size=test_size, random_state=1)
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    # print train_data[train_data.columns[feature_cols_idx]].columns
    return X_train, X_test, y_train, y_test, columns


def get_fitted_classifier_and_test_sets(classifier, train_data, target=None, exclude_columns=None, test_size=0.0):
    X_train, X_test, y_train, y_test = \
        train_test_split(train_data, target_column=target, exclude_columns=exclude_columns, test_size=test_size)
    classifier.fit(X_train, y_train)
    return classifier, X_test, y_test


from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def get_sample_dataset_indices(data, features=0.7, size=0.7, replacement=False):
    # print data.shape
    n_samples = data.shape[0] * size
    n_features = data.shape[1] * features
    samples_ind = np.sort(np.random.choice(np.arange(data.shape[0]), size=n_samples, replace=replacement), axis=0)
    feature_ind = np.sort(np.random.choice(np.arange(data.shape[1]), size=n_features, replace=replacement), axis=0)
    
    return samples_ind, feature_ind
