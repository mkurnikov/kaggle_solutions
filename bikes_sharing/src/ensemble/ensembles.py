import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class EnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimators=None, weights=None, subsample=1.0, prediction_transform=None):
        self.estimators = estimators
        self.weights = weights
        self.subsample = subsample
        self.preds_transform = prediction_transform


    def fit(self, X, y):
        for estimator in self.estimators:
            if hasattr(estimator, 'fit'):
                raise RuntimeError('Base estimator %s doesn\'t have fit() method' % str(estimator))

            estimator.fit(X, y)


    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.estimators)), dtype=np.float32)
        for i, estimator in enumerate(self.estimators):
            if hasattr(estimator, 'predict'):
                raise RuntimeError('Base estimator %s doesn\'t have predict() method' % str(estimator))

            prediction = estimator.predict(X)
            if self.preds_transform is not None:
                if not hasattr(self.preds_transform, '__call__'):
                    raise RuntimeError('preds_transform is not callable')
                prediction = self.preds_transform(prediction)

            predictions[:, i] = prediction

        return predictions.mean(axis=1)

if __name__ == '__main__':
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.datasets import make_regression
    from sklearn.cross_validation import train_test_split

    X, y = make_regression(random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf1 = Ridge()
    clf2 = Lasso()

    ensemble = EnsembleRegressor(estimators=[clf1, clf2, []])
    ensemble.fit(X_train, y_train)
    print ensemble.score(X_test, y_test)



