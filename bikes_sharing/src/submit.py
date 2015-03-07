import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import ExtraTreeRegressor

from constants import TRAINING_SET_FILE, TEST_SET_FILE, OUTPUT_FILE
from feature_extraction import extract_features
from helpers import train_test_split
from src.ensemble.ensembles import EnsembleRegressor

initial_train_data = pd.read_csv(TRAINING_SET_FILE)
train_data = extract_features(initial_train_data, target_column='count')

X_train, _, y_train, _ = train_test_split(train_data, target_column='count', exclude_columns=['casual', 'registered'], 
                                          test_size=0.0, random_state=10)

print X_train.shape, y_train.shape

# classifier = GradientBoostingRegressor(loss='lad', learning_rate=0.0025, n_estimators=4000, max_features=0.5,
#                                         min_samples_split=2, min_samples_leaf=5, max_depth=8, subsample=0.95)
# 
# 
# classifier.fit(X_train, y_train)
clfs = []
gboost_clf = GradientBoostingRegressor(loss='lad', learning_rate=0.01, n_estimators=1000, max_features=0.5,
                                        min_samples_split=2, min_samples_leaf=5, max_depth=8, subsample=0.95, random_state=1)
clfs.append(gboost_clf)

adaboost_clf = AdaBoostRegressor(base_estimator=ExtraTreeRegressor(min_samples_leaf=1, min_samples_split=5),
                        n_estimators=750, loss='linear', learning_rate=0.01, random_state=0)
clfs.append(adaboost_clf)

def negative_to_zero(preds):
    preds[preds < 0] = 0
    return preds

ensemble = EnsembleRegressor(estimators=clfs, prediction_transform=negative_to_zero)
ensemble.fit(X_train, y_train)
#
# X_train, _, y_train, _ = train_test_split(train_data, target_column='registered', exclude_columns=['count', 'casual'], test_size=0.0)
# cas_classifier = GradientBoostingRegressor(loss='lad', n_estimators=1000, max_depth=9, min_samples_split=5)
# cas_classifier.fit(X_train, y_train)

initial_test_data = pd.read_csv(TEST_SET_FILE)
test_data = extract_features(initial_test_data, target_column='count')
# test_data['registered'] = cas_classifier.predict(test_data)

X_test = test_data.values

output_df = pd.DataFrame(columns=['datetime', 'count'])

output_df['datetime'] = initial_test_data['datetime']
# output_df['count'] = predictions
output_df['count'] = ensemble.predict(X_test)

output_df.to_csv(OUTPUT_FILE, index=False)




