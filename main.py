import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV


# load up the Data
data = pd.read_csv("winequality-red.csv", sep= ";")

# data visualization:
# print(data.head())
# print(data.shape)

# split data into training and testing:
y = data.quality
X = data.drop('quality', axis =1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

# create a transformer to use on training and later testing sets. Perform Standardization
scaler = preprocessing.StandardScaler().fit(X_train)

# scale the X_train and X_test datasets:
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# make pipeline object:
my_regressor = RandomForestRegressor(n_estimators=100)
pipeline = make_pipeline(preprocessing.StandardScaler(), my_regressor)

# declare hyperparameter tuning for cross validation:
# hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
#                   'randomforestregressor__max_depth': [None, 5, 3, 1],
#                   'randomforestregressor__criterion': ['mse', 'mae']}

# clf = GridSearchCV(pipeline, hyperparameters, cv=3, verbose=1, n_jobs=-1)

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

clf = RandomizedSearchCV(estimator = my_regressor, param_distributions = random_grid,
                   n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)


# Fit and tune model. Update with best Hyperparameters.
clf.fit(X_train, y_train)
print(clf.best_params_)


# evaluate model before and after random grid search:
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy

base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_test, y_test)

best_random = clf.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test)
print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))


# Create the parameter grid based on the results of random search
param_grid = {
    'bootstrap': [True, False],
    'max_depth': [80, 90, 100, 110, None],
    'max_features': [2, 3],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [3, 5, ],
    'n_estimators': [1000, 1500, 2000, 2500, 3000]
}

# Create a based model
rf = RandomForestRegressor()
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 1)

grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, X_test, y_test)
print('Improvement of {:0.2f}%.'.format(100 * (grid_accuracy - base_accuracy) / base_accuracy))

# fit the model to testing data for final prediction
y_pred = best_grid.predict(X_test)
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))