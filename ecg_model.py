
##############################################################
####################### Libraries ############################
##############################################################

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import warnings
import pickle
import os

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

##############################################################
####################### Functions ############################
##############################################################
os.getcwd()
os.chdir("../datasets")
file_path = os.getcwd() + "\\"
def load_dataset(filename, extension = '.csv'):
    """
    Iports the dataset
    Parameters
    ----------
    dataset

    Returns
    -------
    dataframe
    """
    if 'csv' in extension:
        data = pd.read_csv(file_path+filename+extension)
    elif 'xls' in extension:
        data = pd.read_excel(file_path+filename+extension)
    elif 'pkl' in extension:
        data = pd.DataFrame(pickle.load(open(file_path+filename+extension, 'rb')))
    return data

def save_dataset(data, filename, extension = '.csv'):
    """
    Iports the dataset
    Parameters
    ----------
    dataset

    Returns
    -------
    dataframe
    """
    if 'csv' in extension:
        data.to_csv(file_path+filename+extension)
    elif 'xls' in extension:
        data.to_excel(file_path+filename+extension, index=False)
    elif 'pkl' in extension:
        pickle.dump(data, open(file_path+filename+extension, 'wb'))


##############################################################
################## Data preparation ##########################
##############################################################

df = load_dataset('ecg_feature', extension = '.pkl')

Y = df['y']
X_ = df.drop(['y', 'y_label'], axis=1)
for wpt in [col for col in X_.columns if 'wpt' in col]:
    X_ = X_.drop(wpt, axis=1)
for dwt in [col for col in X_.columns if 'dwt' in col]:
    X_ = X_.drop(dwt, axis=1)
rs = RobustScaler()
X = rs.fit_transform(X_)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)


##############################################################
################## Base Models ##########################
##############################################################
import sklearn
sklearn.__version__
classifiers = [('LR', LogisticRegression()),
               ('KNN', KNeighborsClassifier()),
               ('SVC', SVC()),
               ('CART', DecisionTreeClassifier()),
               ('RF', RandomForestClassifier()),
               ('Adaboost', AdaBoostClassifier()),
               ('GBM', GradientBoostingClassifier()),
               ('LightGBM', LGBMClassifier()),
               ]

for name, classifier in classifiers:
    cv_results = cross_validate(classifier, X, Y, cv=3, scoring=['f1_micro', 'f1_macro', 'accuracy'], error_score="raise")
    print(f' ############# {name} #############')
    print(f" test accuracy: {round(cv_results['test_accuracy'].mean(), 4)} ")
    print(f" f1_micro: {round(cv_results['test_f1_micro'].mean(), 4)} ")
    print(f" f1_macro: {round(cv_results['test_f1_macro'].mean(), 4)} ")


##############################################################
############## Hyperparameter Optimization ###################
##############################################################

lr_params = {'C': [0.001, 0.01, 0.1, 1, 10]}

svc_params = {'C': [0.1, 1, 10, 100],
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['rbf']}

gbm_params = {'learning_rate': [0.001, 0.001, 0.1],
              'max_depth': [3, 5, 8],
              'n_estimators': [10, 50, 100],
              'subsample': [0.5, 0.8, 1.0]}

lightgbm_params = {'learning_rate': [0.01, 0.1],
                   'n_estimators': [100, 200, 500],
                   'colsample_bytree': [0.3, 0.5, 0.7]}

classifiers = [('LR', LogisticRegression(), lr_params),
               ('SVC', SVC(), svc_params),
               ('GBM', GradientBoostingClassifier(), gbm_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]

best_models = {}

for name, classifier, params in classifiers:
    print(f'########## {name} Before ##########')
    cv_results = cross_validate(classifier, X, Y, cv=3, scoring=['f1_micro', 'f1_macro', 'accuracy'])

    print(f" test accuracy : {round(cv_results['test_accuracy'].mean(), 4)} ")
    print(f" f1_micro : {round(cv_results['test_f1_micro'].mean(), 4)} ")
    print(f" f1_macro : {round(cv_results['test_f1_macro'].mean(), 4)} ")

    print(f'########## {name} After ##########')
    gs_best = GridSearchCV(classifier, params, cv=3, n_jobs=-1, verbose=False).fit(X, Y)
    final_model = classifier.set_params(**gs_best.best_params_)

    cv_results = cross_validate(final_model, X, Y, cv=3, scoring=['f1_micro', 'f1_macro', 'accuracy'])
    print(f" test accuracy : {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f" f1_micro : {round(cv_results['test_f1_micro'].mean(), 4)} ")
    print(f" f1_macro : {round(cv_results['test_f1_macro'].mean(), 4)} ")
    print(f"{name} best params: {gs_best.best_params_}", end='\n\n')

    best_models[name] = final_model


##############################################################
################### Plot Importance ##########################
##############################################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by='Value', ascending=False)[0:num])
    plt.title('Feature Importance List')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

final_model_ = final_model.fit(X,Y)
plot_importance(final_model_, X_, 15)
