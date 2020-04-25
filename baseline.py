import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import cross_val_score

from sklearn.base import BaseEstimator,TransformerMixin


class AttribsValueAdder(BaseEstimator, TransformerMixin):
    def __init__(self, col_ixs):
        self.col_ixs = col_ixs
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        addition = None
        try:
            for col_ix in col_ixs:
                if addition == None:
                    addition = X[:,col_ix]
                else:
                    addition = np.add(addition, X[:,col_ix])
        except Exception as e:
            print(e)
        return np.c_[X, addition]

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.cols]

path = "../datasets/titanic/"
train_file = path+"train.csv"
test_file = path+"test.csv"

def load_data(filename):
    '''
    data_train = load_data(train_file)
    '''
    return pd.read_csv(filename, index_col='PassengerId')

'''
feature engineering
'''

data_train = load_data(train_file)
y_full = data_train['Survived']
X_full = data_train.drop(['Survived'], axis=1)

#Pclass	Name Sex Age SibSp Parch Ticket Fare Cabin Embarked
num_attribs = ['Age', 'Fare', 'SibSp', 'Parch']
cat_attribs = ['Sex', 'Embarked']
pclass_attribs = ['Pclass']

col_ixs = [2, 3] # indices of SibSp and Parch in num_attribs
num_transformers = [('columns_select', ColumnSelector(num_attribs)), ('data_cleaning', SimpleImputer()), ('attribs_addition', AttribsValueAdder(col_ixs))]
cat_transformers = [('columns_select', ColumnSelector(cat_attribs)), ('data_cleaning', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder())]
pclass_transformer = [('columns_select', ColumnSelector(pclass_attribs)), ('data_cleaning', SimpleImputer(strategy='most_frequent'))]

num_pip = Pipeline(num_transformers)
cat_pip = Pipeline(cat_transformers)
pclass_pip = Pipeline(pclass_transformer)
parallel_pips = [('num_pip', num_pip), ('cat_pip', cat_pip), ('pclass_pip', pclass_pip)]
full_pip = FeatureUnion(transformer_list=parallel_pips)


'''
models selection
'''

X_prepared = full_pip.fit_transform(X_full)

models = {"random_forest":RandomForestClassifier(),
"adaboost":AdaBoostClassifier(),
"xgboost":XGBClassifier(),
"lightgbm":LGBMClassifier(),
"svm":SVC(),
#"bayes":GaussianNB(),
"knn":KNeighborsClassifier()}

scores = {}

for model_name, model in models.items():
    score = cross_val_score(model, X_prepared, y_full, cv=5, scoring='accuracy')
    scores[model_name] = score

for model_name, score in scores.items():
    print("{}: {} - mean: {}".format(model_name, score, np.mean(score)))


'''
random_forest: [0.78212291 0.81460674 0.84831461 0.7752809  0.8258427 ] - mean: 0.8092335697696316
adaboost: [0.75418994 0.79775281 0.82022472 0.81460674 0.8258427 ] - mean: 0.8025233820852427
xgboost: [0.7877095  0.82022472 0.85955056 0.79775281 0.83707865] - mean: 0.8204632477559475
lightgbm: [0.81564246 0.84269663 0.85955056 0.79775281 0.84831461] - mean: 0.8327914129684263

svm: [0.58659218 0.71348315 0.69101124 0.68539326 0.69101124] - mean: 0.6734982110350888
knn: [0.66480447 0.69101124 0.73595506 0.75280899 0.71910112] - mean: 0.7127361747536249
'''


'''
Grid Search Hyperparameters
'''
