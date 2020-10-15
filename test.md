# hello world

```
class TfIdfHandler(BaseEstimator, TransformerMixin):
    def __init__(self, uuid):
        self.uuid = uuid
    def fit(self, data_origin):
        tfidf_train_features = tfidf_extractor(
            self.uuid, data_origin,
            ngram_range=(1, 1), max_features=None)
        logger.info('cut text to words successfully')
        return tfidf_train_features
    def transform(self, data_origin):
        tfidf_train_features = tfidf_transform(self.uuid, data_origin)
        logger.info('cut text to words successfully')
        return tfidf_train_features    
def save_to_pickle(save_dir, save_data):
    with open(save_dir, 'wb') as f:
        joblib.dump(save_data, f)
def read_from_pickle(pickle_file):
    try:
        with open(os.path.realpath(pickle_file), 'rb') as f:
            pickle_data = joblib.load(f)
        return pickle_data
    except Exception as e:
        logger.info("wrong path:{}".format(e))
def pipeline_time(col_time):
    time_pipe = Pipeline(steps=[
        ("select", DataFrameSelector(col_time)),
        ("feature_handler", TimeFeatureHandler()),
        ("min_max_scalar", MinMaxScaler())
    ], memory=None) if col_time else None
    return time_pipe
def pipeline_all_tf(col_num, col_cat, col_str, col_time, uuid):

    all_pipe = FeatureUnion(transformer_list=[
        ("num_pipe", pipeline_num(col_num)),
        ("cat_pipe", pipeline_cat(col_cat)),
        ("str_pipe", pipeline_str_tfidf(col_str, uuid)),
        ("time_pipe", pipeline_time(col_time))
    ])
    return all_pipe
```
## oputna
```
import optuna
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from utils import waste_time


def common_train(func):
    """
    :param func:
    :return:
    """

    def _wrapper(*args, **kwargs):
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: func(trial, *args, **kwargs),
                       n_trials=100)
        return study

    return _wrapper


@common_train
def objective_ada(trial, x, y):
    """

    :param trial:
    :param x:
    :param y:
    :return:
    """
    param = {
        "n_estimators": trial.suggest_int('n_estimators', 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 2.0),
        "algorithm": trial.suggest_categotical("algorithm",
                                               ["SAMME.R", "SAMME"])
    }
    model = AdaBoostClassifier(**param)
    model.fit(x, y)
    score = cross_val_score(model, x, y, n_jobs=3, cv=3)
    accuracy = score.mean()
    return accuracy


@common_train
def objective_gb(trial, x, y):
    """

    :param trial:
    :param x:
    :param y:
    :return:
    """
    param = {
        "n_estimators": trial.suggest_int('n_estimators', 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 2.0),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "criterion": trial.suggest_categotical("criterion",
                                               ["friedman_mse", "mse", "mae"]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0)
    }
    model = GradientBoostingClassifier(**param)
    model.fit(x, y)
    score = cross_val_score(model, x, y, n_jobs=3, cv=3)
    accuracy = score.mean()
    return accuracy


@waste_time
@common_train
def objective_xbg(trial, x, y):
    """

    :param trial:
    :param x:
    :param y:
    :return:
    """
    param = {
        "n_estimators": trial.suggest_int('n_estimators', 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 25),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 2.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 20),
        "reg_alpha": trial.suggest_int("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_int("reg_lambda", 0, 5),
        "gamma": trial.suggest_int("gamma", 0, 5),
        "colsample_bytree": trial.suggest_discrete_uniform("colsample_bytree",
                                                           0.1, 1, 0.01),
        "n_jobs": 2
    }
    model = XGBClassifier(**param)
    model.fit(x, y)
    score = cross_val_score(model, x, y, cv=3, n_jobs=3)
    accuracy = score.mean()
    return accuracy


@common_train
def objective_rf(trial, x, y):
    """
    :param trial:
    :param x:
    :param y:
    :return:
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "criterion": trial.suggest_categorical("criterion",
                                               ["gini", "entropy"]),
        "max_features": trial.suggest_float("max_features", 0.0, 1.0),
        "max_depth": trial.suggest_int("max_depth", 3, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "n_jobs": 2
    }
    model = RandomForestClassifier(**params)
    model.fit(x, y)
    score = cross_val_score(model, x, y, n_jobs=3, cv=3)
    accuracy = score.mean()
    return accuracy


@common_train
def objective_svm(trial, x, y):
    """
    :param trial:
    :param x:
    :param y:
    :return:
    """
    params = {
        "C": trial.suggest_loguniform("C", 0.001, 1000),
        "kernel": trial.suggest_categorical("kernel",
                                            ["rbf", "poly", "sigmoid"]),
        "degree": trial.suggest_int("degree", 1, 5),
        "gamma": trial.suggest_loguniform("gamma", 3.0517578125e-05, 10),
        "coef0": trial.suggest_float("coef0", 0.0, 10.0),
        "tol": trial.suggest_loguniform("tol", 1e-5, 1e-1)}
    model = SVC(**params)
    model.fit(x, y)
    score = cross_val_score(model, x, y, n_jobs=3, cv=3)
    accuracy = score.mean()
    return accuracy


@common_train
def objective_lr(trial, x, y):
    """
    :param trial:
    :param x:
    :param y:
    :return:
    """
    params = {
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        "dual": trial.suggest_categorical("dual", [True, False]),
        "tol": trial.suggest_loguniform("tol", 1e-5, 1e-1),
        "C": trial.suggest_loguniform("C", 0.03125, 32768),
        "fit_intercept": trial.suggest_categorical("fit_intercept",
                                                   [True, False]),
        "n_jobs": 2}
    model = LogisticRegression(**params)
    model.fit(x, y)
    score = cross_val_score(model, x, y, n_jobs=3, cv=3)
    accuracy = score.mean()
    return accuracy
```
