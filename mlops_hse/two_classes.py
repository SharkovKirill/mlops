from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


class MyKeyError(KeyError):
    def __init__(self, text):
        self.txt = text


class MyAlreadyExitsError(Exception):
    def __init__(self, text):
        self.txt = text


class MyTypeError(KeyError):
    def __init__(self, text):
        self.txt = text


class InvalidData(Exception):
    def __init__(self, text):
        self.txt = text


class AllModels(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(AllModels, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.__available_model_types = {
            "cb": CatBoostClassifier,
            "rf": RandomForestClassifier,
        }
        self.__models = {}
        self.__names_fitted_models = []

    def get_avaliable_model_types(self):
        return self.__available_model_types

    def get_models(self, only_fitted=False):
        if only_fitted:
            return self.__names_fitted_models
        else:
            return list(self.__models.keys())

    def delete_model(self, user_model_name):
        try:
            del self.__models[user_model_name]
            self.__names_fitted_models.remove(user_model_name)
        except KeyError:
            raise MyKeyError("Модели с таким названием не существует")

    def init_new_model(self, type_model, user_model_name: str, params={}):
        if user_model_name in self.__models.keys():
            raise MyAlreadyExitsError("Модель с таким именем уже существует")
        if type_model not in self.__available_model_types.keys():
            raise MyKeyError("Выбранной модели нет в списке доступных")
        self.__models[user_model_name] = Model(
            self.__available_model_types[type_model],
            user_model_name,
            params=params,
        )

    def model_fit(self, X, y, user_model_name):
        try:
            self.__models[user_model_name].fit(X, y)
            self.__names_fitted_models.append(user_model_name)
        except KeyError:
            raise MyKeyError("Модель с таким имененем не найдена")

    def model_predict(self, X, user_model_name):
        if user_model_name in self.__names_fitted_models:
            return self.__models[user_model_name].predict(X)
        else:
            raise MyKeyError(
                "Модель с таким имененем не найдена или не обучена"
            )

    def get_params(self, user_model_name, all: bool = False):
        return self.__models[user_model_name].get_params(all)


class Model:
    getting_params_func_names = {
        CatBoostClassifier: CatBoostClassifier.get_all_params,
        RandomForestClassifier: RandomForestClassifier.get_params,
    }

    def __init__(self, base_model, user_model_name: str, params) -> None:
        self.params = params
        self.user_model_name = user_model_name
        try:
            self.base_model = base_model(**self.params)
        except TypeError:
            raise MyTypeError("Переданы неправильные гиперпараметры модели")

    def fit(self, X, y):
        try:
            self.base_model.fit(X=X, y=y)
        except:
            raise InvalidData("Некорректные данные для обучения")

    def predict(self, X):
        try:
            return self.base_model.predict(X)
        except:
            raise InvalidData("Некорректные данные для предсказания")

    def get_params(self, all):
        if all is True:
            return self.getting_params_func_names[type(self.base_model)](
                self.base_model
            )
        elif all is False:
            return self.params


df = pd.DataFrame({"col1": [1, 2, 4], "col2": [1, 2, 4], "target": [4, 5, 6]})
print(df)

allm = AllModels()

print(allm.get_avaliable_model_types())
print(allm.get_models())

allm.init_new_model("cb", "cb1", {"n_estimators": 131, "random_state": 42})
allm.init_new_model("rf", "rf1", {"n_estimators": 132, "random_state": 42})
allm.model_fit(X=df[["col1", "col2"]], y=df["target"], user_model_name="cb1")
allm.model_fit(X=df[["col1", "col2"]], y=df["target"], user_model_name="rf1")
print(allm.model_predict(pd.Series([4, 5, 6]), "cb1"))
print(allm.get_models())
print(allm.get_params("cb1"))
print(allm.get_params("rf1"))
