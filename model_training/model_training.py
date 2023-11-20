import os
import tempfile
from typing import Dict, List

import boto3
import joblib
import numpy as np
from botocore.client import Config
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from model_training.exceptions import (
    AlreadyExistsError,
    ConnectionError,
    InvalidData,
    NameKeyError,
    ParamsTypeError,
)

ACCESS_KEY = "user_login"
SECRET_KEY = "user_password"
ENDPOINT_URL = "http://127.0.0.1:9000"
MODELs_BUCKET_NAME = "models"


class Storage:
    """
    Minio controller

    Raises
    ------
    ConnectionError
        Occurs when you can't connect to minio
    """

    @staticmethod
    def get_s3_client():
        try:
            s3_client = boto3.client(
                "s3",
                endpoint_url=ENDPOINT_URL,
                aws_access_key_id=ACCESS_KEY,
                aws_secret_access_key=SECRET_KEY,
                config=Config(signature_version="s3v4"),
                region_name="us-east-1",
            )
        except:
            raise ConnectionError("Connection error")
        return s3_client

    @staticmethod
    def create_bucket_if_not_exists(s3_client):
        if MODELs_BUCKET_NAME not in [
            b["Name"] for b in s3_client.list_buckets()["Buckets"]
        ]:
            s3_client.create_bucket(
                ACL="public-read-write",
                Bucket=MODELs_BUCKET_NAME,
                CreateBucketConfiguration={"LocationConstraint": "us-east-1"},
            )

    @staticmethod
    def model_dump_s3(model, user_model_name: str):
        s3_client = Storage.get_s3_client()
        Storage.create_bucket_if_not_exists(s3_client)
        with tempfile.TemporaryFile() as fp:
            joblib.dump(model, fp)
            fp.seek(0)
            s3_client.put_object(
                Body=fp.read(), Bucket=MODELs_BUCKET_NAME, Key=f"/{user_model_name}.pkl"
            )

    @staticmethod
    def reload_models_from_s3(bucket_name: str = MODELs_BUCKET_NAME):
        s3_client = Storage.get_s3_client()
        Storage.create_bucket_if_not_exists(s3_client)
        try:
            model_dict = dict()
            model_keys = [
                obj["Key"]
                for obj in s3_client.list_objects_v2(Bucket=bucket_name)[
                    "Contents"
                ]
            ]

            if len(model_keys) != 0:
                for key in model_keys:
                    with tempfile.TemporaryFile() as fp:
                        s3_client.download_fileobj(
                            Fileobj=fp, Bucket=bucket_name, Key=key
                        )
                        fp.seek(0)
                        model_dict[key[:-4]] = joblib.load(fp)
            return model_dict
        # except Exception as e:
        #     raise logging.exception(e)
        except KeyError:
            return {}

    def del_model_s3(user_model_name: str):
        s3_client = Storage.get_s3_client()
        s3_client.delete_object(Bucket=MODELs_BUCKET_NAME, Key=user_model_name + ".pkl")


class ModelFactory(object):
    """
    Class for working with models

    """

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(ModelFactory, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.__available_model_types = {
            "cb": CatBoostClassifier,
            "rf": RandomForestClassifier,
        }
        self.__names_fitted_models: List[str] = []
        self.__models: Dict[Model] = Storage.reload_models_from_s3()

        if len(self.__models.keys()) != 0:
            for model in self.__models.values():
                if model.fiited:
                    self.__names_fitted_models.append(model.user_model_name)
        # self.__names_fitted_models: List[str] = []
        # self.__models: Dict[Model] = {}

        # self.PATH = os.getcwd()
        # self.PATH_MODELS = os.path.join(self.PATH, "models")

        # if not os.path.isdir(self.PATH_MODELS):
        #     os.mkdir(self.PATH_MODELS)

        # if len(os.listdir(self.PATH_MODELS)) != 0:
        #     for model_name in os.listdir(self.PATH_MODELS):
        #         loaded_model = joblib.load(
        #             os.path.join(self.PATH_MODELS, model_name)
        #         )
        #         self.__models[model_name[:-4]] = loaded_model
        #     for model in self.__models.values():
        #         if model.fiited:
        #             self.__names_fitted_models.append(model.user_model_name)

    def get_available_model_types(self, show: bool = False):
        """
        Getting available model types

        Parameters
        ----------
        show : bool, optional
            True for representation, else False, by default False

        Returns
        -------
        Dict
            If show==True return dict for representaion, else return dict
            with available model classes
        """
        if show:
            return [
                {"model_name": key, "model_type": str(value)}
                for (key, value) in self.__available_model_types.items()
            ]
        else:
            return self.__available_model_types

    def get_models(
        self,
        only_fitted: bool = False,
        all_params: bool = False,
        name_models: str | None = None,
    ):
        """
        Getting all user models

        Parameters
        ----------
        only_fitted : bool, optional
            True if you want to get only fitted model(s), else False,
            by default False
        all_params : bool, optional
            True if you want to get all params, else False,
            by default False
        name_models : str | None, optional
            User's model name of only ONE particular model you want to get,
            by default None

        Returns
        -------
        Dict
            Dictionary with user model name, type of model, params dict,
            fitting bool inidactor

        Raises
        ------
        NameKeyError
            Occurs if the model with the same name was not found
            or was not fitted
        """
        self.__init__()
        if name_models is not None and only_fitted:
            if name_models in self.__names_fitted_models:
                name_models = [name_models]
            else:
                raise NameKeyError(
                    "A model with the same name was not found or was not fitted"
                )
        elif name_models is not None:
            if name_models in list(self.__models.keys()):
                name_models = [name_models]
            else:
                raise NameKeyError("There is no model with this name")
        else:
            if only_fitted:
                name_models = self.__names_fitted_models
            else:
                name_models = list(self.__models.keys())
        return [
            {
                "user_model_name": user_model_name,
                "type_model": self.__models[user_model_name].type_model,
                "params": self.get_params(user_model_name, all_params),
                "fitted": self.__models[user_model_name].fiited,
            }
            for user_model_name in name_models
        ]

    def init_new_model(
        self, type_model: str, user_model_name: str, params: dict = {}
    ):
        """
        Initialize one model and store it in memory

        Parameters
        ----------
        type_model : str
            Shortname of base model type
        user_model_name : str
            User's model name
        params : dict, optional
            User's params for model, by default {}

        Returns
        -------
        Dict
            Dictionary with user model name, type of model, params dict,
            fitting bool inidactor

        Raises
        ------
        AlreadyExistsError
            Occurs if a model with same name already exists

        NameKeyError
            Occurs if there is an error in model type or model name
        """
        self.__init__()
        if user_model_name in self.__models.keys():
            raise AlreadyExistsError(
                "A model with the same name already exists"
            )
        if type_model not in self.__available_model_types.keys():
            raise NameKeyError(
                "The selected model is not in the list of available ones"
            )
        self.__models[user_model_name] = Model(
            self.__available_model_types[type_model],
            type_model,
            user_model_name,
            params=params,
        )
        # joblib.dump(
        #     self.__models[user_model_name],
        #     self.PATH_MODELS + f"/{user_model_name}.pkl",
        # )
        Storage.model_dump_s3(self.__models[user_model_name], user_model_name)
        return {
            "user_model_name": user_model_name,
            "type_model": type_model,
            "params": params,
            "fitted": False,
        }

    def model_fit(self, X: np.array, y: np.array, user_model_name: str):
        """
        Model fitting

        Parameters
        ----------
        X : np.array
            Training data
        y : np.array
            Target data
        user_model_name : str
            Name of the model to be fitted

        Raises
        ------
        NameKeyError
            Occurs if a model with the same name was not found
        """
        try:
            self.__init__()
            self.__models[user_model_name].fit(X, y)
            self.__names_fitted_models.append(user_model_name)
            self.fitted = True
            Storage.model_dump_s3(
                self.__models[user_model_name], user_model_name
            )
            # joblib.dump(
            #     self.__models[user_model_name],
            #     self.PATH_MODELS + f"/{user_model_name}.pkl",
            # )
        except KeyError:
            raise NameKeyError("There is no model with this name")

    def model_predict(self, X: np.array, user_model_name: str):
        """
        Making prediction on data

        Parameters
        ----------
        X : np.array
            Data to predict on
        user_model_name : str
            Name of the model to use for prediction

        Returns
        -------
        np.array
            Predictions

        Raises
        ------
        NameKeyError
            Occurs if a model with the same name was not found
            or was not fitted
        """
        self.__init__()
        if user_model_name in self.__names_fitted_models:
            return self.__models[user_model_name].predict(X)
        else:
            raise NameKeyError(
                "A model with the same name was not found or was not fitted"
            )

    def get_params(self, user_model_name: str, all: bool = False) -> dict:
        """
        Getting params of model by name

        Parameters
        ----------
        user_model_name : str
            User model name
        all : bool, optional
            True if you want to get all params, False if you want to get
            params setted by user, by default False

        Returns
        -------
        dict
            Dict with params
        """
        self.__init__()
        return self.__models[user_model_name].get_params(all)

    def delete_model(self, user_model_name: str):
        """
        Deleting model by name

        Parameters
        ----------
        user_model_name : str
            Model name to delete

        Raises
        ------
        NameKeyError
            Occurs if there is no model with same name
        """
        try:
            self.__init__()
            del self.__models[user_model_name]
            if user_model_name in self.__names_fitted_models:
                self.__names_fitted_models.remove(user_model_name)
            # os.remove(os.path.join(self.PATH_MODELS, f"{user_model_name}.pkl"))
            Storage.del_model_s3(user_model_name)
        except KeyError:
            raise NameKeyError("There is no model with this name")


class Model:
    """
    Class describing one model

    Raises
    ------
    ParamsTypeError
        Error im params
    InvalidData
        Error in data
    """

    getting_params_func_names = {
        CatBoostClassifier: CatBoostClassifier.get_all_params,
        RandomForestClassifier: RandomForestClassifier.get_params,
    }

    def __init__(
        self, base_model, type_model: str, user_model_name: str, params: Dict
    ) -> None:
        """
        Constructs all the necessary attributes for the Model object.

        Parameters
        ----------
        base_model : _type_
            Base model class
        type_model : str
            Shortname for base model
        user_model_name : str
            User-specified model name
        params : Dict
            Params for model

        Raises
        ------
        ParamsTypeError
            Error in params
        """
        self.type_model: str = type_model
        self.params: Dict = params
        self.user_model_name: str = user_model_name
        self.fiited: bool = False
        try:
            self.base_model = base_model(**self.params)
        except TypeError:
            raise ParamsTypeError("Incorrect model hyperparameters passed")

    def fit(self, X: np.array, y: np.array):
        """
        Fit one model

        Parameters
        ----------
        X : np.array
            Train data
        y : np.array
            Target data

        Raises
        ------
        InvalidData
            Error in data
        """
        try:
            self.base_model.fit(X=X, y=y)
            self.fiited = True
        except:
            raise InvalidData("Incorrect training data")

    def predict(self, X: np.array):
        """
        Return prediction on X

        Parameters
        ----------
        X : np.array
            Data to predict on

        Raises
        ------
        InvalidData
            Error in data
        """
        try:
            return self.base_model.predict(X)
        except:
            raise InvalidData("Incorrect data for prediction")

    def get_params(self, all: bool = False) -> dict:
        """
        Getting params of model

        Parameters
        ----------
        all : bool, optional
            True if you want to get all params, False if you want to get
            params setted by user, by default False

        Returns
        -------
        dict
            Params dict
        """
        if all is True:
            return self.getting_params_func_names[type(self.base_model)](
                self.base_model
            )
        elif all is False:
            return self.params
