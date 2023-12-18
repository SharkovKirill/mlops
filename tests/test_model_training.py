import pytest
from model_training.model_training import ModelFactory, Model
from model_training.model_training import Storage
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from model_training.exceptions import ParamsTypeError


def test_get_models(mocker):
    allmodels = ModelFactory()
    mocker.patch(
        "model_training.model_training.Storage.reload_models_from_s3",
        return_value={
            "cb_test": Model(
                type_model="cb",
                user_model_name="cb_test",
                base_model=CatBoostClassifier,
                params={"random_state": 42, "n_estimators": 101},
            ),
            "rf_test": Model(
                type_model="rf",
                user_model_name="rf_test",
                base_model=RandomForestClassifier,
                params={"random_state": 42, "n_estimators": 102},
            ),
        },
    )
    assert [model["user_model_name"] for model in allmodels.get_models()] == [
        "cb_test",
        "rf_test",
    ]


def test_model_init_params_type_error():
    with pytest.raises(
        ParamsTypeError, match="Incorrect model hyperparameters passed"
    ):
        Model(
            type_model="rf",
            user_model_name="rf_test",
            base_model=RandomForestClassifier,
            params={"random_stateeeeeeeeeeee": 42, "n_estimators": 102},
        )


def test_get_params():
    test_model = Model(
        type_model="rf",
        user_model_name="rf_test",
        base_model=RandomForestClassifier,
        params={"random_state": 42, "n_estimators": 102},
    )
    assert test_model.get_params() == {"random_state": 42, "n_estimators": 102}
