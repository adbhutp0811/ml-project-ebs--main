import os
import sys
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys) from e


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():

            # Get hyperparameters for the current model
            para = param.get(model_name, {})

            # Perform hyperparameter tuning
            gs = GridSearchCV(
                estimator=model,
                param_grid=para,
                cv=3,
                scoring="r2",
                n_jobs=-1
            )

            gs.fit(X_train, y_train)

            # Get the best model from GridSearchCV
            best_model = gs.best_estimator_

            # Make predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Calculate R2 scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store test score
            report[model_name] = test_model_score

            print(f"Model: {model_name}")
            print(f"Best Parameters: {gs.best_params_}")
            print(f"Training R2 Score: {train_model_score:.4f}")
            print(f"Testing R2 Score: {test_model_score:.4f}")
            print("-" * 50)

        return report

    except Exception as e:
        raise CustomException(e, sys) from e


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys) from e