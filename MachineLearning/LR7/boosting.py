from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        """
        Обучает новую базовую модель и добавляет ее в ансамбль.

        Параметры
        ----------
        x : array-like, форма (n_samples, n_features)
            Массив признаков для набора данных.
        y : array-like, форма (n_samples,)
            Массив целевых значений.
        predictions : array-like, форма (n_samples,)
            Предсказания текущего ансамбля.

        Примечания
        ----------
        Эта функция добавляет новую модель и обновляет ансамбль.
        """
        # self.gammas.append()
        # self.models.append()

        n_samples = x.shape[0]
        bootstrap_indices = np.random.choice(
            n_samples, size=int(n_samples * self.subsample), replace=True
        )
        x_bootstrap, y_bootstrap = x[bootstrap_indices], y[bootstrap_indices]
        residuals = y_bootstrap - predictions[bootstrap_indices]

        new_model = self.base_model_class(**self.base_model_params)
        new_model.fit(x_bootstrap, residuals)

        def gamma_objective(gamma):
            return self.loss_fn(y_bootstrap, predictions[bootstrap_indices] + gamma * new_model.predict(x_bootstrap))

        gammas = np.linspace(-1, 1, 100)
        losses = [gamma_objective(gamma) for gamma in gammas]
        best_gamma = gammas[np.argmin(losses)]

        self.gammas.append(best_gamma * self.learning_rate)
        self.models.append(new_model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        Обучает модель на тренировочном наборе данных и выполняет валидацию на валидационном наборе.

        Параметры
        ----------
        x_train : array-like, форма (n_samples, n_features)
            Массив признаков для тренировочного набора.
        y_train : array-like, форма (n_samples,)
            Массив целевых значений для тренировочного набора.
        x_valid : array-like, форма (n_samples, n_features)
            Массив признаков для валидационного набора.
        y_valid : array-like, форма (n_samples,)
            Массив целевых значений для валидационного набора.
        """
        train_predictions = np.zeros(len(y_train))
        valid_predictions = np.zeros(len(y_valid))

        if self.early_stopping_rounds is not None:
            best_valid_loss = np.inf
            best_iteration = 0

        for i in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)

            new_model_prediction_train = self.models[-1].predict(x_train)
            new_model_prediction_valid = self.models[-1].predict(x_valid)

            train_predictions += self.gammas[-1] * new_model_prediction_train
            valid_predictions += self.gammas[-1] * new_model_prediction_valid

            train_loss = self.loss_fn(y_train, train_predictions)
            valid_loss = self.loss_fn(y_valid, valid_predictions)

            self.history['train_loss'].append(train_loss)
            self.history['valid_loss'].append(valid_loss)

            print(f"Iteration {i + 1}: Train Loss = {train_loss:.4f}; Validation Loss = {valid_loss:.4f}")

            if self.early_stopping_rounds is not None:
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_iteration = i
                elif i - best_iteration > self.early_stopping_rounds:
                    print(f"Early stopping at iteration {i}")
                    break

        if self.plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(self.history['train_loss'], label='Train Loss')
            plt.plot(self.history['valid_loss'], label='Validation Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.show()

    def predict_proba(self, x):
        """
        Вычисляет вероятности принадлежности классу для каждого образца.

        Параметры
        ----------
        x : array-like, форма (n_samples, n_features)
            Массив признаков для набора данных.

        Возвращает
        ----------
        probabilities : array-like, форма (n_samples, n_classes)
            Вероятности для каждого класса.
        """
        predictions = np.zeros(x.shape[0])
        for model, gamma in zip(self.models, self.gammas):
            predictions += gamma * model.predict(x)
        return np.vstack([1 - self.sigmoid(predictions), self.sigmoid(predictions)]).T

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        """
        Находит оптимальное значение гаммы для минимизации функции потерь.

        Параметры
        ----------
        y : array-like, форма (n_samples,)
            Целевые значения.
        old_predictions : array-like, форма (n_samples,)
            Предыдущие предсказания ансамбля.
        new_predictions : array-like, форма (n_samples,)
            Новые предсказания базовой модели.

        Возвращает
        ----------
        gamma : float
            Оптимальное значение гаммы.

        Примечания
        ----------
        Значение гаммы определяется путем минимизации функции потерь.
        """
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        """
        Возвращает важность признаков в обученной модели.

        Возвращает
        ----------
        importances : array-like, форма (n_features,)
            Важность каждого признака.

        Примечания
        ----------
        Важность признаков определяется по вкладу каждого признака в финальную модель.
        """
        if not self.models:
            return None

        n_features = self.models[0].n_features_in_
        importances = np.zeros(n_features)
        for model in self.models:
            importances += model.feature_importances_

        importances /= len(self.models)
        importances /= importances.sum()
        return importances
