from __future__ import annotations

from typing import List

import numpy as np

from descents import BaseDescent
from descents import get_descent


class LinearRegression:
    """
    Класс линейной регрессии.

    Parameters
    ----------
    descent_config : dict
        Конфигурация градиентного спуска.
    tolerance : float, optional
        Критерий остановки для квадрата евклидова нормы разности весов. По умолчанию равен 1e-4.
    max_iter : int, optional
        Критерий остановки по количеству итераций. По умолчанию равен 300.

    Attributes
    ----------
    descent : BaseDescent
        Экземпляр класса, реализующего градиентный спуск.
    tolerance : float
        Критерий остановки для квадрата евклидова нормы разности весов.
    max_iter : int
        Критерий остановки по количеству итераций.
    loss_history : List[float]
        История значений функции потерь на каждой итерации.

    """

    def __init__(self, descent_config: dict, tolerance: float = 1e-4, max_iter: int = 300):
        """
        :param descent_config: gradient descent config
        :param tolerance: stopping criterion for square of euclidean norm of weight difference (float)
        :param max_iter: stopping criterion for iterations (int)
        """
        self.descent: BaseDescent = get_descent(descent_config)

        self.tolerance: float = tolerance
        self.max_iter: int = max_iter

        self.loss_history: List[float] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRegression:
        """
        Обучение модели линейной регрессии, подбор весов для наборов данных x и y.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        self : LinearRegression
            Возвращает экземпляр класса с обученными весами.

        """
        self.loss_history.append(self.calc_loss(x, y))

        for _ in range(self.max_iter):
            gradient = self.descent.calc_gradient(x, y)
            weight_update = self.descent.update_weights(gradient)

            # Отслеживание процесса сходимости
            self.loss_history.append(self.calc_loss(x, y))

            # Проверка на сходимость
            if np.linalg.norm(weight_update) < self.tolerance:
                print("Обучение остановлено: Евклидова норма разности векторов между двумя последовательными итерациями меньше заданного порога")
                break
            if np.any(np.isnan(self.descent.w)):
                print("Обучение остановлено: в векторе весов появился NaN")
                break
        
        print("Обучение остановлено: достигнуто максимальное количество итераций max_iter")
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Прогнозирование целевых переменных для набора данных x.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.

        Returns
        -------
        prediction : np.ndarray
            Массив прогнозируемых значений.
        """
        return self.descent.predict(x)

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Расчёт значения функции потерь для наборов данных x и y.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        loss : float
            Значение функции потерь.
        """
        return self.descent.calc_loss(x, y)
