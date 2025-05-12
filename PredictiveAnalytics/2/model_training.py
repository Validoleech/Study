from datetime import datetime
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate_model(df_for_model):
    df = df_for_model.dropna()

    # Определение целевой переменной и признаков
    df = df.drop(columns=['Unnamed: 0'])
    target = 'raw_mix.lab.measure.sito_009'
    features = df.columns[df.columns != target]

    # Разделение данных
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # Скалирование данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Построение модели
    model = LinearRegression()
    
    model.fit(X_train_scaled, y_train)

    # Оценка модели
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mae_baseline = mean_absolute_error(y_test, np.full_like(y_test, y_test.mean()))
    mae_baseline_shift = mean_absolute_error(y_test[:-1], y_test[1:])

    # Вычисление разностей для реальных и предсказанных значений
    y_test_diff = y_test[1:].values - y_test[:-1].values
    y_pred_diff = y_pred[1:] - y_pred[:-1]
    # Подсчет доли сонаправленных изменений
    same_direction = np.sum((y_test_diff * y_pred_diff) > 0) / len(y_test_diff)
    

    print(f'Доля сонаправленных изменений: {same_direction:.3f}')
    print(f'Mean Absolute Error Baseline: {mae_baseline}')
    print(f'Mean Absolute Error Baseline shifted: {mae_baseline_shift}')
    print(f'Mean Absolute Error: {mae}')
    # print(f'Mean Squared Error: {mse}')
    # Сохранение результатов в файл    
    
    # Формируем словарь с результатами
    results = {
        'features': list(features),
        'metrics': {
            'mae': float(mae),
            'mae_baseline': float(mae_baseline), 
            'mae_baseline_shift': float(mae_baseline_shift),
            'same_direction_ratio': float(same_direction),
            'mse': float(mse)
        },
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    
    # Сохраняем результаты в JSON файл
    filename = f'../results/metrics/result_{results["timestamp"]}.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # Визуализация результатов
    # plt.scatter(y_test, y_pred)
    # plt.xlabel('True Values')
    # plt.ylabel('Predictions')
    # plt.title('True vs Predicted Values')
    # plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    # plt.show()

    # fig, ax = plt.subplots(figsize=(20, 5))
    # y_train.plot(ax=ax, label='y_train')
    # pd.DataFrame(model.predict(X_train_scaled), index=y_train.index, columns=['train_pred']).plot(ax=ax)
    # y_test.plot(ax=ax, label='y_test')
    # pd.DataFrame(y_pred, index=y_test.index, columns=['test_pred']).plot(ax=ax)
    # ax.legend()
    # plt.show()

    # Визуализация важности признаков для линейной регрессии
    feature_importances = model.coef_  # Получаем коэффициенты модели
    indices = np.argsort(feature_importances)[::-1]  # Сортируем по важности

    plt.figure(figsize=(12, 6))
    plt.title('Feature Importances (Linear Regression)')
    plt.bar(range(X.shape[1]), feature_importances[indices], align='center')
    plt.xticks(range(X.shape[1]), features[indices], rotation=45, ha='right')  # Угол поворота и выравнивание
    plt.xlim([-1, X.shape[1]])
    plt.ylabel('Coefficient Value')  # Подпись оси Y
    plt.tight_layout()  # Автоматическая подгонка элементов графика
    plt.show()  # Отображение графика

    # # Визуализация распределения признаков и целевой переменной
    # for feature in features:
    #     plt.figure(figsize=(12, 6))
    #     sns.kdeplot(data=X_train[feature], label='Train', color='blue', fill=True, alpha=0.5)
    #     sns.kdeplot(data=X_test[feature], label='Test', color='orange', fill=True, alpha=0.5)
    #     plt.title(f'Distribution of {feature}')
    #     plt.xlabel(feature)
    #     plt.ylabel('Density')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()

    # # График распределения целевой переменной
    # plt.figure(figsize=(12, 6))
    # sns.kdeplot(data=y_train, label='Train', color='blue', fill=True, alpha=0.5)
    # sns.kdeplot(data=y_test, label='Test', color='orange', fill=True, alpha=0.5)
    # plt.title('Distribution of Target Variable')
    # plt.xlabel('Target Variable')
    # plt.ylabel('Density')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    # Загрузка данных из CSV файла
    data_path = './data/processed/mart.csv'
    df_for_model = pd.read_csv(data_path)
    
    # Запуск функции
    train_and_evaluate_model(df_for_model)