from datetime import datetime
import json
import os
import joblib
import logging
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import (train_test_split, TimeSeriesSplit,
                                     RandomizedSearchCV, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Вспомогательные классы и функции

os.environ["LOKY_MAX_CPU_COUNT"] = "16"

class BlockedTimeSeriesSplit:
    def __init__(self, n_splits=5, gap=0):
        self.n_splits, self.gap = n_splits, gap

    def get_n_splits(self, *_, **kwargs):
        return self.n_splits

    def split(self, X, *_, **kwargs):
        n = len(X)
        fold = (n - self.gap) // (self.n_splits + 1)
        for i in range(self.n_splits):
            st = (i + 1) * fold
            en = st + fold
            train = np.arange(0, st - self.gap)
            test = np.arange(st, en)
            yield train, test
# ----------------------------------------------------------------


def make_preproc(num_cols, cat_cols, for_linear: bool):
    num_tr = Pipeline(
        [('imp', SimpleImputer(strategy='median')),
         ('sc',  StandardScaler())
         ]) if for_linear else SimpleImputer(strategy='median')
    ct = ColumnTransformer(
        [('num', num_tr, num_cols),
         ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), cat_cols)],
        sparse_threshold=0.3, remainder='drop'
    )
    ct.set_output(transform="pandas")
    return ct
# ----------------------------------------------------------------


def train_better_model(
    df_for_model: pd.DataFrame,
    target: str = 'raw_mix.lab.measure.sito_009',
    test_size: float = .2,
    random_state: int = 42,
    n_splits: int = 5,
    max_lag: int = 3,
    rolling_windows: tuple = (3, 6),
    gap: int = 0,
    results_dir: str = './results/metrics',
    ensemble_type: str = 'weighted_mean',     # 'mean' | 'weighted_mean'
    cv_override: str = 'blk' # 'blk' | 'exp'
):
    os.makedirs(results_dir, exist_ok=True)

    # -------------- Логирование --------------
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(
                results_dir, 'main.log'), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    log = logging.getLogger(__name__)
    log.info('=========== NEW RUN===========')

    # -------------- Подготовка данных --------------
    df = df_for_model.drop(
        columns=[c for c in ['Unnamed: 0', 'index'] if c in df_for_model]).copy()

    num_orig = df.select_dtypes('number').columns.difference([target]).tolist()

    # -------------- Добавление лагов и роллов --------------
    def add_lags_rolls(dset, cols):
        out = dset.copy()
        for col in cols:
            for lag in range(1, max_lag + 1):
                out[f'{col}_lag{lag}'] = out[col].shift(lag)
            for win in rolling_windows:
                out[f'{col}_roll{win}_mean'] = out[col].rolling(win).mean()
                out[f'{col}_roll{win}_std'] = out[col].rolling(win).std()
        return out

    df = add_lags_rolls(df, num_orig).dropna().reset_index(drop=True)

    X, y = df.drop(columns=[target]), df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False)

    num_cols = X.select_dtypes('number').columns.tolist()
    cat_cols = X.select_dtypes(exclude='number').columns.tolist()

    # -------------- Выбор кросс-валидации --------------
    tscv_exp = TimeSeriesSplit(n_splits=n_splits)
    tscv_blk = BlockedTimeSeriesSplit(n_splits=n_splits, gap=gap)

    baseline = Pipeline([
        ('prep', make_preproc(num_cols, cat_cols, True)),
        ('model', Ridge(alpha=1.0))
    ])
    cv_mae_exp = -cross_val_score(baseline, X_train, y_train,
                                  cv=tscv_exp,
                                  scoring='neg_mean_absolute_error',
                                  n_jobs=-1).mean()
    cv_mae_blk = -cross_val_score(baseline, X_train, y_train,
                                  cv=tscv_blk,
                                  scoring='neg_mean_absolute_error',
                                  n_jobs=-1).mean()
    
    logging.info(f'CV-MAE expanding = {cv_mae_exp:.4f}')
    logging.info(f'CV-MAE blocked   = {cv_mae_blk:.4f}')
    # cv = tscv_blk if abs(cv_mae_exp - cv_mae_blk) / \
    #     cv_mae_blk > cv_threshold else tscv_exp

    cv = (tscv_blk if cv_override == 'blk' else
          tscv_exp if cv_override == 'exp' else
          tscv_exp if cv_mae_exp > cv_mae_blk else tscv_blk)

    log.info(f"CV used: {'Blocked' if cv is tscv_blk else 'Expanding'}")

    # -------------- Гиперпараметры --------------
    search_space = {
        'ridge': (
            Pipeline([
                ('prep', make_preproc(num_cols, cat_cols, True)),
                ('model', Ridge(random_state=random_state))
            ]),
            {'model__alpha': [0.1, 1.0, 10]}
        ),
        'rf': (
            Pipeline([
                ('prep', make_preproc(num_cols, cat_cols, False)),
                ('model', RandomForestRegressor(
                    n_jobs=-1,
                    random_state=random_state))
            ]),
            {'model__n_estimators': [400, 800],
             'model__max_depth': [None, 10, 20]}
        ),
        'lgb': (
            Pipeline([
                ('prep', make_preproc(num_cols, cat_cols, False)),
                ('model', LGBMRegressor(objective='mae',
                                        device_type='gpu',
                                        verbosity=-1,
                                        # force_row_wise=True,
                                        random_state=random_state))
            ]),
            dict(model__n_estimators=[200, 400, 800, 1200],
                 model__learning_rate=np.linspace(.03, .1, 8),
                 model__num_leaves=[31, 63, 127],
                 model__feature_fraction=[.7, .8, .9],
                 model__bagging_fraction=[.7, .8, .9],
                 model__bagging_freq=[1])
        ),

        'xgb': (
            Pipeline([
                ('prep', make_preproc(num_cols, cat_cols, False)),
                ('model', XGBRegressor(objective='reg:squarederror',
                                       eval_metric='mae',
                                       tree_method='hist',
                                       random_state=random_state))
            ]),
            dict(model__n_estimators=[200, 400, 800, 1200],
                 model__learning_rate=np.linspace(.03, .1, 8),
                 model__max_depth=[4, 6],
                 model__colsample_bytree=[.7, .8, .9],
                 model__subsample=[.7, .8, .9])
        ),

        'cat': (
            Pipeline([
                ('prep', make_preproc(num_cols, cat_cols, False)),
                ('model', CatBoostRegressor(loss_function='MAE',
                                            verbose=False,
                                            allow_writing_files=False,
                                            random_state=random_state))
            ]),
            dict(model__iterations=[200, 400, 800, 1200],
                 model__learning_rate=np.linspace(.03, .1, 8),
                 model__depth=[4, 6])
        )
    }

    # -------------- Обучение --------------
    model_results, preds_dict = {}, {}

    for name, (pipe, param_dist) in search_space.items():
        log.info(f'Fitting {name}')
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=30,
            cv=cv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=0,
            random_state=random_state
        )
        search.fit(X_train, y_train)

        best_pipe = search.best_estimator_
        y_pred = best_pipe.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        cv_mae = -search.best_score_
        same_dir = float(((y_test.values[1:] - y_test.values[:-1]) *
                          (y_pred[1:] - y_pred[:-1]) > 0).mean())

        model_results[name] = dict(
            cv_mae=cv_mae,
            best_params=search.best_params_,
            full_params=best_pipe.named_steps['model'].get_params(),
            metrics=dict(mae=mae, mse=mse,
                         same_direction_ratio=same_dir)
        )
        preds_dict[name] = y_pred
        joblib.dump(best_pipe, os.path.join(results_dir, f'{name}_model.pkl'))
        log.info(f'{name}: CV-MAE={cv_mae:.4f} | test MAE={mae:.4f}')

    # -------------- Ансамбль --------------
    if ensemble_type == 'mean':
        ens_pred = np.column_stack(list(preds_dict.values())).mean(axis=1)
    else:
        weights = np.array([1 / model_results[m]['cv_mae']
                           for m in preds_dict])
        ens_pred = (np.column_stack(list(preds_dict.values())) *
                    (weights / weights.sum())).sum(axis=1)

    mae_ens = mean_absolute_error(y_test, ens_pred)
    mse_ens = mean_squared_error(y_test, ens_pred)
    same_dir = float(((y_test.values[1:] - y_test.values[:-1]) *
                      (ens_pred[1:] - ens_pred[:-1]) > 0).mean())
    log.info(f'ENSEMBLE {ensemble_type}: test MAE={mae_ens:.4f}')

    # -------------- Результаты --------------
    results = dict(
        cv_strategy=('Blocked' if cv is tscv_blk else 'Expanding'),
        models=model_results,
        ensemble=dict(method=ensemble_type,
                      used_models=list(preds_dict),
                      metrics=dict(mae=mae_ens, mse=mse_ens,
                                   same_direction_ratio=same_dir)),
        features=list(X.columns),
        timestamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    json_path = os.path.join(
        results_dir, f"result_{results['timestamp']}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    log.info(f'Results saved → {json_path}')

    return results


# ----------------------------------------------------------------
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    df = pd.read_csv('./data/processed/mart.csv')
    train_better_model(df)
