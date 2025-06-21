"""
Otimização de Hiperparâmetros para Modelo LightGBM com Optuna

Descrição
---------
Este módulo realiza a busca e otimização de hiperparâmetros para um modelo LightGBM de classificação binária,
utilizando Optuna para maximizar a métrica AUC-ROC no conjunto de validação.

Funcionalidades
---------------
- Define uma função objetivo para o estudo Optuna que treina um modelo LightGBM com parâmetros sugeridos.
- Executa a busca de hiperparâmetros em um número configurável de trials.
- Retorna os melhores parâmetros encontrados, combinados com parâmetros fixos definidos.
- Garante reprodutibilidade com semente fixa global.

Funções
--------
objective(trial: optuna.trial.Trial, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> float
    Função objetivo que treina o modelo e retorna o AUC no conjunto de validação para cada trial.

tune_lgbm_hyperparameters(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, n_trials: int = 50) -> dict
    Executa a otimização dos hiperparâmetros usando Optuna e retorna os melhores parâmetros encontrados.
"""

import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import numpy as np
import random

# Semente global para reprodutibilidade
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Parâmetros fixos para o modelo LightGBM
FIXED_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'is_unbalance': True,
    'seed': SEED  # Garante reprodutibilidade no LightGBM
}

def objective(trial: optuna.trial.Trial, 
              X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> float:
    """
    Função objetivo para otimização dos hiperparâmetros do LightGBM.

    Parâmetros
    ----------
    trial : optuna.trial.Trial
        Objeto do trial fornecido pelo Optuna para sugestão dos hiperparâmetros.
    X_train : np.ndarray
        Dados de treino (features).
    y_train : np.ndarray
        Rótulos de treino.
    X_val : np.ndarray
        Dados de validação (features).
    y_val : np.ndarray
        Rótulos de validação.

    Retorna
    -------
    float
        AUC-ROC obtida no conjunto de validação com os parâmetros do trial.
    """
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 15, 150),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 15.0),
    }
    param.update(FIXED_PARAMS)

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    gbm = lgb.train(
        param,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(0)
        ]
    )

    preds = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    auc = roc_auc_score(y_val, preds)
    return auc

def tune_lgbm_hyperparameters(X_train: np.ndarray, y_train: np.ndarray, 
                              X_val: np.ndarray, y_val: np.ndarray, 
                              n_trials: int = 50) -> dict:
    """
    Executa a otimização dos hiperparâmetros do LightGBM usando Optuna.

    Parâmetros
    ----------
    X_train : np.ndarray
        Dados de treino (features).
    y_train : np.ndarray
        Rótulos de treino.
    X_val : np.ndarray
        Dados de validação (features).
    y_val : np.ndarray
        Rótulos de validação.
    n_trials : int, opcional
        Número de trials para a busca (default é 50).

    Retorna
    -------
    dict
        Dicionário com os melhores hiperparâmetros encontrados, incluindo os parâmetros fixos.
    """
    sampler = optuna.samplers.TPESampler(seed=SEED)  # Amostragem reprodutível
    study = optuna.create_study(direction='maximize', sampler=sampler)

    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)

    best_params = study.best_params
    best_params.update(FIXED_PARAMS)

    print(f"[INFO] Melhor AUC: {study.best_value:.4f}")
    print(f"[INFO] Melhores parâmetros (com fixos): {best_params}")

    return best_params
