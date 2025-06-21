"""
Treinamento de Modelo LightGBM com Otimização de Hiperparâmetros via Optuna

Descrição
---------
Este programa realiza o treinamento de um modelo LightGBM para classificação binária, utilizando datasets de treino e validação pré-processados. Ele busca os melhores hiperparâmetros com Optuna e salva o modelo treinado, o scaler para normalização e as predições no conjunto de treino.

Funcionalidades
---------------
- Carrega datasets pré-processados no formato pickle contendo colunas: image_name, label, features.
- Aplica normalização dos dados com StandardScaler, ajustado no treino e aplicado na validação.
- Executa busca de hiperparâmetros para LightGBM via Optuna, com número configurável de trials.
- Treina o modelo final com os melhores parâmetros encontrados e semente fixa para reprodutibilidade.
- Salva scaler, modelo treinado e predições de treino em arquivos organizados em pastas.
- Gera arquivo CSV com predições de treino (nome da imagem, label real, score predito).

Argumentos de Linha de Comando
------------------------------
--train_dataset : str
    Caminho para o arquivo pickle contendo o dataset de treino (features + labels).
--val_dataset : str
    Caminho para o arquivo pickle contendo o dataset de validação (features + labels).
--output_dir : str, opcional
    Diretório base para salvar scaler, modelo e resultados (default: artifacts).
--tune_trials : int, opcional
    Número de trials para busca de hiperparâmetros com Optuna (default: 50).

Uso Exemplo
-----------
python train_proj1.py --train_dataset caminho/treino.pkl --val_dataset caminho/val.pkl --output_dir artifacts --tune_trials 50
"""

import argparse
import os
import pandas as pd
import joblib
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from utils.hyperparameter_tuning import tune_lgbm_hyperparameters

# Semente global para reprodutibilidade
SEED: int = 42
np.random.seed(SEED)
random.seed(SEED)

# Parsing dos argumentos da linha de comando
parser = argparse.ArgumentParser(description="Treina modelo LightGBM com Optuna.")

parser.add_argument(
    "--train_dataset",
    type=str,
    default="artifacts/preprocessed_dataset/train_dataset.pkl",
    help="Arquivo .pkl com dados de treino."
)
parser.add_argument(
    "--val_dataset",
    type=str,
    default="artifacts/preprocessed_dataset/val_dataset.pkl",
    help="Arquivo .pkl com dados de validação."
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="artifacts",
    help="Diretório base para salvar artefatos."
)

parser.add_argument(
    "--tune_trials",
    type=int,
    default=50,
    help="Número de trials para busca com Optuna."
)

args = parser.parse_args()

# Criar diretórios para salvar artefatos
scaler_dir: str = os.path.join(args.output_dir, "scaler")
model_dir: str = os.path.join(args.output_dir, "lightgbm")
results_dir: str = os.path.join(args.output_dir, "results")

os.makedirs(scaler_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Carregar datasets pré-processados
df_train: pd.DataFrame = pd.read_pickle(args.train_dataset)
df_val: pd.DataFrame = pd.read_pickle(args.val_dataset)

X_train: pd.DataFrame = pd.DataFrame(df_train["features"].tolist())
y_train: pd.Series = df_train["label"].astype(int)
X_val: pd.DataFrame = pd.DataFrame(df_val["features"].tolist())
y_val: pd.Series = df_val["label"].astype(int)

# Normalização dos dados
scaler: StandardScaler = StandardScaler()
X_train_scaled: np.ndarray = scaler.fit_transform(X_train)
X_val_scaled: np.ndarray = scaler.transform(X_val)

# Salvar scaler
scaler_path: str = os.path.join(scaler_dir, "standard_scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"[INFO] Scaler salvo em {scaler_path}")

# Busca dos melhores hiperparâmetros com Optuna
final_params: dict = tune_lgbm_hyperparameters(X_train_scaled, y_train, X_val_scaled, y_val, n_trials=args.tune_trials)
final_params["seed"] = SEED

# Treinamento do modelo LightGBM com os melhores parâmetros
train_data: lgb.Dataset = lgb.Dataset(X_train_scaled, label=y_train)
model: lgb.Booster = lgb.train(
    final_params,
    train_data,
    num_boost_round=10000,
    callbacks=[lgb.log_evaluation(100)]
)

# Salvar modelo treinado
model_path: str = os.path.join(model_dir, "lightgbm_model.pkl")
joblib.dump(model, model_path)
print(f"[INFO] Modelo salvo em {model_path}")

# Gerar predições no conjunto de treino
y_train_pred: np.ndarray = model.predict(X_train_scaled)
train_preds_df: pd.DataFrame = pd.DataFrame({
    "image_name": df_train["image_name"],
    "y_test": y_train,
    "y_pred": y_train_pred
})

# Salvar predições em CSV
train_pred_path: str = os.path.join(results_dir, "train_predictions.csv")
train_preds_df.to_csv(train_pred_path, index=False)
print(f"[INFO] Predições de treino salvas em {train_pred_path}")
