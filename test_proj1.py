"""
Execução de Teste com Modelo LightGBM Treinado

Descrição
---------
Este programa carrega um dataset de teste pré-processado, aplica normalização usando um scaler previamente salvo, carrega um modelo LightGBM treinado e gera predições para o conjunto de teste. As predições são salvas em um arquivo CSV em um diretório configurável.

Funcionalidades
---------------
- Carrega dataset de teste em formato pickle contendo colunas: image_name, label, features.
- Aplica transformação de normalização usando StandardScaler previamente ajustado.
- Carrega modelo LightGBM treinado salvo em arquivo pickle.
- Gera predições probabilísticas para as amostras do conjunto de teste.
- Salva um arquivo CSV com as colunas: image_name, y_test (rótulo verdadeiro), y_pred (score previsto).

Argumentos de Linha de Comando
------------------------------
--test_dataset : str
    Arquivo pickle com dataset de teste (features + labels).
--scaler_path : str
    Caminho para scaler StandardScaler salvo (.pkl).
--model_path : str
    Caminho para modelo LightGBM treinado salvo (.pkl).
--output_dir : str, opcional
    Diretório para salvar predições (default: artifacts/results).

Uso Exemplo
-----------
python test_proj1.py --test_dataset caminho/teste.pkl --scaler_path caminho/scaler.pkl --model_path caminho/modelo.pkl --output_dir artifacts/results
"""

import argparse
import os
import pandas as pd
import joblib
from typing import NoReturn

# Parsing dos argumentos da linha de comando
parser = argparse.ArgumentParser(description="Executa teste no modelo LightGBM treinado.")

parser.add_argument(
    "--test_dataset",
    type=str,
    default="artifacts/preprocessed_dataset/test_dataset.pkl",
    help="Arquivo .pkl com dataset de teste (features + labels)."
)

parser.add_argument(
    "--scaler_path",
    type=str,
    default="artifacts/scaler/standard_scaler.pkl",
    help="Caminho para o scaler salvo (StandardScaler .pkl)."
)

parser.add_argument(
    "--model_path",
    type=str,
    default="artifacts/lightgbm/lightgbm_model.pkl",
    help="Caminho para o modelo LightGBM salvo (.pkl)."
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="artifacts/results",
    help="Diretório onde salvar as predições."
)

args = parser.parse_args()

print("[INFO] Carregando dataset de teste...")
df_test: pd.DataFrame = pd.read_pickle(args.test_dataset)
X_test: pd.DataFrame = pd.DataFrame(df_test["features"].tolist())
y_test: pd.Series = df_test["label"].astype(int)

print("[INFO] Carregando scaler e aplicando normalização...")
scaler = joblib.load(args.scaler_path)
X_test_scaled = scaler.transform(X_test)

print("[INFO] Carregando modelo treinado e gerando predições...")
model = joblib.load(args.model_path)
y_pred = model.predict(X_test_scaled)

print("[INFO] Salvando predições em CSV...")
os.makedirs(args.output_dir, exist_ok=True)
output_df: pd.DataFrame = pd.DataFrame({
    "image_name": df_test["image_name"],
    "y_test": y_test,
    "y_pred": y_pred
})
output_path: str = os.path.join(args.output_dir, "test_predictions.csv")
output_df.to_csv(output_path, index=False)

print(f"[INFO] Teste concluído. Resultados salvos em: {output_path}")
