"""
Geração de Gráficos e Métricas para Conjuntos de Treino e Teste

Descrição
---------
Este programa processa arquivos contendo datasets e predições para os conjuntos de treino e teste, calculando métricas de desempenho de classificação. Ele gera gráficos (matriz de confusão, curvas ROC e PR) e salva relatórios resumidos com as principais métricas.

Funcionalidades
---------------
- Carrega datasets (.pkl) com features e labels.
- Carrega predições (.csv) contendo image_name, y_test e y_pred (scores).
- Calcula métricas: acurácia, F1-score, recall, precisão, AUC-ROC, AUC-PR, entre outras.
- Gera e salva gráficos de matriz de confusão, curva ROC e curva precision-recall.
- Salva resumo de métricas detalhado para cada conjunto (train e test).
- Exibe principais métricas no terminal.

Argumentos de Linha de Comando
------------------------------
--train_dataset : str
    Arquivo pickle do dataset de treino (features + labels).
--train_predictions : str
    Arquivo CSV com predições de treino.
--test_dataset : str
    Arquivo pickle do dataset de teste (features + labels).
--test_predictions : str
    Arquivo CSV com predições de teste.

Uso Exemplo
-----------
python postprocess_proj1.py --train_dataset train.pkl --train_predictions train_preds.csv --test_dataset test.pkl --test_predictions test_preds.csv
"""

import argparse
import os
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score
)
from typing import NoReturn
from utils.metrics import save_metrics_summary, plot_confusion_matrix, plot_roc_curve, plot_pr_curve

def process_split(name: str, dataset_path: str, prediction_path: str, output_dir: str) -> None:
    """
    Processa métricas, gera gráficos e salva relatórios para um conjunto de dados.

    Parâmetros
    ----------
    name : str
        Nome do conjunto ("train" ou "test").
    dataset_path : str
        Caminho para o arquivo pickle com dataset (features + labels).
    prediction_path : str
        Caminho para arquivo CSV com predições (image_name, y_test, y_pred).
    output_dir : str
        Diretório onde os gráficos e relatórios serão salvos.

    Retorna
    -------
    None
        Salva gráficos e relatórios no disco e imprime métricas no terminal.
    """
    print(f"\n[INFO] Processando métricas para: {name}")

    df_dataset: pd.DataFrame = pd.read_pickle(dataset_path)
    df_preds: pd.DataFrame = pd.read_csv(prediction_path)

    y_true = df_preds["y_test"]
    y_scores = df_preds["y_pred"]
    y_pred = (y_scores >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    # Salvar gráficos usando nomes específicos para cada split
    plot_confusion_matrix(cm, output_dir, filename=f"confusion_matrix_{name}.png")
    plot_roc_curve(fpr, tpr, roc_auc, output_dir, filename=f"roc_curve_{name}.png")
    plot_pr_curve(rec, prec, pr_auc, output_dir, filename=f"pr_curve_{name}.png")

    # Salvar resumo de métricas com prefixo para evitar sobrescrita
    save_metrics_summary(report, acc, f1, recall, precision, roc_auc, pr_auc, output_dir, prefix=name)

    # Exibir principais métricas no terminal
    print(f"[INFO] Acurácia:           {acc:.4f}")
    print(f"[INFO] F1-score:           {f1:.4f}")
    print(f"[INFO] Recall (Sensib.):   {recall:.4f}")
    print(f"[INFO] Precisão:           {precision:.4f}")
    print(f"[INFO] AUC-ROC:            {roc_auc:.4f}")
    print(f"[INFO] AUC-PR:             {pr_auc:.4f}")
    print(f"[INFO] Resultados salvos em: {output_dir}/ (arquivos iniciados com '{name}_')")


# Argumentos
parser = argparse.ArgumentParser(description="Gera gráficos e métricas para treino e teste.")

parser.add_argument(
    "--train_dataset",
    type=str,
    default="artifacts/preprocessed_dataset/train_dataset.pkl",
    help="Arquivo .pkl com dataset de treino (features + labels)."
)

parser.add_argument(
    "--train_predictions",
    type=str,
    default="artifacts/results/train_predictions.csv",
    help="Arquivo .csv com predições do treino."
)

parser.add_argument(
    "--test_dataset",
    type=str,
    default="artifacts/preprocessed_dataset/test_dataset.pkl",
    help="Arquivo .pkl com dataset de teste (features + labels)."
)

parser.add_argument(
    "--test_predictions",
    type=str,
    default="artifacts/results/test_predictions.csv",
    help="Arquivo .csv com predições do teste."
)

args = parser.parse_args()

# Criar pasta de saída
output_dir = "artifacts/postprocess"
os.makedirs(output_dir, exist_ok=True)

# Processar métricas dos dois conjuntos
process_split("train", args.train_dataset, args.train_predictions, output_dir)
process_split("test", args.test_dataset, args.test_predictions, output_dir)
