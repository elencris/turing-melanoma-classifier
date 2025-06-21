"""
Visualização e Salvamento de Métricas de Avaliação de Modelos de Classificação

Descrição
---------
Este módulo contém funções para salvar um resumo das métricas de avaliação e gerar gráficos
para matriz de confusão, curva ROC e curva Precision-Recall, salvando-os em arquivos PNG.

Funções
--------
save_metrics_summary(report: dict, accuracy: float, f1: float, recall: float, precision: float, roc_auc: float, pr_auc: float, output_dir: str, prefix: str) -> None
    Salva resumo das métricas em arquivo JSON com prefixo para diferenciar splits.

plot_confusion_matrix(cm: np.ndarray, output_dir: str, filename: str) -> None
    Plota e salva a matriz de confusão como imagem.

plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, output_dir: str, filename: str) -> None
    Plota e salva a curva ROC como imagem.

plot_pr_curve(recall: np.ndarray, precision: np.ndarray, pr_auc: float, output_dir: str, filename: str) -> None
    Plota e salva a curva Precision-Recall como imagem.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def save_metrics_summary(report: dict, accuracy: float, f1: float, recall: float, precision: float, roc_auc: float, pr_auc: float, output_dir: str, prefix: str) -> None:
    """
    Salva resumo das métricas de avaliação em arquivo JSON.

    Parâmetros
    ----------
    report : dict
        Relatório completo da classificação.
    accuracy : float
        Acurácia do modelo.
    f1 : float
        F1-score.
    recall : float
        Recall.
    precision : float
        Precisão.
    roc_auc : float
        Área sob a curva ROC.
    pr_auc : float
        Área sob a curva Precision-Recall.
    output_dir : str
        Diretório onde o arquivo será salvo.
    prefix : str
        Prefixo para o nome do arquivo (ex: 'train' ou 'test').
    """
    summary = {
        "accuracy": accuracy,
        "f1_score": f1,
        "recall": recall,
        "precision": precision,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "classification_report": report
    }
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"metrics_summary_{prefix}.json")
    with open(filename, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"[INFO] Métricas salvas em {filename}")

def plot_confusion_matrix(cm: np.ndarray, output_dir: str, filename: str) -> None:
    """
    Plota e salva a matriz de confusão como imagem PNG.

    Parâmetros
    ----------
    cm : np.ndarray
        Matriz de confusão.
    output_dir : str
        Diretório onde a imagem será salva.
    filename : str
        Nome do arquivo PNG para salvar.
    """
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"[INFO] Matriz de confusão salva em {filename}.")

def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, output_dir: str, filename: str) -> None:
    """
    Plota e salva a curva ROC como imagem PNG.

    Parâmetros
    ----------
    fpr : np.ndarray
        False Positive Rate.
    tpr : np.ndarray
        True Positive Rate.
    roc_auc : float
        Área sob a curva ROC.
    output_dir : str
        Diretório onde a imagem será salva.
    filename : str
        Nome do arquivo PNG para salvar.
    """
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"[INFO] Curva ROC salva em {filename}.")

def plot_pr_curve(recall: np.ndarray, precision: np.ndarray, pr_auc: float, output_dir: str, filename: str) -> None:
    """
    Plota e salva a curva Precision-Recall como imagem PNG.

    Parâmetros
    ----------
    recall : np.ndarray
        Valores de recall.
    precision : np.ndarray
        Valores de precisão.
    pr_auc : float
        Área sob a curva Precision-Recall.
    output_dir : str
        Diretório onde a imagem será salva.
    filename : str
        Nome do arquivo PNG para salvar.
    """
    plt.figure()
    plt.plot(recall, precision, color="blue", lw=2, label=f"PR curve (area = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"[INFO] Curva Precision-Recall salva em {filename}.")
