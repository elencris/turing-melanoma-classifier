"""
Módulo de Pré-processamento de Imagens para Classificação de Melanoma

Descrição
---------
Este módulo realiza o pré-processamento dos conjuntos de imagens de treino, validação e teste para um modelo de classificação binária de melanoma. Ele processa as imagens, extrai características relevantes e gera datasets prontos para treinamento e avaliação.

Variáveis Globais
-----------------
IMAGE_SIZE : tuple[int, int]
    Dimensão padrão para redimensionamento das imagens (200, 200).
RANDOM_STATE : int
    Semente fixa para garantir reprodutibilidade na subamostragem.

Funções
-------
process_dataset(split_name: str, image_dir: str, metadata_path: str, train_fraction: float = 1.0) -> None
    Processa o dataset especificado aplicando pré-processamento e extração de características.

Uso na Linha de Comando
-----------------------
python preprocess_proj1.py --train_data_dir <diretório_imagens_treino> --val_data_dir <diretório_imagens_validação> --test_data_dir <diretório_imagens_teste> --train_metadata_path <csv_metadados_treino> --val_metadata_path <csv_metadados_validação> --test_metadata_path <csv_metadados_teste> --train_fraction <fração_treino> --output_dir <diretório_saida>
"""

import argparse
import os
import cv2
import pandas as pd
import numpy as np
from utils.image_processing import (
    resize_image,
    data_augmentation,
    reduce_background,
    apply_clahe,
    segment_kmeans,
    extract_features
)

# Variáveis Globais
IMAGE_SIZE: tuple[int, int] = (200, 200)
RANDOM_STATE: int = 42

# Argumentos da linha de comando
parser = argparse.ArgumentParser(description="Pré-processa imagens para treino, validação e teste.")

parser.add_argument(
    "--train_data_dir",
    type=str,
    default="data/train",
    help="Diretório com as imagens de treino."
)
parser.add_argument(
    "--val_data_dir",
    type=str,
    default="data/val",
    help="Diretório com as imagens de validação."
)
parser.add_argument(
    "--test_data_dir",
    type=str,
    default="data/test",
    help="Diretório com as imagens de teste."
)

parser.add_argument(
    "--train_metadata_path",
    type=str,
    default="metadata/train.csv",
    help="CSV com metadados do conjunto de treino."
)
parser.add_argument(
    "--val_metadata_path",
    type=str,
    default="metadata/val.csv",
    help="CSV com metadados do conjunto de validação."
)
parser.add_argument(
    "--test_metadata_path",
    type=str,
    default="metadata/test.csv",
    help="CSV com metadados do conjunto de teste."
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="artifacts/preprocessed_dataset",
    help="Pasta para salvar os arquivos de saída."
)

parser.add_argument(
    "--train_fraction",
    type=float,
    default=1.0,
    help="Fração dos dados de treino a usar (0 < x ≤ 1)."
)

args = parser.parse_args()


# Validação da fração
if args.train_fraction == 0:
    raise ValueError("[ERRO] O argumento --train_fraction não pode ser zero.")
if not (0 < args.train_fraction <= 1.0):
    raise ValueError("[ERRO] O argumento --train_fraction deve ser maior que 0 e menor ou igual a 1.0.")

# Criação da pasta de saída
os.makedirs(args.output_dir, exist_ok=True)

def process_dataset(split_name: str, image_dir: str, metadata_path: str, train_fraction: float = 1.0) -> None:
    """
    Processa o dataset especificado aplicando pré-processamento e extração de características.

    Parâmetros
    ----------
    split_name : str
        Nome do dataset ('train', 'val' ou 'test').
    image_dir : str
        Diretório onde estão as imagens.
    metadata_path : str
        Caminho para o arquivo CSV com os metadados.
    train_fraction : float, opcional
        Fração para subamostragem balanceada no treino (default 1.0).

    Retorna
    -------
    None
        Salva o dataset processado em arquivo pickle no diretório especificado.
    """

    print(f"[INFO] Processando {split_name}...")

    df_meta = pd.read_csv(metadata_path)

    diagnosis_cols = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    missing_cols = [col for col in diagnosis_cols if col not in df_meta.columns]
    if missing_cols:
        raise ValueError(f"[ERRO] Colunas ausentes no arquivo CSV: {missing_cols}")

    # Cria coluna 'diagnosis' e label binário 'label' (1 para MEL, 0 para outros)
    df_meta = df_meta.rename(columns={"image": "image_name"})
    df_meta["diagnosis"] = df_meta[diagnosis_cols].idxmax(axis=1)
    df_meta["label"] = (df_meta["diagnosis"] == "MEL").astype(int)

    # Subamostragem balanceada para treino
    if split_name == "train":
        if train_fraction < 1.0:
            sampled_list = [
                group.sample(frac=train_fraction, random_state=RANDOM_STATE)
                for _, group in df_meta.groupby("diagnosis")
            ]
            df_meta = pd.concat(sampled_list).reset_index(drop=True)
            print(f"[INFO] Subamostragem aplicada ao treino ({train_fraction:.0%} por classe). Total: {len(df_meta)}")
        else:
            print("[INFO] Usando 100% dos dados de treino (sem subamostragem).")

    df_meta = df_meta.drop(columns=["diagnosis"])

    processed_rows = []

    # Processa cada imagem
    for _, row in df_meta.iterrows():
        img_name = row["image_name"]
        label = row["label"]
        img_path = os.path.join(image_dir, f"{img_name}.jpg")

        if not os.path.exists(img_path):
            print(f"[AVISO] Imagem não encontrada: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERRO] Falha ao carregar: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = resize_image(img_rgb, size=IMAGE_SIZE)

        # Data augmentation somente para melanoma no treino
        if split_name == "train" and label == 1:
            augmented_images = data_augmentation(resized)
        else:
            augmented_images = [resized]

        # Pré-processamento e extração de features para cada imagem (original + augmentadas)
        for idx, aug_img in enumerate(augmented_images):
            gray = cv2.cvtColor(aug_img, cv2.COLOR_RGB2GRAY)
            bg_reduced = reduce_background(gray)
            clahe_img = apply_clahe(bg_reduced)
            lesion_mask = segment_kmeans(clahe_img)

            features = extract_features(aug_img, lesion_mask)

            if features is None or any(np.isnan(features)):
                print(f"[AVISO] Falha ao extrair features de: {img_name}")
                continue

            img_id = f"{img_name}_aug{idx}" if len(augmented_images) > 1 else img_name
            processed_rows.append({
                "image_name": img_id,
                "label": label,
                "features": features
            })

    df_out = pd.DataFrame(processed_rows)[["image_name", "label", "features"]]
    output_path = os.path.join(args.output_dir, f"{split_name}_dataset.pkl")
    df_out.to_pickle(output_path)
    print(f"[INFO] Dataset '{split_name}' salvo em: {output_path}")
    print(f"[INFO] Total de amostras processadas: {len(df_out)}")

# Execução principal
process_dataset("train", args.train_data_dir, args.train_metadata_path, train_fraction=args.train_fraction)
process_dataset("val", args.val_data_dir, args.val_metadata_path)
process_dataset("test", args.test_data_dir, args.test_metadata_path)

print("[INFO] Todos os datasets foram processados com sucesso.")
