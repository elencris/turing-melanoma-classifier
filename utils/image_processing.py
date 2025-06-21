"""
Processamento e Extração de Características de Imagens para Diagnóstico de Melanoma

Descrição
---------
Este módulo contém funções para pré-processamento, segmentação, extração de características morfológicas,
de cor, textura (LBP) e assimetria a partir de imagens dermatológicas.

Funções
--------
resize_image(img_array: np.ndarray, size: tuple = (200, 200)) -> np.ndarray
    Redimensiona a imagem para o tamanho especificado usando interpolação LANCZOS.

reduce_background(img_gray: np.ndarray) -> np.ndarray
    Reduz o fundo da imagem aplicando desfoque gaussiano e subtração.

apply_clahe(channel: np.ndarray) -> np.ndarray
    Aplica CLAHE (Equalização Adaptativa de Histograma) para melhorar contraste.

segment_kmeans(img_gray: np.ndarray, k: int = 2) -> np.ndarray
    Segmenta a imagem usando K-means para separar áreas de interesse.

remove_background(image: np.ndarray, mask: np.ndarray) -> np.ndarray
    Aplica a máscara binária para remover o fundo da imagem.

extract_lbp_features(image: np.ndarray, P: int = 8, R: int = 1) -> list[float]
    Extrai histograma LBP normalizado de uma imagem em escala de cinza.

calculate_asymmetry(mask: np.ndarray) -> list[float]
    Calcula medidas de assimetria vertical e horizontal da máscara binária.

extract_color_features(image_rgb: np.ndarray, mask: np.ndarray) -> list[float]
    Extrai médias e desvios padrão dos canais RGB dentro da lesão.

extract_features(processed_img: np.ndarray, mask_img: np.ndarray) -> list[float] | None
    Extrai vetor de características combinando morfologia, cor, textura e assimetria.

load_image_as_flat_array(img_array: np.ndarray) -> list[float]
    Converte imagem RGB ou grayscale em vetor 1D normalizado.

data_augmentation(img: np.ndarray) -> list[np.ndarray]
    Gera aumentações da imagem: flips, rotações e ajuste de brilho.
"""

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern
from typing import List, Optional, Union

def resize_image(img_array: np.ndarray, size: tuple[int, int] = (200, 200)) -> np.ndarray:
    """Redimensiona a imagem para o tamanho especificado usando interpolação LANCZOS.

    Parâmetros
    ----------
    img_array : np.ndarray
        Imagem de entrada.
    size : tuple[int, int], opcional
        Tamanho alvo (largura, altura). Padrão é (200, 200).

    Retorna
    -------
    np.ndarray
        Imagem redimensionada.
    """
    img = Image.fromarray(img_array)
    img = img.resize(size, Image.LANCZOS)
    return np.array(img)

def reduce_background(img_gray: np.ndarray) -> np.ndarray:
    """Reduz o fundo da imagem aplicando desfoque gaussiano e subtração.

    Parâmetros
    ----------
    img_gray : np.ndarray
        Imagem em escala de cinza.

    Retorna
    -------
    np.ndarray
        Imagem com fundo reduzido.
    """
    blurred = cv2.GaussianBlur(img_gray, (15, 15), 0)
    subtracted = cv2.subtract(img_gray, blurred)
    return cv2.normalize(subtracted, None, 0, 255, cv2.NORM_MINMAX)

def apply_clahe(channel: np.ndarray) -> np.ndarray:
    """Aplica CLAHE (Equalização Adaptativa de Histograma) para melhorar contraste.

    Parâmetros
    ----------
    channel : np.ndarray
        Canal da imagem em escala de cinza.

    Retorna
    -------
    np.ndarray
        Canal com contraste melhorado.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(channel)

def segment_kmeans(img_gray: np.ndarray, k: int = 2) -> np.ndarray:
    """Segmenta a imagem usando K-means para separar possíveis áreas de interesse.

    Parâmetros
    ----------
    img_gray : np.ndarray
        Imagem em escala de cinza.
    k : int, opcional
        Número de clusters para K-means. Padrão é 2.

    Retorna
    -------
    np.ndarray
        Máscara binária segmentada da lesão.
    """
    pixels = img_gray.reshape(-1, 1)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    labels = kmeans.labels_.reshape(img_gray.shape)

    cluster_means = [np.mean(pixels[labels.flatten() == i]) for i in range(k)]
    lesion_cluster = np.argmax(cluster_means)
    mask = np.uint8(labels == lesion_cluster) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

def remove_background(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Aplica a máscara binária à imagem, removendo o fundo.

    Parâmetros
    ----------
    image : np.ndarray
        Imagem original RGB.
    mask : np.ndarray
        Máscara binária da lesão.

    Retorna
    -------
    np.ndarray
        Imagem com fundo removido.
    """
    return cv2.bitwise_and(image, image, mask=mask)

def extract_lbp_features(image: np.ndarray, P: int = 8, R: int = 1) -> List[float]:
    """Extrai histograma LBP normalizado de uma imagem em escala de cinza.

    Parâmetros
    ----------
    image : np.ndarray
        Imagem em escala de cinza.
    P : int, opcional
        Número de pontos para LBP. Padrão é 8.
    R : int, opcional
        Raio para LBP. Padrão é 1.

    Retorna
    -------
    List[float]
        Histograma normalizado de LBP.
    """
    lbp = local_binary_pattern(image, P, R, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist.tolist()

def calculate_asymmetry(mask: np.ndarray) -> List[float]:
    """Calcula medidas de assimetria vertical e horizontal da máscara binária.

    Parâmetros
    ----------
    mask : np.ndarray
        Máscara binária da lesão.

    Retorna
    -------
    List[float]
        Assimetria vertical e horizontal normalizadas.
    """
    h, w = mask.shape
    left = mask[:, :w // 2]
    right = np.fliplr(mask[:, w // 2:])
    top = mask[:h // 2, :]
    bottom = np.flipud(mask[h // 2:, :])

    vertical_diff = np.sum(np.abs(left.astype(np.int16) - right.astype(np.int16)))
    horizontal_diff = np.sum(np.abs(top.astype(np.int16) - bottom.astype(np.int16)))

    return [vertical_diff / 255, horizontal_diff / 255]

def extract_color_features(image_rgb: np.ndarray, mask: np.ndarray) -> List[float]:
    """Extrai médias e desvios padrão dos canais RGB dentro da lesão.

    Parâmetros
    ----------
    image_rgb : np.ndarray
        Imagem RGB.
    mask : np.ndarray
        Máscara binária da lesão.

    Retorna
    -------
    List[float]
        Lista contendo médias e desvios padrão dos canais R, G, B.
    """
    lesion_pixels = image_rgb[mask > 0]
    if lesion_pixels.size == 0:
        return [0.0] * 6
    means = np.mean(lesion_pixels, axis=0)
    stds = np.std(lesion_pixels, axis=0)
    return means.tolist() + stds.tolist()

def extract_features(processed_img: np.ndarray, mask_img: np.ndarray) -> Optional[List[float]]:
    """Extrai vetor de características morfológicas, cor, textura e assimetria da lesão.

    Parâmetros
    ----------
    processed_img : np.ndarray
        Imagem RGB processada.
    mask_img : np.ndarray
        Máscara binária da lesão.

    Retorna
    -------
    Optional[List[float]]
        Lista de características extraídas ou None se a extração falhar.
    """
    flat = processed_img.flatten().astype(np.float32)
    lesion_area = np.sum(mask_img > 0)
    total_area = mask_img.shape[0] * mask_img.shape[1]

    if lesion_area == 0:
        return None

    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = cv2.arcLength(contours[0], True) if contours else 0
    x, y, w, h = cv2.boundingRect(mask_img)
    aspect_ratio = w / h if h > 0 else 0
    compactness = (perimeter ** 2) / lesion_area if lesion_area > 0 else 0

    M = max(w, h)
    m = min(w, h)
    excentricidade = np.sqrt(1 - (m / M) ** 2) if M > 0 else 0

    color_features = extract_color_features(processed_img, mask_img)
    asymmetry_features = calculate_asymmetry(mask_img)

    gray_for_lbp = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
    lbp_features = extract_lbp_features(gray_for_lbp)

    features = [
        np.mean(flat),
        np.std(flat),
        np.min(flat),
        np.max(flat),
        skew(flat),
        kurtosis(flat),
        lesion_area / total_area,
        perimeter,
        compactness,
        aspect_ratio,
        excentricidade
    ] + color_features + asymmetry_features + lbp_features

    return features

def load_image_as_flat_array(img_array: np.ndarray) -> List[float]:
    """Converte uma imagem RGB ou escala de cinza em vetor 1D normalizado.

    Parâmetros
    ----------
    img_array : np.ndarray
        Imagem RGB ou grayscale.

    Retorna
    -------
    List[float]
        Vetor 1D normalizado da imagem.
    """
    if img_array.ndim == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    flattened = img_array.astype(np.float32).flatten() / 255.0
    return flattened.tolist()

def data_augmentation(img: np.ndarray) -> List[np.ndarray]:
    """Gera aumentações da imagem: flips, rotações e ajuste de brilho.

    Parâmetros
    ----------
    img : np.ndarray
        Imagem RGB original.

    Retorna
    -------
    List[np.ndarray]
        Lista de imagens aumentadas.
    """
    augmented = [img]
    flipped_h = cv2.flip(img, 1)
    flipped_v = cv2.flip(img, 0)
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], 30)
    bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    augmented.extend([flipped_h, flipped_v, rotated, bright])
    return augmented
