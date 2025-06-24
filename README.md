# 🔬 Projeto de Classificação de Imagens Médicas - Equipe Turing

## 🎯 Objetivo

Desenvolver um sistema em **Python** para **classificação binária de imagens dermatológicas** com foco na detecção de **melanoma (label `MEL`)**, utilizando dados da competição **ISIC 2018 - Task 3**.

O problema original de múltiplas classes foi convertido em um desafio binário:
- `1` → Melanoma;
- `0` → Outras condições dermatológicas.

---

## 👩‍💻 Integrantes

- **Elen Cristina Rego Gomes** – Matrícula: 202206840014;
- **Christhian Swami Zhamir da Costa Lima** – Matrícula: 202206840030.

---

## ✅ Requisitos

- Python >= 3.12
- pip >= 25.1

---

## ⚙️ Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/elencris/turing-melanoma-classifier.git
cd turing-melanoma-classifier
```

### 2. Crie e ative o ambiente virtual

```bash
python -m venv ml-env

# Linux
source ml-env/bin/activate

# Windows
.\ml-env\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

> 💡 Para sair do ambiente virtual:
> ```bash
> deactivate
> ```

### 4. Baixe o dataset

Baixe e descompacte os dados da [ISIC 2018 - Task 3](https://challenge.isic-archive.com/data/#2018) na pasta de sua preferência.

| Conteúdo             | Fonte                                                               |
|----------------------|---------------------------------------------------------------------|
| Imagens de treino    | [Arquivo compactado](https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip) |
| Imagens de validação | [Arquivo compactado](https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Validation_Input.zip) |
| Imagens de teste     | [Arquivo compactado](https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Test_Input.zip) |
| Rótulos de treino    | [Arquivo compactado](https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip) |
| Rótulos de validação | [Arquivo compactado](https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Validation_GroundTruth.zip) |
| Rótulos de teste     | [Arquivo compactado](https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Test_GroundTruth.zip) |

---

## 📁 Estrutura sugerida

A pasta `artifacts/` e seu conteúdo são gerados automaticamente em tempo de execução.

```
turing-melanoma-classifier/
├── data/
│   ├── train/                   # imagens de treino
│   ├── val/                     # imagens de validação
│   └── test/                    # imagens de teste
├── metadata/                    # arquivos CSV com rótulos
├── utils/
│   ├── hyperparameter_tuning.py # biblioteca de seleção de hiperparâmetros
│   ├── image_processing.py      # biblioteca de processamento de imagens
│   └── metrics.py               # biblioteca de métricas de desempenho
├── preprocess_proj1.py
├── train_proj1.py
├── test_proj1.py
├── postprocess_proj1.py          
├── requirements.txt
└── README.md
```

---

## 🚀 Execução

Envie os comandos abaixo em seu terminal Linux ou Windows para executar cada etapa, os parâmetros são opcionais, utilize-os caso deseje alterar alguma informação, estes comandos utilizam os valores padrão.

### 1. Pré-processamento

```bash
python preprocess_proj1.py --train_data data/train --val_data data/val --test_data data/test --train_metadata metadata/train.csv --val_metadata metadata/val.csv --test_metadata metadata/test.csv --train_fraction 1.0 --output_dir artifacts/preprocessed_dataset
```

### 2. Treinamento

```bash
python train_proj1.py --train_dataset artifacts/preprocessed_dataset/train_dataset.pkl --val_dataset artifacts/preprocessed_dataset/val_dataset.pkl --output_dir artifacts --tune_trials 50
```

### 3. Teste

```bash
python test_proj1.py --test_dataset artifacts/preprocessed_dataset/test_dataset.pkl --scaler_path artifacts/scaler/standard_scaler.pkl --model_path artifacts/lightgbm/lightgbm_model.pkl --output_dir artifacts/results
```

### 4. Pós-processamento

```bash
python postprocess_proj1.py --train_dataset artifacts/preprocessed_dataset/train_dataset.pkl --train_predictions artifacts/results/train_predictions.csv --test_dataset artifacts/preprocessed_dataset/test_dataset.pkl --test_predictions artifacts/results/test_predictions.csv
```

---

## 📌 Tabela de Uso dos Scripts

| Script | Entrada(s) | Saída(s) | Descrição curta |
|--------|------------|----------|-----------------|
| `preprocess_proj1.py` | Pastas de imagens e arquivos `.csv` com rótulos | `.pkl` dos datasets pré-processados | Extrai features das imagens e salva conjuntos prontos |
| `train_proj1.py` | `train_dataset.pkl`, `val_dataset.pkl` | Modelo `.pkl`, scaler `.pkl`, histórico de treino | Treina o modelo LightGBM com busca de hiperparâmetros via Optuna |
| `test_proj1.py` | `test_dataset.pkl`, modelo `.pkl`, scaler `.pkl` | CSV com predições | Gera predições com o modelo treinado |
| `postprocess_proj1.py`| Datasets e arquivos de predição (`.csv`) | Métricas, gráficos e relatórios | Calcula métricas e gera análise do desempenho final |

---

## ⚡ Sobre o LightGBM

Este projeto utiliza o **[LightGBM](https://lightgbm.readthedocs.io/)** como modelo principal de aprendizado supervisionado.

O LightGBM é um framework de **gradient boosting** que utiliza algoritmos de árvores de decisão e é projetado para ser **rápido, eficiente e escalável**, com os seguintes benefícios:

- 🚀 **Treinamento mais rápido** e maior eficiência;
- 💾 **Baixo consumo de memória**;
- 🎯 **Alta acurácia** em tarefas de classificação;
- ⚙️ Suporte a **aprendizado paralelo, distribuído e com GPU**;
- 🧠 Ideal para **grandes volumes de dados**.

> Neste projeto, o LightGBM é combinado com **Optuna** para otimização automática de hiperparâmetros, melhorando ainda mais a performance do modelo na detecção de melanoma.

---

## 🧰 Bibliotecas

- [`joblib`](https://joblib.readthedocs.io/en/latest/)
- [`lightgbm`](https://lightgbm.readthedocs.io/en/latest/)
- [`matplotlib`](https://matplotlib.org/stable/contents.html)
- [`numpy`](https://numpy.org/doc/)
- [`opencv-python`](https://docs.opencv.org/4.x/)
- [`optuna`](https://optuna.readthedocs.io/en/stable/)
- [`pandas`](https://pandas.pydata.org/docs/)
- [`Pillow`](https://pillow.readthedocs.io/en/stable/)
- [`scikit-learn`](https://scikit-learn.org/stable/documentation.html)
- [`scipy`](https://docs.scipy.org/doc/scipy/)
- [`seaborn`](https://seaborn.pydata.org/)
- [`scikit-image`](https://scikit-image.org/docs/stable/)

---

## 🔄 Git & Versionamento

### 📥 Atualizar repositório local

```bash
git pull origin main
```

### 📤 Enviar alterações

```bash
git add .
git commit -m "mensagem"
git push origin <nome_da_branch>
```

> ✅ Use sempre `git pull` antes do `push` para evitar conflitos.

---

## 📝 Convenção de Commits

Adote o padrão [Conventional Commits](https://www.conventionalcommits.org/pt-br/v1.0.0/):

### Exemplo

```
feat(model): adiciona classificador SVM
fix(preprocess): corrige bug RGB
docs(readme): atualiza instruções
```

| Tipo      | Significado                          |
|-----------|--------------------------------------|
| feat      | Nova funcionalidade                  |
| fix       | Correção de bug                      |
| docs      | Documentação                         |
| style     | Alterações visuais ou de formatação  |
| refactor  | Refatoração sem nova feature         |
| test      | Adição ou modificação de testes      |
| chore     | Tarefas auxiliares/misc.             |

---

## 📚 Fontes e Dados

- Artigo: [arXiv:1803.10417](https://arxiv.org/pdf/1803.10417)
- ISIC 2018 - Task 3: [challenge.isic-archive.com](https://challenge.isic-archive.com/data/#2018)
- Dataverse: [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

### Créditos dos Dados

HAM10000 Dataset: (c) by ViDIR Group, Department of Dermatology, Medical University of Vienna; https://doi.org/10.1038/sdata.2018.161
MSK Dataset: (c) Anonymous; https://arxiv.org/abs/1710.05006; https://arxiv.org/abs/1902.03368