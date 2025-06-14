# 🧠 Projeto de Classificação de Imagens Médicas - Equipe Turing

## 🎯 Objetivo

Desenvolver um sistema em **Python** para **classificação binária de imagens dermatológicas** com foco na detecção de **melanoma (label `MEL`)**, utilizando dados da competição **ISIC 2018 - Task 3**.

O problema original de múltiplas classes foi convertido em um desafio binário:
- `1` → Melanoma
- `0` → Outras condições dermatológicas

---

## 👩‍💻 Integrantes

- **Elen Cristina Rego Gomes** – Matrícula: 202206840014  
- **Christhian Swami Zhamir da Costa Lima** – Matrícula: 202206840030

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

### 3. Execute o script de inicialização

```bash
# Linux
bash scripts/linux/init.sh

# Windows
scripts\windows\init.bat
```

Caso prefira instalar manualmente, baixe os dados da [ISIC 2018 - Task 3](https://challenge.isic-archive.com/data/#2018), descompacte (atualize o caminho no `project.config` para poder utilizar os demais scripts) e execute:

```bash
pip install -r requirements.txt
```

> 💡 Para sair do ambiente virtual:
> ```bash
> deactivate
> ```

---

## 📁 Estrutura Esperada

```
turing-melanoma-classifier/
├── data/
│   ├── train/            # imagens de treino
│   ├── val/              # imagens de validação
│   └── test/             # imagens de teste
├── metadata/             # arquivos CSV com rótulos
├── scripts/
│   ├── project.config
│   ├── linux/
│   └── windows/
├── models/               # modelos treinados
├── results/              # resultados do teste
├── preprocess_proj1.py   # scripts Python principais
├── train_proj1.py
├── test_proj1.py
├── postprocess_proj1.py          
├── requirements.txt
└── README.md
```

---

## 🚀 Execução

### Usando os scripts

Execute os arquivos `run_*.sh` (Linux) ou `run_*.bat` (Windows), conforme desejado.

### Manualmente pelo terminal

#### 1. Pré-processamento

```bash
python preprocess_proj1.py \
  --train_data data/train \
  --val_data data/val \
  --test_data data/test \
  --train_metadata metadata/train.csv \
  --val_metadata metadata/val.csv \
  --test_metadata metadata/test.csv
```

#### 2. Treinamento

```bash
python train_proj1.py \
  --train_data data/train \
  --val_data data/val \
  --train_metadata metadata/train.csv \
  --val_metadata metadata/val.csv
```

#### 3. Teste

```bash
python test_proj1.py \
  --test_data data/test \
  --test_metadata metadata/test.csv
```

#### 4. Pós-processamento

```bash
python postprocess_proj1.py \
  --results results/predictions.csv \
  --labels metadata/test.csv
```

---

## 🧰 Principais Bibliotecas

- `scikit-learn`, `joblib`
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `Pillow`
- `argparse`, `os`
- `pipreqs` (para gerar `requirements.txt` minimalista)

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

- **HAM10000**: © ViDIR Group – Medical University of Vienna ([DOI](https://doi.org/10.1038/sdata.2018.161))
- **MSK Dataset**: © Memorial Sloan Kettering Cancer Center  
  - https://arxiv.org/abs/1710.05006  
  - https://arxiv.org/abs/1902.03368

---

## ✅ Checklist

- [x] Classificação binária (melanoma vs. não melanoma)
- [] Uso de modelos tradicionais (sem redes neurais)
- [x] Código estruturado em scripts Python puros
- [x] Execução via linha de comando com parâmetros
- [] Avaliação com ROC, PR Curve, matriz de confusão
- [x] Geração de `requirements.txt` com `pipreqs`
- [x] Organização do repositório com README
