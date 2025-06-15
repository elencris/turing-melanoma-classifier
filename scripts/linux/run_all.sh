#!/bin/bash
set -e  # Encerra ao primeiro erro

CONFIG_PATH="scripts/project.config"

# Verifica se o arquivo de configuração existe
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERRO: Arquivo de configuração não encontrado: $CONFIG_PATH"
    exit 1
fi

# Importa diretamente as variáveis do arquivo de configuração, removendo \r
source <(sed 's/\r$//' "$CONFIG_PATH")

# 1. Pré-processamento
echo "Executando pre-processamento..."
python3 preprocess_proj1.py \
  --train_data "$TRAIN_DIR" \
  --val_data "$VAL_DIR" \
  --test_data "$TEST_DIR" \
  --train_metadata "$TRAIN_METADATA" \
  --val_metadata "$VAL_METADATA" \
  --test_metadata "$TEST_METADATA"

# 2. Treinamento
echo "Executando treinamento..."
python3 train_proj1.py \
  --train_data "$TRAIN_DIR" \
  --val_data "$VAL_DIR" \
  --train_metadata "$TRAIN_METADATA" \
  --val_metadata "$VAL_METADATA"

# 3. Teste
echo "Executando teste..."
python3 test_proj1.py \
  --test_data "$TEST_DIR" \
  --test_metadata "$TEST_METADATA"

# 4. Pós-processamento
echo "Executando pos-processamento..."
python3 postprocess_proj1.py \
  --results "$RESULTS_DIR" \
  --labels "$TEST_METADATA"

echo "Pipeline completo."
