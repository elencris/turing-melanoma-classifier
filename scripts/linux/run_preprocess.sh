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

# Executa pré-processamento
echo "Executando pre-processamento..."
python3 preprocess_proj1.py \
  --train_data "$TRAIN_DIR" \
  --val_data "$VAL_DIR" \
  --test_data "$TEST_DIR" \
  --train_metadata "$TRAIN_METADATA" \
  --val_metadata "$VAL_METADATA" \
  --test_metadata "$TEST_METADATA"

echo "Pre-processamento concluido."