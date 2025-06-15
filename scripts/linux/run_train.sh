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

# Executa treinamento
echo "Executando treinamento..."
python3 train_proj1.py \
  --train_data "$TRAIN_DIR" \
  --val_data "$VAL_DIR" \
  --train_metadata "$TRAIN_METADATA" \
  --val_metadata "$VAL_METADATA"

echo "Treinamento concluido."