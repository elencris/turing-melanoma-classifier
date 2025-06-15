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

# Executa pós-processamento
echo "Executando pos-processamento..."
python3 postprocess_proj1.py \
  --results "$RESULTS_DIR" \
  --labels "$TEST_METADATA"

echo "Pos-processamento concluido."