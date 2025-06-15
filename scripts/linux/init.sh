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

# Cria os diretórios principais (caso ainda não existam)
mkdir -p "$DATA_DIR" "$TRAIN_DIR" "$VAL_DIR" "$TEST_DIR" "$METADATA_DIR" "$MODELS_DIR" "$RESULTS_DIR"

# Função para baixar e extrair o primeiro .csv de um .zip
DownloadAndExtractCSV() {
    local url="$1"
    local zip_name="$2"
    local output_csv="$3"

    echo "Baixando $zip_name ..."
    wget -q "$url" -O "$zip_name" || {
        echo "ERRO: Falha ao baixar $zip_name"
        exit 1
    }

    echo "Extraindo .csv de $zip_name ..."
    local tmp_dir
    tmp_dir=$(mktemp -d)
    unzip -q "$zip_name" -d "$tmp_dir"

    local csv_file
    csv_file=$(find "$tmp_dir" -type f -iname '*.csv' | head -n 1)

    if [[ -n "$csv_file" ]]; then
        mkdir -p "$(dirname "$output_csv")"
        mv "$csv_file" "$output_csv"
        echo "CSV renomeado para $output_csv"
    else
        echo "ERRO: Nenhum arquivo CSV encontrado."
        rm -rf "$tmp_dir" "$zip_name"
        exit 1
    fi

    rm -rf "$tmp_dir" "$zip_name"
}

# Função para baixar e extrair arquivos .jpg de um .zip
DownloadAndExtractJPG() {
    local url="$1"
    local zip_name="$2"
    local dest_dir="$3"

    echo "Baixando $zip_name ..."
    wget -q "$url" -O "$zip_name" || {
        echo "ERRO: Falha ao baixar $zip_name"
        exit 1
    }

    echo "Extraindo imagens de $zip_name ..."
    mkdir -p "$dest_dir"
    unzip -q -j "$zip_name" '*.jpg' -d "$dest_dir"  # <- usa -j para extrair direto

    rm -f "$zip_name"
}

# Baixa e extrai os metadados CSV
#DownloadAndExtractCSV "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip" "train_gt.zip" "$TRAIN_METADATA"
#DownloadAndExtractCSV "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Validation_GroundTruth.zip" "val_gt.zip" "$VAL_METADATA"
#DownloadAndExtractCSV "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Test_GroundTruth.zip" "test_gt.zip" "$TEST_METADATA"

# Baixa e extrai imagens
#DownloadAndExtractJPG "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Validation_Input.zip" "val.zip" "$VAL_DIR"
#DownloadAndExtractJPG "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Test_Input.zip" "test.zip" "$TEST_DIR"
#DownloadAndExtractJPG "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip" "train.zip" "$TRAIN_DIR"

# Instala dependências Python
pip install -r requirements.txt

echo "Estrutura criada, dados baixados e dependências instaladas com sucesso."
