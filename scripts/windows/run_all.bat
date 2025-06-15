@echo off
setlocal enabledelayedexpansion

REM Caminho para o arquivo de configuracao
set CONFIG_PATH=scripts\project.config

if not exist "%CONFIG_PATH%" (
    echo ERRO: Arquivo de configuracao nao encontrado: %CONFIG_PATH%
    exit /b
)

REM Le variaveis do arquivo .config
for /f "usebackq tokens=*" %%L in ("%CONFIG_PATH%") do (
    set "LINE=%%L"
    for /f "tokens=* delims= " %%X in ("!LINE!") do set "LINE=%%X"
    if not "!LINE!"=="" if not "!LINE:~0,1!"=="#" (
        for /f "tokens=1,* delims==" %%A in ("!LINE!") do (
            set "%%A=%%B"
        )
    )
)

REM 1. Pré-processamento
echo Executando pre-processamento...
python3 preprocess_proj1.py --train_data "%TRAIN_DIR%" --val_data "%VAL_DIR%" --test_data "%TEST_DIR%" --train_metadata "%TRAIN_METADATA%" --val_metadata "%VAL_METADATA%" --test_metadata "%TEST_METADATA%"

REM 2. Treinamento
echo Executando treinamento...
python3 train_proj1.py --train_data "%TRAIN_DIR%" --val_data "%VAL_DIR%" --train_metadata "%TRAIN_METADATA%" --val_metadata "%VAL_METADATA%"

REM 3. Teste
echo Executando teste...
python3 test_proj1.py --test_data "%TEST_DIR%" --test_metadata "%TEST_METADATA%"

REM 4. Pós-processamento
echo Executando pos-processamento...
python3 postprocess_proj1.py --results "%RESULTS_DIR%" --labels "%TEST_METADATA%"

echo Pipeline completo.
pause
