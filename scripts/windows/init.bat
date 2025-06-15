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

REM Cria os diretorios principais (sem erro caso existam)
mkdir "%DATA_DIR%" 2>nul
mkdir "%TRAIN_DIR%" 2>nul
mkdir "%VAL_DIR%" 2>nul
mkdir "%TEST_DIR%" 2>nul
mkdir "%METADATA_DIR%" 2>nul
mkdir "%MODELS_DIR%" 2>nul
mkdir "%RESULTS_DIR%" 2>nul

REM Baixa e extrai os metadados CSV
call :DownloadAndExtractCSV "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip" "train_gt.zip" "%TRAIN_METADATA%"
call :DownloadAndExtractCSV "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Validation_GroundTruth.zip" "val_gt.zip" "%VAL_METADATA%"
call :DownloadAndExtractCSV "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Test_GroundTruth.zip" "test_gt.zip" "%TEST_METADATA%"

REM Baixa e extrai imagens
call :DownloadAndExtractJPG "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Validation_Input.zip" "val.zip" "%VAL_DIR%"
call :DownloadAndExtractJPG "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Test_Input.zip" "test.zip" "%TEST_DIR%"
call :DownloadAndExtractJPG "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip" "train.zip" "%TRAIN_DIR%"

REM Instala dependencias Python
pip install -r requirements.txt

echo Estrutura criada, dados baixados e dependencias instaladas.
pause
exit /b


:DownloadAndExtractCSV
REM %1 = URL, %2 = nome zip, %3 = caminho destino com nome final do CSV
echo Baixando %~2 ...
powershell -Command "try { Invoke-WebRequest -OutFile '%~2' '%~1' } catch { Start-BitsTransfer -Source '%~1' -Destination '%~2' }"
if errorlevel 1 (
    echo ERRO: Falha ao baixar %~2
    exit /b
)

echo Extraindo .csv de %~2 ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Add-Type -AssemblyName System.IO.Compression.FileSystem; $zip = [System.IO.Compression.ZipFile]::OpenRead('%~2'); $entry = $zip.Entries | Where-Object { $_.Name.ToLower().EndsWith('.csv') } | Select-Object -First 1; if ($entry -ne $null) { [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, '%~3', $true); Write-Host 'CSV renomeado para %~3'; } else { Write-Error 'Nenhum arquivo CSV encontrado.' }; $zip.Dispose()"
if errorlevel 1 (
    echo ERRO: Falha ao extrair %~2
    del "%~2"
    exit /b
)

del "%~2"
goto :eof

:DownloadAndExtractJPG
REM %1 = URL, %2 = nome zip, %3 = destino
echo Baixando %~2 ...
powershell -Command "try { Invoke-WebRequest -OutFile '%~2' '%~1' } catch { Start-BitsTransfer -Source '%~1' -Destination '%~2' }"
if errorlevel 1 (
    echo ERRO: Falha ao baixar %~2
    exit /b
)

echo Extraindo imagens de %~2 ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Add-Type -AssemblyName 'System.IO.Compression.FileSystem'; $zip = [System.IO.Compression.ZipFile]::OpenRead('%~2'); foreach ($entry in $zip.Entries) { if ($entry.Name.ToLower().EndsWith('.jpg')) { $path = Join-Path '%~3' $entry.Name; [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $path, $true) } }; $zip.Dispose()"
if errorlevel 1 (
    echo ERRO: Falha ao extrair %~2
    del "%~2"
    exit /b
)

del "%~2"
goto :eof

