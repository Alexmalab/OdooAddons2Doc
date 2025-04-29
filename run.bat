@echo off
setlocal enabledelayedexpansion

REM Check if API key is set
if "%OPENAI_API_KEY%"=="" (
    echo Error: OPENAI_API_KEY environment variable is not set.
    echo In CMD, use: set OPENAI_API_KEY=your_api_key
    echo In PowerShell, use: $env:OPENAI_API_KEY = "your_api_key"
    exit /b 1
)

REM Default values
set "ADDONS_PATH=C:\Users\win11\PycharmProjects\pythonProject\chatgpt_prompt\gpt-repository-loader-main\OdooSrcs\odoo-13.0\test_addons"
set "OUTPUT_PATH=./output"
set "MODEL=grok-3-beta"
set "MAX_WORKERS=4"
set "MAX_FILE_TOKENS=10000"
set "BATCH_SIZE=10"
set "RATE_LIMIT=7.0"
set "BASE_URL=https://api.x.ai/v1"
set "MERGE_ONLY=0"

REM Set HTTP proxy if needed (this will be used by OpenAI library automatically)
set "HTTP_PROXY=127.0.0.1:8848"
set "HTTPS_PROXY=127.0.0.1:8848"

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :check_args
if "%~1"=="--addons-path" (
    set "ADDONS_PATH=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--output-path" (
    set "OUTPUT_PATH=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--model" (
    set "MODEL=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--max-workers" (
    set "MAX_WORKERS=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--max-file-tokens" (
    set "MAX_FILE_TOKENS=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--batch-size" (
    set "BATCH_SIZE=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--rate-limit" (
    set "RATE_LIMIT=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--base-url" (
    set "BASE_URL=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--merge-only" (
    set "MERGE_ONLY=1"
    shift
    goto :parse_args
)
echo Unknown option: %~1
exit /b 1

:check_args
REM Check if addons path is provided
if "%ADDONS_PATH%"=="" if "%MERGE_ONLY%"=="0" (
    echo Error: --addons-path is required.
    echo Usage: run.bat --addons-path /path/to/addons [--output-path ./output] [--model gpt-4o] [--max-workers 4] 
    echo                [--max-file-tokens 3500] [--batch-size 10] [--rate-limit 10.0] [--base-url URL] [--merge-only]
    echo.
    echo Note: HTTP proxy should be set as environment variables:
    echo       set HTTP_PROXY=127.0.0.1:7890
    echo       set HTTPS_PROXY=127.0.0.1:7890
    exit /b 1
)

REM Construct the command
set "CMD=python odoo_docs_generator.py --addons-path %ADDONS_PATH% --output-path %OUTPUT_PATH% --api-key %OPENAI_API_KEY% --model %MODEL% --max-workers %MAX_WORKERS% --max-file-tokens %MAX_FILE_TOKENS% --batch-size %BATCH_SIZE% --rate-limit %RATE_LIMIT%"

REM Add base_url if specified
if not "%BASE_URL%"=="" (
    set "CMD=%CMD% --base-url %BASE_URL%"
)

REM Add merge-only flag if specified
if "%MERGE_ONLY%"=="1" (
    set "CMD=%CMD% --merge-only"
)

REM Print the command (without API key)
set "DISPLAY_CMD=!CMD:%OPENAI_API_KEY%=*****!"
echo Running: !DISPLAY_CMD!

REM Print proxy settings if set
if not "%HTTP_PROXY%"=="" (
    echo Using HTTP proxy: %HTTP_PROXY%
)
if not "%HTTPS_PROXY%"=="" (
    echo Using HTTPS proxy: %HTTPS_PROXY%
)

REM Execute the command
%CMD%

endlocal 