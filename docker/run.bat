@echo off
setlocal enabledelayedexpansion

:: --- Default values ---
set "TICKER=AAPL,MSFT,NVDA"
set "USE_OLLAMA="
set "START_DATE="
set "END_DATE="
set "INITIAL_AMOUNT=100000.0"
set "MARGIN_REQUIREMENT=0.0"
set "SHOW_REASONING="
set "COMMAND="
set "MODEL_NAME="
set "COMPOSE_CMD=docker compose"

:: --- Start script execution ---
call :parse_args %*
goto :run_%COMMAND%

:: ============================================================================
:: --- HELP FUNCTION ---
:: ============================================================================
:show_help
echo AI Hedge Fund Docker Runner
echo(
echo Usage: run.bat [OPTIONS] COMMAND
echo(
echo Options:
echo   --ticker SYMBOLS    Comma-separated list of ticker symbols (e.g., AAPL,MSFT,NVDA^)
echo   --start-date DATE   Start date in YYYY-MM-DD format
echo   --end-date DATE     End date in YYYY-MM-DD format
echo   --initial-cash AMT  Initial cash position (default: 100000.0^)
echo   --margin-requirement RATIO  Margin requirement ratio (default: 0.0^)
echo   --ollama            Use Ollama for local LLM inference
echo   --show-reasoning    Show reasoning from each agent
echo(
echo Commands:
echo   main                Run the main hedge fund application
echo   backtest            Run the backtester
echo   build               Build the Docker image
echo   compose             Run using Docker Compose with integrated Ollama
echo   ollama              Start only the Ollama container for model management
echo   pull MODEL          Pull a specific model into the Ollama container
echo   help                Show this help message
echo(
echo Examples:
echo   run.bat --ticker "AAPL,MSFT,NVDA" main
echo   run.bat --ticker "AAPL,MSFT,NVDA" --ollama main
echo   run.bat compose
echo   run.bat pull llama3
goto :eof

:: ============================================================================
:: --- ARGUMENT PARSER ---
:: ============================================================================
:parse_args
if "%~1"=="" goto :check_command
if /i "%~1"=="--ticker"           (set "TICKER=%~2" & shift & shift & goto :parse_args)
if /i "%~1"=="--start-date"       (set "START_DATE=--start-date %~2" & shift & shift & goto :parse_args)
if /i "%~1"=="--end-date"         (set "END_DATE=--end-date %~2" & shift & shift & goto :parse_args)
if /i "%~1"=="--initial-cash"     (set "INITIAL_AMOUNT=%~2" & shift & shift & goto :parse_args)
if /i "%~1"=="--margin-requirement" (set "MARGIN_REQUIREMENT=%~2" & shift & shift & goto :parse_args)
if /i "%~1"=="--ollama"           (set "USE_OLLAMA=--ollama" & shift & goto :parse_args)
if /i "%~1"=="--show-reasoning"   (set "SHOW_REASONING=--show-reasoning" & shift & goto :parse_args)

:: Commands
if /i "%~1"=="main"     (set "COMMAND=main" & shift & goto :parse_args)
if /i "%~1"=="backtest" (set "COMMAND=backtest" & shift & goto :parse_args)
if /i "%~1"=="build"    (set "COMMAND=build" & shift & goto :parse_args)
if /i "%~1"=="compose"  (set "COMMAND=compose" & shift & goto :parse_args)
if /i "%~1"=="ollama"   (set "COMMAND=ollama" & shift & goto :parse_args)
if /i "%~1"=="pull"     (set "COMMAND=pull" & set "MODEL_NAME=%~2" & shift & shift & goto :parse_args)
if /i "%~1"=="help"     (set "COMMAND=help" & shift & goto :parse_args)
if /i "%~1"=="--help"   (set "COMMAND=help" & shift & goto :parse_args)

echo Unknown option: %~1
call :show_help
exit /b 1

:: ============================================================================
:: --- PRE-RUN CHECKS ---
:: ============================================================================
:check_command
if "!COMMAND!"=="" (
    echo Error: No command specified.
    call :show_help
    exit /b 1
)

:: Check for Docker Compose existence
docker compose version >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    docker-compose --version >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        set "COMPOSE_CMD=docker-compose"
    ) else (
        echo Error: Docker Compose is not installed.
        exit /b 1
    )
)
goto :eof

:: ============================================================================
:: --- COMMANDS ---
:: ============================================================================
:run_help
call :show_help
exit /b 0

:run_build
echo Building the Docker image...
docker build -t ai-hedge-fund -f Dockerfile ..
exit /b 0

:run_compose
echo Running with Docker Compose (includes Ollama)...
!COMPOSE_CMD! up --build
exit /b 0

:run_ollama
echo Starting Ollama container...
!COMPOSE_CMD! up -d ollama
echo(
echo Waiting for Ollama to start...
for /l %%i in (1, 1, 30) do (
    timeout /t 1 /nobreak >nul
    !COMPOSE_CMD! exec ollama curl -s http://localhost:11434/api/version >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo Ollama is now running.
        echo Available models:
        !COMPOSE_CMD! exec ollama ollama list
        echo(
        echo Manage your models using:
        echo   run.bat pull ^<model-name^>
        echo   run.bat ollama
        exit /b 0
    )
)
echo Failed to start Ollama within the expected time. You may need to check the container logs.
exit /b 1

:run_pull
if "!MODEL_NAME!"=="" (
    echo Error: No model name specified for 'pull' command.
    echo Usage: run.bat pull ^<model-name^>
    exit /b 1
)
!COMPOSE_CMD! up -d ollama >nul
echo Ensuring Ollama is running...
for /l %%i in (1, 1, 30) do (
    !COMPOSE_CMD! exec ollama curl -s http://localhost:11434/api/version >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo Ollama is running.
        goto :pull_model_now
    )
    timeout /t 1 /nobreak >nul
)
echo Failed to start Ollama. Cannot pull model.
exit /b 1
:pull_model_now
echo Pulling model: !MODEL_NAME!
!COMPOSE_CMD! exec ollama ollama pull "!MODEL_NAME!"
exit /b 0

:run_main
:run_backtest
:: Check for .env file in the parent directory
if not exist ..\.env (
    if exist ..\.env.example (
        echo No .env file found. Creating from ..\.env.example...
        copy ..\.env.example ..\.env
        echo Please edit .env file in the project root to add your API keys.
    ) else (
        echo Error: No .env or .env.example file found.
        exit /b 1
    )
)

:: Set script path and parameters based on command
if "!COMMAND!"=="main" (
    set "SCRIPT_PATH=src/main.py"
    set "INITIAL_PARAM=--initial-cash !INITIAL_AMOUNT!"
) else (
    set "SCRIPT_PATH=src/backtester.py"
    set "INITIAL_PARAM=--initial-capital !INITIAL_AMOUNT!"
)

:: If using Ollama, run via Docker Compose
if not "!USE_OLLAMA!"=="" (
    echo Setting up Ollama container for local LLM inference...
    !COMPOSE_CMD! up -d ollama >nul
    echo Waiting for Ollama to start...
    for /l %%i in (1, 1, 30) do (
        !COMPOSE_CMD! exec ollama curl -s http://localhost:11434/api/version >nul 2>&1
        if !ERRORLEVEL! EQU 0 (
            echo Ollama is running.
            goto :continue_ollama
        )
        timeout /t 1 /nobreak >nul
    )
    echo Failed to start Ollama.
    exit /b 1
    :continue_ollama
    set "COMMAND_OVERRIDE=!START_DATE! !END_DATE! !INITIAL_PARAM! --margin-requirement !MARGIN_REQUIREMENT!"
    
    echo Running AI Hedge Fund with Ollama...
    if "!COMMAND!"=="main" (
        !COMPOSE_CMD! run --rm hedge-fund-ollama python !SCRIPT_PATH! --ticker "!TICKER!" !COMMAND_OVERRIDE! !SHOW_REASONING! --ollama
    ) else (
        !COMPOSE_CMD! run --rm backtester-ollama python !SCRIPT_PATH! --ticker "!TICKER!" !COMMAND_OVERRIDE! !SHOW_REASONING! --ollama
    )
    exit /b 0
)

:: Standard Docker run (without Ollama)
set "CMD=docker run -it --rm -v "%~dp0..\.env:/app/.env" ai-hedge-fund python !SCRIPT_PATH! --ticker "!TICKER!" !START_DATE! !END_DATE! !INITIAL_PARAM! --margin-requirement !MARGIN_REQUIREMENT! !SHOW_REASONING!"echo Running: !CMD!
!CMD!
exit /b 0