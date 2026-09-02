@echo off
REM Script to run the Vietnamese QA web application on Windows

echo ========================================
echo  Vietnamese Question Answering System
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo [INFO] Virtual environment not found. Creating one...
    python -m venv venv
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt
) else (
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate
)

echo.
echo [INFO] Starting Streamlit application...
echo [INFO] Application will be available at: http://localhost:8501
echo.

REM Run the Streamlit app
streamlit run app/app.py --server.headless=true

pause
