@echo off
REM Launches the sci_rehab Streamlit app using the project's Python install.
REM Double-click this file, or run it from a terminal, to start the app.
REM IMPORTANT: keep this console window open while using the app -- closing
REM it stops the Streamlit server immediately. The app opens in your browser
REM automatically once the server has finished starting (a few seconds).

title SCI Rehab Streamlit App
setlocal

set "PYTHON_EXE=C:\Users\fatim\AppData\Local\Python\pythoncore-3.14-64\python.exe"
set "APP_FILE=%~dp0PYTHON\applications\sci_rehab\streamlit_app.py"

if not exist "%PYTHON_EXE%" (
    echo ERROR: Python executable not found at "%PYTHON_EXE%"
    pause
    exit /b 1
)

if not exist "%APP_FILE%" (
    echo ERROR: Streamlit app not found at "%APP_FILE%"
    pause
    exit /b 1
)

"%PYTHON_EXE%" -m streamlit run "%APP_FILE%"

echo.
echo Streamlit server has stopped.
pause

endlocal
