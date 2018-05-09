@setlocal
@echo off

python -m venv venv
start "install" /B /W venv\Scripts\activate ^&^& python -m pip install --upgrade pip ^&^& pip install matplotlib numpy scipy seaborn pandas 
endlocal