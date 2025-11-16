@echo off
REM Batch Training Script for Domain Adaptation CT
REM This script runs multiple training configurations sequentially

echo ========================================
echo Domain Adaptation CT - Batch Training
echo ========================================
echo.

REM Check if virtual environment is activated
if "%VIRTUAL_ENV%"=="" (
    echo ERROR: Virtual environment not detected!
    echo Please activate your virtual environment first:
    echo    venv\Scripts\activate.bat
    echo.
    pause
    exit /b 1
)

echo Virtual environment detected: %VIRTUAL_ENV%
echo.

REM Set the start time
echo Starting batch training at %date% %time%
echo.

REM Configuration files to run (add/remove as needed)
set CONFIG1=experiment_configs\dann_train_val_D21_v2_config_linear_decreasing.yaml
set CONFIG2=experiment_configs\dann_train_val_D21_v2_config_constant.yaml
set CONFIG3=experiment_configs\dann_train_val_D21_v2_config_linear_increasing.yaml
set CONFIG4=experiment_configs\dann_train_val_D21_v2_config_parabolic_increasing.yaml
set CONFIG5=experiment_configs\dann_train_val_D21_v2_config_logistic_increasing.yaml
set CONFIG6=experiment_configs\dann_train_val_D21_v2_config_parabolic_decreasing.yaml

REM Counter for completed runs
set COMPLETED=0
set FAILED=0

echo ========================================
echo Running Configuration 1/5
echo Config: %CONFIG1%
echo ========================================
python scripts\run_training.py %CONFIG1%
if %ERRORLEVEL% EQU 0 (
    echo SUCCESS: %CONFIG1%
    set /a COMPLETED+=1
) else (
    echo FAILED: %CONFIG1% (Exit code: %ERRORLEVEL%)
    set /a FAILED+=1
)
echo.
echo Pausing 10 seconds before next configuration...
timeout /t 10 /nobreak > nul
echo.

echo ========================================
echo Running Configuration 2/5
echo Config: %CONFIG2%
echo ========================================
python scripts\run_training.py %CONFIG2%
if %ERRORLEVEL% EQU 0 (
    echo SUCCESS: %CONFIG2%
    set /a COMPLETED+=1
) else (
    echo FAILED: %CONFIG2% (Exit code: %ERRORLEVEL%)
    set /a FAILED+=1
)
echo.
echo Pausing 10 seconds before next configuration...
timeout /t 10 /nobreak > nul
echo.

echo ========================================
echo Running Configuration 3/5
echo Config: %CONFIG3%
echo ========================================
python scripts\run_training.py %CONFIG3%
if %ERRORLEVEL% EQU 0 (
    echo SUCCESS: %CONFIG3%
    set /a COMPLETED+=1
) else (
    echo FAILED: %CONFIG3% (Exit code: %ERRORLEVEL%)
    set /a FAILED+=1
)
echo.
echo Pausing 10 seconds before next configuration...
timeout /t 10 /nobreak > nul
echo.

echo ========================================
echo Running Configuration 4/5
echo Config: %CONFIG4%
echo ========================================
python scripts\run_training.py %CONFIG4%
if %ERRORLEVEL% EQU 0 (
    echo SUCCESS: %CONFIG4%
    set /a COMPLETED+=1
) else (
    echo FAILED: %CONFIG4% (Exit code: %ERRORLEVEL%)
    set /a FAILED+=1
)
echo.
echo Pausing 10 seconds before next configuration...
timeout /t 10 /nobreak > nul
echo.

echo ========================================
echo Running Configuration 5/5
echo Config: %CONFIG5%
echo ========================================
python scripts\run_training.py %CONFIG5%
if %ERRORLEVEL% EQU 0 (
    echo SUCCESS: %CONFIG5%
    set /a COMPLETED+=1
) else (
    echo FAILED: %CONFIG5% (Exit code: %ERRORLEVEL%)
    set /a FAILED+=1
)
echo.

REM Summary
echo ========================================
echo BATCH TRAINING COMPLETE
echo ========================================
echo Finished at %date% %time%
echo.
echo SUMMARY:
echo   Completed successfully: %COMPLETED%/5
echo   Failed: %FAILED%/5
echo.

if %FAILED% GTR 0 (
    echo WARNING: Some configurations failed!
    echo Check the output above for details.
) else (
    echo All configurations completed successfully!
)

echo.
echo Press any key to exit...
pause > nul