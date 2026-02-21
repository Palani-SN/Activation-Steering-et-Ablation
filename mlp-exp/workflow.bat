@echo off
SETLOCAL EnableDelayedExpansion

:: Capture Start Time in total seconds (Unix-like timestamp)
for /f "tokens=*" %%a in ('powershell -command "[DateTimeOffset]::Now.ToUnixTimeSeconds()"') do set START_UNIX=%%a
for /f "tokens=*" %%a in ('powershell -command "Get-Date -Format 'HH:mm:ss'"') do set START_HUMAN=%%a

:: --- YOUR CODE STARTS HERE ---
echo [1/7] Activating Environment...
call conda activate mlp || (echo Failed to activate conda & exit /b 1)

echo [2/7] Generating Dataset...
cd dataset
python data_generator.py || (echo Data generation failed & exit /b 1)
cd ..

echo [3/7] Training MLP...
python train_mlp.py || (echo MLP training failed & exit /b 1)

echo [4/7] Harvesting Activations...
python harvest_activations.py || (echo Activation harvest failed & exit /b 1)

echo [5/7] Training Sparse Autoencoder (SAE)...
python train_sae.py || (echo SAE training failed & exit /b 1)

echo [6/7] Running Feature Probe...
python feature_probe.py || (echo Feature probe failed & exit /b 1)

echo [7/7] Generating Feature Reports...
python feature_reports.py || (echo Feature report generation failed & exit /b 1)

echo.
echo ======================================================
echo Pipeline Complete: Monosemantic Features Identified.
echo ======================================================
call conda deactivate
:: --- YOUR CODE ENDS HERE ---

:: Capture End Time
for /f "tokens=*" %%a in ('powershell -command "[DateTimeOffset]::Now.ToUnixTimeSeconds()"') do set END_UNIX=%%a
for /f "tokens=*" %%a in ('powershell -command "Get-Date -Format 'HH:mm:ss'"') do set END_HUMAN=%%a

:: Calculate Duration (Seconds to Mins/Secs)
for /f "tokens=*" %%a in ('powershell -command "$diff = %END_UNIX% - %START_UNIX%; $m = [Math]::Floor($diff/60); $s = $diff %% 60; \"$m m $s s\""') do set DURATION=%%a

echo.
echo ------------------------------------------------------
echo Execution Summary:
echo Started:  %START_HUMAN%
echo Finished: %END_HUMAN%
echo Duration: %DURATION%
echo ------------------------------------------------------
pause