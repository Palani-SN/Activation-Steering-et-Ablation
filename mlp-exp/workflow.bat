@echo off
SETLOCAL EnableDelayedExpansion

:: Capture Start Time in total seconds (Unix-like timestamp)
for /f "tokens=*" %%a in ('powershell -command "[DateTimeOffset]::Now.ToUnixTimeSeconds()"') do set START_UNIX=%%a
for /f "tokens=*" %%a in ('powershell -command "Get-Date -Format 'HH:mm:ss'"') do set START_HUMAN=%%a

:: --- YOUR CODE STARTS HERE ---
echo.
echo ============================================================================
echo      ACTIVATION STEERING AND ABLATION - PIPELINE EXECUTION
echo ============================================================================
echo.

echo [1/8] ^> Activating Environment...
call conda activate act-abl || (echo [FAIL] Activation failed & exit /b 1)
echo [OK] Environment activated successfully
echo.

echo [2/8] ^> Generating Dataset...
cd dataset
python data_generator.py || (echo [FAIL] Data generation failed & exit /b 1)
echo [OK] Primary dataset generated
python variant_generator.py || (echo [FAIL] Variant generation failed & exit /b 1)
echo [OK] OOD variants generated
cd ..
echo.

echo [3/8] ^> Training MLP...
python train_mlp.py || (echo [FAIL] MLP training failed & exit /b 1)
echo [OK] MLP trained to perfection
echo.

echo [4/8] ^> Harvesting Activations...
python harvest_activations.py || (echo [FAIL] Activation harvest failed & exit /b 1)
echo [OK] Activations harvested
echo.

echo [5/8] ^> Training Sparse Autoencoder (SAE)...
python train_sae.py || (echo [FAIL] SAE training failed & exit /b 1)
echo [OK] SAE trained successfully
echo.

echo [6/8] ^> Running Feature Probe...
python feature_probe.py || (echo [FAIL] Feature probing failed & exit /b 1)
echo [OK] Feature analysis complete
echo.

echo [7/8] ^> Running Consistency of Compliance Checks...
python consistency_compliance.py || (echo [FAIL] Consistency Compliance checks failed & exit /b 1)
echo [OK] Steering validation complete
echo.

echo [8/8] ^> Generating Feature Reports...
python feature_reports.py || (echo [FAIL] Feature report generation failed & exit /b 1)
echo [OK] Reports generated successfully
echo.
echo ============================================================================
echo                    PIPELINE COMPLETED SUCCESSFULLY ^>
echo           Activation Steering in Latent Space - All Phases Done
echo ============================================================================
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