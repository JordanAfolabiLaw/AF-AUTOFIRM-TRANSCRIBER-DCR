@echo off
REM deploy.bat - Deploy AF-AUTOFIRM-TRANSCRIBER-DCR to Cloud Run
REM
REM Usage:
REM   deploy.bat              - Deploy only
REM   deploy.bat test FILE_ID - Test with a specific file

setlocal enabledelayedexpansion

set PROJECT_ID=firmlink-alpha
set REGION=us-central1
set SERVICE_NAME=af-autofirm-transcriber-dcr
set IMAGE_URI=%REGION%-docker.pkg.dev/%PROJECT_ID%/cloud-run-source-deploy/%SERVICE_NAME%
set GCS_BUCKET=af-autofirm-transcriber-dcr-bucket

echo ================================================
echo Deploying AF-AUTOFIRM-TRANSCRIBER-DCR
echo ================================================
echo Project: %PROJECT_ID%
echo Region: %REGION%
echo Service: %SERVICE_NAME%
echo.

REM Check if this is a test run
if "%1"=="test" (
    if "%2"=="" (
        echo ERROR: test mode requires FILE_ID
        echo Usage: deploy.bat test FILE_ID [OUTPUT_FOLDER_ID]
        exit /b 1
    )
    echo Running test transcription for file: %2
    
    set URL=https://%SERVICE_NAME%-%PROJECT_ID%.%REGION%.run.app
    
    if not "%3"=="" (
        curl -X POST "!URL!/process/%2?output_folder=%3" ^
            -H "Authorization: Bearer $(gcloud auth print-identity-token)" ^
            -H "Content-Type: application/json"
    ) else (
        curl -X POST "!URL!/process/%2" ^
            -H "Authorization: Bearer $(gcloud auth print-identity-token)" ^
            -H "Content-Type: application/json"
    )
    exit /b 0
)

REM Build and push container
echo [1/3] Building and pushing container...
call gcloud builds submit ^
    --tag "%IMAGE_URI%" ^
    --project "%PROJECT_ID%"

if errorlevel 1 (
    echo ERROR: Container build failed
    exit /b 1
)

REM Create GCS bucket if needed
echo.
echo [2/3] Ensuring GCS bucket exists...
call gsutil ls -b "gs://%GCS_BUCKET%" >nul 2>&1
if errorlevel 1 (
    echo Creating bucket %GCS_BUCKET%...
    call gsutil mb -l "%REGION%" -p "%PROJECT_ID%" "gs://%GCS_BUCKET%"
)

REM Deploy to Cloud Run
echo.
echo [3/3] Deploying to Cloud Run...
call gcloud run deploy %SERVICE_NAME% ^
    --image "%IMAGE_URI%" ^
    --region "%REGION%" ^
    --project "%PROJECT_ID%" ^
    --set-env-vars "PROJECT_ID=%PROJECT_ID%" ^
    --set-env-vars "GCS_BUCKET=%GCS_BUCKET%" ^
    --set-env-vars "LOCATION=%REGION%" ^
    --memory 8Gi ^
    --cpu 4 ^
    --timeout 3600 ^
    --max-instances 10 ^
    --allow-unauthenticated ^
    --no-cpu-throttling

if errorlevel 1 (
    echo ERROR: Deployment failed
    exit /b 1
)

echo.
echo ================================================
echo Deployment complete!
echo ================================================
echo.
echo Service URL:
call gcloud run services describe %SERVICE_NAME% --region %REGION% --format "value(status.url)"
echo.
echo Test with:
echo   deploy.bat test YOUR_DRIVE_FILE_ID [OUTPUT_FOLDER_ID]
echo.
echo Or via curl:
echo   curl -X POST "https://%SERVICE_NAME%-%PROJECT_ID%.%REGION%.run.app/process/FILE_ID"
echo.

endlocal
