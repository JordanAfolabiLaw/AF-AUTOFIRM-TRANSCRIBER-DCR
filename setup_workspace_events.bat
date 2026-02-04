@echo off
REM ============================================================================
REM setup_workspace_events.bat
REM ============================================================================
REM Sets up Pub/Sub topic and subscription for Workspace Events API integration
REM with the AF-AUTOFIRM-TRANSCRIBER-DCR service.
REM
REM Prerequisites:
REM   - gcloud CLI authenticated with appropriate permissions
REM   - Workspace Events API enabled in Google Cloud Console
REM   - Service already deployed to Cloud Run
REM ============================================================================

setlocal enabledelayedexpansion

set PROJECT_ID=firmlink-alpha
set PROJECT_NUMBER=437295194277
set TOPIC_NAME=af-autofirm-dcr-drive-events
set SUBSCRIPTION_NAME=af-autofirm-dcr-push
set SERVICE_URL=https://af-autofirm-transcriber-dcr-wzmrud4omq-uc.a.run.app
set REGION=us-central1
set FOLDER_ID=1YUWj5t13Kstl5Vwv_Ee5mJpF2H9eYWaI

echo ================================================
echo Setting up Workspace Events API Integration
echo ================================================
echo Project: %PROJECT_ID%
echo Topic: %TOPIC_NAME%
echo Subscription: %SUBSCRIPTION_NAME%
echo Service URL: %SERVICE_URL%
echo Watched Folder: %FOLDER_ID%
echo.

REM Step 1: Create Pub/Sub topic
echo [1/5] Creating Pub/Sub topic...
call gcloud pubsub topics create %TOPIC_NAME% --project=%PROJECT_ID% 2>nul
if errorlevel 1 (
    echo Topic may already exist, continuing...
)

REM Step 2: Grant Pub/Sub permission to invoke Cloud Run
echo.
echo [2/5] Granting Pub/Sub permission to invoke Cloud Run...
call gcloud run services add-iam-policy-binding af-autofirm-transcriber-dcr ^
    --member="serviceAccount:service-%PROJECT_NUMBER%@gcp-sa-pubsub.iam.gserviceaccount.com" ^
    --role=roles/run.invoker ^
    --region=%REGION% ^
    --project=%PROJECT_ID%

REM Step 3: Create push subscription
echo.
echo [3/5] Creating Pub/Sub push subscription...
call gcloud pubsub subscriptions create %SUBSCRIPTION_NAME% ^
    --topic=%TOPIC_NAME% ^
    --push-endpoint=%SERVICE_URL%/webhook ^
    --ack-deadline=600 ^
    --message-retention-duration=1d ^
    --project=%PROJECT_ID% 2>nul
if errorlevel 1 (
    echo Subscription may already exist, continuing...
)

REM Step 4: Enable required APIs
echo.
echo [4/5] Enabling required APIs...
call gcloud services enable workspaceevents.googleapis.com --project=%PROJECT_ID%
call gcloud services enable driveactivity.googleapis.com --project=%PROJECT_ID%

echo.
echo [5/5] MANUAL STEP REQUIRED:
echo ================================================
echo Create Workspace Events subscription via API or Console.
echo.
echo Option A: Use Google Cloud Console
echo   1. Go to https://console.cloud.google.com/apis/library/workspaceevents.googleapis.com
echo   2. Navigate to Workspace Events API and create a subscription
echo.
echo Option B: Use REST API (via curl or Postman)
echo.
echo   POST https://workspaceevents.googleapis.com/v1/subscriptions
echo   Authorization: Bearer $(gcloud auth print-access-token)
echo   Content-Type: application/json
echo.
echo   {
echo     "targetResource": "//drive.googleapis.com/drives/%FOLDER_ID%",
echo     "eventTypes": [
echo       "google.workspace.drive.file.v1.created",
echo       "google.workspace.driveactivity.v2.activity_create"
echo     ],
echo     "notificationEndpoint": {
echo       "pubsubTopic": "projects/%PROJECT_ID%/topics/%TOPIC_NAME%"
echo     },
echo     "payloadOptions": {
echo       "includeResource": true
echo     }
echo   }
echo.
echo ================================================
echo.
echo Setup complete! Test by uploading a file to the watched folder.
echo Monitor logs with:
echo   gcloud run logs read af-autofirm-transcriber-dcr --region=%REGION% --limit=50

endlocal
