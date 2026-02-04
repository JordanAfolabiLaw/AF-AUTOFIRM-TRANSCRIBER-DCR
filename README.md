# AF-AUTOFIRM-TRANSCRIBER-DCR

Audio transcription service using **Gemini 2.5 Pro** with native DCR court recording support.

## Features

| Feature | Description |
|---------|-------------|
| **Standard Audio** | Transcribes m4a, wav, mp3, flac, ogg, etc. |
| **DCR Support** | Native Speex decoding for Liberty Court Recorder files |
| **Registrar Notes** | Auto-detects sidecar .txt files for speaker context |
| **REV009 Schema** | JSON output compatible with transcript library |
| **Sequential Chunking** | 60-minute chunks with speaker context passing |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  AF-AUTOFIRM-TRANSCRIBER-DCR                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Detection                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Standard Audio (.mp3, .wav, etc.)                   │   │
│  │   → FFmpeg extraction → Gemini 2.5 Pro              │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ DCR File (.dcr)                                     │   │
│  │   → Parse XML metadata                              │   │
│  │   → Extract Speex data                              │   │
│  │   → Decode with libspeex                            │   │
│  │   → Find registrar notes (.txt sidecar)             │   │
│  │   → Gemini 2.5 Pro with context                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Output                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ {filename}_TRANSCRIPT.json  (REV009 schema)         │   │
│  │ {filename}_TRANSCRIPT.txt   (Human-readable)        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## DCR Format

DCR files from Liberty Court Recorder contain:

- **Magic header**: `HGCRLCRS`
- **XML metadata**: Channel definitions, codec info, timestamps
- **Audio data**: Speex-encoded mono stream (16kHz, Q5)

The audio is a **single mixed mono stream** (not 8 separate channels). The 8 channel labels (JD, CR, DF, WT, PD, AC, AM, RT) indicate which microphones contributed to the mix.

### Registrar Notes

DCR files are often accompanied by `.txt` sidecar files containing court registrar notes:

```
14:33 ~ 2025-12-19 9:02:26 AM ~ Crown ~ Opening statement
14:35 ~ 2025-12-19 9:04:26 AM ~  ~ Witness #1 sworn in
14:40 ~ 2025-12-19 9:09:26 AM ~ Defence ~ Cross-examination begins
```

The transcriber automatically finds and uses these notes to provide context for speaker identification.

## Model Choice

Using **Gemini 2.5 Pro** (`gemini-2.5-pro-preview-05-06`) instead of Gemini 3 due to documented timestamp accuracy issues in Gemini 3 preview models. Court transcripts require precise timestamps.

| Model | Timestamp Accuracy | Status |
|-------|-------------------|--------|
| Gemini 2.5 Pro | ✅ Reliable | **Selected** |
| Gemini 3 Pro | ⚠️ Issues reported | Not used |
| Gemini 3 Flash | ⚠️ Issues reported | Not used |

## Deployment

```bash
# Deploy to Cloud Run
deploy.bat

# Test with a file
deploy.bat test YOUR_DRIVE_FILE_ID

# Test with specific output folder
deploy.bat test YOUR_DRIVE_FILE_ID OUTPUT_FOLDER_ID
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/process/<file_id>` | POST | Process file by Drive ID |
| `/process` | POST | Process with JSON body |
| `/webhook` | POST | Pub/Sub push notifications |
| `/drive-webhook` | POST/GET | Drive API push notifications |

### Request Examples

```bash
# Process by file ID
curl -X POST "https://SERVICE_URL/process/1abc123def456"

# Process with output folder
curl -X POST "https://SERVICE_URL/process/1abc123def456?output_folder=1xyz789"

# Process via JSON body
curl -X POST "https://SERVICE_URL/process" \
  -H "Content-Type: application/json" \
  -d '{"file_id": "1abc123def456", "output_folder_id": "1xyz789"}'
```

## Output Format

### JSON (REV009 Schema)

```json
{
  "library_meta": {
    "generated_at_utc": "2026-02-04T12:00:00.000Z",
    "source_scope": "AF-AUTOFIRM-TRANSCRIBER-DCR (gemini-2.5-pro-preview-05-06)"
  },
  "transcripts": [{
    "doc_id": "dcr-hearing_2025-12-19",
    "doc_type": "COURT_RECORDING",
    "lines": ["[00:00:05] CROWN: Good morning, Your Honour..."],
    "dcr_metadata": {
      "creation_time": "2025-12-19 9:02:26 AM",
      "channels": [{"code": "JD", "description": "Judge"}]
    }
  }]
}
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `PROJECT_ID` | `firmlink-alpha` | GCP project |
| `LOCATION` | `us-central1` | Vertex AI location |
| `GCS_BUCKET` | `af-autofirm-transcriber-dcr-bucket` | Temp storage |
| `MODEL_ID` | `gemini-2.5-pro-preview-05-06` | Gemini model |
| `ENGINES` | `G2,G3,V1` | Transcription engines to use |
| `WATCH_FOLDER_ID` | | Folder to watch for new files |

## Files

```
AF-AUTOFIRM-TRANSCRIBER-DCR/
├── main.py                     # Core application
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container with libspeex
├── deploy.bat                  # Windows deployment script
├── drive_watch_setup.py        # Drive webhook setup
├── setup_workspace_events.bat  # Workspace Events setup
└── README.md                   # This file
```

## Cost Estimate

| Component | Cost per Hour of Audio |
|-----------|------------------------|
| Gemini 2.5 Pro | ~$0.30 |
| Cloud Run | ~$0.10 |
| Cloud Storage | ~$0.01 |
| **Total** | **~$0.41/hour** |

For an 18-hour court recording: ~$7.50
