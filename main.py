"""
AF-AUTOFIRM-TRANSCRIBER-DCR
============================
Audio transcription service using Gemini 2.5 Pro.

Supports:
- Standard audio files (m4a, wav, mp3, etc.)
- DCR court recordings with Speex decoding
- Registrar notes context for DCR files

Outputs:
- JSON (REV009 schema-compatible)
- TXT (human-readable transcript)

Model: gemini-2.5-pro-preview-05-06 (better timestamp accuracy than Gemini 3)
"""

import os
import io
import re
import json
import uuid
import wave
import struct
import hashlib
import tempfile
import subprocess
import base64
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple
from functools import wraps
from dataclasses import dataclass
from pathlib import Path
import time
import ctypes
import ctypes.util

from flask import Flask, request, jsonify
import google.auth
from google.cloud import storage
from google.cloud import speech_v1 as speech
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ID = os.getenv("PROJECT_ID", "firmlink-alpha")
LOCATION = os.getenv("LOCATION", "us-central1")
GCS_BUCKET = os.getenv("GCS_BUCKET", "af-autofirm-transcriber-dcr-bucket")

# Workspace Events API / Pub/Sub webhook settings
WATCH_FOLDER_ID = os.getenv("WATCH_FOLDER_ID", "1YUWj5t13Kstl5Vwv_Ee5mJpF2H9eYWaI")
PROCESSED_EVENTS = {}  # In-memory cache for idempotency
EVENT_CACHE_TTL_MINUTES = 60

# Google Drive folder IDs (optional - can process files by ID directly)
INPUT_FOLDER_ID = os.getenv("INPUT_FOLDER_ID", "")
OUTPUT_FOLDER_ID = os.getenv("OUTPUT_FOLDER_ID", "")

# Model - using Gemini 2.5 Pro for better timestamp accuracy
MODEL_ID = os.getenv("MODEL_ID", "gemini-2.5-pro-preview-05-06")

# ============================================================================
# SPLIT TESTING - MULTIPLE ENGINES
# ============================================================================
# G2 = Gemini 2.5 Pro, G3 = Gemini 3 Pro, V1 = Vertex AI Speech-to-Text

TRANSCRIPTION_ENGINES = {
    'G2': {
        'name': 'Gemini 2.5 Pro',
        'model': 'gemini-2.5-pro-preview-05-06',
        'type': 'gemini'
    },
    'G3': {
        'name': 'Gemini 3 Pro',
        'model': 'gemini-3-pro-preview',
        'type': 'gemini'
    },
    'V1': {
        'name': 'Vertex AI Speech-to-Text',
        'model': 'latest_long',
        'type': 'speech_to_text'
    }
}

# Which engines to use (comma-separated, e.g., "G2,G3,V1")
ENABLED_ENGINES = os.getenv("ENGINES", "G2,G3,V1").split(",")
RUN_PARALLEL = os.getenv("RUN_PARALLEL", "true").lower() == "true"

# Chunking configuration
CHUNK_DURATION_MINUTES = int(os.getenv("CHUNK_DURATION_MINUTES", "60"))
CHUNK_DURATION_SECONDS = CHUNK_DURATION_MINUTES * 60
BLOCK_TARGET_WORDS = 200

# Supported MIME types
AUDIO_MIME_TYPES = {
    'audio/mpeg': '.mp3',
    'audio/mp3': '.mp3',
    'audio/wav': '.wav',
    'audio/x-wav': '.wav',
    'audio/wave': '.wav',
    'audio/mp4': '.m4a',
    'audio/x-m4a': '.m4a',
    'audio/aac': '.aac',
    'audio/ogg': '.ogg',
    'audio/flac': '.flac',
    'audio/webm': '.webm',
    'application/octet-stream': '.dcr',  # DCR files
}

DRIVE_SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.file',
]

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DCRMetadata:
    """Parsed DCR file metadata."""
    channels: List[Tuple[str, str]]  # (code, description)
    speex_format: str
    sample_rate: int
    app_version: str
    creation_time: str
    user: str


@dataclass
class RegistrarNote:
    """Single registrar note entry."""
    offset: str           # MM:SS format
    timestamp: str        # Full timestamp
    speaker: str          # Speaker name (may be empty)
    content: str          # Note content
    offset_seconds: float # Calculated seconds from start


# =============================================================================
# RETRY DECORATOR
# =============================================================================

def retry_with_backoff(max_retries: int = 3, delay: int = 30):
    """Decorator for retrying functions with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    wait_time = delay * (2 ** attempt)
                    print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
            raise last_exception
        return wrapper
    return decorator


# =============================================================================
# GOOGLE SERVICES
# =============================================================================

def get_drive_service():
    """Get authenticated Drive service."""
    credentials, _ = google.auth.default(scopes=DRIVE_SCOPES)
    return build('drive', 'v3', credentials=credentials)


def get_storage_client():
    """Get Cloud Storage client."""
    return storage.Client(project=PROJECT_ID)


def ensure_bucket_exists():
    """Create GCS bucket if needed."""
    client = get_storage_client()
    try:
        return client.get_bucket(GCS_BUCKET)
    except Exception:
        print(f"Creating bucket {GCS_BUCKET}...")
        return client.create_bucket(GCS_BUCKET, location=LOCATION)


# =============================================================================
# FILE OPERATIONS
# =============================================================================

def download_from_drive(file_id: str) -> Tuple[bytes, str, str, Optional[str]]:
    """
    Download file from Drive.
    
    Returns:
        Tuple of (bytes, filename, mime_type, folder_id)
    """
    drive_service = get_drive_service()
    
    file_info = drive_service.files().get(
        fileId=file_id,
        fields='id, name, mimeType, size, parents',
        supportsAllDrives=True
    ).execute()
    
    filename = file_info['name']
    mime_type = file_info['mimeType']
    parents = file_info.get('parents', [])
    folder_id = parents[0] if parents else None
    
    print(f"Downloading: {filename} ({mime_type})")
    
    request = drive_service.files().get_media(fileId=file_id, supportsAllDrives=True)
    file_content = io.BytesIO()
    downloader = MediaIoBaseDownload(file_content, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        if status:
            print(f"Download progress: {int(status.progress() * 100)}%")
    
    file_content.seek(0)
    return file_content.read(), filename, mime_type, folder_id


def find_sidecar_notes(file_id: str, filename: str, folder_id: Optional[str]) -> Optional[bytes]:
    """Find and download registrar notes sidecar file (.txt)."""
    if not folder_id:
        return None
    
    drive_service = get_drive_service()
    base_name = os.path.splitext(filename)[0]
    
    # Search for matching .txt file
    query = f"'{folder_id}' in parents and name contains '{base_name}' and mimeType = 'text/plain' and trashed = false"
    
    try:
        results = drive_service.files().list(
            q=query,
            fields="files(id, name)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        
        files = results.get('files', [])
        
        # Find exact match or close match
        for f in files:
            if f['name'] == f"{base_name}.txt":
                print(f"Found registrar notes: {f['name']}")
                request = drive_service.files().get_media(fileId=f['id'], supportsAllDrives=True)
                content = io.BytesIO()
                downloader = MediaIoBaseDownload(content, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                content.seek(0)
                return content.read()
    except Exception as e:
        print(f"Warning: Could not search for sidecar notes: {e}")
    
    return None


def upload_to_gcs(content: bytes, blob_path: str, 
                  content_type: str = 'application/octet-stream') -> str:
    """Upload content to GCS. Returns gs:// URI."""
    bucket = ensure_bucket_exists()
    blob = bucket.blob(blob_path)
    blob.upload_from_string(content, content_type=content_type)
    return f"gs://{GCS_BUCKET}/{blob_path}"


def delete_from_gcs(blob_path: str):
    """Delete temp file from GCS."""
    try:
        client = get_storage_client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(blob_path)
        blob.delete()
    except Exception as e:
        print(f"Warning: Could not delete {blob_path}: {e}")


def upload_to_drive(filename: str, content: str, folder_id: str, 
                    mime_type: str = 'text/plain') -> dict:
    """Upload file to Drive folder."""
    drive_service = get_drive_service()
    
    file_metadata = {
        'name': filename,
        'parents': [folder_id]
    }
    
    media = MediaIoBaseUpload(
        io.BytesIO(content.encode('utf-8') if isinstance(content, str) else content),
        mimetype=mime_type,
        resumable=True
    )
    
    print(f"Uploading to Drive: {filename}")
    
    return drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id, name, webViewLink',
        supportsAllDrives=True
    ).execute()


# =============================================================================
# DCR PARSING
# =============================================================================

def parse_dcr_metadata(data: bytes) -> Optional[DCRMetadata]:
    """Parse XML metadata from DCR header."""
    import xml.etree.ElementTree as ET
    
    start_marker = b'meta#bgn'
    end_marker = b'meta#end'
    
    start_idx = data.find(start_marker)
    end_idx = data.find(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        return None
    
    xml_start = start_idx + len(start_marker) + 4
    xml_end = end_idx
    
    xml_bytes = data[xml_start:xml_end]
    
    try:
        xml_str = xml_bytes.decode('utf-8')
    except UnicodeDecodeError:
        xml_str = xml_bytes.decode('latin-1')
    
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as e:
        print(f"XML Parse Error: {e}")
        return None
    
    # Extract channels
    channels = []
    for item in root.findall('.//LISTCHNL/ITEM'):
        name_elem = item.find('CHNAME')
        desc_elem = item.find('CHDESC')
        if name_elem is not None and desc_elem is not None:
            channels.append((name_elem.text or "", desc_elem.text or ""))
    
    # Extract format
    audio_elem = root.find('.//AUDIO')
    fmt_name = ""
    if audio_elem is not None:
        fmt_elem = audio_elem.find('FMTNAME')
        if fmt_elem is not None:
            fmt_name = fmt_elem.text or ""
    
    # Extract app info
    prop_elem = root.find('.//PROP')
    app_version = ""
    time_cr = ""
    user_log = ""
    
    if prop_elem is not None:
        app_version = prop_elem.get('VERAPP', '')
        name_app = prop_elem.find('NAMEAPP')
        if name_app is not None and name_app.text:
            app_version = f"{name_app.text} v{app_version}"
        
        time_elem = prop_elem.find('TIMECR')
        if time_elem is not None:
            time_cr = time_elem.text or ""
        
        user_elem = prop_elem.find('USERLOG')
        if user_elem is not None:
            user_log = user_elem.text or ""
    
    # Parse sample rate
    sample_rate = 16000
    if "8.0 kHz" in fmt_name:
        sample_rate = 8000
    elif "32.0 kHz" in fmt_name:
        sample_rate = 32000
    
    return DCRMetadata(
        channels=channels,
        speex_format=fmt_name,
        sample_rate=sample_rate,
        app_version=app_version,
        creation_time=time_cr,
        user=user_log
    )


def extract_dcr_audio_data(dcr_bytes: bytes) -> Tuple[bytes, int]:
    """
    Extract raw Speex audio data from DCR file.
    
    Returns:
        Tuple of (audio_data, audio_offset)
    """
    data_marker = b'data#bgn'
    data_idx = dcr_bytes.find(data_marker)
    
    if data_idx == -1:
        raise ValueError("Could not find audio data section in DCR")
    
    audio_offset = data_idx + len(data_marker) + 4
    audio_data = dcr_bytes[audio_offset:]
    
    return audio_data, audio_offset


def parse_registrar_notes(notes_bytes: bytes) -> List[RegistrarNote]:
    """
    Parse registrar notes from TXT sidecar file.
    
    Format: MM:SS ~ YYYY-MM-DD HH:MM:SS AM/PM ~ [speaker] ~ [note]
    """
    pattern = r'^(\d+:\d+) ~ (\d{4}-\d{2}-\d{2} \d+:\d+:\d+ [AP]M)(?: ~ ([^~]*))?(?: ~ (.*))?$'
    notes = []
    
    try:
        text = notes_bytes.decode('utf-8')
    except UnicodeDecodeError:
        text = notes_bytes.decode('latin-1')
    
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        
        match = re.match(pattern, line)
        if match:
            offset, timestamp, speaker, content = match.groups()
            
            # Parse offset to seconds
            parts = offset.split(':')
            offset_seconds = int(parts[0]) * 60 + int(parts[1])
            
            notes.append(RegistrarNote(
                offset=offset,
                timestamp=timestamp,
                speaker=(speaker or '').strip(),
                content=(content or '').strip(),
                offset_seconds=offset_seconds
            ))
    
    return notes


# =============================================================================
# SPEEX DECODING
# =============================================================================

def decode_speex_with_libspeex(speex_data: bytes, sample_rate: int = 16000) -> Optional[bytes]:
    """
    Decode raw Speex frames using libspeex via ctypes.
    
    Returns:
        PCM audio data (16-bit signed, mono) or None on failure
    """
    # Try to load libspeex
    lib_paths = [
        ctypes.util.find_library('speex'),
        'libspeex.so.1',
        'libspeex.so',
        '/usr/lib/x86_64-linux-gnu/libspeex.so.1',
        '/usr/lib/libspeex.so.1',
    ]
    
    lib = None
    for path in lib_paths:
        if path:
            try:
                lib = ctypes.CDLL(path)
                print(f"Loaded libspeex: {path}")
                break
            except OSError:
                continue
    
    if lib is None:
        print("Could not load libspeex, falling back to FFmpeg")
        return None
    
    try:
        # Define Speex functions
        # speex_decoder_init(const SpeexMode *mode)
        lib.speex_decoder_init.restype = ctypes.c_void_p
        lib.speex_decoder_init.argtypes = [ctypes.c_void_p]
        
        # speex_decode_int(void *state, SpeexBits *bits, spx_int16_t *out)
        lib.speex_decode_int.restype = ctypes.c_int
        lib.speex_decode_int.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int16)]
        
        # speex_bits_init(SpeexBits *bits)
        lib.speex_bits_init.restype = None
        lib.speex_bits_init.argtypes = [ctypes.c_void_p]
        
        # speex_bits_read_from(SpeexBits *bits, char *bytes, int len)
        lib.speex_bits_read_from.restype = None
        lib.speex_bits_read_from.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        
        # Get the wideband mode (16kHz)
        if sample_rate == 16000:
            mode = ctypes.c_void_p.in_dll(lib, 'speex_wb_mode')
        else:
            mode = ctypes.c_void_p.in_dll(lib, 'speex_nb_mode')
        
        # Initialize decoder
        decoder = lib.speex_decoder_init(mode)
        if not decoder:
            print("Failed to initialize Speex decoder")
            return None
        
        # SpeexBits structure (simplified - actual size is larger)
        bits = (ctypes.c_char * 200)()  # Allocate enough space
        lib.speex_bits_init(ctypes.cast(bits, ctypes.c_void_p))
        
        # Frame parameters
        frame_size = 42  # Bytes per Speex frame (from DCR FMTDATA)
        output_frame_samples = 320  # Samples per decoded frame (wideband)
        
        pcm_data = bytearray()
        output_buffer = (ctypes.c_int16 * output_frame_samples)()
        
        offset = 0
        frame_count = 0
        
        while offset + frame_size <= len(speex_data):
            frame = speex_data[offset:offset + frame_size]
            
            # Read frame into bits
            lib.speex_bits_read_from(
                ctypes.cast(bits, ctypes.c_void_p),
                frame,
                len(frame)
            )
            
            # Decode
            result = lib.speex_decode_int(
                decoder,
                ctypes.cast(bits, ctypes.c_void_p),
                output_buffer
            )
            
            if result == 0:
                # Success - copy samples
                for i in range(output_frame_samples):
                    pcm_data.extend(struct.pack('<h', output_buffer[i]))
                frame_count += 1
            
            offset += frame_size
            
            if frame_count % 1000 == 0:
                print(f"  Decoded {frame_count} frames...")
        
        # Cleanup
        lib.speex_decoder_destroy(decoder)
        
        print(f"Decoded {frame_count} frames, {len(pcm_data)} bytes PCM")
        return bytes(pcm_data)
        
    except Exception as e:
        print(f"libspeex decoding error: {e}")
        return None


def decode_speex_with_ffmpeg(speex_data: bytes, sample_rate: int = 16000) -> Optional[bytes]:
    """
    Attempt to decode Speex data using FFmpeg.
    
    This is a fallback method that may not work for raw Speex frames.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = os.path.join(tmpdir, "audio.raw")
        wav_path = os.path.join(tmpdir, "output.wav")
        
        with open(raw_path, 'wb') as f:
            f.write(speex_data)
        
        # Try different FFmpeg approaches
        attempts = [
            # Raw Speex (unlikely to work without container)
            ['ffmpeg', '-y', '-f', 'speex', '-ar', str(sample_rate), '-ac', '1',
             '-i', raw_path, '-acodec', 'pcm_s16le', '-ar', str(sample_rate), wav_path],
            # Try as raw audio
            ['ffmpeg', '-y', '-f', 's16le', '-ar', str(sample_rate), '-ac', '1',
             '-i', raw_path, '-acodec', 'pcm_s16le', wav_path],
        ]
        
        for cmd in attempts:
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=60)
                if result.returncode == 0 and os.path.exists(wav_path):
                    with open(wav_path, 'rb') as f:
                        return f.read()
            except Exception:
                continue
    
    return None


def dcr_to_wav(dcr_bytes: bytes) -> Tuple[bytes, DCRMetadata]:
    """
    Convert DCR file bytes to WAV format.
    
    Returns:
        Tuple of (wav_bytes, metadata)
    """
    # Parse metadata
    metadata = parse_dcr_metadata(dcr_bytes[:16384])
    if metadata is None:
        raise ValueError("Failed to parse DCR metadata")
    
    print(f"DCR Metadata:")
    print(f"  Format: {metadata.speex_format}")
    print(f"  Sample Rate: {metadata.sample_rate} Hz")
    print(f"  Created: {metadata.creation_time}")
    print(f"  Channels: {len(metadata.channels)}")
    
    # Extract audio data
    audio_data, offset = extract_dcr_audio_data(dcr_bytes)
    print(f"  Audio data: {len(audio_data):,} bytes at offset 0x{offset:X}")
    
    # Try libspeex first
    pcm_data = decode_speex_with_libspeex(audio_data, metadata.sample_rate)
    
    # Fallback to FFmpeg
    if pcm_data is None:
        print("Trying FFmpeg fallback...")
        wav_data = decode_speex_with_ffmpeg(audio_data, metadata.sample_rate)
        if wav_data:
            return wav_data, metadata
        raise RuntimeError("Failed to decode Speex audio - libspeex not available and FFmpeg fallback failed")
    
    # Create WAV file
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(metadata.sample_rate)
        wav.writeframes(pcm_data)
    
    wav_buffer.seek(0)
    return wav_buffer.read(), metadata


# =============================================================================
# AUDIO PROCESSING
# =============================================================================

def get_audio_duration(audio_bytes: bytes, filename: str) -> float:
    """Get duration of audio file in seconds using FFprobe."""
    ext = os.path.splitext(filename)[1].lower() or '.wav'
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, f"input{ext}")
        
        with open(input_path, 'wb') as f:
            f.write(audio_bytes)
        
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            input_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFprobe failed: {result.stderr[:500]}")
        
        return float(result.stdout.strip())


def extract_audio_chunk(audio_bytes: bytes, filename: str,
                        start_seconds: int, duration_seconds: int) -> bytes:
    """Extract a chunk of audio."""
    ext = os.path.splitext(filename)[1].lower() or '.wav'
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, f"input{ext}")
        output_path = os.path.join(tmpdir, f"output{ext}")
        
        with open(input_path, 'wb') as f:
            f.write(audio_bytes)
        
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_seconds),
            '-i', input_path,
            '-t', str(duration_seconds),
            '-c', 'copy',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            # Fallback with re-encoding
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_seconds),
                '-i', input_path,
                '-t', str(duration_seconds),
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr[:500]}")
        
        with open(output_path, 'rb') as f:
            return f.read()


def detect_input_type(filename: str, mime_type: str) -> str:
    """Detect if input is standard audio or DCR."""
    if filename.lower().endswith('.dcr'):
        return 'dcr'
    if mime_type == 'application/octet-stream' and filename.lower().endswith('.dcr'):
        return 'dcr'
    return 'standard'


# =============================================================================
# TRANSCRIPTION
# =============================================================================

def get_transcription_prompt(chunk_index: int, total_chunks: int,
                             registrar_notes: Optional[List[RegistrarNote]] = None,
                             speaker_context: Optional[str] = None,
                             is_court_recording: bool = False) -> str:
    """Generate transcription prompt with context."""
    
    context_section = ""
    if speaker_context:
        context_section = f"""
## SPEAKER CONTEXT FROM PREVIOUS CHUNKS
{speaker_context}

CRITICAL: Use the EXACT same speaker labels for the same voices.
"""
    
    notes_section = ""
    if registrar_notes:
        notes_text = "\n".join([
            f"  [{n.offset}] {n.speaker}: {n.content}" if n.speaker else f"  [{n.offset}] {n.content}"
            for n in registrar_notes[:50]  # Limit to first 50 notes
        ])
        notes_section = f"""
## REGISTRAR NOTES (for context and speaker identification)
These are official court notes with timestamps. Use them to identify speakers and key moments:
{notes_text}
"""
    
    court_context = ""
    if is_court_recording:
        court_context = """
## COURT RECORDING CONTEXT
This is a court proceeding. Common speakers include:
- Judge (JD) - Presiding over proceedings
- Crown (CR) - Crown counsel/prosecutor  
- Defence (DF) - Defence counsel
- Witness (WT) - Person giving testimony
- Accused (AC) - Defendant
- Clerk - Court clerk administering oaths

Use these role labels when you can identify speakers (e.g., "CROWN:", "JUDGE:", "WITNESS:").
"""
    
    return f"""You are an expert legal transcriptionist. Transcribe this audio with precise speaker diarization.

## CHUNK INFORMATION
This is chunk {chunk_index + 1} of {total_chunks}.
{context_section}
{court_context}
{notes_section}

## TRANSCRIPTION REQUIREMENTS

1. **Speaker Identification**
   - For court recordings: Use role labels (JUDGE, CROWN, DEFENCE, WITNESS, CLERK, ACCUSED)
   - For other audio: Use Speaker A, Speaker B, etc.
   - Be consistent across the transcript

2. **Timestamps**
   - Include timestamp at speaker changes: [HH:MM:SS] format
   - Timestamps are relative to THIS chunk (starting from 00:00:00)

3. **Verbatim Accuracy**
   - Transcribe EVERY word exactly as spoken
   - Include filler words (um, uh, like)
   - Include false starts and self-corrections
   - Mark unclear audio as [inaudible] or [unclear: best guess]

4. **Legal Proceedings**
   - Note key procedural moments (objections, rulings, exhibits)
   - Preserve legal terminology exactly

## OUTPUT FORMAT
Respond with ONLY valid JSON:

{{
  "utterances": [
    {{
      "speaker": "CROWN",
      "start_time": "00:00:05",
      "end_time": "00:00:45",
      "text": "Good morning, Your Honour. The Crown calls its first witness."
    }}
  ],
  "speaker_summary": {{
    "CROWN": "Male voice, Crown counsel",
    "JUDGE": "Female voice, presiding judge"
  }},
  "chunk_notes": "Any observations about audio quality, proceedings, etc."
}}

CRITICAL: Output ONLY valid JSON. No markdown, no explanations."""


@retry_with_backoff(max_retries=3, delay=60)
def transcribe_chunk_with_gemini(audio_bytes: bytes, filename: str, mime_type: str,
                                  chunk_index: int, total_chunks: int,
                                  registrar_notes: Optional[List[RegistrarNote]] = None,
                                  speaker_context: Optional[str] = None,
                                  is_court_recording: bool = False) -> dict:
    """Transcribe a single audio chunk using Gemini 2.5 Pro."""
    
    model = GenerativeModel(MODEL_ID)
    
    # Upload to GCS temporarily
    ensure_bucket_exists()
    blob_path = f"temp/{uuid.uuid4()}/{filename}"
    gcs_uri = upload_to_gcs(audio_bytes, blob_path, mime_type)
    
    try:
        audio_part = Part.from_uri(uri=gcs_uri, mime_type=mime_type)
        
        prompt = get_transcription_prompt(
            chunk_index, total_chunks,
            registrar_notes, speaker_context, is_court_recording
        )
        
        print(f"Transcribing chunk {chunk_index + 1}/{total_chunks} with {MODEL_ID}...")
        
        response = model.generate_content(
            [audio_part, prompt],
            generation_config={
                "max_output_tokens": 65536,
                "temperature": 0.1,
            }
        )
        
        return parse_gemini_response(response.text, chunk_index)
        
    finally:
        delete_from_gcs(blob_path)


def parse_gemini_response(response_text: str, chunk_index: int) -> dict:
    """Parse Gemini response JSON."""
    text = response_text.strip()
    
    # Remove markdown fences
    if text.startswith('```'):
        first_newline = text.find('\n')
        if first_newline > 0:
            text = text[first_newline + 1:]
        if text.rstrip().endswith('```'):
            text = text.rstrip()[:-3]
        text = text.strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"JSON parse error for chunk {chunk_index}: {e}")
        
        # Try to find JSON boundaries
        try:
            start = text.index('{')
            end = text.rindex('}') + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return {
                "utterances": [],
                "speaker_summary": {},
                "chunk_notes": f"Parse error: {str(e)[:200]}",
                "raw_text": text[:5000]
            }


def transcribe_audio(audio_bytes: bytes, filename: str, mime_type: str,
                     registrar_notes: Optional[List[RegistrarNote]] = None,
                     is_court_recording: bool = False) -> dict:
    """
    Transcribe audio file with sequential chunking and context passing.
    """
    # Get duration
    total_duration = get_audio_duration(audio_bytes, filename)
    print(f"Audio duration: {total_duration:.1f}s ({total_duration/3600:.2f}h)")
    
    # Calculate chunks
    num_chunks = max(1, int(total_duration / CHUNK_DURATION_SECONDS) + 
                     (1 if total_duration % CHUNK_DURATION_SECONDS > 60 else 0))
    
    print(f"Processing {num_chunks} chunks of ~{CHUNK_DURATION_MINUTES} minutes each")
    
    all_utterances = []
    speaker_context = None
    speaker_mapping = {}
    
    for chunk_idx in range(num_chunks):
        start_seconds = chunk_idx * CHUNK_DURATION_SECONDS
        chunk_duration = min(CHUNK_DURATION_SECONDS, total_duration - start_seconds)
        
        if chunk_duration <= 0:
            break
        
        print(f"\n--- Chunk {chunk_idx + 1}/{num_chunks} ---")
        print(f"Time: {start_seconds}s - {start_seconds + chunk_duration}s")
        
        # Get relevant registrar notes for this chunk
        chunk_notes = None
        if registrar_notes:
            chunk_end = start_seconds + chunk_duration
            chunk_notes = [
                n for n in registrar_notes
                if start_seconds <= n.offset_seconds <= chunk_end
            ]
        
        # Extract chunk
        chunk_audio = extract_audio_chunk(
            audio_bytes, filename, int(start_seconds), int(chunk_duration)
        )
        
        # Transcribe
        chunk_result = transcribe_chunk_with_gemini(
            chunk_audio, f"chunk_{chunk_idx:02d}.wav",
            'audio/wav', chunk_idx, num_chunks,
            chunk_notes, speaker_context, is_court_recording
        )
        
        # Process utterances with absolute timestamps
        for utt in chunk_result.get('utterances', []):
            utt_copy = utt.copy()
            utt_copy['chunk_index'] = chunk_idx
            utt_copy['absolute_start_seconds'] = start_seconds + parse_timestamp(utt.get('start_time', '00:00:00'))
            utt_copy['absolute_end_seconds'] = start_seconds + parse_timestamp(utt.get('end_time', '00:00:00'))
            all_utterances.append(utt_copy)
        
        # Update speaker context
        if chunk_result.get('speaker_summary'):
            speaker_mapping.update(chunk_result['speaker_summary'])
            speaker_context = "\n".join([
                f"- {speaker}: {desc}"
                for speaker, desc in speaker_mapping.items()
            ])
        
        print(f"Chunk {chunk_idx + 1}: {len(chunk_result.get('utterances', []))} utterances")
    
    return {
        "utterances": all_utterances,
        "speaker_mapping": speaker_mapping,
        "total_duration_seconds": total_duration,
        "chunks_processed": num_chunks
    }


def parse_timestamp(ts: str) -> float:
    """Parse timestamp string to seconds."""
    if not ts:
        return 0.0
    
    parts = ts.replace('[', '').replace(']', '').split(':')
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except (ValueError, IndexError):
        return 0.0


# =============================================================================
# VERTEX AI SPEECH-TO-TEXT (V1 Engine)
# =============================================================================

@retry_with_backoff(max_retries=3, delay=60)
def transcribe_with_speech_to_text(audio_bytes: bytes, filename: str, 
                                    mime_type: str,
                                    registrar_notes: Optional[List[RegistrarNote]] = None,
                                    is_court_recording: bool = False) -> dict:
    """
    Transcribe audio using Vertex AI Speech-to-Text.
    
    This is the V1 engine for split testing comparison.
    """
    print(f"Transcribing with Vertex AI Speech-to-Text (V1)...")
    
    # Upload to GCS for long audio
    blob_path = f"temp/v1/{uuid.uuid4()}/{filename}"
    gcs_uri = upload_to_gcs(audio_bytes, blob_path, mime_type)
    
    try:
        client = speech.SpeechClient()
        
        # Configure recognition
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16 if mime_type == 'audio/wav' \
                else speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
            sample_rate_hertz=16000 if mime_type == 'audio/wav' else None,
            language_code="en-US",
            enable_speaker_diarization=True,
            diarization_speaker_count=8,  # Expect up to 8 speakers (court)
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,
            model="latest_long",
            use_enhanced=True
        )
        
        audio = speech.RecognitionAudio(uri=gcs_uri)
        
        # Use long running recognize for longer audio
        operation = client.long_running_recognize(config=config, audio=audio)
        
        print("Waiting for Speech-to-Text operation to complete...")
        response = operation.result(timeout=7200)  # 2 hour timeout
        
        # Process results into utterance format
        all_utterances = []
        speaker_mapping = {}
        
        for result in response.results:
            if not result.alternatives:
                continue
            
            alternative = result.alternatives[0]
            
            # Get words with speaker tags
            words_by_speaker = {}
            for word in alternative.words:
                speaker_tag = getattr(word, 'speaker_tag', 1) or 1
                speaker = f"Speaker {speaker_tag}"
                
                if speaker not in words_by_speaker:
                    words_by_speaker[speaker] = []
                
                words_by_speaker[speaker].append({
                    'word': word.word,
                    'start_time': word.start_time.total_seconds() if hasattr(word.start_time, 'total_seconds') else 0,
                    'end_time': word.end_time.total_seconds() if hasattr(word.end_time, 'total_seconds') else 0
                })
        
        # Group consecutive words by speaker into utterances
        current_speaker = None
        current_words = []
        current_start = 0
        
        for word in alternative.words:
            speaker_tag = getattr(word, 'speaker_tag', 1) or 1
            speaker = f"Speaker {speaker_tag}"
            
            if speaker != current_speaker:
                if current_words:
                    # Save previous utterance
                    hours = int(current_start // 3600)
                    minutes = int((current_start % 3600) // 60)
                    seconds = int(current_start % 60)
                    end_time = current_words[-1]['end_time']
                    end_h = int(end_time // 3600)
                    end_m = int((end_time % 3600) // 60)
                    end_s = int(end_time % 60)
                    
                    all_utterances.append({
                        'speaker': current_speaker,
                        'start_time': f"{hours:02d}:{minutes:02d}:{seconds:02d}",
                        'end_time': f"{end_h:02d}:{end_m:02d}:{end_s:02d}",
                        'text': ' '.join([w['word'] for w in current_words]),
                        'absolute_start_seconds': current_start,
                        'absolute_end_seconds': end_time
                    })
                    speaker_mapping[current_speaker] = f"Detected speaker {speaker_tag}"
                
                current_speaker = speaker
                current_words = []
                current_start = word.start_time.total_seconds() if hasattr(word.start_time, 'total_seconds') else 0
            
            current_words.append({
                'word': word.word,
                'start_time': word.start_time.total_seconds() if hasattr(word.start_time, 'total_seconds') else 0,
                'end_time': word.end_time.total_seconds() if hasattr(word.end_time, 'total_seconds') else 0
            })
        
        # Handle last utterance
        if current_words:
            hours = int(current_start // 3600)
            minutes = int((current_start % 3600) // 60)
            seconds = int(current_start % 60)
            end_time = current_words[-1]['end_time']
            end_h = int(end_time // 3600)
            end_m = int((end_time % 3600) // 60)
            end_s = int(end_time % 60)
            
            all_utterances.append({
                'speaker': current_speaker,
                'start_time': f"{hours:02d}:{minutes:02d}:{seconds:02d}",
                'end_time': f"{end_h:02d}:{end_m:02d}:{end_s:02d}",
                'text': ' '.join([w['word'] for w in current_words]),
                'absolute_start_seconds': current_start,
                'absolute_end_seconds': end_time
            })
            speaker_mapping[current_speaker] = f"Detected speaker"
        
        print(f"V1 transcription complete: {len(all_utterances)} utterances")
        
        return {
            "utterances": all_utterances,
            "speaker_mapping": speaker_mapping,
            "total_duration_seconds": all_utterances[-1]['absolute_end_seconds'] if all_utterances else 0,
            "chunks_processed": 1,
            "engine": "V1"
        }
        
    finally:
        delete_from_gcs(blob_path)


def transcribe_with_engine(engine_id: str, audio_bytes: bytes, filename: str,
                           mime_type: str, registrar_notes: Optional[List[RegistrarNote]] = None,
                           is_court_recording: bool = False) -> Tuple[str, dict]:
    """
    Transcribe audio with a specific engine.
    
    Args:
        engine_id: One of 'G2', 'G3', 'V1'
        
    Returns:
        Tuple of (engine_id, transcription_result)
    """
    engine = TRANSCRIPTION_ENGINES.get(engine_id)
    if not engine:
        raise ValueError(f"Unknown engine: {engine_id}")
    
    print(f"\n{'='*60}")
    print(f"Starting transcription with {engine['name']} ({engine_id})")
    print(f"{'='*60}")
    
    if engine['type'] == 'speech_to_text':
        result = transcribe_with_speech_to_text(
            audio_bytes, filename, mime_type,
            registrar_notes, is_court_recording
        )
    else:
        # Gemini (G2 or G3)
        # Temporarily override MODEL_ID for this transcription
        original_model = globals().get('MODEL_ID')
        globals()['MODEL_ID'] = engine['model']
        
        try:
            result = transcribe_audio(
                audio_bytes, filename, mime_type,
                registrar_notes, is_court_recording
            )
        finally:
            globals()['MODEL_ID'] = original_model
    
    result['engine'] = engine_id
    result['engine_name'] = engine['name']
    result['model'] = engine['model']
    
    return (engine_id, result)


# =============================================================================
# OUTPUT GENERATION (REV009 Schema)
# =============================================================================

def format_timestamp(seconds: float) -> str:
    """Format seconds to HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def create_blocks(utterances: List[dict], lines: List[str], 
                  target_words: int = BLOCK_TARGET_WORDS) -> List[dict]:
    """Create blocks targeting ~200 words each."""
    if not utterances:
        return []
    
    blocks = []
    current_block_utterances = []
    current_word_count = 0
    doc_uuid = uuid.uuid4().hex[:8]
    
    for idx, utt in enumerate(utterances):
        text = utt.get('text', '')
        word_count = len(text.split())
        
        current_block_utterances.append((idx, utt))
        current_word_count += word_count
        
        if current_word_count >= target_words or idx == len(utterances) - 1:
            block_start = current_block_utterances[0][0]
            block_end = current_block_utterances[-1][0]
            
            block_text = '\n'.join(lines[block_start:block_end + 1])
            speakers = list(set(u[1].get('speaker', 'Unknown') for u in current_block_utterances))
            
            context_prev = lines[block_start - 1] if block_start > 0 else ""
            context_next = lines[block_end + 1] if block_end < len(lines) - 1 else ""
            
            block = {
                "block_id": f"{doc_uuid}-{len(blocks)+1:04d}",
                "ln": {"start": block_start + 1, "end": block_end + 1},
                "text": block_text,
                "ctx": {
                    "prev_ln": block_start if block_start > 0 else None,
                    "next_ln": block_end + 2 if block_end < len(lines) - 1 else None
                },
                "context_prev": context_prev,
                "context_next": context_next,
                "people_tags": [],
                "time_tags": [],
                "contradiction_topics": [],
                "assertion_types": [],
                "speaker_tags": speakers,
                "start_seconds": current_block_utterances[0][1].get('absolute_start_seconds', 0),
                "end_seconds": current_block_utterances[-1][1].get('absolute_end_seconds', 0),
                "word_count": current_word_count,
                "contains_page_markers": False
            }
            
            blocks.append(block)
            current_block_utterances = []
            current_word_count = 0
    
    return blocks


def generate_json_output(transcription: dict, filename: str, 
                         dcr_metadata: Optional[DCRMetadata] = None,
                         registrar_notes: Optional[List[RegistrarNote]] = None) -> dict:
    """Generate REV009 schema-compatible JSON."""
    
    utterances = transcription.get('utterances', [])
    date_prefix = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    base_name = os.path.splitext(filename)[0]
    
    # Determine doc_type
    doc_type = "AUDIO_TRANSCRIPT"
    if dcr_metadata:
        doc_type = "COURT_RECORDING"
    
    # Generate lines
    lines = []
    for utt in utterances:
        ts = format_timestamp(utt.get('absolute_start_seconds', 0))
        speaker = utt.get('speaker', 'Unknown')
        text = utt.get('text', '')
        lines.append(f"[{ts}] {speaker}: {text}")
    
    # Generate blocks
    blocks = create_blocks(utterances, lines)
    
    # Get participants
    participants = list(set(utt.get('speaker', 'Unknown') for utt in utterances))
    
    # Hash
    full_text = '\n'.join(lines)
    text_hash = hashlib.sha256(full_text.encode('utf-8')).hexdigest()
    
    # Build doc_id
    if dcr_metadata and dcr_metadata.creation_time:
        # Parse DCR creation time for doc_id
        doc_id = f"dcr-{base_name.replace(' ', '_')}"
    else:
        doc_id = f"{date_prefix}_{base_name.replace(' ', '_')}_Transcript"
    
    result = {
        "library_meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat() + "Z",
            "source_scope": f"AF-AUTOFIRM-TRANSCRIBER-DCR ({MODEL_ID})",
            "ingested_files": [filename],
            "revisions": []
        },
        "tag_legend": {
            "time_tags": {},
            "people_tags": {p: f"Speaker: {p}" for p in participants},
            "weights": {
                "0.2": "brief mention",
                "0.5": "material detail",
                "0.8-1.0": "core detail"
            }
        },
        "transcripts": [{
            "doc_id": doc_id,
            "doc_title": f"Audio Transcript — {base_name}",
            "original_filename": filename,
            "standard_filename": f"{date_prefix}-{doc_type}_{base_name.replace(' ', '_')}.txt",
            "doc_type": doc_type,
            "date_prefix": date_prefix,
            "primary_person": participants[0] if participants else "Unknown",
            "participants": participants,
            "source_role": "court_recording" if dcr_metadata else "audio_recording",
            "source_format": f"Gemini_2.5_Pro_Transcription",
            "full_text_hash": text_hash,
            "lines": lines,
            "blocks": blocks,
            "duration_seconds": transcription.get('total_duration_seconds', 0),
            "speaker_mapping": transcription.get('speaker_mapping', {})
        }],
        "indexes": {
            "by_speaker": {},
            "by_time": {}
        }
    }
    
    # Add DCR-specific metadata
    if dcr_metadata:
        result["transcripts"][0]["dcr_metadata"] = {
            "creation_time": dcr_metadata.creation_time,
            "user": dcr_metadata.user,
            "app_version": dcr_metadata.app_version,
            "channels": [{"code": c[0], "description": c[1]} for c in dcr_metadata.channels]
        }
    
    # Add registrar notes if available
    if registrar_notes:
        result["transcripts"][0]["registrar_notes"] = [
            {
                "offset": n.offset,
                "timestamp": n.timestamp,
                "speaker": n.speaker,
                "content": n.content
            }
            for n in registrar_notes
        ]
    
    # Build indexes
    for block in blocks:
        for speaker in block.get('speaker_tags', []):
            if speaker not in result["indexes"]["by_speaker"]:
                result["indexes"]["by_speaker"][speaker] = []
            result["indexes"]["by_speaker"][speaker].append(block["block_id"])
    
    return result


def generate_txt_output(json_data: dict) -> str:
    """Generate human-readable TXT transcript."""
    lines = []
    transcript = json_data.get('transcripts', [{}])[0]
    
    # Header
    lines.append("=" * 80)
    lines.append("AUDIO TRANSCRIPTION")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Document ID: {transcript.get('doc_id', 'Unknown')}")
    lines.append(f"Source File: {transcript.get('original_filename', 'Unknown')}")
    lines.append(f"Generated: {json_data.get('library_meta', {}).get('generated_at_utc', 'Unknown')}")
    
    duration = transcript.get('duration_seconds', 0)
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    lines.append(f"Duration: {hours}h {minutes}m")
    lines.append(f"Model: {MODEL_ID}")
    lines.append("")
    
    # DCR Metadata
    if 'dcr_metadata' in transcript:
        dcr = transcript['dcr_metadata']
        lines.append("DCR RECORDING INFO:")
        lines.append("-" * 40)
        lines.append(f"  Created: {dcr.get('creation_time', 'Unknown')}")
        lines.append(f"  User: {dcr.get('user', 'Unknown')}")
        lines.append(f"  Application: {dcr.get('app_version', 'Unknown')}")
        if dcr.get('channels'):
            lines.append("  Channels:")
            for ch in dcr['channels']:
                lines.append(f"    [{ch['code']}] {ch['description']}")
        lines.append("")
    
    # Speaker Legend
    speaker_mapping = transcript.get('speaker_mapping', {})
    if speaker_mapping:
        lines.append("SPEAKER LEGEND:")
        lines.append("-" * 40)
        for speaker, desc in speaker_mapping.items():
            lines.append(f"  {speaker}: {desc}")
        lines.append("")
    
    # Registrar Notes Summary
    if 'registrar_notes' in transcript:
        notes = transcript['registrar_notes']
        lines.append(f"REGISTRAR NOTES ({len(notes)} entries):")
        lines.append("-" * 40)
        for note in notes[:20]:  # First 20
            speaker_str = f"{note['speaker']}: " if note['speaker'] else ""
            lines.append(f"  [{note['offset']}] {speaker_str}{note['content']}")
        if len(notes) > 20:
            lines.append(f"  ... and {len(notes) - 20} more")
        lines.append("")
    
    # Transcript
    lines.append("=" * 80)
    lines.append("TRANSCRIPT")
    lines.append("=" * 80)
    lines.append("")
    
    for line in transcript.get('lines', []):
        lines.append(line)
        lines.append("")
    
    return '\n'.join(lines)


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_file(file_id: str, output_folder_id: Optional[str] = None) -> dict:
    """
    Process a single audio file (standard or DCR).
    
    Args:
        file_id: Google Drive file ID
        output_folder_id: Optional output folder (uses source folder if not provided)
    
    Returns:
        Processing result dict
    """
    result = {
        'file_id': file_id,
        'status': 'processing',
        'started_at': datetime.now(timezone.utc).isoformat()
    }
    
    try:
        print(f"\n{'='*60}")
        print(f"Processing: {file_id}")
        print(f"{'='*60}")
        
        # Download file
        file_bytes, filename, mime_type, folder_id = download_from_drive(file_id)
        result['original_filename'] = filename
        result['mime_type'] = mime_type
        result['file_size_mb'] = round(len(file_bytes) / (1024 * 1024), 2)
        
        print(f"File: {filename}")
        print(f"Size: {result['file_size_mb']} MB")
        print(f"Type: {mime_type}")
        
        # Determine output folder
        target_folder = output_folder_id or OUTPUT_FOLDER_ID or folder_id
        if not target_folder:
            raise ValueError("No output folder specified and no source folder found")
        
        # Detect input type
        input_type = detect_input_type(filename, mime_type)
        result['input_type'] = input_type
        print(f"Input type: {input_type}")
        
        dcr_metadata = None
        registrar_notes = None
        audio_bytes = file_bytes
        audio_filename = filename
        
        if input_type == 'dcr':
            # Process DCR file
            print("\nProcessing DCR file...")
            
            # Try to find registrar notes
            notes_bytes = find_sidecar_notes(file_id, filename, folder_id)
            if notes_bytes:
                registrar_notes = parse_registrar_notes(notes_bytes)
                print(f"Found {len(registrar_notes)} registrar notes")
                result['registrar_notes_count'] = len(registrar_notes)
            
            # Convert DCR to WAV
            audio_bytes, dcr_metadata = dcr_to_wav(file_bytes)
            audio_filename = os.path.splitext(filename)[0] + ".wav"
            mime_type = 'audio/wav'
            
            result['dcr_creation_time'] = dcr_metadata.creation_time
            result['dcr_channels'] = len(dcr_metadata.channels)
        
        # =================================================================
        # MULTI-ENGINE SPLIT TESTING
        # =================================================================
        engines_to_run = [e.strip() for e in ENABLED_ENGINES if e.strip() in TRANSCRIPTION_ENGINES]
        print(f"\nRunning {len(engines_to_run)} engines for split testing: {', '.join(engines_to_run)}")
        
        transcription_results = {}
        
        if RUN_PARALLEL and len(engines_to_run) > 1:
            # Run engines in parallel
            print("Running engines in PARALLEL mode...")
            with ThreadPoolExecutor(max_workers=len(engines_to_run)) as executor:
                futures = {
                    executor.submit(
                        transcribe_with_engine,
                        engine_id, audio_bytes, audio_filename, mime_type,
                        registrar_notes, (input_type == 'dcr')
                    ): engine_id for engine_id in engines_to_run
                }
                
                for future in as_completed(futures):
                    engine_id = futures[future]
                    try:
                        _, trans_result = future.result()
                        transcription_results[engine_id] = trans_result
                        print(f"Engine {engine_id} completed")
                    except Exception as e:
                        print(f"Engine {engine_id} failed: {e}")
                        transcription_results[engine_id] = {'error': str(e)}
        else:
            # Run engines sequentially
            print("Running engines SEQUENTIALLY...")
            for engine_id in engines_to_run:
                try:
                    _, trans_result = transcribe_with_engine(
                        engine_id, audio_bytes, audio_filename, mime_type,
                        registrar_notes, (input_type == 'dcr')
                    )
                    transcription_results[engine_id] = trans_result
                except Exception as e:
                    print(f"Engine {engine_id} failed: {e}")
                    transcription_results[engine_id] = {'error': str(e)}
        
        # =================================================================
        # GENERATE OUTPUTS FOR EACH ENGINE (6 files total)
        # =================================================================
        print("\nGenerating outputs for all engines...")
        base_name = os.path.splitext(filename)[0]
        all_outputs = []
        
        for engine_id, transcription in transcription_results.items():
            if 'error' in transcription:
                print(f"Skipping output for {engine_id} due to error")
                continue
            
            print(f"Generating {engine_id} outputs...")
            
            # Add engine info to transcription
            transcription['engine_id'] = engine_id
            transcription['engine_name'] = TRANSCRIPTION_ENGINES[engine_id]['name']
            
            # Generate JSON and TXT
            json_data = generate_json_output(
                transcription, filename, dcr_metadata, registrar_notes
            )
            
            # Add engine metadata to JSON
            json_data['transcription_engine'] = {
                'id': engine_id,
                'name': TRANSCRIPTION_ENGINES[engine_id]['name'],
                'model': TRANSCRIPTION_ENGINES[engine_id]['model']
            }
            
            txt_content = generate_txt_output(json_data)
            
            # Use engine suffix in filenames
            json_filename = f"{base_name}_TRANSCRIPT_{engine_id}.json"
            txt_filename = f"{base_name}_TRANSCRIPT_{engine_id}.txt"
            
            # Upload to Drive
            json_file = upload_to_drive(
                json_filename,
                json.dumps(json_data, indent=2),
                target_folder,
                'application/json'
            )
            
            txt_file = upload_to_drive(
                txt_filename,
                txt_content,
                target_folder,
                'text/plain'
            )
            
            all_outputs.append({
                'engine': engine_id,
                'json': {'name': json_filename, 'id': json_file['id'], 'link': json_file.get('webViewLink')},
                'txt': {'name': txt_filename, 'id': txt_file['id'], 'link': txt_file.get('webViewLink')},
                'utterance_count': len(json_data['transcripts'][0].get('lines', [])),
                'block_count': len(json_data['transcripts'][0].get('blocks', []))
            })
        
        result['status'] = 'success'
        result['engines_used'] = engines_to_run
        result['outputs'] = all_outputs
        result['total_files'] = len(all_outputs) * 2
        
        print(f"\n{'='*60}")
        print("SUCCESS! SPLIT TEST COMPLETE")
        print(f"{'='*60}")
        print(f"Engines: {', '.join(engines_to_run)}")
        print(f"Total output files: {len(all_outputs) * 2}")
        for out in all_outputs:
            print(f"  {out['engine']}: {out['utterance_count']} utterances, {out['block_count']} blocks")
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    result['completed_at'] = datetime.now(timezone.utc).isoformat()
    return result


# =============================================================================
# FLASK ENDPOINTS
# =============================================================================

@app.route('/')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'af-autofirm-transcriber-dcr',
        'model': MODEL_ID,
        'features': ['standard_audio', 'dcr_speex', 'registrar_notes', 'webhook_events'],
        'chunk_duration_minutes': CHUNK_DURATION_MINUTES
    })


@app.route('/process/<file_id>', methods=['POST'])
def process_single(file_id):
    """Process a single file by Drive ID."""
    try:
        output_folder = request.args.get('output_folder')
        result = process_file(file_id, output_folder)
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/process', methods=['POST'])
def process_from_body():
    """Process file with parameters in request body."""
    try:
        data = request.get_json() or {}
        file_id = data.get('file_id')
        if not file_id:
            return jsonify({'status': 'error', 'error': 'file_id required'}), 400
        
        output_folder = data.get('output_folder_id')
        result = process_file(file_id, output_folder)
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


# =============================================================================
# WORKSPACE EVENTS WEBHOOK (Pub/Sub Push)
# =============================================================================

def cleanup_old_events():
    """Remove processed events older than TTL to prevent memory bloat."""
    global PROCESSED_EVENTS
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=EVENT_CACHE_TTL_MINUTES)
    PROCESSED_EVENTS = {
        k: v for k, v in PROCESSED_EVENTS.items() 
        if v > cutoff
    }


def extract_file_id_from_event(event: dict) -> Optional[str]:
    """
    Extract file ID from various Workspace Events API event formats.
    
    Supports both:
    - google.workspace.events.subscription.v1.driveActivity format
    - google.workspace.drive.file.v3 format
    """
    try:
        # Try Drive Activity format: targets[0].driveItem.name = "items/FILE_ID"
        targets = event.get('targets', [])
        if targets:
            for target in targets:
                drive_item = target.get('driveItem', {})
                name = drive_item.get('name', '')
                if name.startswith('items/'):
                    return name.split('/')[-1]
                # Direct file ID
                if drive_item.get('id'):
                    return drive_item['id']
        
        # Try resourceId format (older Drive push)
        if event.get('resourceId'):
            return event['resourceId']
        
        # Try file.v3 format: file.id
        file_obj = event.get('file', {})
        if file_obj.get('id'):
            return file_obj['id']
        
        # Try resource format
        resource = event.get('resource', {})
        if resource.get('id'):
            return resource['id']
        
        # Fallback: look for any 'id' or 'fileId' field
        for key in ['id', 'fileId', 'file_id']:
            if event.get(key):
                return event[key]
        
        print(f"Could not extract file ID from event: {json.dumps(event, indent=2)[:500]}")
        return None
        
    except Exception as e:
        print(f"Error extracting file ID: {e}")
        return None


def is_file_in_watched_folder(file_id: str) -> bool:
    """Check if file is in the watched folder (optional security check)."""
    try:
        service = get_drive_service()
        file_meta = service.files().get(
            fileId=file_id,
            fields='parents'
        ).execute()
        parents = file_meta.get('parents', [])
        return WATCH_FOLDER_ID in parents
    except Exception as e:
        print(f"Error checking folder: {e}")
        return True  # Allow processing if we can't verify


@app.route('/webhook', methods=['POST'])
def webhook():
    """
    Handle Pub/Sub push notifications from Workspace Events API.
    
    Event flow:
    1. File uploaded to watched Drive folder
    2. Workspace Events API detects change
    3. Event published to Pub/Sub topic
    4. Pub/Sub pushes to this endpoint
    5. Webhook triggers transcription
    """
    try:
        envelope = request.get_json()
        
        # Handle subscription verification (GET-style in POST body)
        if not envelope:
            return jsonify({'error': 'Empty request body'}), 400
        
        # Pub/Sub message format
        if 'message' not in envelope:
            # Could be a subscription verification
            print(f"Webhook received non-message: {json.dumps(envelope)[:500]}")
            return jsonify({'status': 'acknowledged'}), 200
        
        message = envelope['message']
        message_id = message.get('messageId', 'unknown')
        
        # Idempotency check - prevent duplicate processing
        if message_id in PROCESSED_EVENTS:
            print(f"Duplicate event ignored: {message_id}")
            return jsonify({
                'status': 'duplicate',
                'message_id': message_id
            }), 200
        
        # Decode message data
        data_b64 = message.get('data', '')
        if not data_b64:
            return jsonify({'error': 'No data in message'}), 400
        
        try:
            data_json = base64.b64decode(data_b64).decode('utf-8')
            event = json.loads(data_json)
        except Exception as e:
            print(f"Failed to decode message data: {e}")
            return jsonify({'error': f'Invalid message data: {e}'}), 400
        
        print(f"Webhook received event: {json.dumps(event, indent=2)[:1000]}")
        
        # Check event type - only process file creation events
        event_type = event.get('type', event.get('@type', ''))
        action_detail = event.get('primaryActionDetail', {})
        
        # Filter for file creation events
        is_create_event = any([
            'create' in event_type.lower(),
            'add' in event_type.lower(),
            action_detail.get('create'),
            action_detail.get('upload'),
            event.get('actions', [{}])[0].get('detail', {}).get('create') if event.get('actions') else False
        ])
        
        if not is_create_event:
            print(f"Ignoring non-create event: {event_type}")
            return jsonify({
                'status': 'ignored',
                'reason': 'Not a file creation event',
                'event_type': event_type
            }), 200
        
        # Extract file ID
        file_id = extract_file_id_from_event(event)
        if not file_id:
            return jsonify({'error': 'Could not extract file ID from event'}), 400
        
        # Mark as processed BEFORE async processing
        PROCESSED_EVENTS[message_id] = datetime.now(timezone.utc)
        cleanup_old_events()
        
        print(f"Starting transcription for file: {file_id}")
        
        # Process asynchronously (return 200 immediately to Pub/Sub)
        def async_process():
            try:
                result = process_file(file_id)
                print(f"Transcription complete for {file_id}: {result.get('status')}")
            except Exception as e:
                print(f"Async transcription failed for {file_id}: {e}")
        
        thread = threading.Thread(target=async_process)
        thread.start()
        
        return jsonify({
            'status': 'accepted',
            'file_id': file_id,
            'message_id': message_id
        }), 200
        
    except Exception as e:
        print(f"Webhook error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'error': str(e)}), 500


# =============================================================================
# DRIVE API PUSH NOTIFICATIONS (changes.watch)
# =============================================================================

# Store last processed change token
LAST_PAGE_TOKEN = {}

@app.route('/drive-webhook', methods=['POST', 'GET'])
def drive_webhook():
    """
    Handle Google Drive API push notifications (changes.watch).
    
    Drive sends notifications with headers:
    - X-Goog-Channel-ID: The channel ID
    - X-Goog-Resource-State: sync, change, or exists
    - X-Goog-Resource-ID: The resource being watched
    """
    try:
        # Get Drive push headers
        channel_id = request.headers.get('X-Goog-Channel-ID', '')
        resource_state = request.headers.get('X-Goog-Resource-State', '')
        resource_id = request.headers.get('X-Goog-Resource-ID', '')
        
        print(f"Drive webhook: channel={channel_id}, state={resource_state}")
        
        # Handle sync - just acknowledge
        if resource_state == 'sync':
            print("Drive webhook sync received")
            return '', 200
        
        # Only process 'change' events
        if resource_state != 'change':
            print(f"Ignoring drive webhook state: {resource_state}")
            return '', 200
        
        # Get changes from Drive API
        def process_drive_changes():
            try:
                service = get_drive_service()
                
                # Get or initialize page token for this channel
                page_token = LAST_PAGE_TOKEN.get(channel_id)
                if not page_token:
                    # Get start page token
                    response = service.changes().getStartPageToken(
                        supportsAllDrives=True
                    ).execute()
                    page_token = response.get('startPageToken')
                    LAST_PAGE_TOKEN[channel_id] = page_token
                
                # List changes
                changes_response = service.changes().list(
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                    fields='newStartPageToken,changes(file(id,name,parents,mimeType,createdTime))'
                ).execute()
                
                # Update page token for next time
                new_token = changes_response.get('newStartPageToken')
                if new_token:
                    LAST_PAGE_TOKEN[channel_id] = new_token
                
                changes = changes_response.get('changes', [])
                print(f"Found {len(changes)} changes")
                
                for change in changes:
                    file_info = change.get('file', {})
                    if not file_info:
                        continue
                    
                    file_id = file_info.get('id')
                    file_name = file_info.get('name', '')
                    parents = file_info.get('parents', [])
                    mime_type = file_info.get('mimeType', '')
                    
                    # Check if in watched folder
                    if WATCH_FOLDER_ID not in parents:
                        continue
                    
                    # Check if it's an audio/DCR file (not a folder)
                    if mime_type == 'application/vnd.google-apps.folder':
                        continue
                    
                    # Skip Google Docs types
                    if mime_type.startswith('application/vnd.google-apps.'):
                        continue
                    
                    # Idempotency check
                    event_key = f"drive-{file_id}"
                    if event_key in PROCESSED_EVENTS:
                        print(f"Already processed: {file_name}")
                        continue
                    
                    PROCESSED_EVENTS[event_key] = datetime.now(timezone.utc)
                    cleanup_old_events()
                    
                    print(f"Processing new file: {file_name} ({file_id})")
                    
                    try:
                        result = process_file(file_id)
                        print(f"Transcription complete: {result.get('status')}")
                    except Exception as e:
                        print(f"Processing failed for {file_id}: {e}")
                        
            except Exception as e:
                print(f"Drive changes processing error: {e}")
                import traceback
                traceback.print_exc()
        
        # Process asynchronously
        thread = threading.Thread(target=process_drive_changes)
        thread.start()
        
        return '', 200
        
    except Exception as e:
        print(f"Drive webhook error: {e}")
        return '', 200  # Always return 200 to prevent retries


# =============================================================================
# CLI MODE
# =============================================================================

def main():
    """CLI entry point for testing."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python main.py <drive_file_id> [output_folder_id]")
        print("\nOr run as web server:")
        print("  python main.py --serve")
        sys.exit(1)
    
    if sys.argv[1] == '--serve':
        port = int(os.environ.get('PORT', 8080))
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        file_id = sys.argv[1]
        output_folder = sys.argv[2] if len(sys.argv) > 2 else None
        
        result = process_file(file_id, output_folder)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
