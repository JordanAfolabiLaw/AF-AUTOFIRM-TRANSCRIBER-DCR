"""
drive_watch_setup.py
====================
Setup and renew Drive API push notifications for the AF-AUTOFIRM-DCR service.

This uses the Drive API changes.watch endpoint which works with Shared Drives.
The watch channel expires after 24 hours and must be renewed.

Run this script:
1. Initially to create the watch channel
2. Via Cloud Scheduler daily to renew before expiration

Usage:
    python drive_watch_setup.py
"""

import os
import uuid
import json
from datetime import datetime, timezone, timedelta
import google.auth
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Configuration
PROJECT_ID = "firmlink-alpha"
FOLDER_ID = "1YUWj5t13Kstl5Vwv_Ee5mJpF2H9eYWaI"
WEBHOOK_URL = "https://af-autofirm-transcriber-dcr-wzmrud4omq-uc.a.run.app/drive-webhook"

# Channel expiration: 24 hours minus 1 hour buffer
EXPIRATION_HOURS = 23


def get_drive_service():
    """Get authenticated Drive service."""
    credentials, _ = google.auth.default(
        scopes=['https://www.googleapis.com/auth/drive']
    )
    return build('drive', 'v3', credentials=credentials)


def get_start_page_token(service, drive_id=None):
    """Get the starting page token for watching changes."""
    params = {'supportsAllDrives': True}
    if drive_id:
        params['driveId'] = drive_id
    response = service.changes().getStartPageToken(**params).execute()
    return response.get('startPageToken')


def create_watch_channel(service, page_token, drive_id=None):
    """Create a watch channel for Drive changes."""
    channel_id = f"af-autofirm-dcr-{uuid.uuid4().hex[:8]}"
    expiration_time = datetime.now(timezone.utc) + timedelta(hours=EXPIRATION_HOURS)
    expiration_ms = int(expiration_time.timestamp() * 1000)
    
    body = {
        'id': channel_id,
        'type': 'web_hook',
        'address': WEBHOOK_URL,
        'expiration': str(expiration_ms),
        'payload': True
    }
    
    params = {
        'pageToken': page_token,
        'supportsAllDrives': True,
        'includeItemsFromAllDrives': True
    }
    if drive_id:
        params['driveId'] = drive_id
    
    response = service.changes().watch(body=body, **params).execute()
    
    return {
        'channel_id': response.get('id'),
        'resource_id': response.get('resourceId'),
        'expiration': response.get('expiration'),
        'created_at': datetime.now(timezone.utc).isoformat()
    }


def stop_watch_channel(service, channel_id, resource_id):
    """Stop an existing watch channel."""
    try:
        service.channels().stop(body={
            'id': channel_id,
            'resourceId': resource_id
        }).execute()
        print(f"Stopped channel: {channel_id}")
    except Exception as e:
        print(f"Failed to stop channel: {e}")


def main():
    """Setup or renew Drive watch channel."""
    print("=" * 60)
    print("Drive API Push Notifications Setup")
    print("=" * 60)
    print(f"Folder: {FOLDER_ID}")
    print(f"Webhook: {WEBHOOK_URL}")
    print()
    
    service = get_drive_service()
    
    # Get folder details to find drive ID
    folder = service.files().get(
        fileId=FOLDER_ID,
        supportsAllDrives=True,
        fields='id,name,driveId'
    ).execute()
    
    print(f"Folder name: {folder.get('name')}")
    drive_id = folder.get('driveId')
    if drive_id:
        print(f"Shared Drive ID: {drive_id}")
    print()
    
    # Get start page token
    page_token = get_start_page_token(service, drive_id)
    print(f"Start page token: {page_token}")
    
    # Create watch channel
    print("\nCreating watch channel...")
    channel = create_watch_channel(service, page_token, drive_id)
    
    print("\n" + "=" * 60)
    print("Watch Channel Created!")
    print("=" * 60)
    print(f"Channel ID: {channel['channel_id']}")
    print(f"Resource ID: {channel['resource_id']}")
    print(f"Expires: {datetime.fromtimestamp(int(channel['expiration'])/1000, tz=timezone.utc).isoformat()}")
    print()
    print("IMPORTANT: This channel expires in ~24 hours.")
    print("Set up Cloud Scheduler to run this script daily.")
    
    # Save channel info for later management
    with open('channel_info.json', 'w') as f:
        json.dump(channel, f, indent=2)
    print("\nChannel info saved to channel_info.json")
    
    return channel


if __name__ == '__main__':
    main()
