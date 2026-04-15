#!/bin/bash
# Download EMatch checkpoints from Google Drive
# Source: https://drive.google.com/drive/folders/1geAMTtWiySkU17SbWa3PH4mkSCinCEk6

FOLDER_ID="1geAMTtWiySkU17SbWa3PH4mkSCinCEk6"
DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/checkpoints/ematch"

# Check / install gdown
if ! python3 -c "import gdown" 2>/dev/null; then
    echo "[INFO] gdown not found. Installing..."
    pip install gdown --quiet
fi

echo "[INFO] Downloading checkpoints to: $DEST_DIR"
mkdir -p "$DEST_DIR"

python3 - <<EOF
import gdown, os, sys

folder_url = "https://drive.google.com/drive/folders/${FOLDER_ID}"
dest = "${DEST_DIR}"

try:
    # List folder contents without downloading (returns GoogleDriveFileToDownload objects)
    files = gdown.download_folder(folder_url, output=dest, quiet=True,
                                  use_cookies=False, skip_download=True)
    if not files:
        print("[INFO] No files found in folder.")
        sys.exit(0)

    downloaded = 0
    skipped = 0
    for f in files:
        name = os.path.basename(f.local_path)
        if os.path.exists(f.local_path):
            print(f"[SKIP] Already exists: {name}")
            skipped += 1
        else:
            print(f"[DOWN] Downloading: {name}")
            os.makedirs(os.path.dirname(f.local_path), exist_ok=True)
            gdown.download(id=f.id, output=f.local_path, quiet=False)
            downloaded += 1

    print(f"\n[INFO] Done — {downloaded} downloaded, {skipped} skipped.")
except Exception as e:
    print(f"[ERROR] {e}", file=sys.stderr)
    sys.exit(1)
EOF
