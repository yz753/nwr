#!/bin/bash
# Requires the DANDI client: pip install -U dandi

set -euo pipefail

# Paste your DANDI API key between the single quotes below.
export DANDI_API_KEY='1b1f9a18d7bca944dc364d80f0f7629a91dc10f7'

if (( $# != 3 )); then
    echo "Usage: $0 DANDISET_ID PROCESSED_FOLDER UPLOAD_FOLDER" >&2
    exit 1
fi

dandiset_id="$1"
processed_folder="$2"
upload_folder="$3"

if [[ ! "$dandiset_id" =~ ^[0-9]{6}$ ]]; then
    echo "Dandiset ID must contain exactly six digits." >&2
    exit 1
fi

command -v dandi >/dev/null 2>&1 || {
    echo "Install the DANDI client first: pip install -U dandi" >&2
    exit 1
}

if [[ ! -d "$processed_folder" ]]; then
    echo "Processed folder does not exist: $processed_folder" >&2
    exit 1
fi
processed_folder=$(cd "$processed_folder" && pwd -P)

mkdir -p "$upload_folder"
upload_folder=$(cd "$upload_folder" && pwd -P)
dandiset_folder="$upload_folder/$dandiset_id"

# Avoid recursively organizing an output directory inside the source tree.
case "$dandiset_folder/" in
    "$processed_folder/"*)
        echo "The Dandiset upload folder must be outside the processed folder." >&2
        exit 1
        ;;
esac

if [[ -z "$DANDI_API_KEY" || "$DANDI_API_KEY" == 'PASTE_YOUR_DANDI_API_KEY_HERE' ]]; then
    echo "Set DANDI_API_KEY near the top of this script before running it." >&2
    exit 1
fi

cd "$upload_folder"
if [[ ! -f "$dandiset_folder/dandiset.yaml" ]]; then
    # Intended for a newly created Dandiset; an existing remote Dandiset
    # may also download its existing assets.
    dandi download "https://dandiarchive.org/dandiset/$dandiset_id/draft"
fi
cd "$dandiset_folder"

echo "Copying NWB files into subject folders; original files stay unchanged..."
dandi organize -f symlink "$processed_folder"

echo "Validating the Dandiset..."
dandi validate .

echo "Uploading to Dandiset $dandiset_id (draft)..."
dandi upload

echo "Upload command completed. Review the Dandiset at:"
echo "https://dandiarchive.org/dandiset/$dandiset_id/draft"