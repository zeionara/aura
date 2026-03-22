#!/bin/bash

# Usage: ./backup.sh 2026.02.01.03

validate_json() {
    local file="$1"

    if command -v jq &>/dev/null; then
        jq empty "$file" 2>/dev/null
    elif command -v python3 &>/dev/null; then
        python -c "import json; json.load(open('$file'))" 2>/dev/null
    else
        echo "Error: Neither jq nor python is installed." >&2
        exit 1
    fi
}

set -euo pipefail

src_path=assets/data/$1
max_retries=${2:-5}

if [ ! -d "$src_path" ]; then
    echo "Error: '$src_path' is not a directory"
    exit 1
fi

dst_path="${src_path}-backup-$(date +%d.%m.%Y)" # destination folder

if [ -d "$dst_path" ]; then
    echo "Error: '$dst_path' already exists"
    exit 1
fi

mkdir -p "$dst_path/annotations" || exit 1
ln -s "../$1/source" $dst_path/

n_files=0

for f in "$src_path"/annotations/*.json; do
    name=$(basename "$f")
    echo "Processing: $name"
    retry=0
    while [ $retry -lt $max_retries ]; do
        cp "$f" "$dst_path/annotations/$name"
        # if json_pp < "$dst_path/annotations/$name" >/dev/null 2>&1; then
        if validate_json "$dst_path/annotations/$name"; then
            echo "  OK"
            break
        fi
        echo "  invalid, retrying..."
        rm -f "$dst_path/annotations/$name"            # remove bad copy
        ((retry++)) || true
        sleep 1
    done

    if [ $retry -eq $max_retries ]; then
        echo "  ERROR: failed after $max_retries attempts"
        echo "FAIL"
        exit 1
    fi

    ((n_files++)) || true
done

echo "SUCCESS: copied $n_files files to $dst_path"
