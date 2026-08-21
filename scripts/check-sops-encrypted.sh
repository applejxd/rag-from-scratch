#!/usr/bin/env bash

secret_file="${1:-.env.json}"

if ! mise exec -- sops filestatus "$secret_file" |
    grep -Eq '"encrypted"[[:space:]]*:[[:space:]]*true'; then
    echo "$secret_file must be encrypted with SOPS." >&2
    exit 1
fi
