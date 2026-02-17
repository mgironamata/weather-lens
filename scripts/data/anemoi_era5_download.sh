#!/usr/bin/env bash
# Minimal retry wrapper for anemoi-datasets copy + resume

set -uo pipefail

# ---- CONFIG ----
URL="https://data.ecmwf.int/anemoi-datasets/era5-o96-1979-2023-6h-v8.zarr"
DEST="era5-o96-1979-2023-6h-v8.zarr"

# Optional: activate your env (comment out if not needed)
# if command -v micromamba >/dev/null 2>&1; then
#   eval "$(micromamba shell hook -s bash)"
#   micromamba activate regrid
# fi

MAX_ATTEMPTS=0        # 0 = infinite retries; otherwise set a number (e.g., 50)
BACKOFF_START=10      # seconds (first wait)
BACKOFF_MAX=300       # cap in seconds
LOGFILE="copy_era5_retry.log"

# ---- FUNCTIONS ----
log(){ echo "[$(date +'%F %T')] $*" | tee -a "$LOGFILE" >&2; }

backoff_seconds(){
  local n=$1
  # exponential backoff: BACKOFF_START * 2^(n-1), capped at BACKOFF_MAX
  local s=$(( BACKOFF_START << (n-1) ))
  if (( s > BACKOFF_MAX )); then s=$BACKOFF_MAX; fi
  echo "$s"
}

cleanup(){
  log "Received interrupt. Exiting."
  exit 130
}
trap cleanup INT TERM

# ---- MAIN LOOP ----
attempt=1
while :; do
  log "Attempt $attempt: anemoi-datasets copy (resume) → $DEST"
  # Run the command; --resume ensures partial progress is reused
  if anemoi-datasets copy "$URL" "$DEST" --resume 2>&1 | tee -a "$LOGFILE"; then
    log "Copy completed successfully."
    exit 0
  else
    rc=$?
    log "Copy failed with exit code $rc."
    # Stop if we hit a fixed max attempts
    if (( MAX_ATTEMPTS > 0 && attempt >= MAX_ATTEMPTS )); then
      log "Reached MAX_ATTEMPTS=$MAX_ATTEMPTS. Giving up."
      exit "$rc"
    fi
    # Wait with backoff and retry
    wait_s=$(backoff_seconds "$attempt")
    log "Retrying in ${wait_s}s…"
    sleep "$wait_s"
    attempt=$((attempt + 1))
  fi
done