#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

set -euo pipefail

# Colors
declare -A LOG_COLORS=(
    ["INFO"]="\033[0;32m"
    ["WARN"]="\033[1;33m"
    ["ERROR"]="\033[0;31m"
)
NC="\033[0m"

TS_FORMAT="%Y-%m-%d %H:%M:%S"

log() {
    local level=$1
    shift
    local msg="$@"
    local ts=$(date +"${TS_FORMAT}")
    echo -e "${LOG_COLORS[$level]}${ts} [${level}] ${msg}${NC}"
}

SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")
JOB_URL=""
METRICS_DIR=""
PROM_FILE_NAME="alpasim-runtime.prom"
ALPASIM_DOCKER_PROJECT_ID="132833"

show_help() {
   cat << EOF
Usage: $0 [OPTIONS]

Options:
   --job-url     URL     Gitlab Job URL to fetch artifacts from. Should be a job under the Alpasim-docker project.
   --metrics-dir DIR     Absolute path to metrics directory on your local FS containing alpasim-runtime.prom file.
   --help               Show this help message

EOF
   exit 0
}


# Show help if no arguments
[[ $# -eq 0 ]] && show_help


while [[ $# -ge 1 ]]; do
    case "$1" in
        --help)
            show_help
            exit 0
            ;;
        --job-url)
            JOB_URL="$2"
            shift 2
            ;;
        --metrics-dir)
            METRICS_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate mutually exclusive flags
if [ -n "$METRICS_DIR" ] && [ -n "$JOB_URL" ]; then
    log "ERROR" "Provide either the absolute path to a metrics directory containing *.prom files, or a Job URL."
    exit 1
fi

if [ -n "$JOB_URL" ] && [ -z "${GITLAB_TOKEN:-}" ]; then
    GITLAB_TOKEN=$(cat ~/.netrc | xargs | grep -Po "(?<=machine gitlab-master.nvidia.com).*?password \K[^ ]*")

    if [[ -z "${GITLAB_TOKEN}" ]]; then
        log "ERROR" "To download artifacts from gitlab, place GITLAB_TOKEN on your env before running this script."
        exit 1
    fi
fi

# Parse GitLab job URL
if [ -n "$JOB_URL" ]; then
    JOB_ID=$(echo "$JOB_URL" | sed -E 's|.*/jobs/([0-9]+)|\1|')

    # Create temporary directory
    TEMP_DIR=$(mktemp -d)

    remote_path="https://gitlab-master.nvidia.com/api/v4/projects/${ALPASIM_DOCKER_PROJECT_ID}/jobs/${JOB_ID}/artifacts/output/$PROM_FILE_NAME"

    log "INFO" "Downloading artifacts for job ID: $JOB_ID from project: $ALPASIM_DOCKER_PROJECT_ID at path $remote_path"
    curl --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
         --output "$TEMP_DIR/$PROM_FILE_NAME" \
         "$remote_path"

    log "INFO" "Artifacts downloaded to: $TEMP_DIR/$PROM_FILE_NAME"
    chmod -R 777 $TEMP_DIR
    METRICS_DIR="$TEMP_DIR"
fi

cleanup() {
    log "INFO" "Shutting down"
    pushd $SCRIPT_DIR > /dev/null
    METRICS_PATH="$METRICS_DIR" docker compose down
    popd > /dev/null
    log "INFO" "Done."
}

log "INFO" "Starting metrics services..."
log "INFO" "The first time you run this, Grafana may take multiple minutes to start."

pushd $SCRIPT_DIR > /dev/null
METRICS_PATH="$METRICS_DIR" docker compose up -d --wait
popd > /dev/null

log "INFO" "Browse to http://localhost:3000/?orgId=1&from=now-6h&to=now&timezone=browser&var-service=driver&var-method=\$__all&refresh=auto to view grafana dashboard."
log "INFO" "Press any key to stop services."

trap 'cleanup; exit' SIGINT SIGTERM
trap cleanup EXIT

read -n 1 -s key
