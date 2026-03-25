#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GEAK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONTAINER_A="${HARNESSGEN_CONTAINER_A:-harness-gen}"
CONTAINER_B="${HARNESSGEN_CONTAINER_B:-harness-gen2}"
KEY_A="${HARNESSGEN_KEY_A:-}"
KEY_B="${HARNESSGEN_KEY_B:-}"
MODEL="${HARNESSGEN_MODEL:-claude-opus-4.6}"
GATEWAY_URL="${HARNESSGEN_GATEWAY_URL:-https://llm-api.amd.com/Anthropic}"

SKIP_INSTALL=false
VALIDATE_ONLY=false
SKIP_GATEWAY_CHECK=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Set up and validate two pre-existing harness-generation containers for GEAK
preprocess work on a new machine.

What this script does:
  1. Starts \`harness-gen\` / \`harness-gen2\` if they are stopped
  2. Verifies the GEAK checkout is mounted into both containers
  3. Runs \`python3 -m pip install -e .\` inside both containers
  4. Persists per-container \`AMD_LLM_API_KEY\`, \`GEAK_MODEL\`, and
     \`MSWEA_MODEL_NAME\` to both \`~/.profile\` and \`~/.bashrc\`
  5. Validates imports, preprocess CLI availability, env vars, and optional
     gateway reachability

Options:
  --container-a NAME        First container name (default: ${CONTAINER_A})
  --container-b NAME        Second container name (default: ${CONTAINER_B})
  --key-a KEY               API key for container A
  --key-b KEY               API key for container B
  --model NAME              Default preprocess model (default: ${MODEL})
  --gateway-url URL         Gateway URL to probe (default: ${GATEWAY_URL})
  --geak-root PATH          GEAK checkout path inside the containers
                            (default: current host GEAK root = ${GEAK_ROOT})
  --skip-install            Skip \`pip install -e .\`
  --validate-only           Skip setup actions and only run validation
  --skip-gateway-check      Skip HTTP reachability probe for the AMD gateway
  -h, --help                Show this help

Preferred key input:
  HARNESSGEN_KEY_A=... HARNESSGEN_KEY_B=... ./$(basename "$0")

Examples:
  HARNESSGEN_KEY_A=key1 HARNESSGEN_KEY_B=key2 ./$(basename "$0")
  ./$(basename "$0") --validate-only --skip-gateway-check
  ./$(basename "$0") --container-a harness-gen --container-b harness-gen2
EOF
}

require_command() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: Required command not found: $cmd"
        exit 1
    fi
}

container_exists() {
    local container="$1"
    docker inspect "$container" >/dev/null 2>&1
}

ensure_container_running() {
    local container="$1"
    local running
    if ! container_exists "$container"; then
        echo "ERROR: Docker container '$container' does not exist."
        exit 1
    fi

    running="$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)"
    if [[ "$running" != "true" ]]; then
        echo "Starting container '$container'..."
        docker start "$container" >/dev/null
    fi
}

assert_geak_root_visible() {
    local container="$1"
    if ! docker exec "$container" bash -lc "test -d '$GEAK_ROOT'"; then
        echo "ERROR: GEAK root '$GEAK_ROOT' is not visible inside '$container'."
        echo "       Recreate the container with the repo mounted at the same path."
        exit 1
    fi
}

install_geak_editable() {
    local container="$1"
    echo "Installing GEAK editable package in '$container'..."
    docker exec "$container" bash -lc "
        cd '$GEAK_ROOT'
        export PIP_DISABLE_PIP_VERSION_CHECK=1
        python3 -m pip install -e .
    "
}

persist_env_exports() {
    local container="$1"
    local key="$2"
    local model="$3"

    docker exec \
        -e TARGET_KEY="$key" \
        -e TARGET_MODEL="$model" \
        "$container" \
        bash -lc "python3 - <<'PY'
from pathlib import Path
import os

updates = {
    'AMD_LLM_API_KEY': os.environ['TARGET_KEY'],
    'GEAK_MODEL': os.environ['TARGET_MODEL'],
    'MSWEA_MODEL_NAME': os.environ['TARGET_MODEL'],
}

for file_name in ('.profile', '.bashrc'):
    path = Path.home() / file_name
    text = path.read_text() if path.exists() else ''
    lines = []
    for line in text.splitlines():
        if any(line.startswith(f'export {name}=') for name in updates):
            continue
        lines.append(line)
    for name, value in updates.items():
        lines.append(f\"export {name}='{value}'\")
    path.write_text('\\n'.join(lines) + '\\n')
PY"
}

validate_env_visibility() {
    local container="$1"
    local expected_model="$2"

    docker exec \
        -e EXPECTED_MODEL="$expected_model" \
        "$container" \
        bash -lc "python3 - <<'PY'
import os
import sys

missing = []
for name in ('AMD_LLM_API_KEY', 'GEAK_MODEL', 'MSWEA_MODEL_NAME'):
    if not os.getenv(name):
        missing.append(name)

if missing:
    raise SystemExit(f'Missing env vars: {missing}')

expected = os.environ['EXPECTED_MODEL']
if os.environ['GEAK_MODEL'] != expected:
    raise SystemExit(f'GEAK_MODEL mismatch: {os.environ[\"GEAK_MODEL\"]} != {expected}')
if os.environ['MSWEA_MODEL_NAME'] != expected:
    raise SystemExit(f'MSWEA_MODEL_NAME mismatch: {os.environ[\"MSWEA_MODEL_NAME\"]} != {expected}')

print('env_ok')
PY"
}

validate_python_and_cli() {
    local container="$1"
    docker exec "$container" bash -lc "
        cd '$GEAK_ROOT'
        python3 - <<'PY'
import anthropic
import minisweagent
import minisweagent.run.preprocess.preprocessor
print('import_ok')
PY
        python3 -m minisweagent.run.preprocess.preprocessor --help >/dev/null
        echo cli_ok
    "
}

validate_gateway() {
    local container="$1"
    local gateway_url="$2"
    docker exec \
        -e GATEWAY_URL_ENV="$gateway_url" \
        "$container" \
        bash -lc "python3 - <<'PY'
import os
import urllib.error
import urllib.request

url = os.environ['GATEWAY_URL_ENV']
req = urllib.request.Request(url, method='GET')
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        print(f'gateway_http_{resp.status}')
except urllib.error.HTTPError as exc:
    # HTTP error still proves DNS/TLS/network path is working.
    print(f'gateway_http_{exc.code}')
except Exception as exc:
    raise SystemExit(f'gateway_unreachable: {type(exc).__name__}: {exc}')
PY"
}

setup_container() {
    local container="$1"
    local key="$2"
    local model="$3"

    ensure_container_running "$container"
    assert_geak_root_visible "$container"

    if [[ "$SKIP_INSTALL" != "true" ]]; then
        install_geak_editable "$container"
    fi
    persist_env_exports "$container" "$key" "$model"
}

validate_container() {
    local container="$1"
    local model="$2"

    ensure_container_running "$container"
    assert_geak_root_visible "$container"

    echo "Validating '$container'..."
    validate_env_visibility "$container" "$model"
    validate_python_and_cli "$container"
    if [[ "$SKIP_GATEWAY_CHECK" != "true" ]]; then
        validate_gateway "$container" "$GATEWAY_URL"
    else
        echo "gateway_check_skipped"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --container-a)
            CONTAINER_A="$2"
            shift 2
            ;;
        --container-b)
            CONTAINER_B="$2"
            shift 2
            ;;
        --key-a)
            KEY_A="$2"
            shift 2
            ;;
        --key-b)
            KEY_B="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --gateway-url)
            GATEWAY_URL="$2"
            shift 2
            ;;
        --geak-root)
            GEAK_ROOT="$2"
            shift 2
            ;;
        --skip-install)
            SKIP_INSTALL=true
            shift
            ;;
        --validate-only)
            VALIDATE_ONLY=true
            shift
            ;;
        --skip-gateway-check)
            SKIP_GATEWAY_CHECK=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

require_command docker

if [[ "$VALIDATE_ONLY" != "true" ]]; then
    if [[ -z "$KEY_A" || -z "$KEY_B" ]]; then
        echo "ERROR: Both API keys are required unless --validate-only is used."
        echo "       Use --key-a/--key-b or HARNESSGEN_KEY_A/HARNESSGEN_KEY_B."
        exit 1
    fi
fi

echo "=== Harnessgen Container Setup ==="
echo "  GEAK root:       ${GEAK_ROOT}"
echo "  Container A:     ${CONTAINER_A}"
echo "  Container B:     ${CONTAINER_B}"
echo "  Model:           ${MODEL}"
echo "  Gateway URL:     ${GATEWAY_URL}"
echo "  Skip install:    ${SKIP_INSTALL}"
echo "  Validate only:   ${VALIDATE_ONLY}"
echo "  Gateway check:   $([[ "$SKIP_GATEWAY_CHECK" == "true" ]] && echo 'skipped' || echo 'enabled')"
echo ""

if [[ "$VALIDATE_ONLY" != "true" ]]; then
    setup_container "$CONTAINER_A" "$KEY_A" "$MODEL"
    setup_container "$CONTAINER_B" "$KEY_B" "$MODEL"
fi

validate_container "$CONTAINER_A" "$MODEL"
validate_container "$CONTAINER_B" "$MODEL"

echo ""
echo "Both containers are configured and passed validation."
