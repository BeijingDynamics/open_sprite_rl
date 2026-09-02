#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SPRITE_RL_ROOT="${SPRITE_RL_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
export ISAACLAB_ROOT="${ISAACLAB_ROOT:-$HOME/IsaacLab}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-yes}"

ISAACLAB_SH="$ISAACLAB_ROOT/isaaclab.sh"
TRAIN_PY="$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/train.py"

require_file() {
  [[ -f "$1" ]] || {
    echo "Required file not found: $1" >&2
    exit 2
  }
}

verify_sha256() {
  local file="$1"
  local expected="$2"
  local actual
  require_file "$file"
  actual="$(sha256sum "$file" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "SHA256 mismatch: $file" >&2
    echo "expected: $expected" >&2
    echo "actual:   $actual" >&2
    exit 4
  }
}

require_file "$ISAACLAB_SH"
require_file "$TRAIN_PY"
mkdir -p "$SPRITE_RL_ROOT/outputs"
