#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-$HOME/IsaacLab}"
OVERLAY="$ROOT/isaaclab_overlay"
LOCOMOTION="$ISAACLAB_ROOT/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion"
REGISTRY="$LOCOMOTION/__init__.py"

[[ -d "$ISAACLAB_ROOT/.git" ]] || {
  echo "ISAACLAB_ROOT is not an Isaac Lab checkout: $ISAACLAB_ROOT" >&2
  exit 2
}

actual_commit="$(git -C "$ISAACLAB_ROOT" rev-parse HEAD)"
expected_commit="b4c321024792976150ca55fddb26fa34480d974e"
[[ "$actual_commit" == "$expected_commit" ]] || {
  echo "Isaac Lab commit mismatch" >&2
  echo "expected: $expected_commit" >&2
  echo "actual:   $actual_commit" >&2
  exit 3
}

mkdir -p "$LOCOMOTION"
cp -a "$OVERLAY/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/sprite0615_tracking" "$LOCOMOTION/"
cp -a "$OVERLAY/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/sprite0615_amp_locomotion" "$LOCOMOTION/"

tracking_import='from .sprite0615_tracking.config.sprite0615 import *  # noqa'
amp_import='from .sprite0615_amp_locomotion.config.sprite0615 import *  # noqa'
grep -qxF "$tracking_import" "$REGISTRY" || printf '\n%s\n' "$tracking_import" >> "$REGISTRY"
grep -qxF "$amp_import" "$REGISTRY" || printf '%s\n' "$amp_import" >> "$REGISTRY"

trainer_src="$OVERLAY/scripts/reinforcement_learning/rsl_rl/train_actor_critic_only.py"
trainer_dst="$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/train_actor_critic_only.py"
if [[ -f "$trainer_src" ]]; then
  cp "$trainer_src" "$trainer_dst"
fi

export SPRITE_RL_ROOT="$ROOT"
export SPRITE_OVERLAY_LOCOMOTION="$LOCOMOTION"
"$ISAACLAB_ROOT/isaaclab.sh" -p -c '
import ast
import os
from pathlib import Path

root = Path(os.environ["SPRITE_OVERLAY_LOCOMOTION"])
files = sorted((root / "sprite0615_tracking").rglob("*.py"))
files += sorted((root / "sprite0615_amp_locomotion").rglob("*.py"))
if not files:
    raise SystemExit("No Sprite overlay Python files were installed")
for path in files:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print(f"Sprite overlay static check OK ({len(files)} Python files)")
'

echo "Installed Sprite task overlay into $ISAACLAB_ROOT"
echo "Set these variables before training:"
echo "  export ISAACLAB_ROOT=$ISAACLAB_ROOT"
echo "  export SPRITE_RL_ROOT=$ROOT"
