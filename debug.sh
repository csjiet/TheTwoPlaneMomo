#!/usr/bin/env bash
# ==================================================================================================
# Usage:
#   ./debug.sh <jobid> [lines]
#
# Minimal commands:
#   ./debug.sh <int_jobid>
#     - Tail last 200 lines of logs for <int_jobid> (stderr/stdout/gpu), then show Slurm job info.
#
#   ./debug.sh <int_jobid> <n>
#     - Same, but tail the last <n> lines of each log.
#
#   DEBUG_COLORS=0 ./debug.sh <int_jobid>
#     - Turn off fancy colors 
#
# What it does (order):
#   1) Derives StdOut/StdErr/WorkDir from scontrol (best-effort).
#   2) Shows STDERR (red), STDOUT (green), GPU_LOG (blue) with tail + quick error grep.
#   3) Shows Slurm summaries 
#
# Color customization:
#   - Set DEBUG_COLORS=0 to disable colors entirely.
#   - Or override any of the vars below, e.g.:
#       C_SQUEUE=$'\033[95m'  ./debug.sh 1706399
#       C_STDERR=$'\033[91m'  ./debug.sh 1706399 400
# ==================================================================================================

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <jobid> [lines]"
  exit 2
fi

jid="$1"
lines="${2:-20}"

# ------------------------------------------------------------------------------------------
# Defaults (override by env var; set DEBUG_COLORS=0 for no color)
# ------------------------------------------------------------------------------------------
DEBUG_COLORS="${DEBUG_COLORS:-1}"

# If colors on: use ANSI; if off: empty strings
if [[ "$DEBUG_COLORS" == "1" ]]; then
  C_RESET="${C_RESET:-$'\033[0m'}"
  C_DIM="${C_DIM:-$'\033[2m'}"

  # Section colors (override any of these via env)
	C_STDERR="${C_STDERR:-$'\033[91m'}" # light red (bright red)
  C_STDOUT="${C_STDOUT:-$'\033[32m'}" # green
	C_GPU="${C_GPU:-$'\033[94m'}"       # light blue
  C_QUEUE="${C_QUEUE:-$'\033[95m'}"   # bright magenta/ pink 
	C_SQUEUE="${C_SQUEUE:-$'\033[93m'}" # orange-ish (bright yellow)
else
  C_RESET=""
  C_DIM=""
  C_STDERR=""
  C_STDOUT=""
  C_GPU=""
  C_QUEUE=""
  C_SQUEUE=""
fi

hdr () { # hdr <color> <title>
  printf "\n%s================================= %s =================================%s\n" "$1" "$2" "$C_RESET"
}

# Run a whole command block in one color
block () { # block <color> <cmd...>
  local color="$1"
  shift
  printf "%s" "$color"
  "$@"
  printf "%s" "$C_RESET"
}

# Fetch scontrol once (best-effort)
sc="$(scontrol show job -dd "$jid" 2>/dev/null || true)"

# Extract paths if present
stdout="$(echo "$sc" | tr ' ' '\n' | sed -n 's/^StdOut=//p' | head -n 1)"
stderr="$(echo "$sc" | tr ' ' '\n' | sed -n 's/^StdErr=//p' | head -n 1)"
workdir="$(echo "$sc" | tr ' ' '\n' | sed -n 's/^WorkDir=//p' | head -n 1)"

# Try to find a matching GPU log in the same directory as stdout/stderr
gpu=""
if [[ -n "${stdout:-}" ]]; then
  cand="${stdout%.out}.gpu"
  [[ -f "$cand" ]] && gpu="$cand"
fi
if [[ -z "$gpu" && -n "${stderr:-}" ]]; then
  cand="${stderr%.err}.gpu"
  [[ -f "$cand" ]] && gpu="$cand"
fi

show_file () { # show_file <label> <path> <color>
  local label="$1" path="$2" color="$3"
  if [[ -n "$path" && -f "$path" ]]; then
    hdr "$color" "$label (tail -n $lines) : $path"
    block "$color" tail -n "$lines" "$path" || true
    # grep: keep it "dim" but still inside the same section color
    printf "%s%s" "$color" "$C_DIM"
    grep -nE "Traceback|ERROR|Exception|Killed|OOM|out of memory" "$path" | tail -n 50 || true
    printf "%s" "$C_RESET"
  else
    hdr "$color" "$label : not found (${path:-<empty>})"
    printf "%s(not found)%s\n" "$color" "$C_RESET"
  fi
}

# ==================================================================================================
# 1) STDERR (red)
show_file "STDERR output" "$stderr" "$C_STDERR"

# ==================================================================================================
# 2) STDOUT (green)
show_file "STDOUT output" "$stdout" "$C_STDOUT"

# ==================================================================================================
# 3) GPU log (blue)
show_file "GPU_LOG output" "$gpu" "$C_GPU"

# ==================================================================================================
# 4) Queue/Slurm info (last)
hdr "$C_SQUEUE" "squeue (your jobs)"
block "$C_SQUEUE" squeue -u "$USER" || true

hdr "$C_QUEUE" "sacct (Slurm accounting summary of jobs)"
block "$C_QUEUE" sacct -j "$jid" -a -X \
  --format=JobID%20,JobName%35,State,ExitCode,Reason%80,NodeList%12,Elapsed,Start,End \
  || true

hdr "$C_QUEUE" "scontrol show job -dd (authoritative paths)"
if [[ -z "$sc" ]]; then
  printf "%sscontrol couldn't find job %s (maybe purged).%s\n" "$C_QUEUE" "$jid" "$C_RESET"
else
  printf "%s" "$C_QUEUE"
  echo "$sc"
  printf "%s" "$C_RESET"
fi

hdr "$C_QUEUE" "derived"
printf "%sWorkDir: %s%s\n" "$C_QUEUE" "${workdir:-<unknown>}" "$C_RESET"
printf "%sStdOut : %s%s\n" "$C_QUEUE" "${stdout:-<unknown>}" "$C_RESET"
printf "%sStdErr : %s%s\n" "$C_QUEUE" "${stderr:-<unknown>}" "$C_RESET"
printf "%sGPU_LOG: %s%s\n" "$C_QUEUE" "${gpu:-<unknown>}" "$C_RESET"

