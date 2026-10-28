#!/usr/bin/env bash
# bench.sh — measured stone-llama vs ollama comparison, localhost, same machine.
#
# Dependencies: bash, curl, python3 (stdlib), nvidia-smi. Nothing is downloaded.
# Only reads: HTTP probes to the two endpoints and read-only nvidia-smi queries.
#
# Usage:
#   ./bench.sh                      # measure both servers, print markdown + JSON
#   STONE_START=1 ./bench.sh        # also start/stop a stone-llama daemon on STONE_PORT
#
# Env knobs: STONE_PORT OLLAMA_PORT STONE_MODEL OLLAMA_MODEL PROMPT MAX_TOKENS
#            TEMP REPEATS OUTDIR STONE_START
set -euo pipefail

OUTDIR=${OUTDIR:-/home/anon/ai/bench}
mkdir -p "$OUTDIR"

STONE_PORT=${STONE_PORT:-5199}
OLLAMA_PORT=${OLLAMA_PORT:-11434}
STONE_MODEL=${STONE_MODEL:-SmolLM3-3B-exl3_4.0bpw}
OLLAMA_MODEL=${OLLAMA_MODEL:-smollm3:3b}
PROMPT=${PROMPT:-"Explain quantum entanglement in simple terms."}
MAX_TOKENS=${MAX_TOKENS:-128}
TEMP=${TEMP:-0.7}
REPEATS=${REPEATS:-5}
STONE_START=${STONE_START:-0}
SAMPLE_MS=${SAMPLE_MS:-100}

STONE_URL="http://127.0.0.1:${STONE_PORT}"
OLLAMA_URL="http://127.0.0.1:${OLLAMA_PORT}"
DATE=$(date -u +%Y%m%dT%H%M%SZ)

STONE_OWNED=0
SAMPLER_PID=""
SAMPLES="$(mktemp)"

cleanup() {
  [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null || true
  if [ "$STONE_OWNED" = 1 ]; then stone-llama stop >/dev/null 2>&1 || true; fi
  rm -f "$SAMPLES"
}
trap cleanup EXIT

# --- preflight: endpoints must answer, else this is impossible -> exit non-zero ---
preflight() {
  curl -sf -m 5 "${STONE_URL}/v1/models" >/dev/null || {
    echo "bench: stone-llama not answering on ${STONE_URL}" >&2
    echo "bench: start it first: stone-llama serve ${STONE_MODEL} --port ${STONE_PORT} --backend tabby" >&2
    echo "bench: or re-run with STONE_START=1" >&2
    exit 2
  }
  curl -sf -m 5 "${OLLAMA_URL}/v1/models" >/dev/null || {
    echo "bench: ollama not answering on ${OLLAMA_URL} (probe-only; never restarted by this script)" >&2
    exit 2
  }
  command -v nvidia-smi >/dev/null || { echo "bench: nvidia-smi missing; VRAM impossible" >&2; exit 2; }
}

if [ "$STONE_START" = 1 ]; then
  stone-llama serve "$STONE_MODEL" --port "$STONE_PORT" --backend tabby \
    >"$OUTDIR/stone-serve-$DATE.log" 2>&1 &
  STONE_OWNED=1
  for _ in $(seq 1 120); do
    curl -sf -m 2 "${STONE_URL}/v1/models" >/dev/null 2>&1 && break
    sleep 1
  done
  # model load: chat answers 503 until the backend is resident
  for _ in $(seq 1 180); do
    code=$(curl -s -o /dev/null -w "%{http_code}" -m 10 -X POST "${STONE_URL}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"model":"'"$STONE_MODEL"'","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"stream":false}' 2>/dev/null || echo 000)
    [ "$code" = "200" ] && break
    sleep 1
  done
fi

preflight

# --- inventory (printed, and captured into the JSON) ---
GPU_JSON=$(nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version \
  --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "=== inventory ==="
echo "gpu (name,total,used,free,driver MiB): $GPU_JSON"
echo "stone /v1/models: $(curl -s -m 5 ${STONE_URL}/v1/models | python3 -c 'import sys,json;print([m["id"] for m in json.load(sys.stdin)["data"]])' 2>/dev/null)"
echo "ollama /v1/models: $(curl -s -m 5 ${OLLAMA_URL}/v1/models | python3 -c 'import sys,json;print([m["id"] for m in json.load(sys.stdin)["data"]])' 2>/dev/null)"
echo ""

# --- VRAM sampler: one line per tick -> ts_ns gpu_used_mib per-proc "pid,name,mib;..." ---
(
  while :; do
    ts=$(date +%s%N)
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    procs=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';' | tr -d ' ')
    echo "${ts} ${used:-0} ${procs}"
    sleep "$(python3 -c "print($SAMPLE_MS/1000.0)")"
  done
) >>"$SAMPLES" &
SAMPLER_PID=$!
sleep 0.5

# --- measurement: streaming request loop, per-run TTFT/wall/token-chunks ---
measure() {
  local label="$1" url="$2" model="$3"
  local raw="$OUTDIR/raw-${label}-${DATE}.jsonl"
  : >"$raw"
  echo "[$label] ${url} model=${model} repeats=${REPEATS}" >&2
  for i in $(seq 1 "$REPEATS"); do
    python3 - "$url" "$model" "$PROMPT" "$MAX_TOKENS" "$TEMP" <<'PY' >>"$raw"
import json, sys, time, urllib.request
url, model, prompt, max_tokens, temp = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5])
body = json.dumps({"model": model,
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens, "temperature": temp, "stream": True}).encode()
req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
t0 = time.time()
ttft = None
chunks = 0
pis = []          # prompt tokens (usage) when reported
gen = 0
try:
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                j = json.loads(payload)
            except Exception:
                continue
            if ttft is None:
                ttft = time.time() - t0
            ch = (j.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            if delta.get("content"):
                chunks += 1
            u = j.get("usage")
            if u:
                pis.append(u.get("prompt_tokens") or 0)
                gen = u.get("completion_tokens") or gen
    end = time.time()
    print(json.dumps({"ok": True, "ttft_s": ttft, "wall_s": end - t0,
                      "chunks": chunks, "completion_tokens": gen,
                      "prompt_tokens": (pis[-1] if pis else None),
                      "t0": t0, "t1": end}))
except Exception as e:
    print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}",
                      "wall_s": time.time() - t0, "t0": t0, "t1": time.time()}))
PY
    last=$(tail -n1 "$raw")
    echo "  run $i: $last" >&2
    sleep 1
  done
  echo "$raw"
}

STONE_RAW=$(measure stone "$STONE_URL" "$STONE_MODEL")

# stop the daemon WE started before measuring ollama, so ollama gets the full
# card — otherwise its numbers are confounded by stone's resident weights.
if [ "$STONE_OWNED" = 1 ]; then
  stone-llama stop >/dev/null 2>&1 || true
  STONE_OWNED=0
  sleep 3
else
  echo "WARNING: stone daemon still resident during ollama window — ollama" >&2
  echo "WARNING: shares VRAM with stone's loaded model; ollama numbers are" >&2
  echo "WARNING: confounded. Prefer STONE_START=1 for a clean run." >&2
fi

# give ollama a fair shot at the whole card: ask (API-level, service stays up)
# to unload its cached copy, so the next request reloads with full free VRAM.
# This is not a service restart — just the documented keep_alive=0 unload.
if [ "${OLLAMA_RELOAD:-1}" = 1 ]; then
  curl -sf -m 10 "${OLLAMA_URL}/api/generate" \
    -d '{"model":"'"$OLLAMA_MODEL"'","keep_alive":0}' >/dev/null 2>&1 || true
  sleep 2
fi

OLLAMA_RAW=$(measure ollama "$OLLAMA_URL" "$OLLAMA_MODEL")

sleep 0.5
kill "$SAMPLER_PID" 2>/dev/null || true
wait "$SAMPLER_PID" 2>/dev/null || true
SAMPLER_PID=""

# --- summarize: real stats + real peak VRAM inside each run's window ---
python3 - "$STONE_RAW" "$OLLAMA_RAW" "$SAMPLES" "$OUTDIR" "$DATE" \
        "$STONE_MODEL" "$OLLAMA_MODEL" "$PROMPT" "$MAX_TOKENS" "$TEMP" "$REPEATS" \
        "$STONE_PORT" "$OLLAMA_PORT" "$GPU_JSON" <<'PY'
import json, os, statistics as st, sys

(stone_raw, ollama_raw, samples, outdir, date, stone_model, ollama_model,
 prompt, max_tokens, temp, repeats, stone_port, ollama_port, gpu_csv) = sys.argv[1:]

def load(path):
    runs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs

def summarize(runs):
    ok = [r for r in runs if r.get("ok")]
    bad = [r for r in runs if not r.get("ok")]
    out = {"runs_ok": len(ok), "runs_failed": len(bad),
           "failures": [r.get("error") for r in bad]}
    if not ok:
        return out
    ttft = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
    wall = [r["wall_s"] for r in ok]
    # decode tok/s uses completion_tokens when the server reports usage,
    # else the streamed content-chunk count (labelled as approximation)
    tps, basis = [], None
    for r in ok:
        if r.get("completion_tokens"):
            n, basis = r["completion_tokens"], "completion_tokens"
        else:
            n, basis = r["chunks"], "stream_chunks(approx)"
        dec = r["wall_s"] - (r.get("ttft_s") or 0.0)
        if dec > 0.01 and n > 1:
            tps.append((n - 1) / dec)
        elif r["wall_s"] > 0:
            tps.append(n / r["wall_s"])
    out.update({
        "ttft_s": {"mean": st.mean(ttft), "min": min(ttft), "max": max(ttft)} if ttft else None,
        "wall_s": {"mean": st.mean(wall), "min": min(wall), "max": max(wall),
                   "median": st.median(wall)},
        "decode_tok_s": {"mean": st.mean(tps), "min": min(tps), "max": max(tps),
                         "basis": basis},
    })
    return out

samples_parsed = []
with open(samples) as f:
    for line in f:
        p = line.split()
        if len(p) >= 2:
            try:
                samples_parsed.append((int(p[0]), int(p[1]), " ".join(p[2:])))
            except ValueError:
                pass

def peak_vram(t0, t1):
    """max GPU used (MiB) over [t0,t1] wall-clock seconds, from the ns sampler."""
    lo, hi = int((t0 - 0.3) * 1e9), int((t1 + 0.3) * 1e9)
    win = [u for (ts, u, _) in samples_parsed if lo <= ts <= hi]
    procs = {}
    for ts, u, pl in samples_parsed:
        if lo <= ts <= hi:
            for part in pl.split(";"):
                fields = [f for f in part.split(",") if f]
                if len(fields) >= 3 and fields[2].isdigit():
                    procs[fields[1]] = max(procs.get(fields[1], 0), int(fields[2]))
    return (max(win) if win else None), len(win), procs

stone_runs, ollama_runs = load(stone_raw), load(ollama_raw)
s_sum, o_sum = summarize(stone_runs), summarize(ollama_runs)
s_peak = peak_vram(min(r["t0"] for r in stone_runs), max(r["t1"] for r in stone_runs))
o_peak = peak_vram(min(r["t0"] for r in ollama_runs), max(r["t1"] for r in ollama_runs))

gpu = gpu_csv.split(",")
gpu_meta = {"name": gpu[0].strip() if len(gpu) > 0 else None,
            "total_mib": int(gpu[1]) if len(gpu) > 1 else None,
            "used_mib_at_start": int(gpu[2]) if len(gpu) > 2 else None,
            "free_mib_at_start": int(gpu[3]) if len(gpu) > 3 else None,
            "driver": gpu[4].strip() if len(gpu) > 4 else None}

result = {
    "timestamp_utc": date,
    "harness": "bench.sh (bash + curl + python3 stdlib)",
    "config": {"prompt": prompt, "max_tokens": int(max_tokens),
               "temperature": float(temp), "repeats": int(repeats),
               "stone_url": f"http://127.0.0.1:{stone_port}",
               "ollama_url": f"http://127.0.0.1:{ollama_port}",
               "vram_sample_ms": 100},
    "models": {"stone": {"id": stone_model, "backend": "tabby / exllamav3",
                         "quant": "EXL3 (see quantization_config.json)"},
               "ollama": {"id": ollama_model, "quant": "Q4_K_M (ollama show)"}},
    "gpu": gpu_meta,
    "measurements": {
        "stone": {**s_sum, "peak_vram_used_mib": s_peak[0],
                  "vram_samples_in_window": s_peak[1], "per_process_peak_mib": s_peak[2]},
        "ollama": {**o_sum, "peak_vram_used_mib": o_peak[0],
                   "vram_samples_in_window": o_peak[1], "per_process_peak_mib": o_peak[2]},
    },
    "raw": {"stone": stone_runs, "ollama": ollama_runs},
}

os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, f"results-{date}.json")
with open(path, "w") as f:
    json.dump(result, f, indent=2)

def strnum(v, fmt):
    return fmt.format(v) if isinstance(v, (int, float)) else "n/a"

s, o = s_sum, o_sum
print()
print("| metric | stone-llama (exl3) | ollama (Q4_K_M) |")
print("|---|---|---|")
print(f"| model | {stone_model} | {ollama_model} |")
print(f"| runs ok | {s['runs_ok']}/{repeats} | {o['runs_ok']}/{repeats} |")
print(f"| TTFT mean (s) | {strnum(s.get('ttft_s',{}).get('mean') if s.get('ttft_s') else None,'{:.3f}')} | {strnum(o.get('ttft_s',{}).get('mean') if o.get('ttft_s') else None,'{:.3f}')} |")
print(f"| TTFT min/max (s) | {strnum(s.get('ttft_s',{}).get('min') if s.get('ttft_s') else None,'{:.3f}')}/{strnum(s.get('ttft_s',{}).get('max') if s.get('ttft_s') else None,'{:.3f}')} | {strnum(o.get('ttft_s',{}).get('min') if o.get('ttft_s') else None,'{:.3f}')}/{strnum(o.get('ttft_s',{}).get('max') if o.get('ttft_s') else None,'{:.3f}')} |")
print(f"| wall mean (s) | {strnum(s.get('wall_s',{}).get('mean'),'{:.3f}')} | {strnum(o.get('wall_s',{}).get('mean'),'{:.3f}')} |")
print(f"| decode tok/s mean | {strnum(s.get('decode_tok_s',{}).get('mean'),'{:.1f}')} | {strnum(o.get('decode_tok_s',{}).get('mean'),'{:.1f}')} |")
print(f"| peak VRAM used (MiB) | {s_peak[0]} | {o_peak[0]} |")
print(f"| per-proc peak (MiB) | {s_peak[2]} | {o_peak[2]} |")
print()
print(f"json: {path}")

# exit non-zero if either side produced no usable measurement
useful = (s.get("runs_ok", 0) > 0 and o.get("runs_ok", 0) > 0)
sys.exit(0 if useful else 3)
PY
RC=$?
if [ "$RC" != 0 ]; then
  echo "bench: incomplete measurement (json written with failures)" >&2
fi
exit "$RC"
