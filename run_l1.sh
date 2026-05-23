cd /data2/yaxuanli/hmgraph_codex

export VIDEO=/data2/yaxuanli/HD-EPIC/Videos/P01/P01-20240202-110250.mp4
export OUT=/data2/yaxuanli/hmgraph_codex/runs/P01-20240202-110250/full
export LOG=/data2/yaxuanli/hmgraph_codex/logs/P01-20240202-110250_full.log
export PIDFILE=/data2/yaxuanli/hmgraph_codex/logs/P01-20240202-110250_full.pid

nohup bash -lc '
set -euo pipefail

cd /data2/yaxuanli/hmgraph_codex
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m code.run_hmgraph_pipeline \
  --video-path "$VIDEO" \
  --output-dir "$OUT" \
  --start-sec 0 \
  --overwrite

python -m code.validate_hmgraph_pipeline_outputs \
  --output-dir "$OUT" \
  --start-sec 0 \
  --video-path "$VIDEO"

echo "DONE"
' > "$LOG" 2>&1 &

echo $! | tee "$PIDFILE"
echo "Log: $LOG"
echo "PID: $(cat $PIDFILE)"