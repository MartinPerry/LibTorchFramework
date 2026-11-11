#!/bin/bash

SESSION="training"
RUN_ID="$1"


# Check if session already exists
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists."
    echo "Attach with: tmux attach -t $SESSION"
    exit 1
fi

CMD="./LibTorchFramework --config exprecast_config.json"

if [ -n "$RUN_ID" ]; then
    CMD+=" --dashboard.run_id \"$RUN_ID\""
fi

tmux new-session -d -s "$SESSION" "$CMD; exec bash"

echo "Training started in tmux session '$SESSION'."

if [ -n "$RUN_ID" ]; then
    echo "Dashboard run_id: $RUN_ID"
fi

echo "Attach with: tmux attach -t $SESSION"