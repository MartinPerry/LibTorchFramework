#!/bin/bash

SESSION="training"

# Check if session already exists
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists."
    echo "Attach with: tmux attach -t $SESSION"
    exit 1
fi

tmux new-session -d -s "$SESSION" \
    "./LibTorchFramework --config exprecast_config.json; exec bash"

echo "Training started in tmux session '$SESSION'."
echo "Attach with: tmux attach -t $SESSION"