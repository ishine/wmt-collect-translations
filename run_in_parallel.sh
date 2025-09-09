#!/bin/bash

SESSION_NAME="gemini2.5"

# Create a new tmux session
tmux new-session -d -s $SESSION_NAME

# 31, 15, 8, 5
for i in $(seq 0 31); do
    CMD="sleep $i; conda activate wmt; source SECRETS.sh; python mist_collection.py --system 'Gemini-2.5-Pro' --parallel --subtask mist_mtqe"

    if [ $i -eq 0 ]; then
        # Send command to the first pane
        tmux send-keys -t $SESSION_NAME "$CMD" C-m
    else
        # Split window and run in new pane
        tmux split-window -t $SESSION_NAME
        tmux select-layout -t $SESSION_NAME tiled
        tmux send-keys -t $SESSION_NAME "$CMD" C-m
    fi
done

# Attach to the session
tmux attach -t $SESSION_NAME
