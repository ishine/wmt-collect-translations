#!/bin/bash

SESSION_NAME="wmt_eval"

# Array of systems (cleaned up and deduplicated)
SYSTEMS=(
    "AyaExpanse-32B"
    "AyaExpanse-8B"
    "Claude-4"
    "CommandA"
    "CommandR7B"
    "DeepL"
    "DeepSeek-V3"
    "Gemini-2.5-Pro"
    "Gemma-3-12B"
    "Gemma-3-27B"
    "GoogleTranslate"
    "GPT-4.1"
    "Llama-3.1-8B"
    "Llama-4-Maverick"
    "Mistral-7B"
    "Mistral-Medium"
    "Qwen2.5-7B"
    "Qwen3-235B"
    "YandexTranslate"
)

# Function to sanitize window name
sanitize_name() {
    echo "$1" | tr -c '[:alnum:]' '_'
}

# Create a new tmux session with the first system
first_system="${SYSTEMS[0]}"
window_name=$(sanitize_name "$first_system")
tmux new-session -d -s $SESSION_NAME
tmux rename-window -t $SESSION_NAME "$window_name"
tmux send-keys -t $SESSION_NAME "conda activate wmt; source SECRETS.sh; python main.py --system '${first_system}'" C-m

# Create a new window for each remaining system
for system in "${SYSTEMS[@]:1}"; do
    # Create a new window with sanitized name
    window_name=$(sanitize_name "$system")
    tmux new-window -t $SESSION_NAME -n "$window_name"
    tmux send-keys -t $SESSION_NAME:"$window_name" "conda activate wmt; source SECRETS.sh; python main.py --system '${system}'" C-m
done

# Select the first window
tmux select-window -t $SESSION_NAME:0

# Attach to the session
tmux attach -t $SESSION_NAME
