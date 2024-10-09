SESSION_NAME="sessionSC_20"

# Start a new tmux session, run the commands, and exit
tmux new -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "source venv-python/bin/activate" C-m
tmux send-keys -t $SESSION_NAME "export CUDA_VISIBLE_DEVICES=2" C-m
tmux send-keys -t $SESSION_NAME "python ../run_exp.py --sweep_id S3_SC_runs/nm0khoy4" C-m

# Optional: Attach to the tmux session to monitor the output
# tmux attach -t $SESSION_NAME

# Wait for a moment to ensure the session started properly
sleep 1
SESSION_NAME="sessionSC_30"

# Start a new tmux session, run the commands, and exit
tmux new -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "source venv-python/bin/activate" C-m
tmux send-keys -t $SESSION_NAME "export CUDA_VISIBLE_DEVICES=3" C-m
tmux send-keys -t $SESSION_NAME "python ../run_exp.py --sweep_id S3_SC_runs/nm0khoy4" C-m

# Optional: Attach to the tmux session to monitor the output
# tmux attach -t $SESSION_NAME

# Wait for a moment to ensure the session started properly
sleep 1

SESSION_NAME="sessionSC_2"
    
# Start a new tmux session, run the commands, and exit
tmux new -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "source venv-python/bin/activate" C-m
tmux send-keys -t $SESSION_NAME "export CUDA_VISIBLE_DEVICES=2" C-m
tmux send-keys -t $SESSION_NAME "python ../run_exp.py --sweep_id S3_SC_runs/nm0khoy4" C-m

# Optional: Attach to the tmux session to monitor the output
# tmux attach -t $SESSION_NAME

# Wait for a moment to ensure the session started properly
sleep 1

  SESSION_NAME="sessionSC_3"

# Start a new tmux session, run the commands, and exit
tmux new -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "source venv-python/bin/activate" C-m
tmux send-keys -t $SESSION_NAME "export CUDA_VISIBLE_DEVICES=3" C-m
tmux send-keys -t $SESSION_NAME "python ../run_exp.py --sweep_id S3_SC_runs/nm0khoy4" C-m

# Optional: Attach to the tmux session to monitor the output
# tmux attach -t $SESSION_NAME

# Wait for a moment to ensure the session started properly
sleep 1