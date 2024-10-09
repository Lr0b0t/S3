
# Loop over GPUDEV values
for GPUDEV in {2..3}
do
  # Loop for 4 iterations per GPUDEV value
  for ITERATION in {1..2}
  do
    # Create a unique tmux session name
    SESSION_NAME="sessionSCxx_${GPUDEV}_${ITERATION}"
    
    # Start a new tmux session, run the commands, and exit
    tmux new -d -s $SESSION_NAME
    tmux send-keys -t $SESSION_NAME "source venv-python/bin/activate" C-m
    tmux send-keys -t $SESSION_NAME "export CUDA_VISIBLE_DEVICES=$GPUDEV" C-m
    tmux send-keys -t $SESSION_NAME "python sparch/run_exp.py --sweep_id S3_SC_runs/iw05npc7" C-m

    # Optional: Attach to the tmux session to monitor the output
    # tmux attach -t $SESSION_NAME
    
    # Wait for a moment to ensure the session started properly
    sleep 1
  done
done