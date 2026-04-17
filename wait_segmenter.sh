for i in {1..30}; do
  tmux capture-pane -t val_seg_v19 -p | tail -n 10
  sleep 4
done
