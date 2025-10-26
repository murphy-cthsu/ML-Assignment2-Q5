#!/bin/bash
# Quick commands to manage training

# Start training in background
start_training() {
    cd /home/cthsu/Workspace/ML_2025/HW2/Q5
    nohup conda run -n ml_hw2 python train_full_scale.py --preset medium > training.log 2>&1 &
    echo "Training started! PID: $!"
    echo "Monitor with: tail -f training.log"
}

# Check if training is running
check_training() {
    if pgrep -f train_full_scale > /dev/null; then
        echo "✓ Training is running"
        ps aux | grep train_full_scale | grep -v grep
    else
        echo "✗ Training is not running"
    fi
}

# Monitor training log
monitor_training() {
    tail -f training.log
}

# Stop training
stop_training() {
    PID=$(pgrep -f train_full_scale)
    if [ -n "$PID" ]; then
        echo "Stopping training (PID: $PID)..."
        kill $PID
        echo "Training stopped"
    else
        echo "No training process found"
    fi
}

# Show usage
case "$1" in
    start)
        start_training
        ;;
    check)
        check_training
        ;;
    monitor)
        monitor_training
        ;;
    stop)
        stop_training
        ;;
    *)
        echo "Usage: $0 {start|check|monitor|stop}"
        echo ""
        echo "  start   - Start training in background"
        echo "  check   - Check if training is running"
        echo "  monitor - Watch training log in real-time"
        echo "  stop    - Stop training process"
        ;;
esac
