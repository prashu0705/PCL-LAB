#!/bin/bash

# run_benchmarks.sh
# Automated benchmark script for traffic simulation
# Tests different configurations of MPI processes and OpenMP threads

echo "======================================================"
echo "Traffic Simulation - Performance Benchmark Suite"
echo "======================================================"
echo ""

# Create results directory
mkdir -p benchmark_results
RESULTS_FILE="benchmark_results/results.csv"

# Write CSV header
echo "Configuration,MPI_Processes,OpenMP_Threads,Execution_Time_Sec,Speedup,Efficiency" > $RESULTS_FILE

# Baseline - Serial execution
echo "Running SERIAL baseline..."
export OMP_NUM_THREADS=1
START_TIME=$(date +%s.%N)
mpirun -np 1 ./traffic_sim > benchmark_results/serial.log 2>&1
END_TIME=$(date +%s.%N)
SERIAL_TIME=$(echo "$END_TIME - $START_TIME" | bc)
echo "Serial,1,1,$SERIAL_TIME,1.00,1.00" >> $RESULTS_FILE
echo "  Time: ${SERIAL_TIME}s"
echo ""

# Test configurations
CONFIGS=(
    "2 1"   # 2 MPI, 1 OpenMP
    "4 1"   # 4 MPI, 1 OpenMP
    "8 1"   # 8 MPI, 1 OpenMP
    "1 2"   # 1 MPI, 2 OpenMP
    "1 4"   # 1 MPI, 4 OpenMP
    "1 8"   # 1 MPI, 8 OpenMP
    "2 2"   # 2 MPI, 2 OpenMP (Hybrid)
    "2 4"   # 2 MPI, 4 OpenMP (Hybrid)
    "4 2"   # 4 MPI, 2 OpenMP (Hybrid)
    "4 4"   # 4 MPI, 4 OpenMP (Hybrid)
)

for CONFIG in "${CONFIGS[@]}"; do
    read -r NP NT <<< "$CONFIG"
    
    echo "Running MPI=$NP, OpenMP=$NT..."
    export OMP_NUM_THREADS=$NT
    
    START_TIME=$(date +%s.%N)
    mpirun -np $NP ./traffic_sim > benchmark_results/mpi${NP}_omp${NT}.log 2>&1
    END_TIME=$(date +%s.%N)
    
    EXEC_TIME=$(echo "$END_TIME - $START_TIME" | bc)
    SPEEDUP=$(echo "scale=2; $SERIAL_TIME / $EXEC_TIME" | bc)
    TOTAL_CORES=$(echo "$NP * $NT" | bc)
    EFFICIENCY=$(echo "scale=4; $SPEEDUP / $TOTAL_CORES" | bc)
    
    CONFIG_NAME="MPI${NP}_OMP${NT}"
    echo "$CONFIG_NAME,$NP,$NT,$EXEC_TIME,$SPEEDUP,$EFFICIENCY" >> $RESULTS_FILE
    
    echo "  Time: ${EXEC_TIME}s | Speedup: ${SPEEDUP}x | Efficiency: ${EFFICIENCY}"
    echo ""
done

echo "======================================================"
echo "Benchmark Complete!"
echo "Results saved to: $RESULTS_FILE"
echo "======================================================"

# Generate summary
echo ""
echo "Summary of Best Configurations:"
echo "--------------------------------"
tail -n +2 $RESULTS_FILE | sort -t',' -k5 -rn | head -5 | \
    awk -F',' '{printf "%-15s | Speedup: %6.2fx | Efficiency: %5.1f%%\n", $1, $5, $6*100}'

echo ""
echo "To visualize results, run: python3 analyze_benchmarks.py"
