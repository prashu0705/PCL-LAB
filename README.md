# Traffic Simulation with MPI + OpenMP

A comprehensive parallel traffic simulation system implementing distributed computing (MPI) and shared-memory parallelization (OpenMP) with advanced visualization and performance analysis.

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [System Components](#system-components)
- [Performance Analysis](#performance-analysis)
- [Visualization Guide](#visualization-guide)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project simulates traffic flow in a city grid network with:
- **10x10 intersection grid** (100 nodes, ~360 bidirectional roads)
- **100+ vehicles** with dynamic routing
- **Real-time congestion tracking**
- **Hybrid MPI+OpenMP parallelization**
- **Comprehensive visualization suite**

### Key Features

✅ **Distributed Computing (MPI)**: Divides city into regions across processes  
✅ **Shared-Memory Parallelization (OpenMP)**: Parallelizes car updates within regions  
✅ **Dynamic Routing**: Cars use shortest-path algorithms considering congestion  
✅ **Real-time Metrics**: Track congestion, throughput, and travel times  
✅ **Rich Visualizations**: Heatmaps, network graphs, animations, and performance plots

## 🏗️ System Architecture

### 1. City Modeler (Graph & Data Structures)
- **Graph Representation**: Adjacency list with road properties
- **Road Properties**: Length, speed limit, capacity, current traffic
- **Car State**: Position, route, speed, destination
- **Routing**: Dijkstra's shortest path with dynamic congestion weighting

### 2. MPI Engineer (Distributed Computing)
- **Domain Decomposition**: City divided by node ranges
- **Boundary Exchange**: MPI_Send/Recv for cars crossing regions
- **Global Metrics**: MPI_Reduce for congestion aggregation
- **Synchronization**: Barrier for timestep coordination

### 3. OpenMP Engineer (Thread Parallelization)
- **Parallel Loops**: `#pragma omp parallel for` on car updates
- **Dynamic Scheduling**: Load balancing for variable car counts
- **Race Condition Prevention**: Atomic operations for shared road state
- **Thread Scaling**: Configurable thread count via OMP_NUM_THREADS

### 4. Visualizer & Analyst (Python Suite)
- **Congestion Heatmaps**: Color-coded traffic density
- **Network Visualizations**: Arrow-based flow diagrams
- **Animations**: Time-series traffic evolution (GIF)
- **Performance Plots**: Scaling, speedup, efficiency graphs
- **Comprehensive Reports**: Multi-metric dashboard

## 📦 Requirements

### Software Dependencies

```bash
# C++ Compiler with OpenMP support
g++ >= 8.0 or clang++ >= 10.0

# MPI Implementation
OpenMPI >= 3.0 or MPICH >= 3.2

# Python 3 with packages
python3 >= 3.7
numpy >= 1.19
matplotlib >= 3.3
seaborn >= 0.11
pandas >= 1.2
```

### Hardware Recommendations

- **Minimum**: 4 cores, 8GB RAM
- **Recommended**: 8+ cores, 16GB RAM
- **Optimal**: 16+ cores, 32GB RAM for large-scale benchmarks

## 🚀 Installation

### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install build-essential libopenmpi-dev openmpi-bin
```

**macOS (Homebrew):**
```bash
brew install gcc open-mpi
```

**Fedora/RHEL:**
```bash
sudo dnf install gcc-c++ openmpi openmpi-devel
```

### 2. Install Python Dependencies

```bash
pip3 install numpy matplotlib seaborn pandas
# Or using conda
conda install numpy matplotlib seaborn pandas
```

### 3. Clone/Download Project Files

```bash
# Organize files
project/
├── traffic_simulation.cpp      # Main simulation code
├── visualize_traffic.py        # Visualization suite
├── Makefile                    # Build system
├── run_benchmarks.sh           # Benchmark script
└── README.md                   # This file
```

### 4. Verify Installation

```bash
make check-deps
```

Expected output:
```
✓ MPI compiler found
✓ Python3 found
✓ Python packages found
✓ All dependencies satisfied
```

## 🎮 Quick Start

### Basic Workflow

```bash
# 1. Build the simulation
make

# 2. Run simulation (4 MPI processes, 4 OpenMP threads)
make run

# 3. Generate visualizations
make visualize
```

### Expected Output

After `make run`:
```
=== Traffic Simulation Started ===
Grid: 10x10
Cars: 100
MPI Processes: 4
OpenMP Threads: 4

Timestep 0 - Global Congestion: 0.234
Timestep 10 - Global Congestion: 0.456
...
=== Simulation Complete ===
Total time: 2.34 seconds
```

After `make visualize`:
```
Generated files:
  ✓ congestion_heatmap_t0.png
  ✓ road_network_t0.png
  ✓ traffic_animation.gif
  ✓ scaling_analysis.png
  ✓ efficiency_analysis.png
  ✓ comprehensive_report.png
```

## 📖 Detailed Usage

### Custom Execution

#### Adjust MPI Processes and OpenMP Threads

```bash
# 8 MPI processes, 2 OpenMP threads each (16 total cores)
make run-custom NP=8 NT=2

# Single process, 16 threads (pure OpenMP)
make run-custom NP=1 NT=16

# 16 processes, 1 thread each (pure MPI)
make run-custom NP=16 NT=1
```

#### Manual Execution

```bash
# Build
mpic++ -std=c++17 -O3 -fopenmp -o traffic_sim traffic_simulation.cpp

# Run with environment control
export OMP_NUM_THREADS=4
mpirun -np 4 ./traffic_sim
```

### Performance Benchmarking

```bash
# Automated benchmark suite
make benchmark

# Results saved to benchmark_results/results.csv
```

The benchmark tests:
- Serial baseline (1 MPI, 1 OpenMP)
- Pure MPI scaling (1, 2, 4, 8 processes)
- Pure OpenMP scaling (1, 2, 4, 8 threads)
- Hybrid configurations (combinations)

### Customizing Simulation Parameters

Edit `traffic_simulation.cpp` main function:

```cpp
const int GRID_SIZE = 10;        // Grid dimensions (NxN)
const int NUM_CARS = 100;        // Number of vehicles
const int NUM_TIMESTEPS = 50;    // Simulation duration
const double DT = 0.1;           // Time step (hours)
```

Then rebuild:
```bash
make clean && make
```

## 🔧 System Components

### City Graph Structure

```
Nodes: Intersections (100 in 10x10 grid)
Edges: Roads (bidirectional, ~360 total)

Road Properties:
- Length: 0.5 - 2.0 km (random)
- Speed Limit: 40 - 80 km/h (random)
- Capacity: 8 - 15 cars (random)
- Congestion: current_cars / capacity
```

### Car Behavior

1. **Initialization**: Random start/destination
2. **Routing**: Shortest path via Dijkstra
3. **Movement**: Speed adjusted by congestion
4. **Re-routing**: Dynamic path updates (optional)
5. **Completion**: Removed upon reaching destination

### MPI Communication Pattern

```
Process 0: Nodes 0-24
Process 1: Nodes 25-49
Process 2: Nodes 50-74
Process 3: Nodes 75-99

Boundary Exchange:
- Car crosses region → MPI_Send to target process
- Target process → MPI_Recv and integrates car
- Synchronization → MPI_Barrier each timestep
```

### OpenMP Parallelization

```cpp
// Car update loop (embarrassingly parallel)
#pragma omp parallel for schedule(dynamic)
for (int i = 0; i < num_cars; ++i) {
    update_car_position(cars[i], graph, dt);
}

// Road congestion update (requires atomics)
#pragma omp atomic
road->current_cars++;
```

## 📊 Performance Analysis

### Expected Scaling Results

| Configuration | Time (s) | Speedup | Efficiency |
|--------------|----------|---------|------------|
| Serial (1x1) | 100.0    | 1.00x   | 100%       |
| MPI 4x1      | 28.0     | 3.57x   | 89%        |
| OMP 1x4      | 30.0     | 3.33x   | 83%        |
| Hybrid 2x2   | 22.0     | 4.55x   | 114%*      |
| Hybrid 4x4   | 12.0     | 8.33x   | 52%        |

*Super-linear speedup due to cache effects

### Speedup Formula

```
Speedup = T_serial / T_parallel
Efficiency = Speedup / (num_processes × num_threads)
```

### Interpretation

- **Good Scaling**: Efficiency > 70%
- **Acceptable**: Efficiency 50-70%
- **Poor**: Efficiency < 50% (communication overhead dominant)

## 🎨 Visualization Guide

### 1. Congestion Heatmap

**File**: `congestion_heatmap_t*.png`

- **Green**: Low traffic (< 30% capacity)
- **Yellow**: Moderate (30-60%)
- **Red**: Heavy congestion (> 60%)

### 2. Road Network Visualization

**File**: `road_network_t*.png`

- **Arrows**: Traffic flow direction
- **Arrow Width**: Traffic volume
- **Arrow Color**: Congestion level
- **Black Circles**: Intersections

### 3. Animation

**File**: `traffic_animation.gif`

- Shows congestion evolution over time
- 5 FPS, entire simulation timeline

### 4. Performance Plots

#### Scaling Analysis (`scaling_analysis.png`)
- **Left**: Execution time vs cores
- **Right**: Speedup vs cores (with ideal line)

#### Efficiency Analysis (`efficiency_analysis.png`)
- Parallel efficiency for each configuration
- Target: Stay above 0.7 (70%)

#### Comprehensive Report (`comprehensive_report.png`)
- 6-panel dashboard
- Execution time, speedup, communication overhead
- Load balance, memory usage, summary table

## 🐛 Troubleshooting

### Common Issues

#### 1. MPI Not Found
```
Error: mpic++ command not found
```
**Solution**: Install MPI
```bash
# Ubuntu
sudo apt-get install libopenmpi-dev openmpi-bin

# macOS
brew install open-mpi
```

#### 2. OpenMP Not Enabled
```
Warning: ignoring #pragma omp
```
**Solution**: Add `-fopenmp` flag
```bash
mpic++ -fopenmp -o traffic_sim traffic_simulation.cpp
```

#### 3. Python Packages Missing
```
ModuleNotFoundError: No module named 'matplotlib'
```
**Solution**: Install packages
```bash
pip3 install numpy matplotlib seaborn pandas
```

#### 4. No Output Files
```
No files found matching pattern: congestion_data_rank0_t*.csv
```
**Solution**: Ensure simulation ran successfully
```bash
# Check if simulation produced CSV files
ls *.csv

# If not, check simulation logs
./traffic_sim 2>&1 | tee simulation.log
```

#### 5. Segmentation Fault
```
mpirun noticed that process rank 0 exited on signal 11 (Segmentation fault)
```
**Solution**: Reduce grid size or car count for initial testing
```cpp
const int GRID_SIZE = 5;   // Smaller grid
const int NUM_CARS = 20;   // Fewer cars
```

### Performance Issues

#### Low Speedup
- **Cause**: Too few cars relative to cores
- **Solution**: Increase `NUM_CARS` to 500-1000

#### High Communication Overhead
- **Cause**: Too many MPI processes
- **Solution**: Use hybrid approach (fewer processes, more threads)

#### Poor Load Balance
- **Cause**: Uneven car distribution
- **Solution**: Implement dynamic load balancing or better domain decomposition

## 📝 Code Structure

### Main Components

```cpp
// 1. Data Structures
struct Road { ... }
struct Car { ... }
class CityGraph { ... }

// 2. MPI Coordination
class MPISimulator {
    void distribute_cars();
    void exchange_boundary_cars();
    double gather_global_congestion();
}

// 3. OpenMP Parallelization
class OpenMPEngine {
    static void update_cars_parallel();
    static void update_road_congestion_parallel();
}

// 4. Main Loop
int main() {
    MPI_Init();
    // Build graph
    // Generate cars
    // Distribute work
    for (timestep) {
        update_cars();      // OpenMP
        exchange_cars();    // MPI
        gather_metrics();   // MPI
    }
    MPI_Finalize();
}
```

## 🎓 Educational Value

### Learning Objectives

1. **Hybrid Parallelization**: Combine distributed + shared memory
2. **Domain Decomposition**: Spatial partitioning strategies
3. **Communication Patterns**: Point-to-point vs collective
4. **Load Balancing**: Dynamic work distribution
5. **Performance Analysis**: Amdahl's Law, scaling limits
6. **Scientific Visualization**: Data-driven graphics

### Experimental Ideas

1. **Vary Grid Size**: 5x5, 10x10, 20x20 → observe scaling
2. **Traffic Patterns**: Rush hour, random, directional flow
3. **Road Failures**: Simulate accidents or construction
4. **Routing Strategies**: Static vs dynamic re-routing
5. **Communication Optimization**: Async MPI, reduced exchanges

## 📄 License & Citation

This project is provided for educational purposes.

If used in academic work, please cite:
```
Traffic Simulation with MPI and OpenMP
GitHub: [Your Repository]
Year: 2024
```

## 🤝 Contributing

Improvements welcome! Areas for enhancement:
- Realistic traffic models (car-following, lane changing)
- GPU acceleration (CUDA/OpenCL)
- Interactive visualization (real-time GUI)
- Machine learning integration (traffic prediction)
- Larger city maps (OpenStreetMap integration)

## 📞 Support

For issues:
1. Check [Troubleshooting](#troubleshooting)
2. Review error logs: `simulation.log`
3. Test with minimal config: 1 process, 1 thread, 5x5 grid
4. Open GitHub issue with full error output

---

**Happy Simulating! 🚗💨**
