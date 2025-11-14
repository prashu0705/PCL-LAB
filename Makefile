# Makefile for Traffic Simulation with MPI + OpenMP
# Usage:
#   make           - Build the simulation
#   make run       - Run with default settings (4 MPI processes, 4 OpenMP threads)
#   make visualize - Run Python visualization
#   make clean     - Remove build files

# Compiler settings
CXX = mpic++
CXXFLAGS = -std=c++17 -O3 -fopenmp -Wall -Wextra
LDFLAGS = -fopenmp

# Target executable
TARGET = traffic_sim

# Source files
SOURCES = traffic_simulation.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# Default target
all: $(TARGET)

# Build executable
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)
	@echo "✓ Build complete: $(TARGET)"
	@echo "  Run with: make run"

# Pattern rule for object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run simulation with default settings
run: $(TARGET)
	@echo "================================"
	@echo "Running Traffic Simulation"
	@echo "MPI Processes: 4"
	@echo "OpenMP Threads: 4"
	@echo "================================"
	mpirun -np 4 ./$(TARGET)
	@echo "\n✓ Simulation complete!"
	@echo "  Visualize results with: make visualize"

# Run with custom settings
# Usage: make run-custom NP=8 NT=2
run-custom: $(TARGET)
	@echo "Running with $(NP) MPI processes and $(NT) OpenMP threads"
	export OMP_NUM_THREADS=$(NT) && mpirun -np $(NP) ./$(TARGET)

# Run scaling benchmarks
benchmark: $(TARGET)
	@echo "Running scaling benchmarks..."
	@echo "This will take several minutes..."
	@./run_benchmarks.sh

# Run Python visualization
visualize:
	@echo "================================"
	@echo "Generating Visualizations"
	@echo "================================"
	python3 visualize_traffic.py
	@echo "\n✓ Visualizations complete!"
	@echo "  Check output PNG and GIF files"

# Clean build artifacts
clean:
	rm -f $(TARGET) $(OBJECTS)
	rm -f *.csv *.png *.gif
	@echo "✓ Clean complete"

# Help target
help:
	@echo "Traffic Simulation Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make           - Build the simulation"
	@echo "  make run       - Run with 4 MPI processes, 4 OpenMP threads"
	@echo "  make visualize - Generate visualizations from simulation data"
	@echo "  make benchmark - Run performance benchmarks"
	@echo "  make clean     - Remove all generated files"
	@echo ""
	@echo "Custom run:"
	@echo "  make run-custom NP=8 NT=2  - Run with 8 MPI processes, 2 OpenMP threads"
	@echo ""
	@echo "Requirements:"
	@echo "  - MPI (OpenMPI or MPICH)"
	@echo "  - OpenMP-enabled compiler"
	@echo "  - Python 3 with numpy, matplotlib, seaborn, pandas"

# Check dependencies
check-deps:
	@echo "Checking dependencies..."
	@which mpic++ > /dev/null || (echo "✗ mpic++ not found. Install MPI." && exit 1)
	@echo "✓ MPI compiler found"
	@which python3 > /dev/null || (echo "✗ Python3 not found" && exit 1)
	@echo "✓ Python3 found"
	@python3 -c "import numpy, matplotlib, seaborn, pandas" 2>/dev/null || \
		(echo "✗ Python packages missing. Install with: pip install numpy matplotlib seaborn pandas" && exit 1)
	@echo "✓ Python packages found"
	@echo "✓ All dependencies satisfied"

.PHONY: all run run-custom benchmark visualize clean help check-deps
