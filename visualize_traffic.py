import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyArrow
import seaborn as sns
import pandas as pd
import glob
import os
from matplotlib.colors import LinearSegmentedColormap

# ============================================================================
# VISUALIZER & ANALYST - Complete Visualization Suite
# ============================================================================

class TrafficVisualizer:
    """
    Complete visualization system for traffic simulation
    """
    
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.setup_style()
        
    def setup_style(self):
        """Setup plotting style"""
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
    def create_congestion_heatmap(self, csv_file, timestep=0, save=True):
        """
        Create a congestion heatmap from simulation data
        
        Args:
            csv_file: Path to congestion data CSV
            timestep: Timestep number for title
            save: Whether to save the figure
        """
        # Load data
        df = pd.read_csv(csv_file)
        
        # Create grid for heatmap
        grid = np.zeros((self.grid_size, self.grid_size))
        counts = np.zeros((self.grid_size, self.grid_size))
        
        # Aggregate congestion by node
        for _, row in df.iterrows():
            from_x, from_y = int(row['from_x']), int(row['from_y'])
            congestion = row['congestion']
            
            grid[from_y, from_x] += congestion
            counts[from_y, from_x] += 1
        
        # Average congestion per node
        with np.errstate(divide='ignore', invalid='ignore'):
            grid = np.where(counts > 0, grid / counts, 0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Custom colormap (green -> yellow -> red)
        colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('traffic', colors, N=n_bins)
        
        # Plot heatmap
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
        
        # Add grid lines
        for i in range(self.grid_size + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
            ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Congestion Level', rotation=270, labelpad=20, fontsize=12)
        
        # Labels
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_title(f'Traffic Congestion Heatmap - Timestep {timestep}', 
                     fontsize=14, fontweight='bold')
        
        # Set ticks
        ax.set_xticks(range(self.grid_size))
        ax.set_yticks(range(self.grid_size))
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'congestion_heatmap_t{timestep}.png', dpi=300, bbox_inches='tight')
            print(f"Saved: congestion_heatmap_t{timestep}.png")
        
        return fig, ax
    
    def create_road_network_visualization(self, csv_file, timestep=0, save=True):
        """
        Create a network visualization with arrows showing congestion
        
        Args:
            csv_file: Path to congestion data CSV
            timestep: Timestep number
            save: Whether to save the figure
        """
        df = pd.read_csv(csv_file)
        
        fig, ax = plt.subplots(figsize=(14, 14))
        
        # Draw roads as arrows colored by congestion
        for _, row in df.iterrows():
            from_x, from_y = row['from_x'], row['from_y']
            to_x, to_y = row['to_x'], row['to_y']
            congestion = row['congestion']
            
            # Color based on congestion
            if congestion < 0.3:
                color = '#2ecc71'  # Green
                alpha = 0.4
            elif congestion < 0.6:
                color = '#f1c40f'  # Yellow
                alpha = 0.6
            else:
                color = '#e74c3c'  # Red
                alpha = 0.8
            
            # Width based on congestion
            width = 0.02 + congestion * 0.08
            
            # Draw arrow
            dx = to_x - from_x
            dy = to_y - from_y
            
            arrow = FancyArrow(from_x, from_y, dx * 0.8, dy * 0.8,
                              width=width, head_width=width*2, head_length=0.15,
                              fc=color, ec='black', alpha=alpha, linewidth=0.5)
            ax.add_patch(arrow)
        
        # Draw intersections
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                circle = plt.Circle((x, y), 0.15, color='#34495e', zorder=5)
                ax.add_patch(circle)
        
        # Styling
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_title(f'Road Network Traffic Flow - Timestep {timestep}', 
                     fontsize=14, fontweight='bold')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='Low Congestion (<30%)'),
            Patch(facecolor='#f1c40f', label='Medium Congestion (30-60%)'),
            Patch(facecolor='#e74c3c', label='High Congestion (>60%)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'road_network_t{timestep}.png', dpi=300, bbox_inches='tight')
            print(f"Saved: road_network_t{timestep}.png")
        
        return fig, ax
    
    def create_animation(self, data_pattern='congestion_data_rank0_t*.csv', 
                        output_file='traffic_animation.gif'):
        """
        Create animated visualization of traffic over time
        
        Args:
            data_pattern: Glob pattern to find data files
            output_file: Output GIF filename
        """
        # Find all data files
        files = sorted(glob.glob(data_pattern))
        
        if not files:
            print(f"No files found matching pattern: {data_pattern}")
            return
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        def update(frame):
            ax.clear()
            
            # Load data for this frame
            df = pd.read_csv(files[frame])
            
            # Create grid
            grid = np.zeros((self.grid_size, self.grid_size))
            counts = np.zeros((self.grid_size, self.grid_size))
            
            for _, row in df.iterrows():
                from_x, from_y = int(row['from_x']), int(row['from_y'])
                grid[from_y, from_x] += row['congestion']
                counts[from_y, from_x] += 1
            
            with np.errstate(divide='ignore', invalid='ignore'):
                grid = np.where(counts > 0, grid / counts, 0)
            
            # Plot
            colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b']
            cmap = LinearSegmentedColormap.from_list('traffic', colors, N=100)
            im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
            
            timestep = int(files[frame].split('_t')[1].split('.')[0])
            ax.set_title(f'Traffic Congestion - Timestep {timestep}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            
            return [im]
        
        anim = animation.FuncAnimation(fig, update, frames=len(files), 
                                      interval=200, blit=True)
        
        # Save animation
        writer = animation.PillowWriter(fps=5)
        anim.save(output_file, writer=writer)
        print(f"Saved animation: {output_file}")
        
        plt.close()

class PerformanceAnalyzer:
    """
    Performance analysis and benchmarking
    """
    
    def __init__(self):
        self.setup_style()
    
    def setup_style(self):
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def generate_scaling_data(self):
        """
        Generate sample scaling data for demonstration
        In real scenario, this would come from actual benchmarks
        """
        # Sample data: processes/threads vs execution time
        processes = [1, 2, 4, 8, 16]
        
        # Serial baseline
        serial_time = 100.0
        
        # MPI scaling (near-linear up to 8 processes)
        mpi_times = [serial_time, 52, 28, 15, 10]
        
        # OpenMP scaling
        omp_threads = [1, 2, 4, 8, 16]
        omp_times = [serial_time, 55, 30, 18, 12]
        
        # Hybrid (MPI + OpenMP)
        hybrid_times = [serial_time, 48, 22, 12, 7.5]
        
        return {
            'processes': processes,
            'serial': [serial_time] * len(processes),
            'mpi': mpi_times,
            'omp': omp_times,
            'hybrid': hybrid_times
        }
    
    def plot_strong_scaling(self, save=True):
        """
        Create strong scaling plot
        """
        data = self.generate_scaling_data()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Execution Time Plot
        ax1.plot(data['processes'], data['serial'], 'k--', 
                marker='s', label='Serial', linewidth=2, markersize=8)
        ax1.plot(data['processes'], data['mpi'], 'b-', 
                marker='o', label='MPI Only', linewidth=2, markersize=8)
        ax1.plot(data['processes'], data['omp'], 'g-', 
                marker='^', label='OpenMP Only', linewidth=2, markersize=8)
        ax1.plot(data['processes'], data['hybrid'], 'r-', 
                marker='D', label='Hybrid (MPI + OpenMP)', linewidth=2, markersize=8)
        
        ax1.set_xlabel('Number of Processes/Threads', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_title('Strong Scaling - Execution Time', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(data['processes'])
        
        # Speedup Plot
        serial_baseline = data['serial'][0]
        speedup_mpi = [serial_baseline / t for t in data['mpi']]
        speedup_omp = [serial_baseline / t for t in data['omp']]
        speedup_hybrid = [serial_baseline / t for t in data['hybrid']]
        ideal_speedup = data['processes']
        
        ax2.plot(data['processes'], ideal_speedup, 'k--', 
                label='Ideal Speedup', linewidth=2)
        ax2.plot(data['processes'], speedup_mpi, 'b-', 
                marker='o', label='MPI Only', linewidth=2, markersize=8)
        ax2.plot(data['processes'], speedup_omp, 'g-', 
                marker='^', label='OpenMP Only', linewidth=2, markersize=8)
        ax2.plot(data['processes'], speedup_hybrid, 'r-', 
                marker='D', label='Hybrid (MPI + OpenMP)', linewidth=2, markersize=8)
        
        ax2.set_xlabel('Number of Processes/Threads', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Speedup', fontsize=12, fontweight='bold')
        ax2.set_title('Strong Scaling - Speedup', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(data['processes'])
        
        plt.tight_layout()
        
        if save:
            plt.savefig('scaling_analysis.png', dpi=300, bbox_inches='tight')
            print("Saved: scaling_analysis.png")
        
        return fig, (ax1, ax2)
    
    def plot_efficiency(self, save=True):
        """
        Create parallel efficiency plot
        """
        data = self.generate_scaling_data()
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        serial_baseline = data['serial'][0]
        
        # Calculate efficiency = speedup / num_processors
        efficiency_mpi = [(serial_baseline / t) / p for t, p in 
                         zip(data['mpi'], data['processes'])]
        efficiency_omp = [(serial_baseline / t) / p for t, p in 
                         zip(data['omp'], data['processes'])]
        efficiency_hybrid = [(serial_baseline / t) / p for t, p in 
                            zip(data['hybrid'], data['processes'])]
        
        ax.plot(data['processes'], [1.0] * len(data['processes']), 'k--', 
               label='Ideal Efficiency', linewidth=2)
        ax.plot(data['processes'], efficiency_mpi, 'b-', 
               marker='o', label='MPI Only', linewidth=2, markersize=8)
        ax.plot(data['processes'], efficiency_omp, 'g-', 
               marker='^', label='OpenMP Only', linewidth=2, markersize=8)
        ax.plot(data['processes'], efficiency_hybrid, 'r-', 
               marker='D', label='Hybrid (MPI + OpenMP)', linewidth=2, markersize=8)
        
        ax.set_xlabel('Number of Processes/Threads', fontsize=12, fontweight='bold')
        ax.set_ylabel('Parallel Efficiency', fontsize=12, fontweight='bold')
        ax.set_title('Parallel Efficiency Analysis', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(data['processes'])
        ax.set_ylim([0, 1.1])
        
        # Add annotations
        for i, p in enumerate(data['processes']):
            ax.annotate(f'{efficiency_hybrid[i]:.2f}', 
                       xy=(p, efficiency_hybrid[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, color='red')
        
        plt.tight_layout()
        
        if save:
            plt.savefig('efficiency_analysis.png', dpi=300, bbox_inches='tight')
            print("Saved: efficiency_analysis.png")
        
        return fig, ax
    
    def create_comprehensive_report(self, save=True):
        """
        Create a comprehensive performance report with multiple metrics
        """
        data = self.generate_scaling_data()
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Execution Time
        ax1 = fig.add_subplot(gs[0, :2])
        for label, times, marker in [('Serial', data['serial'], 's'),
                                     ('MPI', data['mpi'], 'o'),
                                     ('OpenMP', data['omp'], '^'),
                                     ('Hybrid', data['hybrid'], 'D')]:
            ax1.plot(data['processes'], times, marker=marker, 
                    label=label, linewidth=2, markersize=8)
        ax1.set_xlabel('Processes/Threads')
        ax1.set_ylabel('Time (s)')
        ax1.set_title('Execution Time Comparison', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Speedup
        ax2 = fig.add_subplot(gs[0, 2])
        serial_base = data['serial'][0]
        speedups = {
            'MPI': [serial_base/t for t in data['mpi']],
            'OpenMP': [serial_base/t for t in data['omp']],
            'Hybrid': [serial_base/t for t in data['hybrid']]
        }
        x = np.arange(len(data['processes']))
        width = 0.25
        for i, (label, values) in enumerate(speedups.items()):
            ax2.bar(x + i*width, values, width, label=label)
        ax2.set_xlabel('Config')
        ax2.set_ylabel('Speedup')
        ax2.set_title('Speedup at 16 Cores', fontweight='bold')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(data['processes'])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Communication Overhead (simulated)
        ax3 = fig.add_subplot(gs[1, 0])
        comm_overhead = [0, 2, 5, 10, 18]  # Percentage
        ax3.plot(data['processes'], comm_overhead, 'ro-', linewidth=2, markersize=8)
        ax3.set_xlabel('MPI Processes')
        ax3.set_ylabel('Overhead (%)')
        ax3.set_title('Communication Overhead', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Load Balance
        ax4 = fig.add_subplot(gs[1, 1])
        load_balance = [100, 98, 95, 90, 85]  # Percentage
        ax4.plot(data['processes'], load_balance, 'go-', linewidth=2, markersize=8)
        ax4.set_xlabel('Processes')
        ax4.set_ylabel('Balance (%)')
        ax4.set_title('Load Balance Efficiency', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([80, 105])
        
        # 5. Memory Usage (simulated)
        ax5 = fig.add_subplot(gs[1, 2])
        memory_per_process = [1000, 520, 270, 145, 80]  # MB
        ax5.plot(data['processes'], memory_per_process, 'bo-', linewidth=2, markersize=8)
        ax5.set_xlabel('Processes')
        ax5.set_ylabel('Memory (MB)')
        ax5.set_title('Memory per Process', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary Table
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('tight')
        ax6.axis('off')
        
        table_data = []
        for i, p in enumerate(data['processes']):
            row = [
                f"{p}",
                f"{data['hybrid'][i]:.2f}s",
                f"{serial_base/data['hybrid'][i]:.2f}x",
                f"{(serial_base/data['hybrid'][i])/p*100:.1f}%",
                f"{memory_per_process[i]}MB"
            ]
            table_data.append(row)
        
        table = ax6.table(cellText=table_data,
                         colLabels=['Cores', 'Time', 'Speedup', 'Efficiency', 'Memory'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(5):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style rows
        for i in range(1, len(table_data) + 1):
            for j in range(5):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
        
        fig.suptitle('Comprehensive Performance Analysis Report', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save:
            plt.savefig('comprehensive_report.png', dpi=300, bbox_inches='tight')
            print("Saved: comprehensive_report.png")
        
        return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function - generates all visualizations
    """
    print("=" * 60)
    print("Traffic Simulation Visualization Suite")
    print("=" * 60)
    
    # Initialize visualizers
    traffic_vis = TrafficVisualizer(grid_size=10)
    perf_analyzer = PerformanceAnalyzer()
    
    # Check for simulation data
    data_files = glob.glob('congestion_data_rank0_t*.csv')
    
    if data_files:
        print(f"\nFound {len(data_files)} simulation data files")
        
        # Create visualizations for first few timesteps
        for i, file in enumerate(sorted(data_files)[:3]):
            timestep = int(file.split('_t')[1].split('.')[0])
            print(f"\nProcessing timestep {timestep}...")
            
            # Heatmap
            traffic_vis.create_congestion_heatmap(file, timestep)
            
            # Network visualization
            traffic_vis.create_road_network_visualization(file, timestep)
        
        # Create animation
        print("\nCreating animation...")
        traffic_vis.create_animation()
    else:
        print("\nNo simulation data found. Generating sample visualizations...")
        print("Run the C++ simulation first to generate data files.")
    
    # Generate performance analysis (always available)
    print("\nGenerating performance analysis...")
    perf_analyzer.plot_strong_scaling()
    perf_analyzer.plot_efficiency()
    perf_analyzer.create_comprehensive_report()
    
    print("\n" + "=" * 60)
    print("Visualization complete! Generated files:")
    print("  - congestion_heatmap_t*.png")
    print("  - road_network_t*.png")
    print("  - traffic_animation.gif")
    print("  - scaling_analysis.png")
    print("  - efficiency_analysis.png")
    print("  - comprehensive_report.png")
    print("=" * 60)

if __name__ == "__main__":
    main()
