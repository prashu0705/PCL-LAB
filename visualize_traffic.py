#!/usr/bin/env python3
"""
Smart Traffic Visualizer for Hybrid MPI + OpenMP C++ simulator.

Features:
- Auto-detect congestion CSVs (merged or per-rank).
- Merge per-rank CSVs into global per-timestep files.
- 2D heatmaps + arrow road network plots.
- Optional 3D surface congestion view.
- Simple GUI mode with timestep slider.
- Performance plots (simulated scaling data).
- Mode labels: serial / mpi / openmp / hybrid / auto.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrow
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
import pandas as pd
import glob
import os
import re
import argparse
import textwrap

# ---------------------- helpers ---------------------- #

def timestep_from_filename(fname: str) -> int:
    """Extract numeric timestep from filename like *_t10.csv."""
    base = os.path.basename(fname)
    m = re.search(r'_t(\d+)\.', base)
    if m:
        return int(m.group(1))
    nums = re.findall(r'\d+', base)
    return int(nums[-1]) if nums else 0


def detect_data_files() -> dict:
    """
    Auto-detect available data files.
    Returns dict: {
        'merged': [list of congestion_merged_t*.csv],
        'ranked': [list of congestion_data_rank*_t*.csv]
    }
    """
    merged = glob.glob('congestion_merged_t*.csv')
    ranked = glob.glob('congestion_data_rank*_t*.csv')
    return {
        'merged': sorted(merged, key=timestep_from_filename),
        'ranked': sorted(ranked, key=timestep_from_filename),
    }

# ---------------------- visualizer ---------------------- #

class TrafficVisualizer:
    def __init__(self, grid_size=10, mode="auto"):
        self.grid_size = grid_size
        self.mode = mode.lower()
        plt.rcParams['figure.figsize'] = (10, 8)
        plt.rcParams['font.size'] = 10

    # ------- merging per-rank data ------- #
    def merge_rank_files_for_timestep(self, timestep, out_fname=None):
        pattern = f'congestion_data_rank*_t{timestep}.csv'
        files = glob.glob(pattern)
        if not files:
            return None
        dfs = []
        for f in files:
            try:
                dfs.append(pd.read_csv(f))
            except Exception as e:
                print(f"[WARN] Failed to read {f}: {e}")
        if not dfs:
            return None
        merged = pd.concat(dfs, ignore_index=True)
        if out_fname is None:
            out_fname = f'congestion_merged_t{timestep}.csv'
        merged.to_csv(out_fname, index=False)
        print(f"[INFO] Merged {len(files)} rank files into {out_fname}")
        return out_fname

    # ------- grid builder ------- #
    def _grid_from_df(self, df):
        grid = np.zeros((self.grid_size, self.grid_size))
        counts = np.zeros((self.grid_size, self.grid_size))
        for _, row in df.iterrows():
            fx, fy = int(row['from_x']), int(row['from_y'])
            if 0 <= fx < self.grid_size and 0 <= fy < self.grid_size:
                grid[fy, fx] += row['congestion']
                counts[fy, fx] += 1
        with np.errstate(divide='ignore', invalid='ignore'):
            grid = np.where(counts > 0, grid / counts, 0)
        return grid, counts

    # ------- 2D heatmap ------- #
    def plot_heatmap(self, csv_file, timestep=0, save=True, show=False):
        df = pd.read_csv(csv_file)
        grid, _ = self._grid_from_df(df)

        fig, ax = plt.subplots()
        colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b']
        cmap = LinearSegmentedColormap.from_list('traffic', colors, N=100)

        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, origin='lower', interpolation='nearest')
        for i in range(self.grid_size + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
            ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Congestion Level')

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Congestion Heatmap (t={timestep}, mode={self.mode})")
        ax.set_xticks(range(self.grid_size))
        ax.set_yticks(range(self.grid_size))

        plt.tight_layout()
        if save:
            out = f"congestion_heatmap_t{timestep}.png"
            plt.savefig(out, dpi=300, bbox_inches='tight')
            print(f"[INFO] Saved {out}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    # ------- 3D surface ------- #
    def plot_heatmap_3d(self, csv_file, timestep=0, save=True, show=False):
        df = pd.read_csv(csv_file)
        grid, _ = self._grid_from_df(df)
        x = np.arange(self.grid_size)
        y = np.arange(self.grid_size)
        X, Y = np.meshgrid(x, y)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, grid, cmap='viridis', edgecolor='none', antialiased=True)
        fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, label="Congestion")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Congestion")
        ax.set_title(f"3D Congestion Surface (t={timestep}, mode={self.mode})")

        if save:
            out = f"congestion_surface3d_t{timestep}.png"
            plt.savefig(out, dpi=300, bbox_inches='tight')
            print(f"[INFO] Saved {out}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    # ------- road network arrows ------- #
    def plot_road_network(self, csv_file, timestep=0, save=True, show=False):
        df = pd.read_csv(csv_file)
        fig, ax = plt.subplots(figsize=(8, 8))

        for _, row in df.iterrows():
            fx, fy = float(row['from_x']), float(row['from_y'])
            tx, ty = float(row['to_x']), float(row['to_y'])
            c = float(row['congestion'])
            if c < 0.3:
                color = '#2ecc71'; alpha = 0.5
            elif c < 0.6:
                color = '#f1c40f'; alpha = 0.7
            else:
                color = '#e74c3c'; alpha = 0.9
            width = 0.03 + 0.12 * c
            dx = tx - fx
            dy = ty - fy
            arr = FancyArrow(fx + 0.15*dx, fy + 0.15*dy,
                             dx * 0.7, dy * 0.7,
                             width=width,
                             head_width=width*2,
                             head_length=0.12,
                             length_includes_head=True,
                             fc=color, ec='black', alpha=alpha, linewidth=0.4)
            ax.add_patch(arr)

        # nodes
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                ax.add_patch(plt.Circle((x, y), 0.12, color='#34495e', zorder=5))

        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Road Network Flow (t={timestep}, mode={self.mode})")

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='Low (<30%)'),
            Patch(facecolor='#f1c40f', label='Medium (30-60%)'),
            Patch(facecolor='#e74c3c', label='High (>60%)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        plt.tight_layout()
        if save:
            out = f"road_network_t{timestep}.png"
            plt.savefig(out, dpi=300, bbox_inches='tight')
            print(f"[INFO] Saved {out}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    # ------- animation ------- #
    def create_animation(self, files, output="traffic_animation.gif", fps=5):
        files = sorted(files, key=timestep_from_filename)
        if not files:
            print("[WARN] No files for animation")
            return
        first = pd.read_csv(files[0])
        grid, _ = self._grid_from_df(first)

        fig, ax = plt.subplots(figsize=(6, 6))
        colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b']
        cmap = LinearSegmentedColormap.from_list('traffic', colors, N=100)
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, origin='lower')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Congestion")

        def update(frame_idx):
            df = pd.read_csv(files[frame_idx])
            g, _ = self._grid_from_df(df)
            im.set_data(g)
            t = timestep_from_filename(files[frame_idx])
            ax.set_title(f"Timestep {t} (mode={self.mode})")
            return (im,)

        anim = animation.FuncAnimation(fig, update, frames=len(files), interval=200, blit=True)
        try:
            writer = animation.PillowWriter(fps=fps)
            anim.save(output, writer=writer)
            print(f"[INFO] Saved animation: {output}")
        except Exception as e:
            print(f"[WARN] GIF failed ({e}), trying mp4...")
            mp4 = output.rsplit('.', 1)[0] + ".mp4"
            anim.save(mp4, fps=fps)
            print(f"[INFO] Saved animation: {mp4}")
        plt.close(fig)

    # ------- GUI-like interactive view ------- #
    def interactive_gui(self, files):
        """Simple GUI using matplotlib slider over timesteps."""
        from matplotlib.widgets import Slider

        files = sorted(files, key=timestep_from_filename)
        if not files:
            print("[WARN] No files for GUI mode")
            return

        df0 = pd.read_csv(files[0])
        grid0, _ = self._grid_from_df(df0)
        fig, ax = plt.subplots(figsize=(7, 7))
        plt.subplots_adjust(bottom=0.15)

        colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b']
        cmap = LinearSegmentedColormap.from_list('traffic', colors, N=100)
        im = ax.imshow(grid0, cmap=cmap, vmin=0, vmax=1, origin='lower')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Congestion')

        ax.set_title(f"Timestep {timestep_from_filename(files[0])} (mode={self.mode})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Slider
        ax_slider = plt.axes([0.2, 0.03, 0.6, 0.03])
        slider = Slider(ax_slider, 'Frame', 0, len(files) - 1, valinit=0, valstep=1)

        def on_change(val):
            idx = int(slider.val)
            df = pd.read_csv(files[idx])
            g, _ = self._grid_from_df(df)
            im.set_data(g)
            t = timestep_from_filename(files[idx])
            ax.set_title(f"Timestep {t} (mode={self.mode})")
            fig.canvas.draw_idle()

        slider.on_changed(on_change)
        print("[INFO] GUI mode: use slider at bottom to change timestep, close window to exit.")
        plt.show()

# ---------------------- performance analyzer ---------------------- #

class PerformanceAnalyzer:
    def __init__(self, mode="auto"):
        self.mode = mode.lower()
        plt.rcParams['figure.figsize'] = (10, 5)

    def generate_scaling_data(self):
        # Example synthetic data: you can later replace with measured timings
        processes = [1, 2, 4, 8, 16]
        serial_time = 100.0
        mpi_times = [100.0, 52, 28, 15, 10]
        omp_times = [100.0, 55, 30, 18, 12]
        hybrid_times = [100.0, 48, 22, 12, 7.5]
        return {
            "processes": processes,
            "serial": [serial_time] * len(processes),
            "mpi": mpi_times,
            "omp": omp_times,
            "hybrid": hybrid_times,
        }

    def plot_scaling(self, save=True):
        data = self.generate_scaling_data()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # times
        ax1.plot(data['processes'], data['serial'], 'k--o', label='Serial')
        ax1.plot(data['processes'], data['mpi'], 'b-o', label='MPI')
        ax1.plot(data['processes'], data['omp'], 'g-o', label='OpenMP')
        ax1.plot(data['processes'], data['hybrid'], 'r-o', label='Hybrid')
        ax1.set_xlabel("Processes / Threads")
        ax1.set_ylabel("Time (s)")
        ax1.set_title("Strong Scaling - Time")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # speedup
        base = data['serial'][0]
        speedup_mpi = [base / t for t in data['mpi']]
        speedup_omp = [base / t for t in data['omp']]
        speedup_hybrid = [base / t for t in data['hybrid']]
        ideal = data['processes']
        ax2.plot(data['processes'], ideal, 'k--', label='Ideal')
        ax2.plot(data['processes'], speedup_mpi, 'b-o', label='MPI')
        ax2.plot(data['processes'], speedup_omp, 'g-o', label='OpenMP')
        ax2.plot(data['processes'], speedup_hybrid, 'r-o', label='Hybrid')
        ax2.set_xlabel("Processes / Threads")
        ax2.set_ylabel("Speedup")
        ax2.set_title("Strong Scaling - Speedup")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        if save:
            plt.savefig("scaling_analysis.png", dpi=300, bbox_inches='tight')
            print("[INFO] Saved scaling_analysis.png")
        plt.close(fig)

    def plot_efficiency(self, save=True):
        data = self.generate_scaling_data()
        fig, ax = plt.subplots(figsize=(8, 5))

        base = data['serial'][0]
        eff_mpi = [(base / t) / p for t, p in zip(data['mpi'], data['processes'])]
        eff_omp = [(base / t) / p for t, p in zip(data['omp'], data['processes'])]
        eff_hybrid = [(base / t) / p for t, p in zip(data['hybrid'], data['processes'])]

        ax.plot(data['processes'], eff_mpi, 'b-o', label='MPI')
        ax.plot(data['processes'], eff_omp, 'g-o', label='OpenMP')
        ax.plot(data['processes'], eff_hybrid, 'r-o', label='Hybrid')
        ax.plot(data['processes'], [1.0]*len(data['processes']), 'k--', label='Ideal')

        ax.set_xlabel("Processes / Threads")
        ax.set_ylabel("Efficiency")
        ax.set_ylim(0, 1.1)
        ax.set_title("Parallel Efficiency")
        ax.grid(True, alpha=0.3)
        ax.legend()

        if save:
            plt.savefig("efficiency_analysis.png", dpi=300, bbox_inches='tight')
            print("[INFO] Saved efficiency_analysis.png")
        plt.close(fig)

# ---------------------- main ---------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Traffic simulation visualizer (auto-detect, 2D/3D, GUI, scaling plots)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python3 visualize_traffic.py
          python3 visualize_traffic.py --mode hybrid --view both --animate
          python3 visualize_traffic.py --gui --view 3d
        """)
    )
    parser.add_argument("--grid", type=int, default=10, help="Grid size used in C++ simulation")
    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "serial", "mpi", "openmp", "hybrid"],
                        help="Which execution mode this run corresponds to (for labeling)")
    parser.add_argument("--view", type=str, default="2d",
                        choices=["2d", "3d", "both"],
                        help="Type of plots to generate")
    parser.add_argument("--animate", action="store_true", help="Create GIF/MP4 animation")
    parser.add_argument("--gui", action="store_true", help="Interactive GUI with timestep slider")
    parser.add_argument("--no-perf", action="store_true", help="Skip performance plots")
    args = parser.parse_args()

    print("="*60)
    print("Traffic Simulation Visualization Suite")
    print("="*60)
    print(f"Grid size : {args.grid}")
    print(f"Mode label: {args.mode}")
    print(f"View type : {args.view}")
    print(f"GUI mode  : {args.gui}")
    print("="*60)

    data = detect_data_files()
    use_files = []

    if data['merged']:
        print(f"[INFO] Using merged files: {len(data['merged'])} files")
        use_files = data['merged']
    elif data['ranked']:
        # Merge per timestep
        print(f"[INFO] No merged files, found {len(data['ranked'])} per-rank files, merging...")
        timesteps = sorted({timestep_from_filename(f) for f in data['ranked']})
        vis = TrafficVisualizer(grid_size=args.grid, mode=args.mode)
        merged_files = []
        for t in timesteps:
            out = vis.merge_rank_files_for_timestep(t)
            if out:
                merged_files.append(out)
        use_files = merged_files
    else:
        print("[ERROR] No congestion CSV files found.")
        print("Run the C++ simulator first to generate 'congestion_data_rank*_t*.csv'.")
        return

    if not use_files:
        print("[ERROR] No usable timestep files after merge.")
        return

    vis = TrafficVisualizer(grid_size=args.grid, mode=args.mode)

    # Make static plots for first few timesteps
    for f in sorted(use_files, key=timestep_from_filename)[:3]:
        t = timestep_from_filename(f)
        print(f"[INFO] Plotting timestep {t} from {f}")
        if args.view in ("2d", "both"):
            vis.plot_heatmap(f, timestep=t, save=True, show=False)
            vis.plot_road_network(f, timestep=t, save=True, show=False)
        if args.view in ("3d", "both"):
            vis.plot_heatmap_3d(f, timestep=t, save=True, show=False)

    # Animation
    if args.animate:
        print("[INFO] Creating animation...")
        vis.create_animation(use_files, output="traffic_animation.gif", fps=5)

    # GUI interactive mode
    if args.gui:
        print("[INFO] Launching GUI/interactive mode...")
        vis.interactive_gui(use_files)

    # Performance plots
    if not args.no_perf:
        perf = PerformanceAnalyzer(mode=args.mode)
        perf.plot_scaling()
        perf.plot_efficiency()

    print("="*60)
    print("Visualization complete.")
    print("Generated files may include:")
    print("  - congestion_merged_t*.csv")
    print("  - congestion_heatmap_t*.png")
    print("  - road_network_t*.png")
    print("  - congestion_surface3d_t*.png (if view=3d/both)")
    print("  - traffic_animation.gif (if --animate)")
    print("  - scaling_analysis.png, efficiency_analysis.png (if perf not skipped)")
    print("="*60)


if __name__ == "__main__":
    main()

