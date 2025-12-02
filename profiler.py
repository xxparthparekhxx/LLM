import time
from collections import deque
from typing import Dict, Optional, List
import statistics

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

class TrainingProfiler:
    def __init__(self, window_size: int = 20):
        self.timings: Dict[str, deque] = {}
        self.current_starts: Dict[str, float] = {}
        self.window_size = window_size
        self.console = Console() if RICH_AVAILABLE else None
        self.step_count = 0
        self.last_step_time = time.time()
        
    def start(self, name: str):
        """Start timing a section"""
        self.current_starts[name] = time.perf_counter()
        
    def stop(self, name: str):
        """Stop timing a section and record duration"""
        if name in self.current_starts:
            duration = (time.perf_counter() - self.current_starts[name]) * 1000  # ms
            if name not in self.timings:
                self.timings[name] = deque(maxlen=self.window_size)
            self.timings[name].append(duration)
            del self.current_starts[name]
            
    def step(self):
        """Mark end of a training step"""
        now = time.time()
        duration = (now - self.last_step_time) * 1000 # ms
        if "Total Step" not in self.timings:
            self.timings["Total Step"] = deque(maxlen=self.window_size)
        self.timings["Total Step"].append(duration)
        self.last_step_time = now
        self.step_count += 1

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all tracked sections"""
        stats = {}
        for name, times in self.timings.items():
            if not times:
                continue
            stats[name] = {
                "current": times[-1],
                "avg": statistics.mean(times),
                "min": min(times),
                "max": max(times)
            }
        return stats

    def print_stats(self):
        """Print stats to console (fallback if rich not available)"""
        if not self.timings:
            return
            
        print(f"\n--- Step {self.step_count} Stats ---")
        stats = self.get_stats()
        for name, data in stats.items():
            print(f"{name:<20}: {data['current']:.2f}ms (Avg: {data['avg']:.2f}ms)")

    def generate_table(self) -> Table:
        """Generate a rich table with stats"""
        if not RICH_AVAILABLE:
            return None
            
        table = Table(title=f"Training Performance (Step {self.step_count})")
        table.add_column("Section", style="cyan")
        table.add_column("Current (ms)", justify="right", style="green")
        table.add_column("Average (ms)", justify="right", style="yellow")
        table.add_column("Min (ms)", justify="right", style="dim")
        table.add_column("Max (ms)", justify="right", style="dim")
        table.add_column("% of Step", justify="right", style="magenta")
        
        stats = self.get_stats()
        total_avg = stats.get("Total Step", {}).get("avg", 1.0)
        
        # Sort by average duration, but keep Total Step at bottom
        sorted_keys = sorted([k for k in stats.keys() if k != "Total Step"], 
                           key=lambda k: stats[k]['avg'], reverse=True)
        if "Total Step" in stats:
            sorted_keys.append("Total Step")
            
        for name in sorted_keys:
            data = stats[name]
            pct = (data['avg'] / total_avg * 100) if total_avg > 0 else 0
            
            # Highlight bottlenecks
            style = None
            if name != "Total Step" and pct > 30:
                style = "bold red"
            
            table.add_row(
                name,
                f"{data['current']:.1f}",
                f"{data['avg']:.1f}",
                f"{data['min']:.1f}",
                f"{data['max']:.1f}",
                f"{pct:.1f}%",
                style=style
            )
            
        return table
