"""服务器资源监控脚本

在服务器端运行此脚本以监控GPU显存和系统内存占用情况。
可以在推理服务运行时使用此脚本来监控资源使用情况。
"""

import dataclasses
import subprocess
import time
import datetime
import sys
from typing import Optional

import tyro
import rich
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout


@dataclasses.dataclass
class Args:
    """命令行参数"""
    
    # 监控间隔（秒）
    interval: float = 1.0
    # 是否记录到文件
    log_file: Optional[str] = None
    # 是否只监控一次并退出
    once: bool = False


def get_gpu_info() -> list[dict]:
    """获取GPU信息"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        
        gpu_info = []
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 7:
                    gpu_info.append({
                        "index": parts[0],
                        "name": parts[1],
                        "memory_used_mb": float(parts[2]),
                        "memory_total_mb": float(parts[3]),
                        "utilization": float(parts[4]),
                        "temperature": float(parts[5]),
                        "power_draw": float(parts[6]),
                    })
        return gpu_info
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return []


def get_memory_info() -> dict:
    """获取系统内存信息"""
    try:
        with open("/proc/meminfo", "r") as f:
            lines = f.readlines()
        
        mem_info = {}
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip().split()[0]  # 取数值部分
                try:
                    mem_info[key] = int(value)
                except ValueError:
                    pass
        
        total_mb = mem_info.get("MemTotal", 0) / 1024
        available_mb = mem_info.get("MemAvailable", 0) / 1024
        used_mb = total_mb - available_mb
        
        return {
            "total_mb": total_mb,
            "used_mb": used_mb,
            "available_mb": available_mb,
            "percent": (used_mb / total_mb * 100) if total_mb > 0 else 0,
        }
    except Exception as e:
        return {
            "total_mb": 0,
            "used_mb": 0,
            "available_mb": 0,
            "percent": 0,
        }


def get_process_info(search_term: str = "serve_policy") -> list[dict]:
    """获取相关进程信息"""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            check=True,
        )
        
        processes = []
        for line in result.stdout.split("\n"):
            if search_term in line and "grep" not in line:
                parts = line.split(None, 10)
                if len(parts) >= 11:
                    processes.append({
                        "user": parts[0],
                        "pid": parts[1],
                        "cpu": parts[2],
                        "mem": parts[3],
                        "command": parts[10][:80],  # 截断长命令
                    })
        return processes
    except Exception as e:
        return []


def create_display_table(gpu_info: list[dict], mem_info: dict, process_info: list[dict]) -> Layout:
    """创建显示表格"""
    layout = Layout()
    
    # GPU信息表格
    gpu_table = Table(title="GPU 显存使用情况", show_header=True, header_style="bold magenta")
    gpu_table.add_column("GPU", style="cyan")
    gpu_table.add_column("名称", style="cyan")
    gpu_table.add_column("显存使用", justify="right", style="yellow")
    gpu_table.add_column("总显存", justify="right", style="green")
    gpu_table.add_column("使用率", justify="right", style="yellow")
    gpu_table.add_column("GPU利用率", justify="right", style="yellow")
    gpu_table.add_column("温度", justify="right", style="red")
    gpu_table.add_column("功耗", justify="right", style="magenta")
    
    if gpu_info:
        for gpu in gpu_info:
            mem_percent = (gpu["memory_used_mb"] / gpu["memory_total_mb"]) * 100 if gpu["memory_total_mb"] > 0 else 0
            gpu_table.add_row(
                gpu["index"],
                gpu["name"][:30],  # 截断名称
                f"{gpu['memory_used_mb']:.0f} MB",
                f"{gpu['memory_total_mb']:.0f} MB",
                f"{mem_percent:.1f}%",
                f"{gpu['utilization']:.1f}%",
                f"{gpu['temperature']:.1f}°C",
                f"{gpu['power_draw']:.1f}W",
            )
    else:
        gpu_table.add_row("N/A", "未检测到GPU", "", "", "", "", "", "")
    
    # 系统内存表格
    mem_table = Table(title="系统内存使用情况", show_header=True, header_style="bold magenta")
    mem_table.add_column("项目", style="cyan")
    mem_table.add_column("大小", justify="right", style="yellow")
    
    mem_table.add_row("总内存", f"{mem_info['total_mb']:.0f} MB")
    mem_table.add_row("已使用", f"{mem_info['used_mb']:.0f} MB")
    mem_table.add_row("可用", f"{mem_info['available_mb']:.0f} MB")
    mem_table.add_row("使用率", f"{mem_info['percent']:.1f}%")
    
    # 进程信息表格
    proc_table = Table(title="相关进程", show_header=True, header_style="bold magenta")
    proc_table.add_column("PID", style="cyan")
    proc_table.add_column("用户", style="cyan")
    proc_table.add_column("CPU%", justify="right", style="yellow")
    proc_table.add_column("MEM%", justify="right", style="yellow")
    proc_table.add_column("命令", style="green")
    
    if process_info:
        for proc in process_info:
            proc_table.add_row(
                proc["pid"],
                proc["user"],
                proc["cpu"],
                proc["mem"],
                proc["command"],
            )
    else:
        proc_table.add_row("N/A", "未找到相关进程", "", "", "")
    
    # 布局
    layout.split_column(
        Layout(gpu_table, name="gpu"),
        Layout(mem_table, name="mem"),
        Layout(proc_table, name="proc"),
    )
    
    return layout


def monitor_once(console: Console, log_file: Optional[str] = None) -> None:
    """监控一次并输出"""
    gpu_info = get_gpu_info()
    mem_info = get_memory_info()
    process_info = get_process_info()
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.print(f"\n[bold cyan]═══ 资源监控 ({timestamp}) ═══[/bold cyan]\n")
    
    layout = create_display_table(gpu_info, mem_info, process_info)
    console.print(layout)
    
    # 记录到文件
    if log_file:
        with open(log_file, "a") as f:
            f.write(f"\n=== {timestamp} ===\n")
            f.write(f"GPU Info: {gpu_info}\n")
            f.write(f"Memory Info: {mem_info}\n")
            f.write(f"Process Info: {process_info}\n")


def monitor_continuous(args: Args) -> None:
    """持续监控"""
    console = Console()
    
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]    服务器资源监控工具[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print(f"[yellow]监控间隔: {args.interval}秒[/yellow]")
    console.print(f"[yellow]按 Ctrl+C 停止监控[/yellow]\n")
    
    if args.log_file:
        console.print(f"[green]日志文件: {args.log_file}[/green]\n")
    
    try:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                gpu_info = get_gpu_info()
                mem_info = get_memory_info()
                process_info = get_process_info()
                
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # 创建显示内容
                content = Layout()
                content.split_column(
                    Layout(f"[bold cyan]═══ 资源监控 ({timestamp}) ═══[/bold cyan]", size=1),
                    Layout(create_display_table(gpu_info, mem_info, process_info)),
                )
                
                live.update(content)
                
                # 记录到文件
                if args.log_file:
                    with open(args.log_file, "a") as f:
                        f.write(f"\n=== {timestamp} ===\n")
                        if gpu_info:
                            for gpu in gpu_info:
                                f.write(f"GPU {gpu['index']}: {gpu['memory_used_mb']:.0f}/{gpu['memory_total_mb']:.0f} MB "
                                      f"({(gpu['memory_used_mb']/gpu['memory_total_mb']*100):.1f}%), "
                                      f"GPU Util: {gpu['utilization']:.1f}%\n")
                        f.write(f"Memory: {mem_info['used_mb']:.0f}/{mem_info['total_mb']:.0f} MB "
                              f"({mem_info['percent']:.1f}%)\n")
                
                time.sleep(args.interval)
                
    except KeyboardInterrupt:
        console.print("\n[yellow]监控已停止[/yellow]")


def main(args: Args) -> None:
    """主函数"""
    console = Console()
    
    # 检查nvidia-smi是否可用
    try:
        subprocess.run(["nvidia-smi", "-L"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        console.print("[yellow]警告: nvidia-smi不可用，GPU信息将无法显示[/yellow]\n")
    
    if args.once:
        monitor_once(console, args.log_file)
    else:
        monitor_continuous(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
