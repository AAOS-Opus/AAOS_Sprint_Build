# src/utils/process_control.py
import platform
import subprocess
import sys

def stop_orchestrator():
    """Cross-platform orchestrator shutdown"""
    system = platform.system().lower()
    try:
        if "windows" in system:
            # Windows-specific process termination
            subprocess.run(
                ['taskkill', '/F', '/FI', 'WINDOWTITLE eq uvicorn*'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )
            subprocess.run(
                ['taskkill', '/F', '/IM', 'uvicorn.exe'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )
        else:
            # Unix-like systems
            subprocess.run(
                ['pkill', '-f', 'uvicorn src.orchestrator.core:app'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )
        print("Orchestrator shutdown signal sent")
    except Exception as e:
        print(f"Shutdown warning: {e}")

if __name__ == "__main__":
    stop_orchestrator()
