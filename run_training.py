#!/usr/bin/env python3
"""
Python script to run a complete training round locally.
Starts stream service, server, and client in sequence.
"""

import os
import sys
import time
import subprocess
import signal
import csv
import requests
from pathlib import Path

# Colors for output
class Colors:
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color

def print_colored(message, color=Colors.NC):
    print(f"{color}{message}{Colors.NC}")

def check_port(port, timeout=5):
    """Check if a port is available."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result == 0

def wait_for_service(url, max_attempts=30, delay=1):
    """Wait for a service to be ready."""
    for i in range(max_attempts):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(delay)
    return False


def load_client_ids_for_cluster(cluster_id: str, assignments_path: Path) -> list:
    """Load client IDs for a given cluster from the assignment CSV."""
    if not assignments_path.exists():
        return []
    client_ids = []
    with assignments_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("Cluster", "")).strip() == str(cluster_id):
                client_id = str(row.get("ID", "")).strip()
                if client_id:
                    client_ids.append(client_id)
    return client_ids

def main():
    print_colored("=" * 50, Colors.BLUE)
    print_colored("Starting Split-Fed Learning Training", Colors.BLUE)
    print_colored("=" * 50, Colors.BLUE)
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Set environment variables
    env = os.environ.copy()
    env.update({
        "CLIENT_ID": "9900002",
        "CLUSTER_ID": "0",
        "DATA_STREAM_API_URL": "http://localhost:5001",
        "SERVER_PORT": "5002",
        "CLUSTER_ID_SERVER": "0",
        "MODEL_INPUT_SIZE": "1",
        "MODEL_HIDDEN_SIZE": "32",
        "MODEL_NUM_LAYERS": "2",
        "MODEL_SPLIT_LAYER": "1",
        "MODEL_OUTPUT_SIZE": "1",
        "LEARNING_RATE": "0.001",
        "BATCH_SIZE": "32",
        "NUM_EPOCHS": "50",
        "WINDOW_SIZE": "24",
        "USE_LOCALHOST": "true",
    })
    
    print_colored("\nEnvironment variables:", Colors.GREEN)
    for key in ["CLIENT_ID", "CLUSTER_ID", "SERVER_PORT", "BATCH_SIZE", "NUM_EPOCHS", "WINDOW_SIZE"]:
        print(f"  {key}: {env[key]}")
    
    processes = []
    
    try:
        # Start stream service
        print_colored("\n[1/3] Starting stream service on port 5001...", Colors.BLUE)
        stream_log = open(output_dir / "stream_service.log", "w")
        stream_proc = subprocess.Popen(
            [sys.executable, "data/stream_service.py"],
            env=env,
            stdout=stream_log,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd()
        )
        processes.append(("stream_service", stream_proc, stream_log))
        
        if not wait_for_service("http://localhost:5001/health", max_attempts=10):
            print_colored("ERROR: Stream service failed to start", Colors.RED)
            return 1
        print_colored(f"✓ Stream service started (PID: {stream_proc.pid})", Colors.GREEN)
        
        # Start server
        print_colored("\n[2/3] Starting server on port 5002...", Colors.BLUE)
        server_log = open(output_dir / "server.log", "w")
        server_proc = subprocess.Popen(
            [sys.executable, "server/server.py"],
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd()
        )
        processes.append(("server", server_proc, server_log))
        
        if not wait_for_service("http://localhost:5002/healthz", max_attempts=10):
            print_colored("ERROR: Server failed to start", Colors.RED)
            return 1
        print_colored(f"✓ Server started (PID: {server_proc.pid})", Colors.GREEN)
        
        # Wait a bit more
        time.sleep(2)
        
        # Run client training (sequentially if multiple client IDs are provided)
        print_colored("\n[3/3] Starting client training...", Colors.BLUE)
        print_colored("=" * 50, Colors.BLUE)

        client_ids = []
        if env.get("CLIENT_IDS"):
            client_ids = [cid.strip() for cid in env["CLIENT_IDS"].split(",") if cid.strip()]
        if not client_ids:
            assignments_path = Path("dataset") / "cluster-assignment.csv"
            client_ids = load_client_ids_for_cluster(env["CLUSTER_ID"], assignments_path)
        if not client_ids:
            client_ids = [env["CLIENT_ID"]]

        exit_code = 0
        for idx, client_id in enumerate(client_ids, start=1):
            env["CLIENT_ID"] = client_id
            print_colored(f"\n[Client {idx}/{len(client_ids)}] Training CLIENT_ID={client_id}", Colors.BLUE)
            client_proc = subprocess.run(
                [sys.executable, "client/client.py", "train"],
                env=env,
                cwd=os.getcwd()
            )
            exit_code = client_proc.returncode
            if exit_code != 0:
                print_colored(f"Client {client_id} exited with code {exit_code}", Colors.RED)
                break
        
    except KeyboardInterrupt:
        print_colored("\n\nInterrupted by user", Colors.RED)
        exit_code = 1
    except Exception as e:
        print_colored(f"\n\nError: {e}", Colors.RED)
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        # Cleanup
        print_colored("\nCleaning up processes...", Colors.BLUE)
        for name, proc, log_file in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
                print_colored(f"✓ {name} stopped", Colors.GREEN)
            except:
                try:
                    proc.kill()
                except:
                    pass
            finally:
                log_file.close()
    
    # Final status
    if exit_code == 0:
        print_colored("\n" + "=" * 50, Colors.GREEN)
        print_colored("Training completed successfully!", Colors.GREEN)
        print_colored("=" * 50, Colors.GREEN)
    else:
        print_colored("\n" + "=" * 50, Colors.RED)
        print_colored(f"Training completed with exit code: {exit_code}", Colors.RED)
        print_colored("=" * 50, Colors.RED)
    
    print_colored(f"\nCheck logs in {output_dir}/ directory", Colors.BLUE)
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
