import argparse
import os
import subprocess
import sys
import time

import yt.wrapper as yt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--model")
    args = parser.parse_args()

    yt_client = yt.YtClient(proxy=os.environ["YT_PROXY"], token=os.environ["YT_SECURE_VAULT_YT_TOKEN"])
    yt_client.config["proxy"]["url"] = "http://localhost:80"  # FIXME

    operation_id = os.environ["YT_OPERATION_ID"]

    while True:
        op_jobs = yt_client.list_jobs(operation_id)["jobs"]
        if len(op_jobs) == args.num_workers + 1:
            break
        time.sleep(1)

    worker_jobs = [j for j in op_jobs if j["task_name"] == "workers"]
    assert len(worker_jobs) == args.num_workers, len(worker_jobs)

    worker_addresses = []
    for wjob in worker_jobs:
        exec_node_address = wjob["address"]
        job_id = wjob["id"]
        ports = yt_client.get(f"//sys/exec_nodes/{exec_node_address}/orchid/exec_node/job_controller/active_jobs/{job_id}/job_ports")
        port = ports[0]
        host = exec_node_address.split(":")[0]
        worker_addresses.append(f"{host}:{port}")

    run_server_command = [
        "/llama/bin/llama-server", "-m", args.model,
        "-ngl", "99", "--host", "0.0.0.0", "--port", os.environ["YT_PORT_0"],
    ]
    if len(worker_addresses) > 0:
        run_server_command.extend(["--rpc", ",".join(worker_addresses)])

    subprocess.run(run_server_command, stdout=sys.stderr, check=True)


if __name__ == "__main__":
    main()
