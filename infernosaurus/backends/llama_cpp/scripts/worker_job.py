import argparse
import os
import subprocess
import sys

import yt.wrapper as yt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--working-dir")
    parser.add_argument("--num-model-workers", type=int)

    args = parser.parse_args()

    os.environ["YT_TOKEN"] = os.environ["YT_SECURE_VAULT_YT_TOKEN"]
    yt_client = yt.YtClient(config=yt.default_config.get_config_from_env())

    command = ["/llama/bin/rpc-server", "--host", "0.0.0.0", "--port", os.environ["YT_PORT_0"]]
    process = subprocess.Popen(command)
    if process.poll() is not None:
        raise Exception("rpc server process terminated")

    operation_id = os.environ["YT_OPERATION_ID"]
    job_id = os.environ["YT_JOB_ID"]

    with yt_client.Transaction():
        main_job_found = False

        main_jobs = yt_client.list(args.working_dir, attributes=["value"])
        while not main_job_found:
            for main_job_node in main_jobs:
                if len(main_job_node.attributes["value"]["model_workers"]) == args.num_model_workers:
                    continue
                try:
                    yt_client.lock(f"{args.working_dir}/{main_job_node}")
                except yt.errors.YtResponseError:  # TODO: check that it is a lock conflict
                    continue

                try:
                    model_workers = yt_client.get(f"{args.working_dir}/{main_job_node}/@value/model_workers")
                    if len(model_workers) == args.num_model_workers:
                        continue

                    model_workers.append({"operation_id": operation_id, "job_id": job_id})
                    main_job_found = True
                    break
                finally:
                    yt_client.unlock(f"{args.working_dir}/{main_job_node}")

    # TODO: keep checking the main job node still exist

    sys.exit(process.wait())


if __name__ == "__main__":
    main()