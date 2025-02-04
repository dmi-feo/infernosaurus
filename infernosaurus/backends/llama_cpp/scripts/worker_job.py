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
    print("starting rpc server", file=sys.stderr)
    process = subprocess.Popen(command)
    print("rpc server has been started", file=sys.stderr)
    if process.poll() is not None:
        raise Exception("rpc server process terminated")

    operation_id = os.environ["YT_OPERATION_ID"]
    job_id = os.environ["YT_JOB_ID"]

    print("Going to find main_job")
    main_job_found = False

    while not main_job_found:
        for main_job_node in yt_client.list(args.working_dir, attributes=["value"]):
            print(f"working with {main_job_node}", file=sys.stderr)
            if len(main_job_node.attributes["value"]["model_workers"]) == args.num_model_workers:
                print(f"job {main_job_node} has enough model_workers, skipping", file=sys.stderr)
                continue

            with yt_client.Transaction():
                try:
                    yt_client.lock(f"{args.working_dir}/{main_job_node}")
                except yt.errors.YtResponseError as err:  # TODO: check that it is a lock conflict
                    print(f"Got ResponseError: {err}", file=sys.stderr)
                    continue

                print(f"trying to attach to {main_job_node}", file=sys.stderr)
                model_workers = yt_client.get(f"{args.working_dir}/{main_job_node}/@value/model_workers")
                if len(model_workers) == args.num_model_workers:
                    print(f"job {main_job_node} has enough model_workers (while trying to attach), skipping", file=sys.stderr)
                    continue

                model_workers.append({"operation_id": operation_id, "job_id": job_id})
                yt_client.set(f"{args.working_dir}/{main_job_node}/@value/model_workers", model_workers)
                main_job_found = True
                break

    # TODO: keep checking the main job node still exist
    print("Main job found, going to work", file=sys.stderr)
    sys.exit(process.wait())


if __name__ == "__main__":
    main()