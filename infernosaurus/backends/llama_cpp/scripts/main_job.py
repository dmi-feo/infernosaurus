import argparse
import json
import os
import sys
import time

import yt.wrapper as yt
from llama_cpp import Llama


def get_job_hostport(yt_client: yt.YtClient, operation_id: str, job_id: str, port_idx: int = 0) -> tuple[str, int]:
    job = yt_client.get_job(operation_id, job_id)
    exec_node_address = job["address"]
    ports = yt_client.get(
        f"//sys/exec_nodes/{exec_node_address}/orchid/exec_node/job_controller/active_jobs/{job_id}/job_ports"
    )
    port = ports[port_idx]
    host = exec_node_address.split(":")[0]
    return host, port


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-column")
    parser.add_argument("--output-column")
    parser.add_argument("--prompt")
    parser.add_argument("--model-path")
    parser.add_argument("--inference-params")
    parser.add_argument("--echo", action="store_true")
    parser.add_argument("--num-model-workers", type=int)
    parser.add_argument("--working-dir")

    args = parser.parse_args()

    job_id = os.environ["YT_JOB_ID"]

    os.environ["YT_TOKEN"] = os.environ["YT_SECURE_VAULT_YT_TOKEN"]
    yt_client = yt.YtClient(config=yt.default_config.get_config_from_env())

    job_node = f"{args.working_dir}/{job_id}"

    yt_client.create(
        "document",
        job_node,
        attributes={"value": {"model_workers": []}},
    )

    if args.num_model_workers > 0:
        model_workers = []
        while True:
            print("Waiting for model workers...", file=sys.stderr)  # TODO: logging
            model_workers = yt_client.get(f"{args.working_dir}/{job_id}/@value/model_workers")
            if len(model_workers) == args.num_model_workers:
                break
            time.sleep(3)

        worker_fqdns = []
        for mw in model_workers:
            op_id, job_id = mw["operation_id"], mw["job_id"]
            host, port = get_job_hostport(yt_client, op_id, job_id)
            worker_fqdns.append(f"{host}:{port}")

        rpc_servers = ",".join(worker_fqdns)
    else:
        rpc_servers = None

    llm = Llama(model_path=args.model_path, rpc_servers=rpc_servers, n_gpu_layers=-1)

    inference_params = json.loads(args.inference_params)

    for line in sys.stdin:
        data = json.loads(line)
        input_row = str(data[args.input_column])
        prepared_prompt = args.prompt.replace("{{value}}", input_row)
        processed_row = llm.create_completion(
            prepared_prompt, echo=args.echo, **inference_params,
        )["choices"][0]["text"]
        data[args.output_column] = processed_row
        sys.stdout.write(json.dumps(data))

    yt_client.remove(job_node)


if __name__ == "__main__":
    main()
