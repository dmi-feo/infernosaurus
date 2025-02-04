import os
import time
from typing import Any

import openai
import yt.wrapper as yt

from infernosaurus.backends.llama_cpp.backend import LlamaCppOnline, LlamaCppOffline
from infernosaurus.inference_backend_base import OnlineInferenceBackendBase, OfflineInferenceBackendBase
from infernosaurus.models import (
    OnlineInferenceRuntimeInfo,
    OnlineInferenceRuntimeConfig,
    OfflineInferenceRuntimeConfig,
    OfflineInferenceRequest,
)


def wait_for_jobs_to_start(yt_client: yt.YtClient, op_id: str, num_jobs: int):
    for i in range(300):
        op_jobs = yt_client.list_jobs(op_id)["jobs"]
        if len(op_jobs) == num_jobs:
            break

        if i % 10 == 0:
            op_state = yt_client.get_operation_state(op_id)
            if op_state == "failed":
                # TODO: exception class
                # TODO: stderr
                stderr = yt_client.get_operation(op_id)
                raise Exception(f"Operation failed, {stderr}")
        time.sleep(1)


def get_job_hostport(yt_client: yt.YtClient, job: dict[str, Any]) -> tuple[str, int]:
    exec_node_address = job["address"]
    job_id = job["id"]
    ports = yt_client.get(
        f"//sys/exec_nodes/{exec_node_address}/orchid/exec_node/job_controller/active_jobs/{job_id}/job_ports"
    )
    port = ports[0]
    host = exec_node_address.split(":")[0]
    return host, port


class OnlineInferenceOperator:
    _backend: OnlineInferenceBackendBase
    _runtime_info: OnlineInferenceRuntimeInfo | None

    def __init__(self, backend_type: str, runtime_config: OnlineInferenceRuntimeConfig):
        self.yt_client = yt.YtClient(
            proxy=runtime_config.yt_settings.proxy_url, token=runtime_config.yt_settings.token,
            config=runtime_config.yt_settings.client_config_patch,
        )

        self._backend = {
            "llama_cpp": LlamaCppOnline,
        }[backend_type](runtime_config=runtime_config)

    def start(self):
        op_spec = self._backend.get_operation_spec()
        op = self.yt_client.run_operation(op_spec, sync=False)

        wait_for_jobs_to_start(self.yt_client, op.id, self._backend.runtime_config.worker_num + 1)
        op_jobs = self.yt_client.list_jobs(op.id)["jobs"]

        server_job = next(j for j in op_jobs if j["task_name"] == "server")
        server_host, server_port = get_job_hostport(self.yt_client, server_job)

        # FILL IN RUNTIME INFO
        self._runtime_info = OnlineInferenceRuntimeInfo(
            operation_id=op.id,
            server_job_id=server_job["id"],
            server_url=f"http://{server_host}:{server_port}",  # TODO: https
        )

        # check server is ready
        for i in range(600):
            if self._backend.is_ready(self._runtime_info):
                break
        else:
            raise Exception("server is not ready")

    def stop(self) -> None:
        self.yt_client.abort_operation(self._runtime_info.operation_id)
        self._runtime_info = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def get_openai_client(self) -> openai.OpenAI:
        return openai.OpenAI(
            # TODO: https
            base_url=self._runtime_info.server_url,
            api_key="no-key",
        )


class OfflineInferenceOperator:
    _backend: OfflineInferenceBackendBase

    def __init__(self, backend_type: str, runtime_config: OfflineInferenceRuntimeConfig):
        self.yt_client = yt.YtClient(
            proxy=runtime_config.yt_settings.proxy_url, token=runtime_config.yt_settings.token,
            config=runtime_config.yt_settings.client_config_patch,
        )
        self.runtime_config = runtime_config

        self._backend = {
            "llama_cpp": LlamaCppOffline,
        }[backend_type](runtime_config)

    def process(self, request: OfflineInferenceRequest):
        self.yt_client.create("map_node", request.working_dir, ignore_existing=True, recursive=True)

        op_env = {
            "YT_ALLOW_HTTP_REQUESTS_TO_YT_FROM_JOB": "1",
            "YT_PROXY": self.yt_client.config["proxy"]["url"],
        }
        if (yt_proxy_override := os.environ.get("INSA_OVERRIDE_YT_PROXY")) is not None:
            op_env["YT_PROXY"] = yt_proxy_override

        main_op_params = self._backend.get_main_launch_params(request)
        main_op_spec = (
            yt.MapSpecBuilder()
            .begin_mapper()
                .command(main_op_params.command + " >&2")
                .format(yt.JsonFormat(encode_utf8=False))
                .docker_image(main_op_params.docker_image)
                .memory_limit(self.runtime_config.worker_resources.mem)
                .cpu_limit(self.runtime_config.worker_resources.cpu)
                .file_paths(main_op_params.local_files + main_op_params.cypress_files)
                .environment(op_env)
            .end_mapper()
            .secure_vault({"YT_TOKEN": self.yt_client.config["token"]})
            .input_table_paths([request.input_table])
            .output_table_paths([request.output_table])
            .job_count(self.runtime_config.worker_num)
            .max_failed_job_count(1)  # FIXME
        )

        worker_op = None
        if self.runtime_config.model_worker_num > 0:
            worker_op_params = self._backend.get_worker_launch_params(request)
            worker_job_count = self.runtime_config.worker_num * self.runtime_config.model_worker_num
            worker_op_spec = (
                yt.VanillaSpecBuilder()
                .begin_task("model_worker")
                    .command(worker_op_params.command + " >&2")
                    .job_count(worker_job_count)
                    .docker_image(worker_op_params.docker_image)
                    .port_count(1)
                    .memory_limit(self.runtime_config.model_worker_resources.mem)
                    .cpu_limit(self.runtime_config.model_worker_resources.cpu)
                    .file_paths(worker_op_params.local_files + worker_op_params.cypress_files)
                    .environment(op_env)
                .end_task()
                .secure_vault({"YT_TOKEN": self.yt_client.config["token"]})
                .max_failed_job_count(1)  # FIXME
            )

            worker_op = self.yt_client.run_operation(worker_op_spec, sync=False)
            wait_for_jobs_to_start(self.yt_client, worker_op.id, worker_job_count)

        try:
            self.yt_client.run_operation(main_op_spec)
        finally:
            if worker_op is not None:
                self.yt_client.abort_operation(worker_op.id)