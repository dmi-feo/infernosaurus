import contextlib
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

    def _get_job_hostport(self, job: dict[str, Any]) -> tuple[str, int]:
        exec_node_address = job["address"]
        job_id = job["id"]
        ports = self.yt_client.get(
            f"//sys/exec_nodes/{exec_node_address}/orchid/exec_node/job_controller/active_jobs/{job_id}/job_ports"
        )
        port = ports[0]
        host = exec_node_address.split(":")[0]
        return host, port

    def start(self):
        op_spec = self._backend.get_operation_spec()
        op = self.yt_client.run_operation(op_spec, sync=False)

        # WAIT FOR NUM JOBS
        for i in range(300):
            op_jobs = self.yt_client.list_jobs(op.id)["jobs"]
            if len(op_jobs) == self._backend.runtime_config.worker_num + 1:  # FIXME
                break

            if i % 10 == 0:
                op_state = self.yt_client.get_operation_state(op.id)
                if op_state == "failed":
                    # TODO: exception class
                    # TODO: stderr
                    raise Exception("Operation failed")
            time.sleep(1)

        server_job = next(j for j in op_jobs if j["task_name"] == "server")
        server_host, server_port = self._get_job_hostport(server_job)

        # FILL IN RUNTIME INFO
        self._runtime_info = OnlineInferenceRuntimeInfo(
            operation_id=op.id,
            server_job_id=server_job["id"],
            server_url=f"http://{server_host}:{server_port}",  # TODO: https
        )

        # check server is ready
        while True:
            if not self._backend.is_ready(self._runtime_info):
                time.sleep(1)
            else:
                break

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

        self._backend = {
            "llama_cpp": LlamaCppOffline,
        }[backend_type](runtime_config=runtime_config)

    def process(self, request: OfflineInferenceRequest):
        op_spec = self._backend.get_operation_spec(request)
        self.yt_client.run_operation(op_spec)