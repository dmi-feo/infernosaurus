import contextlib
import time
from typing import Any

import openai
import yt.wrapper as yt

from infernosaurus.backends.llama_cpp.backend import LlamaCppBackend
from infernosaurus.llm_backend_base import LLMBackendBase
from infernosaurus.models import LLMRuntimeInfo, LLMRequest

YT_CLIENT_CONFIG = {
    "pickling": {
        "ignore_system_modules": True,
    },
    "is_local_mode": True,  # FIXME
    "proxy": {
        "enable_proxy_discovery": False,  # FIXME
    }
}

class LLMOperator:
    _backend: LLMBackendBase
    _runtime_info: LLMRuntimeInfo | None

    def __init__(self, backend_type: str, request: LLMRequest):
        self.yt_client = yt.YtClient(proxy=request.yt_proxy, token=request.yt_token, config=YT_CLIENT_CONFIG)

        self._backend = {
            "llama_cpp": LlamaCppBackend
        }[backend_type](request=request)

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
            if len(op_jobs) == self._backend.request.resources.worker_num + 1:  # FIXME
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
        self._runtime_info = LLMRuntimeInfo(
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

    @contextlib.contextmanager
    def server(self):
        self.start()
        yield self
        self.stop()

    def get_openai_client(self) -> openai.OpenAI:
        return openai.OpenAI(
            # TODO: https
            base_url=self._runtime_info.server_url,
            api_key="no-key",
        )
