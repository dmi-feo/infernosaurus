import os.path

import httpx
import yt.wrapper as yt

from infernosaurus.llm_backend_base import LLMBackendBase
from infernosaurus import models


YT_CLIENT_CONFIG = {
    "pickling": {
        "ignore_system_modules": True,
    },
    "is_local_mode": True,  # FIXME
    "proxy": {
        "enable_proxy_discovery": False,  # FIXME
    }
}


class LlamaCppOffline:
    def __init__(self, yt_proxy: str, yt_token: str, resources: models.Resources, model_path: str):
        self.resources = resources
        self.model_path = model_path

        self.yt_client = yt.YtClient(proxy=yt_proxy, token=yt_token, config=YT_CLIENT_CONFIG)

    def process(
            self, input_table: str, input_column: str, output_table: str, output_column: str,
            prompt: str,
    ):
        infer_script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "infer.py",
        )
        model_rel_path = "./" + self.model_path.split("/")[-1]

        op_spec = yt.MapSpecBuilder() \
            .begin_mapper() \
                .command(f"python3 ./infer.py --input-column \"{input_column}\" --output-column \"{output_column}\""
                         f" --prompt \"{prompt}\" --model-path {model_rel_path}") \
                .format(yt.JsonFormat()) \
                .docker_image("ghcr.io/dmi-feo/llamosaurus:2") \
                .memory_limit(self.resources.worker_mem) \
                .cpu_limit(self.resources.worker_cpu) \
                .file_paths([self.model_path, yt.LocalFile(infer_script_path)]) \
            .end_mapper() \
            .input_table_paths([input_table]) \
            .output_table_paths([output_table]) \
            .job_count(self.resources.worker_num) \
            .stderr_table_path("//tmp/stderr") \
            .max_failed_job_count(1)

        self.yt_client.run_operation(op_spec)


class LlamaCppBackend(LLMBackendBase):
    def get_operation_spec(self):  # TODO: typing
        op_spec = yt.VanillaSpecBuilder()
        op_spec = self._build_server_task(op_spec)
        if self.request.resources.worker_num > 0:
            op_spec = self._build_workers_task(op_spec)

        op_spec = op_spec \
            .stderr_table_path("//tmp/stderr") \
            .max_failed_job_count(1) \
            .secure_vault({"YT_TOKEN": self.request.yt_token}) \
            .title(self.request.operation_title)

        return op_spec

    def is_ready(self, runtime_info: models.LLMRuntimeInfo) -> bool:
        try:
            resp = httpx.get(f"{runtime_info.server_url}/health")
        except (httpx.NetworkError, httpx.ProtocolError):
            return False
        return resp.status_code == 200

    def _build_server_task(self, op_spec_builder):
        bootstrap_script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "bootstrap_server.py",
        )
        model_rel_path = "./" + self.request.model_path.split("/")[-1]

        return op_spec_builder.begin_task("server") \
            .command(f"python3 ./bootstrap_server.py --num-workers {self.request.resources.worker_num} --model {model_rel_path}") \
            .job_count(1) \
            .docker_image("ghcr.io/dmi-feo/llamosaurus:2") \
            .port_count(1) \
            .memory_limit(self.request.resources.server_mem) \
            .cpu_limit(self.request.resources.server_cpu) \
            .environment({"YT_ALLOW_HTTP_REQUESTS_TO_YT_FROM_JOB": "1", "YT_PROXY": self.request.yt_proxy}) \
            .file_paths([self.request.model_path, yt.LocalFile(bootstrap_script_path)]) \
            .end_task()

    def _build_workers_task(self, op_spec_builder):
        return op_spec_builder.begin_task("workers") \
            .command("/llama/bin/rpc-server --host 0.0.0.0 --port $YT_PORT_0 >&2") \
            .job_count(self.request.resources.worker_num) \
            .docker_image("ghcr.io/dmi-feo/llamosaurus:2") \
            .port_count(1) \
            .memory_limit(self.request.resources.worker_mem) \
            .cpu_limit(self.request.resources.worker_cpu) \
            .end_task()