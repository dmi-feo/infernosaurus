import os.path

import httpx
import yt.wrapper as yt

from infernosaurus.inference_backend_base import OnlineInferenceBackendBase, OfflineInferenceBackendBase
from infernosaurus import models
from infernosaurus.utils import quoted as q


class LlamaCppOffline(OfflineInferenceBackendBase):
    def get_operation_spec(self, request: models.OfflineInferenceRequest):
        infer_script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "infer.py",
        )
        model_rel_path = "./" + request.model_path.split("/")[-1]

        command = " ".join([
            "python3", "./infer.py", "--model-path", q(model_rel_path),
            "--input-column", q(request.input_column), "--output-column", q(request.output_column),
            "--prompt", q(request.prompt), "--echo" if request.echo else "",
            "--max-tokens", str(request.max_tokens),
        ])

        op_spec = yt.MapSpecBuilder() \
            .begin_mapper() \
                .command(command) \
                .format(yt.JsonFormat(encode_utf8=False)) \
                .docker_image("ghcr.io/dmi-feo/llamosaurus:4") \
                .memory_limit(self.runtime_config.worker_resources.mem) \
                .cpu_limit(self.runtime_config.worker_resources.cpu) \
                .file_paths([request.model_path, yt.LocalFile(infer_script_path)]) \
            .end_mapper() \
            .input_table_paths([request.input_table]) \
            .output_table_paths([request.output_table]) \
            .job_count(self.runtime_config.worker_num) \
            .stderr_table_path("//tmp/stderr") \
            .max_failed_job_count(1)

        return op_spec


class LlamaCppOnline(OnlineInferenceBackendBase):
    def get_operation_spec(self):  # TODO: typing
        op_spec = yt.VanillaSpecBuilder()
        op_spec = self._build_server_task(op_spec)
        if self.runtime_config.worker_num > 0:
            op_spec = self._build_workers_task(op_spec)

        op_spec = op_spec \
            .stderr_table_path("//tmp/stderr") \
            .max_failed_job_count(1) \
            .secure_vault({"YT_TOKEN": self.runtime_config.yt_settings.token}) \
            .title(self.runtime_config.operation_title)

        return op_spec

    def is_ready(self, runtime_info: models.OnlineInferenceRuntimeInfo) -> bool:
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
        model_rel_path = "./" + self.runtime_config.model_path.split("/")[-1]

        return op_spec_builder.begin_task("server") \
            .command(f"python3 ./bootstrap_server.py --num-workers {self.runtime_config.worker_num} --model {model_rel_path}") \
            .job_count(1) \
            .docker_image("ghcr.io/dmi-feo/llamosaurus:2") \
            .port_count(1) \
            .memory_limit(self.runtime_config.server_resources.mem) \
            .cpu_limit(self.runtime_config.server_resources.cpu) \
            .environment({"YT_ALLOW_HTTP_REQUESTS_TO_YT_FROM_JOB": "1", "YT_PROXY": self.runtime_config.yt_settings.proxy_url}) \
            .file_paths([self.runtime_config.model_path, yt.LocalFile(bootstrap_script_path)]) \
            .end_task()

    def _build_workers_task(self, op_spec_builder):
        return op_spec_builder.begin_task("workers") \
            .command("/llama/bin/rpc-server --host 0.0.0.0 --port $YT_PORT_0 >&2") \
            .job_count(self.runtime_config.worker_num) \
            .docker_image("ghcr.io/dmi-feo/llamosaurus:2") \
            .port_count(1) \
            .memory_limit(self.runtime_config.worker_resources.mem) \
            .cpu_limit(self.runtime_config.worker_resources.cpu) \
            .end_task()