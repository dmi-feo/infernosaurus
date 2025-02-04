import json
import os.path

import httpx
import yt.wrapper as yt

from infernosaurus.inference_backend_base import OnlineInferenceBackendBase, OfflineInferenceBackendBase
from infernosaurus import models
from infernosaurus.models import LaunchParams
from infernosaurus.utils import quoted as q


class LlamaCppOffline(OfflineInferenceBackendBase):
    def get_main_launch_params(self, request: models.OfflineInferenceRequest) -> LaunchParams:
        job_script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "scripts",
            "main_job.py",
        )
        model_rel_path = "./" + request.model_path.split("/")[-1]

        command_parts = [
            "python3", "./main_job.py", "--model-path", q(model_rel_path),
            "--input-column", q(request.input_column), "--output-column", q(request.output_column),
            "--prompt", q(request.prompt), "--echo" if request.echo else "",
            "--inference-params", q(json.dumps(request.inference_parameters.get_params()), qt="'"),
            "--num-model-workers", str(self.runtime_config.model_worker_num),
            "--working-dir", request.working_dir,
        ]

        command = " ".join(command_parts)

        return LaunchParams(
            command=command,
            local_files=[yt.LocalFile(job_script_path)],
            cypress_files=[request.model_path],
            docker_image="ghcr.io/dmi-feo/llamosaurus:4",
        )

    def get_worker_launch_params(self, request: models.OfflineInferenceRequest) -> LaunchParams:
        job_script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "scripts",
            "worker_job.py",
        )

        command_parts = [
            "python3", "./worker_job.py",
            "--num-model-workers", str(self.runtime_config.model_worker_num),
            "--working-dir", request.working_dir,
        ]
        command = " ".join(command_parts)

        return LaunchParams(
            command=command,
            local_files=[yt.LocalFile(job_script_path)],
            cypress_files=[],
            docker_image="ghcr.io/dmi-feo/llamosaurus:2",
        )


class LlamaCppOnline(OnlineInferenceBackendBase):
    def get_operation_spec(self) -> yt.VanillaSpecBuilder:
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