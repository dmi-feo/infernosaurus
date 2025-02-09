import os

import yt.wrapper as yt

from infernosaurus.inference_backend_base import OfflineInferenceBackendBase
from infernosaurus import models
from infernosaurus.utils import quoted as q


class VLLMOffline(OfflineInferenceBackendBase):
    def get_main_launch_params(self, request: models.OfflineInferenceRequest) -> models.LaunchParams:
        job_script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "scripts",
            "main_job.py",
        )
        model_rel_path = "./" + request.model_path.split("/")[-1]

        command_parts = [
            "python3", "./main_job.py", "--model-path", q(model_rel_path),
            "--input-column", q(request.input_column), "--output-column", q(request.output_column),
            "--prompt", q(request.prompt),

        ]

        command = " ".join(command_parts)

        return models.LaunchParams(
            command=command,
            local_files=[yt.LocalFile(job_script_path)],
            cypress_files=[request.model_path],
            docker_image="ghcr.io/dmi-feo/vllmosaurus:1",
            env_vars={"VLLM_CONFIGURE_LOGGING": "0"},
        )

    def get_worker_launch_params(self, request: models.OfflineInferenceRequest) -> models.LaunchParams:
        pass