from typing import Any

import attr

from infernosaurus import typing as t


@attr.define
class OnlineInferenceRuntimeInfo:
    operation_id: t.OpID = attr.ib()
    server_job_id: t.JobID = attr.ib()
    server_url: str = attr.ib()


@attr.define
class Resources:
    cpu: int = attr.ib()
    mem: int = attr.ib()


@attr.define
class YtSettings:
    proxy_url: str = attr.ib()
    token: str = attr.ib(repr=False)
    client_config_patch: dict[str, Any] = attr.ib(factory=dict)


@attr.define
class InferenceParameters:
    temperature: float | None = attr.ib(default=None)
    max_tokens: int | None = attr.ib(default=None)

    def get_params(self) -> dict[str, Any]:
        params = {}
        for field in attr.fields_dict(self.__class__):
            value = getattr(self, field)
            if value is not None:
                params[field] = value

        return params


@attr.define
class OnlineInferenceRuntimeConfig:
    yt_settings: YtSettings = attr.ib()
    server_resources: Resources = attr.ib()
    model_path: str = attr.ib()
    worker_num: int = attr.ib(default=0)
    worker_resources: Resources | None = attr.ib(default=None)
    operation_title: str = attr.ib(default="llama's ass")


@attr.define
class OfflineInferenceRuntimeConfig:
    yt_settings: YtSettings = attr.ib()
    worker_num: int = attr.ib()
    worker_resources: Resources = attr.ib()
    model_worker_num: int = attr.ib(default=0)
    model_worker_resources: Resources | None = attr.ib(default=None)

    @model_worker_resources.validator
    def check_is_set(self, attribute, value):
        if self.model_worker_num > 0 and value is None:
            raise ValueError("model_worker_resources must be set when model_worker_num > 0")


@attr.define
class OfflineInferenceRequest:
    input_table: str = attr.ib()
    input_column: str = attr.ib()
    output_table: str = attr.ib()
    output_column: str = attr.ib()
    model_path: str = attr.ib()
    prompt: str = attr.ib()
    echo: bool = attr.ib(default=False)
    inference_parameters: InferenceParameters = attr.ib(factory=InferenceParameters)