import attr

from infernosaurus import typing as t


@attr.define
class LLMRuntimeInfo:
    operation_id: t.OpID = attr.ib()
    server_job_id: t.JobID = attr.ib()
    server_url: str = attr.ib()


@attr.define
class Resources:
    server_cpu: int = attr.ib()
    server_mem: int = attr.ib()

    worker_num: int = attr.ib(default=0)
    worker_mem: int = attr.ib(default=0)
    worker_cpu: int = attr.ib(default=0)


@attr.define
class LLMRequest:
    yt_proxy: str = attr.ib()
    yt_token: str = attr.ib(repr=False)
    resources: Resources = attr.ib()
    model_path: str = attr.ib()
    operation_title: str = attr.ib(default="llama's ass")
