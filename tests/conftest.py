import os

import pytest
import yt.wrapper as yt
from testcontainers_yt_local.container import YtLocalContainer


def _get_data_path(rel_path: str):
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        rel_path,
    )


@pytest.fixture(scope="session")
def yt_with_model():
    yt_container = YtLocalContainer(
        use_ng_image=True, enable_cri_jobs=True, privileged=True,
    )
    for port in range(24578, 24578 + 4):
        yt_container = yt_container.with_bind_ports(port, port)

    with yt_container:
        yt_cli: yt.YtClient = yt_container.get_client(token="topsecret")
        yt_cli.write_file("//tmp/the-model.gguf", open(_get_data_path("qwen-494M-0.5-F16.gguf"), "rb"))

        yield yt_container