import random
import string

import yt.wrapper as yt


def quoted(line: str, qt: str = "\"") -> str:
    return f"{qt}{line}{qt}"


def get_random_string(length: int) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=length))


def get_job_hostport(yt_client: yt.YtClient, operation_id: str, job_id: str, port_idx: int = 0) -> tuple[str, int]:
    job = yt_client.get_job(operation_id, job_id)
    exec_node_address = job["address"]
    ports = yt_client.get(
        f"//sys/exec_nodes/{exec_node_address}/orchid/exec_node/job_controller/active_jobs/{job_id}/job_ports"
    )
    port = ports[port_idx]
    host = exec_node_address.split(":")[0]
    return host, port
