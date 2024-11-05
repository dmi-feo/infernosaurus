import random
import string


def quoted(line: str, qt: str = "\"") -> str:
    return f"{qt}{line}{qt}"


def get_random_string(length: int) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=length))