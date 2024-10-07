import abc

import attr

from infernosaurus import typing as t
from infernosaurus import models


@attr.define
class LLMBackendBase(abc.ABC):
    request: models.LLMRequest

    @abc.abstractmethod
    def get_operation_spec(self):
        pass

    @abc.abstractmethod
    def is_ready(self, runtime_info: models.LLMRuntimeInfo) -> bool:
        pass