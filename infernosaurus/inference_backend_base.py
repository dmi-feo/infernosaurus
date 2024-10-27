import abc

import attr
import yt.wrapper as yt

from infernosaurus import models


@attr.define
class OnlineInferenceBackendBase(abc.ABC):
    runtime_config: models.OnlineInferenceRuntimeConfig = attr.ib()

    @abc.abstractmethod
    def get_operation_spec(self):
        pass

    @abc.abstractmethod
    def is_ready(self, runtime_info: models.OnlineInferenceRuntimeInfo) -> bool:
        pass


@attr.define
class OfflineInferenceBackendBase(abc.ABC):
    runtime_config: models.OfflineInferenceRuntimeConfig = attr.ib()

    @abc.abstractmethod
    def get_workers_operation_spec(self, request: models.OfflineInferenceRequest) -> yt.VanillaSpecBuilder | None:
        pass

    @abc.abstractmethod
    def get_operation_spec(self, request: models.OfflineInferenceRequest, workers_fqdns: list[str]):
        pass