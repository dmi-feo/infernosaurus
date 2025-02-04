import abc

import attr

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
    runtime_config: models.OfflineInferenceRuntimeConfig

    @abc.abstractmethod
    def get_main_launch_params(self, request: models.OfflineInferenceRequest) -> models.LaunchParams:
        pass

    @abc.abstractmethod
    def get_worker_launch_params(self, request: models.OfflineInferenceRequest) -> models.LaunchParams:
        pass