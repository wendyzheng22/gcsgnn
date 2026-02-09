import abc
import typing

import torch.nn as nn


class BaseModel(nn.Module, abc.ABC):
    @abc.abstractmethod
    def __init__(self, *_: typing.Any, **__: typing.Any) -> None:
        super().__init__()

    @abc.abstractmethod
    def init_parameters(self) -> None:
        pass

    @abc.abstractmethod
    def forward(self, *_: typing.Any, **__: typing.Any) -> typing.Any:
        pass