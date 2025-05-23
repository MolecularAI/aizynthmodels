from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omegaconf import ListConfig

from aizynthmodels.utils.callbacks.callbacks import __name__ as callback_module
from aizynthmodels.utils.loading import load_item

if TYPE_CHECKING:
    from typing import Any, Dict, List


class CallbackCollection:
    """
    Store callback objects for the chemformer model.

    The callbacks can be obtained by name

    .. code-block::

        callbacks = CallbackCollection()
        callback = callbacks['LearningRateMonitor']
    """

    _collection_name = "callbacks"

    def __init__(self) -> None:
        self._logger = logging.Logger("callback-collection")
        self._items: Dict[str, Any] = {}

    def objects(self) -> List[Any]:
        """Return a list of all the loaded items"""
        return list(self._items.values())

    def load_from_config(self, callbacks_config: ListConfig) -> None:
        """
        Load one or several callbacks from a configuration dictionary

        The keys are the name of callback class. If a callback is not
        defined in the ``molbart.utils.callbacks.callbacks`` module, the module
        name can be appended, e.g. ``mypackage.callbacks.AwesomeCallback``.

        The values of the configuration is passed directly to the callback
        class along with the ``config`` parameter.

        :param callbacks_config: Config of callbacks
        """
        for item in callbacks_config:
            cls, kwargs, config_str = load_item(item, callback_module)
            obj = cls(**kwargs)
            self._items[repr(obj)] = obj
            logging.info(f"Loaded callback: '{repr(obj)}'{config_str}")
