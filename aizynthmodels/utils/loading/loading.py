import importlib
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig

from aizynthmodels.utils.type_utils import StrDict


def load_dynamic_class(
    name_spec: str,
    default_module: Optional[str] = None,
    exception_cls: Any = ValueError,
) -> Any:
    """
    Load an object from a dynamic specification.

    The specification can be either:
        ClassName, in-case the module name is taken from the `default_module` argument
    or
        package_name.module_name.ClassName, in-case the module is taken as `package_name.module_name`

    :param name_spec: the class specification
    :param default_module: the default module. Defaults to None.
    :param exception_cls: the exception class to raise on exception
    :return: the loaded class
    """
    if "." not in name_spec:
        name = name_spec
        if not default_module:
            raise exception_cls("Must provide default_module argument if not given in name_spec")
        module_name = default_module
    else:
        module_name, name = name_spec.rsplit(".", maxsplit=1)

    try:
        loaded_module = importlib.import_module(module_name)
    except ImportError:
        raise exception_cls(f"Unable to load module: {module_name}")

    if not hasattr(loaded_module, name):
        raise exception_cls(f"Module ({module_name}) does not have a class called {name}")

    return getattr(loaded_module, name)


def unravel_list_dict(input_data: List[Dict]) -> StrDict:
    output = {}
    for data in input_data:
        for key, value in data.items():
            output[key] = value
    return output


def load_item(
    item: Union[str, DictConfig], base_module: str, extra_kwargs: StrDict = {}
) -> Tuple[Any, StrDict, DictConfig]:
    """
    Load the class specified in 'item' from 'base_module'. The item is either a str
    representing the class to load, or a DictConfig with one key (the class name) and a
    value (set of corresponding input kwargs), i.e., if the item is a DictConfig,
    only the first key will be loaded.

    :param item: specifying the class to load (name (str) or name with arguments (DictConfig)).
    :param base_module: the module which contains the class.
    :param extra_kwargs: dictionary with arguments specified separately of 'item'.
    :returns: the loaded class, all combined arguments and a string listing the arguments.
    """
    if isinstance(item, str):
        cls = load_dynamic_class(item, base_module)
        kwargs = extra_kwargs
        config_str = ""
        return cls, kwargs, config_str

    item = [(key, item[key]) for key in item.keys()][0]
    name, kwargs = item

    if kwargs is None:
        cls = load_dynamic_class(name, base_module)
        kwargs = extra_kwargs
        config_str = ""
        return cls, kwargs, config_str

    kwargs = unravel_list_dict(kwargs)

    kwargs.update(extra_kwargs)
    config_str = f" with configuration '{kwargs}'"

    cls = load_dynamic_class(name, base_module)
    return cls, kwargs, config_str
