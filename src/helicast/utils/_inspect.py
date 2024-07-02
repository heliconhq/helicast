import inspect
from inspect import FullArgSpec
from typing import Any, ClassVar, Dict, List, _GenericAlias

from pydantic import BaseModel

from helicast.typing import UNSET

__all__ = [
    "get_init_args_with_defaults",
    "get_init_args",
    "get_non_default_init_args",
    "is_classvar",
    "get_param_type_mapping",
    "get_classvar_list",
]


def _get_args_with_defaults(
    arg_spec: FullArgSpec, ignore_self: bool = True
) -> Dict[str, Any]:
    args = arg_spec.args[:]
    if ignore_self and arg_spec.args[0] == "self":
        args = arg_spec.args[1:]

    args = [(i, UNSET) for i in args]
    defaults = [] if arg_spec.defaults is None else arg_spec.defaults

    # Defaults are set from right to left
    args = list(reversed(args))
    defaults = list(reversed(defaults))
    for i in range(min(len(args), len(defaults))):
        args[i] = (args[i][0], defaults[i])

    return dict(reversed(args))


def _get_kwargs_with_defaults(arg_spec: FullArgSpec) -> Dict[str, Any]:
    kwargs = [] if arg_spec.kwonlyargs is None else arg_spec.kwonlyargs
    kwargs = {i: UNSET for i in kwargs}
    if arg_spec.kwonlydefaults is not None:
        kwargs.update(arg_spec.kwonlydefaults)

    return kwargs


def get_init_args_with_defaults(obj) -> Dict[str, Any]:
    """Extract the __init__ arguments from ``obj``. If ``obj`` is an
    instance of a class, we extract ``obj.__class__.__init__``, otherwise we
    expect ``obj`` to be the class itself and get ``obj.__init__``.

    Raises:
        ValueError: If the first argument upon inspection is not ``"self"``.

    Returns:
        A dictionnary where the keys are the __init__ arguments and the values the
        defaults. If there is no default, the singleton ``UNSET`` is used.
    """

    if not isinstance(obj, type):
        arg_spec = inspect.getfullargspec(obj.__class__.__init__)
    else:
        arg_spec = inspect.getfullargspec(obj.__init__)

    if arg_spec.args[0] != "self":
        raise ValueError()

    args = _get_args_with_defaults(arg_spec)
    kwargs = _get_kwargs_with_defaults(arg_spec)

    return dict(**args, **kwargs)


def get_init_args(obj) -> List[str]:
    """Extract the __init__ arguments from ``obj``. If ``obj`` is an
    instance of a class, we extract ``obj.__class__.__init__``, otherwise we
    expect ``obj`` to be the class itself and get ``obj.__init__``.

    Returns:
        A list of the argument names of the  __init__ method. The ``self`` argument
        is ignored.
    """

    if (isinstance(obj, type) and issubclass(obj, BaseModel)) or isinstance(
        obj, BaseModel
    ):
        args = obj.model_fields
        args = {i: j for i, j in args.items() if j.init is not False}
    else:
        args = get_init_args_with_defaults(obj)

    return list(args.keys())


def get_non_default_init_args(obj) -> Dict[str, Any]:
    """Extract the __init__ arguments from the Class of ``obj`` that are NOT default.

    Raises:
        ValueError: If the first argument upon inspection is not ``"self"``.

    Returns:
        A dictionnary where the keys are the __init__ arguments that where used to
        initialize ``obj`` are not the default ones.
    """

    all_args: dict = get_init_args_with_defaults(obj)

    if hasattr(obj, "get_params"):
        params = obj.get_params(deep=False)
    else:
        params = {i: getattr(obj, i) for i in all_args.keys()}

    return {i: params[i] for i, j in all_args.items() if j is UNSET or j != params[i]}


def is_classvar(_type: Any) -> bool:
    """Returns True if the ``_type`` is a ClassVar, False otherwise."""
    if _type == ClassVar:
        return True

    try:
        if _type.__origin__ == ClassVar:
            return True
    except:
        pass

    return False


def get_param_type_mapping(cls) -> Dict[str, type]:
    """Returns the types of the parameters of the estimator. This is useful for
    type checking. Correctly ignores ClassVar fields."""

    __annotations__ = getattr(cls, "__annotations__", {})
    return {
        name: _type for name, _type in __annotations__.items() if not is_classvar(_type)
    }


def get_classvar_list(cls) -> List[str]:
    """Returns the list of ClassVar fields of the dataclass."""
    __annotations__ = getattr(cls, "__annotations__", {})
    return [name for name, _type in __annotations__.items() if is_classvar(_type)]
