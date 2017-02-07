from typing import (Callable, List, Mapping, Sequence, Type, Union, Optional,
                    GenericMeta, MutableSequence, TypeVar, Any, FrozenSet)


def __getattr__(name) -> Any: ...


class Converter(object):
    def register_dumps_hook(self, cls: Type[T], func: Callable[[T], Any]): ...
    def register_loads_hook(self, cls: Type[T],
                            func: Callable[[Type, Any], T]) -> None: ...
    def loads(self, obj, cl: Type): ...
    def _dumps_mapping(self, mapping: Mapping): ...
    def _loads_list(self, cl: Type[GenericMeta], obj: Iterable[T]) -> List[T]: ...
    def _loads_set(self, cl: Type[GenericMeta], obj: Iterable[T]) -> MutableSet[T]: ...
    def _loads_frozenset(self, cl: Type[GenericMeta], obj: Iterable[T]) -> FrozenSet[T]: ...
    def _loads_dict(self, cl: Type[GenericMeta], obj: Mapping[T, V]) -> Dict[T, V]: ...
    def _loads_union(self, union: Type[Union], obj: Union): ...
    def _loads_tuple(self, tup: Type[Tuple], obj: Iterable): ...
    def _get_dis_func(self, union: Type[Union]) -> Callable[..., Type]: ...

