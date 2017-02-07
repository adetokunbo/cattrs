"""Test loading functionality."""
import sys
from string import ascii_letters

from enum import Enum
from hypothesis import assume, given
from hypothesis.strategies import (binary, choices, composite, dictionaries,
                                   floats, frozensets, integers, just, lists,
                                   one_of, sampled_from, sets, text, tuples)
from pytest import mark, raises
from typing import (Any, Dict, FrozenSet, List, Mapping, MutableMapping,
                    MutableSequence, MutableSet, Optional, Sequence, Set,
                    Tuple, Union)

from cattr import Converter
from cattr.compat import bytes, unicode

ints_and_text = [(integers(), int), (text(), unicode)]
primitives = ints_and_text + [(floats(min_value=1.0, allow_nan=False), float)]

ints_and_type = tuples(integers(max_value=sys.maxsize), just(int))
floats_and_type = tuples(floats(min_value=1.0, allow_nan=False), just(float))
strs_and_type = tuples(text(), just(unicode))
bytes_and_type = tuples(binary(), just(bytes))

primitives_and_type = one_of(ints_and_type, floats_and_type, strs_and_type,
                             bytes_and_type)

list_types = sampled_from([List, Sequence, MutableSequence])
mut_set_types = sampled_from([Set, MutableSet])
set_types = one_of(mut_set_types, just(FrozenSet))
dict_types = sampled_from([Dict, MutableMapping, Mapping])

lists_of_primitives = sampled_from(primitives).flatmap(
    lambda e: tuples(lists(e[0]),
                     one_of(list_types.map(lambda t: t[e[1]]), list_types)))

h_tuple_types = sampled_from([Tuple, Sequence])
h_tuples_of_primitives = sampled_from(primitives).flatmap(
    lambda e: tuples(lists(e[0]),
                     one_of(sampled_from([Tuple[e[1], ...], Sequence[e[1]]]),
                            h_tuple_types))).map(
    lambda e: (tuple(e[0]), e[1]))

seqs_of_primitives = one_of(lists_of_primitives, h_tuples_of_primitives)


@composite
def enums_of_primitives(draw):
    names = draw(sets(text(alphabet=ascii_letters, min_size=1), min_size=1))
    n = len(names)
    vals = draw(
        one_of(
            sets(
                one_of(
                    integers(), floats(allow_nan=False), text(min_size=1)),
                min_size=n,
                max_size=n)))
    return Enum('HypEnum', list(zip(names, vals)))


def create_generic_type(generic_types, param_type):
    """Create a strategy for generating parameterized generic types."""
    return one_of(generic_types,
                  generic_types.map(lambda t: t[Any]),
                  generic_types.map(lambda t: t[param_type]))


# N.B.  cannot use floats in these sets, floats that equalish can break
# conversion tests
mut_sets_of_ints_or_text = sampled_from(ints_and_text).flatmap(
    lambda e: tuples(sets(e[0]), create_generic_type(mut_set_types, e[1])))

frozen_sets_of_ints_or_text = sampled_from(ints_and_text).flatmap(
    lambda e: tuples(frozensets(e[0]), create_generic_type(just(FrozenSet),
                                                           e[1]))
)
sets_of_ints_or_text = one_of(mut_sets_of_ints_or_text,
                              frozen_sets_of_ints_or_text)


def create_generic_dict_type(type1, type2):
    """Create a strategy for generating parameterized dict types."""
    return one_of(dict_types,
                  dict_types.map(lambda t: t[type1, type2]),
                  dict_types.map(lambda t: t[Any, type2]),
                  dict_types.map(lambda t: t[type1, Any]))


def create_dict_and_type(tuple_of_strats):
    """Map two primitive strategies into a strategy for dict and type."""
    (prim_strat_1, type_1), (prim_strat_2, type_2) = tuple_of_strats

    return tuples(
        dictionaries(prim_strat_1, prim_strat_2),
        create_generic_dict_type(type_1, type_2))


dicts_of_primitives = (
    tuples(sampled_from(ints_and_text), sampled_from(ints_and_text))
    .flatmap(create_dict_and_type))


@given(primitives_and_type)
def test_loading_primitives(converter, primitive_and_type):
    """Test just loading a primitive value."""
    val, t = primitive_and_type
    assert converter.loads(val, t) == val
    assert converter.loads(val, Any) == val


@given(seqs_of_primitives)
def test_loading_seqs(converter, seq_and_type):
    """Test loading sequence generic types."""
    iterable, t = seq_and_type
    converted = converter.loads(iterable, t)
    for x, y in zip(iterable, converted):
        assert x == y


@given(sets_of_ints_or_text, set_types)
def test_loading_sets(converter, set_and_type, set_type):
    """Test loading generic sets."""
    set_, input_set_type = set_and_type

    if input_set_type.__args__:
        set_type = set_type[input_set_type.__args__[0]]

    wanted_type = getattr(set_type, '__origin__', None) or set_type
    converted = converter.loads(set_, set_type)
    assert converted == set_
    assert isinstance(converted, wanted_type)

    converted = converter.loads(set_, Any)
    assert converted == set_
    assert isinstance(converted, type(set_))


@given(sets_of_ints_or_text)
def test_stringifying_sets(converter, set_and_type):
    """Test loading generic sets and converting the contents to str."""
    set_, input_set_type = set_and_type

    input_set_type.__args__ = (unicode, )
    converted = converter.loads(set_, input_set_type)
    assert len(converted) == len(set_)
    for e in set_:
        assert unicode(e) in converted


@given(lists(primitives_and_type))
def test_loading_hetero_tuples(converter, list_of_vals_and_types):
    """Test loading heterogenous tuples."""
    types = tuple(e[1] for e in list_of_vals_and_types)
    vals = [e[0] for e in list_of_vals_and_types]
    t = Tuple[types]

    converted = converter.loads(vals, t)

    assert isinstance(converted, tuple)

    for x, y in zip(vals, converted):
        assert x == y

    for x, y in zip(types, converted):
        assert isinstance(y, x)


# the given excludes bytes (cf primitives_and_type), as some byte values
# cannot be converted to unicode using the utf-8 encoding
@given(lists(one_of(ints_and_type, floats_and_type, strs_and_type)))
def test_stringifying_tuples(converter, list_of_vals_and_types):
    """Stringify all elements of a heterogeneous tuple."""
    vals = [e[0] for e in list_of_vals_and_types]
    t = Tuple[(unicode, ) * len(list_of_vals_and_types)]

    converted = converter.loads(vals, t)

    assert isinstance(converted, tuple)

    for x, y in zip(vals, converted):
        assert unicode(x) == y

    for x in converted:
        assert isinstance(x, unicode)


@given(dicts_of_primitives)
def test_loading_dicts(converter, dict_and_type):
    d, t = dict_and_type

    converted = converter.loads(d, t)

    assert converted == d
    assert converted is not d


@given(dicts_of_primitives)
def test_stringifying_dicts(converter, dict_and_type):
    d, t = dict_and_type

    converted = converter.loads(d, Dict[unicode, unicode])

    for k, v in d.items():
        assert converted[unicode(k)] == unicode(v)


@given(primitives_and_type)
def test_loading_optional_primitives(converter, primitive_and_type):
    """Test loading Optional primitive types."""
    val, type = primitive_and_type

    assert converter.loads(val, Optional[type]) == val
    assert converter.loads(None, Optional[type]) is None


@given(lists_of_primitives)
def test_loading_lists_of_opt(converter, list_and_type):
    """Test loading Optional primitive types."""
    l, t = list_and_type

    l.append(None)
    args = t.__args__

    if args and args[0] not in (Any, str, unicode, Optional):
        with raises(TypeError):
            converter.loads(l, t)
    assume(args)
    optional_t = Optional[args[0]]
    t.__args__ = (optional_t, )

    converted = converter.loads(l, t)

    for x, y in zip(l, converted):
        assert x == y

    t.__args__ = args


@given(lists_of_primitives)
def test_stringifying_lists_of_opt(converter, list_and_type):
    """Test loading Optional primitive types into strings."""
    l, t = list_and_type

    l.append(None)

    converted = converter.loads(l, List[Optional[unicode]])

    for x, y in zip(l, converted):
        if x is None:
            assert x is y
        else:
            assert unicode(x) == y


@given(lists(integers()))
def test_loading_primitive_union_hook(converter, ints):
    """Test registering a union loading hook."""

    def load_hook(cl, val):
        """Even ints are passed through, odd are stringified."""
        return val if val % 2 == 0 else str(val)

    converter.register_loads_hook(Union[str, int], load_hook)

    converted = converter.loads(ints, List[Union[str, int]])

    for x, y in zip(ints, converted):
        if x % 2 == 0:
            assert x == y
        else:
            assert str(x) == y


@given(choices(), enums_of_primitives())
def test_loading_enums(converter, choice, enum):
    """Test loading enums by their values."""
    val = choice(list(enum))

    assert converter.loads(val.value, enum) == val
