from os import PathLike
from typing import Literal, Tuple, TypeAlias, Union, get_args

PathOrStr: TypeAlias = Union[str, PathLike]
CacheEntryNames: TypeAlias = Literal["k", "v", "ff_output", "update_indicator"]
CACHE_ENTRY_NAMES: Tuple[str, str, str, str] = get_args(CacheEntryNames)
SequenceEntryNames: TypeAlias = Literal["prompt", "gen"]
SEQUENCE_ENTRY_NAMES: Tuple[str, str] = get_args(SequenceEntryNames)