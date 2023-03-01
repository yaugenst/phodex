import re
import sys
from io import TextIOWrapper
from typing import Any, Iterable


class FilteredStream:
    def __init__(self, stream: TextIOWrapper, patterns: Iterable | str) -> None:
        self.stream = stream
        if isinstance(patterns, str):
            patterns = [patterns]
        self.patterns = [re.compile(p) for p in patterns]

    def __getattr__(self, attr_name: str) -> Any:
        return getattr(self.stream, attr_name)

    def write(self, data: str) -> None:
        if any([p.search(data) for p in self.patterns]):
            self.stream.write(data + "\n")

    def flush(self) -> None:
        self.stream.flush()


class filter_stdout:
    """This context manager temporarily hijacks sys.stdout to only allow
    printing of lines that correspond to certain patterns.
    """

    def __init__(self, patterns: Iterable | str) -> None:
        self.stream = FilteredStream(sys.stdout, patterns)

    def __enter__(self) -> None:
        sys.stdout.flush()
        self._old_stdout = sys.stdout
        sys.stdout = self.stream

    def __exit__(self, *_) -> None:
        self.stream.flush()
        sys.stdout = self._old_stdout
