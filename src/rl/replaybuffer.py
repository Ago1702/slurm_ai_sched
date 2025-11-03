from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterable, Union

import numpy as np

import torch


@dataclass
class ReplayBuffer:
    max_size: int = 100

    def __post_init__(self) -> None:
        self.buffers = set()
        self.ptr = 0
        self.size = 0

    def reset(self) -> None:
        """
        Clears all buffer values.
        """
        for buffer in self.buffers:
            self.initialize_buffer(buffer)
    
    def initialize_buffer(self, buffer_name: str) -> None:
        """
        Initialize a new buffer in memory.

        Args:
            buffer_name (str): The name of the buffer.
        """
        setattr(self, buffer_name, np.empty(self.max_size, dtype=object))
        self.buffers.add(buffer_name)

    def store(self, **kwargs: object) -> None:
        """
        Store items in each buffer.

        Example:
            .. code-block:: python

                buffer = ReplayBuffer()
                buffer.initialize_buffer("chain1")
                buffer.initialize_buffer("chain2")

                for i in range(10):
                    buffer.store(
                        chain1 = i,
                        chain2 = 2*i
                    )
        """
        for k, v in kwargs.items():
            getattr(self, k)[self.ptr] = v

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, n: int) -> dict:
        """
        Random uniform sample over all buffers.

        Args:
            n (int): The size of the sample.

        Returns:
            dict: Buffer name to iterable of sample values key, value pairs
                for each buffer.
        """
        idx = np.random.randint(low=0, high=self.size, size=n)
        return self[idx]
    
    def previous(self, n: int) -> dict:
        """
        Get the most recent entries to the buffer.

        Args:
            n (int): Number of previous entries to look back.

        Returns:
            dict: Buffer name to iterable of previous values key, value pairs
                for each buffer.
        """
        assert n <= self.max_size
        return self.take(range(self.ptr - n, self.ptr), mode="wrap")
    
    def take(self, *args, **kwargs) -> dict:
        """
        Wrapper around Numpy's take function. Used to get a range of values, and only
                necessary when wrapping splicing from the end of the buffer
                back to the beginning, or vice versa. Otherwise, __getitem__ is
                preferable.

        Returns:
            dict: Buffer name to iterable of taken values key, value pairs
                for each buffer.

        Example:
            .. code-block:: python

                buffer = ReplayBuffer()
                buffer.initialize_buffer("chain1")
                buffer.initialize_buffer("chain2")

                for i in range(10):
                    buffer.store(
                        chain1 = i,
                        chain2 = 2*i
                    )

                buffer.take(range(0, 5))
        """
        return {k: getattr(self, k).take(*args, **kwargs) for k in self.buffers}

    def set_values(
        self,
        buffer: str,
        idx: Union[int, Iterable[int], Generator[int, Any, Any]],
        val: Union[object, Iterable[object], Generator[object, Any, Any]],
    ) -> None:
        """
        Set explicit buffer value(s) at some index or indices.

        Args:
            buffer (str): The name of the buffer to modify.
            idx (Union[int, Iterable[int], Generator[int, Any, Any]]): The index or indices to modify.
            val (Union[object, Iterable[object], Generator[object]]): The value(s) to set.
        """
        self.__dict__[buffer][idx] = val

    def get_values(
        self, buffer: str, idx: Union[int, Iterable[int], Generator[int, Any, Any]]
    ) -> Union[object, Iterable[object]]:
        """
        Get buffer value(s) at some index or indices.

        Args:
            buffer (str): The name of the buffer to get value(s) from.
            idx (Union[int, Iterable[int], Generator[int, Any, Any]]): The index or indices to get values from.

        Returns:
            Union[object, Iterable[object]]: The value(s) at the given index or indices.
        """
        return self.__dict__[buffer][idx]


    def __getitem__(self, idx: int) -> Dict[str, object]:
        """
        Get items at an index.

        Args:
            idx (int): Index to get data at.

        Returns:
            dict[str, object]: A dictionary containing buffer names as keys
                and their objects at index idx as values.
        """
        return {k: getattr(self, k)[idx] for k in self.buffers}

    def __len__(self) -> int:
        """
        The size of the buffer.

        Returns:
            int: Maximum buffer size.
        """
        return self.size