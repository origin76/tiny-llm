from abc import ABC, abstractmethod
from typing import Optional

import mlx.core as mx

from abc import ABC
from typing import Optional
import mlx.core as mx
    
class TinyKvCache(ABC):

    def __init__(self):
        self.key: Optional[mx.array] = None
        self.value: Optional[mx.array] = None

    @abstractmethod
    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        
        """
        Update the key-value cache and fetch the updated key-value cache.

        Args:
            key: The key to update the cache with.
            value: The value to update the cache with.
            mask_length: The length of the mask (only used in batching mode)
            mask: The mask to use (only used in batching mode)

        Returns:
            A tuple of the updated key-value cache, the updated value, the sequence length, and the mask.
            In week 2 day 1, we only need to return the updated key-value cache, the updated value.
            In week 2 day 6/7, we need to return the updated key-value cache, the updated value, the sequence length, and the mask.
            so that the batching kv cache can use this information to generate the mask.
        """

class BatchingKvCache(TinyKvCache):
    def __init__(self, max_active_requests: int, max_seq_len: int):
        self.max_active_requests = max_active_requests
        self.max_seq_len = max_seq_len

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        pass

    def add_request(self, prefilled: TinyKvCache, id: int):
        pass

    def remove_request(self, id: int):
        pass


class TinyKvFullCache(TinyKvCache):
    def __init__(self):
        self.key: Optional[mx.array] = None
        self.value: Optional[mx.array] = None
        self.offset = 0

    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        # Same implementation as base class for now
        if self.key is None:
            self.key = key
            self.value = value
        else:
            # Concatenate on sequence dimension (axis=2 for shape B x H x L x D)
            self.key = mx.concatenate([self.key, key], axis=2)
            self.value = mx.concatenate([self.value, value], axis=2)

        L = self.key.shape[2]
        self.offset = L

        return self.key, self.value, L, mask
