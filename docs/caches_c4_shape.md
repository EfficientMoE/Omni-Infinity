# C4 VDN shape-cache limits

Flex BlockMask, gather-index, and delta-rule backend caches stay in third_party/vdn-minimax-h3.
The caption is not part of the BlockMask key.
A new caption length misses the inference FLASH compile.
One seq_len is compiled per process.
This issue does not add a shape cache.
This plan does not modify third_party.

Attention still runs on a shape hit; only the mask, gather index, and
delta-rule backend are reused. A second job whose packed text length differs
from the first job in the process misses the FLASH compile and the BlockMask
entry. This note does not propose a new key, a new cache, or a patch under
third_party/.
