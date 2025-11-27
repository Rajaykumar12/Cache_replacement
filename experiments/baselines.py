from collections import OrderedDict, defaultdict
import heapq

class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> value (or just key existence)
        self.freq = defaultdict(int)
        self.time = 0
        self.heap = [] # (freq, time, key)
        self.hits = 0
        self.misses = 0

    def process_request(self, key):
        self.time += 1
        if key in self.cache:
            self.hits += 1
            self.freq[key] += 1
            # We don't update heap immediately to avoid O(N) remove. 
            # We'll handle stale entries during eviction or use a better structure if needed.
            # For strict correctness with O(log N), we push new (freq, time, key) 
            # and pop stale ones. This is "Lazy Removal".
            heapq.heappush(self.heap, (self.freq[key], self.time, key))
        else:
            self.misses += 1
            if len(self.cache) >= self.capacity:
                self._evict()
            self.cache[key] = True
            self.freq[key] = 1
            heapq.heappush(self.heap, (self.freq[key], self.time, key))

    def _evict(self):
        while self.heap:
            f, t, k = heapq.heappop(self.heap)
            # Check if this is the latest info for k
            if k in self.cache and self.freq[k] == f:
                # Found the victim
                del self.cache[k]
                del self.freq[k]
                return
            # Else: it was a stale entry, ignore

    def get_hit_rate(self):
        total = self.hits + self.misses
        return (self.hits / total) * 100 if total > 0 else 0


class ARCCache:
    def __init__(self, capacity: int):
        self.c = capacity
        self.p = 0  # Target size for T1
        
        self.t1 = OrderedDict() # Recent: bottom (LRU) -> top (MRU)
        self.b1 = OrderedDict() # Ghost Recent
        self.t2 = OrderedDict() # Frequent
        self.b2 = OrderedDict() # Ghost Frequent
        
        self.hits = 0
        self.misses = 0

    def process_request(self, key):
        if key in self.t1:
            self.hits += 1
            del self.t1[key]
            self.t2[key] = True # Move to T2 (MRU)
        elif key in self.t2:
            self.hits += 1
            self.t2.move_to_end(key) # Renew in T2
        elif key in self.b1:
            self.misses += 1
            # Adaptation
            delta = 1 if len(self.b1) >= len(self.b2) else len(self.b2) / len(self.b1)
            self.p = min(self.c, self.p + delta)
            self._replace(key)
            del self.b1[key]
            self.t2[key] = True
        elif key in self.b2:
            self.misses += 1
            # Adaptation
            delta = 1 if len(self.b2) >= len(self.b1) else len(self.b1) / len(self.b2)
            self.p = max(0, self.p - delta)
            self._replace(key)
            del self.b2[key]
            self.t2[key] = True
        else:
            self.misses += 1
            # Case IV: New item
            if len(self.t1) + len(self.b1) == self.c:
                if len(self.t1) < self.c:
                    self.b1.popitem(last=False)
                    self._replace(key)
                else:
                    self.t1.popitem(last=False)
            elif len(self.t1) + len(self.b1) < self.c:
                if len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2) >= self.c:
                    if len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2) == 2 * self.c:
                        self.b2.popitem(last=False)
                    self._replace(key)
            
            self.t1[key] = True

    def _replace(self, key):
        if len(self.t1) > 0 and (len(self.t1) > self.p or (key in self.b2 and len(self.t1) == self.p)):
            k, _ = self.t1.popitem(last=False)
            self.b1[k] = True
        else:
            k, _ = self.t2.popitem(last=False)
            self.b2[k] = True

    def get_hit_rate(self):
        total = self.hits + self.misses
        return (self.hits / total) * 100 if total > 0 else 0
