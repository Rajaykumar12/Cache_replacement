"""Real production traces from the libCacheSim public bucket (oracleGeneral format).

  python3 realtraces.py            # download + convert all REAL traces (first N requests each)

Each record is 24 bytes: uint32 timestamp, uint64 obj_id, uint32 size, int64 next_access_vtime. Only obj_id
is kept, remapped to dense ints; next_access_vtime would leak the future and is dropped here, and sizes are
ignored (unit-size, object hit rate). Converted traces are cached as data/real/<name>.npy.
"""
import os
import sys
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(ROOT, 'data', 'real')
BUCKET = 'https://s3.amazonaws.com/cache-datasets/cache_dataset_oracleGeneral/'
N = 10_000_000
REC = np.dtype([('ts', '<u4'), ('id', '<u8'), ('size', '<u4'), ('next', '<i8')])
REAL = {  # name -> (bucket path, domain)
    'msr_hm_0': ('2007_msr/msr_hm_0.oracleGeneral.zst', 'block storage'),
    'twitter_c50': ('2020_twitter/cluster50.oracleGeneral.sample10.zst', 'in-memory key-value'),
    'meta_reag': ('2022_metaCDN/meta_reag.oracleGeneral.zst', 'CDN'),
}


def read_oracle(fh, n=N):
    """First n records of a zstd-compressed oracleGeneral stream -> dense int32 key array."""
    import zstandard
    want, parts, got = n * REC.itemsize, [], 0
    with zstandard.ZstdDecompressor().stream_reader(fh, read_across_frames=True) as r:
        while got < want:  # read() may return short chunks
            b = r.read(min(1 << 24, want - got))
            if not b: break
            parts.append(b); got += len(b)
    buf = b''.join(parts)
    rec = np.frombuffer(buf[:len(buf) // REC.itemsize * REC.itemsize], REC)
    return np.unique(rec['id'], return_inverse=True)[1].astype(np.int32)


def path(name):
    return os.path.join(DIR, name + '.npy')


def fetch(name, n=N):
    os.makedirs(DIR, exist_ok=True)
    zst = os.path.join(DIR, os.path.basename(REAL[name][0]))
    if not os.path.exists(zst):
        urllib.request.urlretrieve(BUCKET + REAL[name][0], zst + '.part')
        os.replace(zst + '.part', zst)
    with open(zst, 'rb') as fh:
        np.save(path(name), read_oracle(fh, n))


def load(name, n=N):
    """First n requests (the cached .npy holds up to N)."""
    if not os.path.exists(path(name)): fetch(name)
    return np.load(path(name), mmap_mode='r')[:n].tolist()


if __name__ == '__main__':
    for name in sys.argv[1:] or REAL:
        fetch(name)
        a = np.load(path(name))
        print(f'{name:14s} {len(a):>10,} requests  {a.max() + 1:>10,} distinct keys')
