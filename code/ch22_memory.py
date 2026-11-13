"""
ch22_memory.py — where the weights live, and what each place costs.   (Chapter 22)

    python code/ch22_memory.py            # measure this machine, then do the sums of the chapter
    python code/ch22_memory.py --save     # also write runs/memory.json for the website

Chapter 18 found that a large model writes as fast as memory can deliver its
weights. This file looks at the memory itself:

    1. the ladder     the same read, on blocks from 256 KB to 1 GB: cache, then RAM
    2. two pools      a model split between a graphics card and ordinary RAM
    3. one pool       unified memory, on published figures
    4. the KV cache   what the context costs on top of the weights
    5. loading        reading a file against mapping it
"""
import argparse, json, mmap, os, tempfile, time
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')            # measure one core, whatever the machine has
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KB, MB, GB = 1024, 1024 ** 2, 1024 ** 3


# ---- 1. the ladder: how fast can this machine read a block of numbers? ---------------
# [snippet:ladder]
def read_speed(nbytes, seconds=0.25):
    """Read a block of `nbytes` again and again. Returns gigabytes read per second."""
    block = np.ones(nbytes // 4, dtype=np.float32)
    np.dot(block, block)                              # once to warm up
    reads, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        np.dot(block, block)                          # touches every number once
        reads += 1
    return reads * nbytes / (time.perf_counter() - t0) / 1e9

def ladder(sizes):
    return [(n, max(read_speed(n) for _ in range(3))) for n in sizes]
# [/snippet]

def caches():
    """What the operating system says about this processor's caches (Linux only)."""
    out = {}
    base = '/sys/devices/system/cpu/cpu0/cache'
    if os.path.isdir(base):
        for d in sorted(os.listdir(base)):
            try:
                kind = open(f'{base}/{d}/type').read().strip()
                if kind != 'Instruction':
                    out['L' + open(f'{base}/{d}/level').read().strip()] = open(f'{base}/{d}/size').read().strip()
            except OSError:
                pass
    return out


# ---- 2 and 3. several memories: times add up, speeds do not -------------------------
# [snippet:split]
def ceiling(places):
    """At most how many tokens per second, when the weights are spread over several memories?
    places: (gigabytes that sit there, gigabytes per second it delivers), one pair per memory."""
    seconds_per_token = sum(gb / speed for gb, speed in places)   # every weight is read once per token
    return 1 / seconds_per_token
# [/snippet]

# Published figures, not measurements: (name, GB a model can use, GB per second)
CARDS = [('GeForce RTX 5070', 12, 672), ('GeForce RTX 5080', 16, 960), ('GeForce RTX 4090', 24, 1008), ('GeForce RTX 5090', 32, 1792)]
UNIFIED = [('Apple M5 Pro', 64, 307), ('Apple M5 Max', 128, 614), ('Apple M5 Ultra', 512, 1200),
           ('AMD Ryzen AI Max+ 395', 96, 256), ('NVIDIA DGX Spark', 128, 273)]
RAM = ('Desktop RAM, DDR5-5600, two channels', 5600e6 * 8 * 2 / 1e9)     # 89.6 GB/s by arithmetic


# ---- 4. the KV cache: what the context costs ------------------------------------------
# [snippet:kv]
def kv_cache_bytes(layers, width, context, bytes_per_number=2):
    """Each layer keeps one key and one value (2 vectors of `width` numbers) for every token in the window."""
    return 2 * layers * width * context * bytes_per_number
# [/snippet]

# layers, width of one key (key/value heads x numbers per head), from the publisher's configuration files
SHAPES = {'Llama 3.1 8B': (32, 8 * 128), 'Llama 3.1 70B': (80, 8 * 128)}


# ---- 5. loading: read the file, or map it ---------------------------------------------
# [snippet:map]
def load_two_ways(nbytes):
    """Write a file of `nbytes`, then open it twice: by reading it all, and by mapping it."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(np.ones(nbytes // 4, dtype=np.float32).tobytes())
    t0 = time.perf_counter()
    with open(f.name, 'rb') as g:
        copy = g.read()                                   # every byte is copied into our memory now
    t_read = time.perf_counter() - t0
    t0 = time.perf_counter()
    with open(f.name, 'rb') as g:
        view = mmap.mmap(g.fileno(), 0, access=mmap.ACCESS_READ)   # nothing is copied: pages arrive when touched
    t_map = time.perf_counter() - t0
    t0 = time.perf_counter()
    first = np.frombuffer(view, dtype=np.float32, count=1024).sum()   # use the first 4 KB
    t_first = time.perf_counter() - t0
    del first, copy
    os.remove(f.name)
    return t_read, t_map, t_first
# [/snippet]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', action='store_true')
    args = ap.parse_args()
    out = {}

    print('1. reading a block of numbers again and again, one core')
    sizes = [256 * KB, MB, 4 * MB, 16 * MB, 64 * MB, 256 * MB, GB]
    rungs = ladder(sizes)
    for n, speed in rungs:
        label = f'{n // GB} GB' if n >= GB else f'{n // MB} MB' if n >= MB else f'{n // KB} KB'
        print(f'   {label:>7}   {speed:6.1f} GB per second   {"#" * int(speed / 2)}')
    print(f'   caches of this processor: {caches() or "not reported"}')
    out['ladder'] = [[n, round(s, 1)] for n, s in rungs]
    out['caches'] = caches()

    print('\n2. a 40 GB model, a 24 GB graphics card (1,008 GB/s) and desktop RAM (89.6 GB/s)')
    card, ram = 1008, RAM[1]
    rows = [('a 24 GB model, all of it on the card', [(24, card)]),
            ('the 40 GB model, 24 GB on the card and 16 GB in RAM', [(24, card), (16, ram)]),
            ('the 40 GB model, all of it in RAM', [(40, ram)])]
    for name, places in rows:
        print(f'   {name:55} at most {ceiling(places):5.1f} tokens per second')
    out['split'] = [[name, round(ceiling(p), 1)] for name, p in rows]

    print('\n3. the same 40 GB model in one pool of unified memory')
    for name, room, speed in UNIFIED:
        print(f'   {name:24} {room:4} GB at {speed:5} GB/s   at most {ceiling([(40, speed)]):5.1f} tokens per second')

    print('\n4. the KV cache')
    ours = kv_cache_bytes(3, 96, 64, 4)
    print(f'   our model: 2 x 3 layers x 96 numbers x 64 tokens x 4 bytes = {ours:,} bytes ({ours / KB:.0f} KB); its weights take {345504 * 4 / KB:,.0f} KB')
    out['kv'] = {'ours': ours}
    for name, (layers, width) in SHAPES.items():
        for context in (8192, 131072):
            b = kv_cache_bytes(layers, width, context)
            print(f'   {name:14} {context:>7,} tokens of context: {b / 1e9:5.1f} GB')
            out['kv'][f'{name} {context}'] = round(b / 1e9, 2)

    print('\n5. opening a 256 MB file')
    t_read, t_map, t_first = min(load_two_ways(256 * MB) for _ in range(3))
    print(f'   read it all       {1000 * t_read:8.2f} ms')
    print(f'   map it            {1000 * t_map:8.3f} ms, then {1000 * t_first:.3f} ms to use its first 4 KB')
    print('   (the file was still in the operating system\'s memory, so this times the copy, not the disk)')
    out['load'] = {'read_ms': round(1000 * t_read, 2), 'map_ms': round(1000 * t_map, 3), 'first_ms': round(1000 * t_first, 3)}

    if args.save:
        path = os.path.join(HERE, '..', 'runs', 'memory.json')
        with open(path, 'w') as f:
            json.dump(out, f)
        print('\nwrote', os.path.normpath(path))
