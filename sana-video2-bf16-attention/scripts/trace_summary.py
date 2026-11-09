"""Aggregate GPU kernel time by name from a chrome trace (.json or .json.gz)."""

import collections
import gzip
import json
import sys

path = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 15
opener = gzip.open if path.endswith(".gz") else open
with opener(path, "rt") as f:
    events = json.load(f)["traceEvents"]
by_name = collections.Counter()
calls = collections.Counter()
for e in events:
    if e.get("cat") == "kernel" and "dur" in e:
        by_name[e["name"]] += e["dur"]
        calls[e["name"]] += 1
total = sum(by_name.values())
print(json.dumps({"total_kernel_ms": total / 1000, "kernels": sum(calls.values())}))
for name, dur in by_name.most_common(top):
    print(f"{dur / 1000:10.1f} ms {100 * dur / total:5.1f}% {calls[name]:6d}x  {name[:110]}")
