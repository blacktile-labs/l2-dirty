"""P-1 (B10): compare SASS of rw_kernel, readback_kernel, flush_l2_read_kernel between two
`cuobjdump -sass` dumps, ignoring instruction addresses. Usage: sass_diff.py <vm_sass> <laptop_sass>"""
import re
import sys

KERNELS = {"rw_kernel": r"_Z\d+rw_kernel", "readback_kernel": r"_Z\d+readback_kernel",
           "flush_l2_read_kernel": r"_ZN3dmk\d+flush_l2_read_kernel"}
INS = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;\s*/\*\s*(0x[0-9a-f]+)\s*\*/")
ENC2 = re.compile(r"^\s*/\*\s*(0x[0-9a-f]+)\s*\*/\s*$")


def parse(path):
    funcs, cur = {}, None
    for line in open(path, errors="replace"):
        m = re.search(r"Function : (\S+)", line)
        if m:
            cur = m.group(1)
            funcs[cur] = {"ins": [], "enc": []}
            continue
        if cur is None:
            continue
        m = INS.search(line)
        if m:
            funcs[cur]["ins"].append(re.sub(r"\s+", " ", m.group(1)))
            funcs[cur]["enc"].append(m.group(2))
            continue
        m = ENC2.match(line)
        if m and funcs[cur]["enc"]:
            funcs[cur]["enc"][-1] += m.group(1)
    return funcs


def pick(funcs, pat):
    names = [n for n in funcs if re.match(pat, n)]
    return (names[0], funcs[names[0]]) if len(names) == 1 else (names, None)


vm, lap = parse(sys.argv[1]), parse(sys.argv[2])
all_ok = True
for k, pat in KERNELS.items():
    nv, fv = pick(vm, pat)
    nl, fl = pick(lap, pat)
    if fv is None or fl is None:
        print(f"{k}: not found uniquely (vm {nv}, laptop {nl})")
        all_ok = False
        continue
    same_ins = fv["ins"] == fl["ins"]
    same_enc = fv["enc"] == fl["enc"]
    all_ok &= same_ins
    print(f"{k}: {len(fv['ins'])} vs {len(fl['ins'])} instructions; text identical={same_ins}; encoding identical={same_enc}")
    if not same_ins:
        for i, (a, b) in enumerate(zip(fv["ins"], fl["ins"])):
            if a != b:
                print(f"  first difference at #{i}: vm '{a}' | laptop '{b}'")
                break
print("P-1:", "PASS (identical SASS)" if all_ok else "FAIL (SASS differs)")
