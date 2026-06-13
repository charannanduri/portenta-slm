"""
Convert c_inference/model.bin into firmware/src/model_data.h --- a C array that
gets compiled into the Portenta's flash. We point our weight pointers directly
into this array (no copy), so it must be 4-byte aligned (alignas(4)) for the
float loads, matching the padding we added in export.py.

Run:  .venv/bin/python train/embed_model.py
"""

SRC = "c_inference/model.bin"
DST = "firmware/src/model_data.h"

data = open(SRC, "rb").read()
with open(DST, "w") as f:
    f.write("#pragma once\n")
    f.write(f"// Auto-generated from {SRC} -- {len(data)} bytes\n")
    f.write(f"const unsigned int model_bin_len = {len(data)};\n")
    f.write("alignas(4) const unsigned char model_bin[] = {\n")
    for i in range(0, len(data), 20):
        f.write("  " + ",".join(str(b) for b in data[i:i + 20]) + ",\n")
    f.write("};\n")

print(f"wrote {DST}  ({len(data)} bytes, {len(data)/1024:.0f} KB in flash)")
