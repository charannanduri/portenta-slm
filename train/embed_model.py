"""
Turn model.bin into a C file (model_data.h) so it gets baked right into the
Portenta's memory along with the program. The C code reads the weights straight
out of this array. We mark it aligned(4) so the decimal numbers sit on tidy
spots (same reason export.py adds those padding bytes).

Run:  .venv/bin/python train/embed_model.py
"""

SRC = "c_inference/model.bin"
DST = "firmware/src/model_data.h"

data = open(SRC, "rb").read()
with open(DST, "w") as f:
    f.write("#pragma once\n")
    f.write(f"// made automatically from {SRC} -- {len(data)} bytes\n")
    f.write(f"const unsigned int model_bin_len = {len(data)};\n")
    f.write("alignas(4) const unsigned char model_bin[] = {\n")
    for i in range(0, len(data), 20):           # 20 numbers per line, just for tidiness
        f.write("  " + ",".join(str(b) for b in data[i:i + 20]) + ",\n")
    f.write("};\n")

print(f"wrote {DST}  ({len(data)} bytes, {len(data)/1024:.0f} KB in flash)")
