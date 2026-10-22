#!/usr/bin/env python3
"""SF-40 helper (run_sf40.sh part controls): write a valid .npy (v1.0, '<f8', C order) of nx*ny*nz zeros.

  make_zeros_npy.py NX NY NZ OUT.npy      shape (nz, ny, nx): C order, so x1 (index i) is the fastest axis, i.e. the
                                          flat layout i + nx (j + ny k) of the driver's --cells reader.
  make_zeros_npy.py --selftest
Stdlib only (no numpy).
"""
import ast
import struct
import sys


def npy_header(shape):
    d = "{'descr': '<f8', 'fortran_order': False, 'shape': (%s), }" % ", ".join(str(s) for s in shape)
    if len(shape) == 1:
        d = d.replace("(%d)" % shape[0], "(%d,)" % shape[0])
    total = 10 + len(d) + 1
    d += " " * ((64 - total % 64) % 64) + "\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(d)) + d.encode("latin1")


def write_zeros(path, nx, ny, nz):
    with open(path, "wb") as f:
        f.write(npy_header((nz, ny, nx)))
        chunk = bytes(8 * 1024 * 1024)
        left = 8 * nx * ny * nz
        while left > 0:
            f.write(chunk[:left])
            left -= min(left, len(chunk))


def read_header(path):
    with open(path, "rb") as f:
        assert f.read(6) == b"\x93NUMPY"
        major, minor = f.read(2)
        (hl,) = struct.unpack("<H", f.read(2))
        hdr = ast.literal_eval(f.read(hl).decode("latin1"))
        data_off = 10 + hl
    return hdr, data_off


def selftest():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "z.npy")
        write_zeros(p, 8, 4, 2)
        hdr, off = read_header(p)
        size = os.path.getsize(p)
        ok = (hdr["descr"] == "<f8" and hdr["fortran_order"] is False and hdr["shape"] == (2, 4, 8)
              and off % 64 == 0 and size - off == 8 * 64)
        with open(p, "rb") as f:
            f.seek(off)
            ok = ok and f.read() == bytes(8 * 64)
    print("SELFTEST OK make_zeros_npy" if ok else "SELFTEST FAILED make_zeros_npy")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        sys.exit(selftest())
    if len(sys.argv) != 5:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    write_zeros(sys.argv[4], int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
