"""Leitura e escrita mínima de NIfTI-1 (.nii / .nii.gz) usando só a biblioteca padrão e NumPy.

Os volumes ficam em memória como arrays (z, y, x) em float32, já com scl_slope/scl_inter aplicados.
"""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HEADER_SIZE = 348
VOX_OFFSET = 352  # cabeçalho + 4 bytes de extensão vazia

DT_UINT8 = 2
DT_INT16 = 4
DT_FLOAT32 = 16

_DTYPES: dict[int, type] = {
    2: np.uint8,
    4: np.int16,
    8: np.int32,
    16: np.float32,
    64: np.float64,
    256: np.int8,
    512: np.uint16,
    768: np.uint32,
}


@dataclass
class NiftiImage:
    data: np.ndarray  # (z, y, x) float32
    spacing: tuple[float, float, float]  # (x, y, z) em mm
    header: bytes  # cabeçalho original de 348 bytes
    endian: str  # "<" ou ">"


def _open(path: Path, mode: str):
    return gzip.open(path, mode) if path.name.endswith(".gz") else open(path, mode)


def _detect_endian(header: bytes) -> str:
    for endian in ("<", ">"):
        if struct.unpack(endian + "i", header[:4])[0] == HEADER_SIZE:
            return endian
    raise ValueError("arquivo não é NIfTI-1 (sizeof_hdr != 348)")


def read_nifti(path: str | Path) -> NiftiImage:
    path = Path(path)
    with _open(path, "rb") as f:
        raw = f.read()
    if len(raw) < HEADER_SIZE:
        raise ValueError(f"{path}: arquivo truncado")
    header = raw[:HEADER_SIZE]
    e = _detect_endian(header)
    if header[344:347] not in (b"n+1", b"ni1"):
        raise ValueError(f"{path}: magic NIfTI inválido")
    if header[344:347] == b"ni1":
        raise ValueError(f"{path}: pares .hdr/.img não são suportados; use .nii")

    dim = struct.unpack(e + "8h", header[40:56])
    datatype = struct.unpack(e + "h", header[70:72])[0]
    pixdim = struct.unpack(e + "8f", header[76:108])
    vox_offset = int(struct.unpack(e + "f", header[108:112])[0])
    slope, inter = struct.unpack(e + "2f", header[112:120])

    if datatype not in _DTYPES:
        raise ValueError(f"{path}: datatype {datatype} não suportado")
    nx = dim[1]
    ny = dim[2] if dim[0] >= 2 else 1
    nz = dim[3] if dim[0] >= 3 else 1  # volumes 4D: usa só o primeiro
    count = nx * ny * nz

    dtype = np.dtype(_DTYPES[datatype]).newbyteorder(e)
    data = np.frombuffer(raw, dtype=dtype, count=count, offset=vox_offset)
    data = data.reshape(nz, ny, nx).astype(np.float32)
    if slope != 0.0 and (slope != 1.0 or inter != 0.0):
        data = data * np.float32(slope) + np.float32(inter)

    spacing = (abs(pixdim[1]) or 1.0, abs(pixdim[2]) or 1.0, abs(pixdim[3]) or 1.0)
    return NiftiImage(data=data, spacing=spacing, header=header, endian=e)


def _build_header(ref: bytes, endian: str, shape_zyx: tuple[int, int, int], datatype: int) -> bytes:
    nz, ny, nx = shape_zyx
    h = bytearray(ref)
    e = endian
    struct.pack_into(e + "8h", h, 40, 3, nx, ny, nz, 1, 1, 1, 1)
    bitpix = np.dtype(_DTYPES[datatype]).itemsize * 8
    struct.pack_into(e + "hh", h, 70, datatype, bitpix)
    struct.pack_into(e + "f", h, 108, float(VOX_OFFSET))
    struct.pack_into(e + "2f", h, 112, 1.0, 0.0)  # scl_slope, scl_inter
    struct.pack_into(e + "2f", h, 124, 0.0, 0.0)  # cal_max, cal_min
    h[344:348] = b"n+1\x00"
    return bytes(h)


def write_nifti(path: str | Path, data: np.ndarray, ref: NiftiImage, datatype: int = DT_UINT8) -> None:
    """Grava `data` (z, y, x) com a geometria do cabeçalho de `ref`."""
    path = Path(path)
    header = _build_header(ref.header, ref.endian, data.shape, datatype)
    dtype = np.dtype(_DTYPES[datatype]).newbyteorder(ref.endian)
    payload = np.ascontiguousarray(data).astype(dtype).tobytes()
    with _open(path, "wb") as f:
        f.write(header)
        f.write(b"\x00" * (VOX_OFFSET - HEADER_SIZE))
        f.write(payload)


def new_image(data: np.ndarray, spacing: tuple[float, float, float]) -> NiftiImage:
    """Cria um cabeçalho NIfTI-1 do zero (usado pelo gerador de fantoma)."""
    e = "<"
    h = bytearray(HEADER_SIZE)
    struct.pack_into(e + "i", h, 0, HEADER_SIZE)
    sx, sy, sz = spacing
    struct.pack_into(e + "8f", h, 76, 1.0, sx, sy, sz, 1.0, 0.0, 0.0, 0.0)
    h[123] = 2  # xyzt_units = mm
    struct.pack_into(e + "hh", h, 252, 0, 1)  # qform_code, sform_code
    struct.pack_into(e + "4f", h, 280, sx, 0.0, 0.0, 0.0)
    struct.pack_into(e + "4f", h, 296, 0.0, sy, 0.0, 0.0)
    struct.pack_into(e + "4f", h, 312, 0.0, 0.0, sz, 0.0)
    h[344:348] = b"n+1\x00"
    return NiftiImage(data=data.astype(np.float32), spacing=spacing, header=bytes(h), endian=e)
