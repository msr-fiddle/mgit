import os
from tqdm import tqdm
import numpy as np
import torch
import sys
import lzma
import multiprocess as mp

__all__ = [
    "compress",
    "decompress",
    "compress_rle",
    "decompress_rle",
    "compress_lzma",
    "decompress_lzma",
    "compress_sparse",
    "decompress_sparse",
    "compress_sparse_dict",
    "decompress_sparse_dict",
    "compress_dict",
    "decompress_dict",
    "delta_compress",
    "delta_decompress",
    "delta_compress_lzma",
    "delta_decompress_lzma",
    "delta_compress_rle",
    "delta_decompress_rle",
    "delta_compress_sparse",
    "delta_decompress_sparse",
    "delta_compress_sparse_dict",
    "delta_decompress_sparse_dict",
    "delta_compress_dict",
    "delta_decompress_dict",
    "quantize",
    "dequantize",
]


def encode(seq, progress_bar=False):
    """
    Encodes run-length encoding of given iterable.

    Parameters
    ----------
    seq: Any Python iterable, e.g.  lists, strings, tuples,
        pandas Series, to perform run-length encoding on.

    Returns
    -------
    values, counts: list of contiguous unique values, and list of
        counts
    """
    assert len(seq) > 0, "Sequence passed has zero length"

    values = []
    counts = []
    start_idxs = []

    # First element
    values.append(int(seq[0]))
    current_count = 1
    current_start_idx = 0

    if progress_bar == True:
        iterator = tqdm(range(1, len(seq)))
    else:
        iterator = range(1, len(seq))

    for idx in iterator:
        # If the current value is the same as the last
        # recorded unique value
        if seq[idx] == values[-1]:
            # Increment current count
            current_count += 1
        else:
            # Close previous count
            counts.append(int(current_count))
            start_idxs.append(current_start_idx)

            # Open new count
            values.append(int(seq[idx]))
            current_count = 1
            current_start_idx = idx

    # Close last count
    counts.append(int(current_count))
    start_idxs.append(current_start_idx)

    return values, counts


def decode(values, counts):
    """
    Decodes run-length encoding of given iterable.

    Parameters
    ----------
    values, counts: List of contiguous unique values, and list of counts

    Returns
    -------
    seq: Decoded sequence
    """
    assert len(values) == len(counts), "len(values) != len(counts)"

    try:
        counts = [int(i) for i in counts]
    except:
        raise ValueError("Counts contain non-integer values")

    seq = [[i] * j for i, j in zip(values, counts)]
    result = []
    for i in range(len(seq)):
        result += seq[i]
    return result


def quantize(x, err=1e-4):
    return torch.floor(x.float() / (2 * np.log(1 + err)) + 0.5).int()


def dequantize(x, err=1e-4):
    return 2 * x * np.log(1 + err)


def compress(x, err=1e-4, quantize_delta=True):
    if quantize_delta:
        x = quantize(x, err)
    else:
        raise Exception(" uncontrolled behaviour for unquantized rle compression")
    encoding = encode(x.ravel().tolist())
    encode_1 = np.array(encoding[0], dtype=int).tobytes()
    encode_2 = np.array(encoding[1], dtype=int).tobytes()
    t_1 = lzma.compress(encode_1)
    t_2 = lzma.compress(encode_2)
    return t_1, t_2, x.shape


def decompress(t_1, t_2, shape, err=1e-4, quantize_delta=True):
    t_1 = np.frombuffer(lzma.decompress(t_1), dtype=int)
    t_2 = np.frombuffer(lzma.decompress(t_2), dtype=int)
    result = np.array(decode(t_1, t_2)).reshape(shape)
    if quantize_delta:
        return dequantize(result, err)
    else:
        raise Exception(" uncontrolled behaviour for unquantized rle compression")
        # return result


def delta_compress(v2, v1, err=1e-4, name_map=None, quantize_delta=True):
    before = 0
    after = 0
    compressed = {}
    if name_map:
        convert = lambda x: name_map[x]
    else:
        convert = lambda x: x

    for name, v2_param in tqdm(v2.items()):
        v1_param = v1[convert(name)]
        if quantize_delta:
            delta = v2_param.cpu() - v1_param.cpu()
        else:
            delta = quantize(v2_param.cpu(), err=err) - quantize(
                v1_param.cpu(), err=err
            )
        if np.count_nonzero(delta) > 0:
            before += sys.getsizeof(v1_param.cpu().untyped_storage())
            t_1, t_2, shape = compress(delta, err=err, quantize_delta=quantize_delta)
            after += len(t_1) + len(t_2) + 4 * len(list(shape))
            compressed[name] = (t_1, t_2, shape)
    if after == 0:
        return compressed, -np.Inf
    else:
        return compressed, before / after


def delta_decompress(compressed, v1, err=1e-4, name_map=None, quantize_delta=True):
    decompressed = {}
    if name_map:
        inv_map = {v: k for k, v in name_map.items()}
        convert = lambda x: inv_map[x]
    else:
        convert = lambda x: x

    for name in tqdm(v1.keys()):
        if convert(name) in compressed:
            t_1, t_2, shape = compressed[convert(name)]
            delta = torch.tensor(
                decompress(t_1, t_2, shape, err=err, quantize_delta=quantize_delta)
            )
            if quantize_delta:
                decompressed[convert(name)] = delta + v1[name].cpu()
            else:
                decompressed[convert(name)] = dequantize(
                    delta + quantize(v1[name].cpu(), err=err), err=err
                )

    return decompressed


def compress_lzma(x, err=1e-4, quantize_delta=True):
    dtype = x.numpy().dtype
    if quantize_delta:
        x = quantize(x, err)
        return lzma.compress(x.ravel().numpy().astype(int).tobytes()), x.shape
    else:
        return lzma.compress(x.ravel().numpy().astype(dtype).tobytes()), x.shape


def decompress_lzma(x, shape, err=1e-4, quantize_delta=True, dtype=np.float32):
    if quantize_delta:
        x = np.frombuffer(lzma.decompress(x), dtype=int)
        return dequantize(x.reshape(shape), err)
    else:
        x = np.frombuffer(lzma.decompress(x), dtype=dtype)
        return x.reshape(shape)


def delta_compress_lzma(v2, v1, err=1e-4, name_map=None, quantize_delta=True):
    before = 0
    after = 0
    compressed = {}

    if name_map:
        convert = lambda x: name_map[x]
    else:
        convert = lambda x: x

    def process_item(item):
        name, v1_param, v2_param, err, quantize_delta = item
        delta = v2_param - v1_param
        if np.count_nonzero(delta) > 0:
            storage_before = sys.getsizeof(v1_param.untyped_storage())
            x, shape = compress_lzma(delta, err=err, quantize_delta=quantize_delta)
            storage_after = len(x) + 4 * len(list(shape))
            return name, x, shape, storage_before, storage_after
        else:
            return None, None, None, None, None

    # cpu_count = mp.cpu_count() 
    # if len(os.sched_getaffinity(0)) < cpu_count:
    #     try:
    #         os.sched_setaffinity(0, range(cpu_count))
    #         print('Using ', len(os.sched_getaffinity(0)), ' processes for compression')
    #     except OSError:
    #         print('Using ', len(os.sched_getaffinity(0)), ' processes for compression')
    # else:
    #     print('Using ', len(os.sched_getaffinity(0)), ' processes for compression')
    
    with mp.get_context("spawn").Pool() as pool:
        for result in tqdm(pool.imap_unordered(process_item, 
            [(name, v1[convert(name)], v2_param, err, quantize_delta) 
             for name, v2_param in v2.items()], 4), total=len(v2)):
            name, x, shape, storage_before, storage_after = result
            if name is not None:
                before += storage_before
                after += storage_after
                compressed[name] = (x, shape)

    if after == 0:
        return compressed, -np.Inf
    else:
        return compressed, before / after

def delta_decompress_lzma(compressed, v1, err=1e-4, name_map=None, quantize_delta=True):
    decompressed = {}
    if name_map:
        inv_map = {v: k for k, v in name_map.items()}
        convert = lambda x: inv_map[x]
    else:
        convert = lambda x: x

    def process_item(item):
        name, x, shape, err, quantize_delta, dtype = item
        delta = torch.tensor(
                decompress_lzma(x, shape, err=err, quantize_delta=quantize_delta, dtype=dtype)
            )
        return name, delta
        
    with mp.get_context("spawn").Pool() as pool:
        for result in tqdm(pool.imap_unordered(process_item, 
                [(name,) + compressed[convert(name)] + 
                 (err, quantize_delta, v1[name].cpu().numpy().dtype) 
                 for name in v1.keys() if convert(name) in compressed], 4), 
                total=len(compressed)):
            name, delta = result
            decompressed[convert(name)] = delta + v1[name].cpu()

    return decompressed


def compress_rle(x, err=1e-4, quantize_delta=True):
    if quantize_delta:
        x = quantize(x, err)
    else:
        raise Exception(" uncontrolled behaviour for unquantized rle compression")
    values, counts = encode(x.ravel().tolist())
    counts = np.array(counts)
    max_count = np.max(counts)
    if max_count <= 255:
        counts = counts.astype(np.uint8)
    elif max_count <= 65535:
        counts = counts.astype(np.uint16)
    elif max_count <= 4294967295:
        counts = counts.astype(np.uint32)
    else:
        counts = counts.astype(int)

    values = np.array(values)
    max_value = np.max(values)
    min_value = np.min(values)

    if max_value <= 127 and min_value >= -128:
        values = values.astype(np.int8)
    elif max_value <= 32767 and min_value >= -32768:
        values = values.astype(np.int16)
    elif max_value <= 2147483647 and min_value >= -2147483648:
        values = values.astype(np.int32)
    else:
        values = values.astype(int)
    return values, counts, x.shape


def decompress_rle(values, counts, shape, err=1e-4, quantize_delta=True):
    result = np.array(decode(values, counts)).reshape(shape)
    if quantize_delta:
        return dequantize(result, err)
    else:
        raise Exception(" uncontrolled behaviour for unquantized rle compression")
        #return result


def delta_compress_rle(v2, v1, err=1e-4, name_map=None, quantize_delta=True):
    before = 0
    after = 0
    compressed = {}
    if name_map:
        convert = lambda x: name_map[x]
    else:
        convert = lambda x: x

    for name, v2_param in tqdm(v2.items()):
        v1_param = v1[convert(name)]
        if quantize_delta:
            delta = v2_param.cpu() - v1_param.cpu()
        else:
            delta = quantize(v2_param.cpu(), err=err) - quantize(
                v1_param.cpu(), err=err
            )
        if np.count_nonzero(delta) > 0:
            before += sys.getsizeof(v1_param.cpu().untyped_storage())
            values, counts, shape = compress_rle(
                delta, err=err, quantize_delta=quantize_delta
            )
            after += (
                values.size * values.itemsize
                + counts.size * counts.itemsize
                + 4 * len(list(shape))
            )
            compressed[name] = (values, counts, shape)
    if after == 0:
        return compressed, -np.Inf
    else:
        return compressed, before / after


def delta_decompress_rle(compressed, v1, err=1e-4, name_map=None, quantize_delta=True):
    decompressed = {}
    if name_map:
        inv_map = {v: k for k, v in name_map.items()}
        convert = lambda x: inv_map[x]
    else:
        convert = lambda x: x

    for name in tqdm(v1.keys()):
        if convert(name) in compressed:
            values, counts, shape = compressed[convert(name)]
            delta = torch.tensor(
                decompress_rle(
                    values, counts, shape, err=err, quantize_delta=quantize_delta
                )
            )
            if quantize_delta:
                decompressed[convert(name)] = delta + v1[name].cpu()
            else:
                decompressed[convert(name)] = dequantize(
                    delta + quantize(v1[name].cpu(), err=err), err=err
                )
    return decompressed


def compress_sparse(x, err=1e-4, quantize_delta=True):
    if quantize_delta:
        return quantize(x, err).to_sparse()
    else:
        return x.to_sparse()


def decompress_sparse(x, err=1e-4, quantize_delta=True):
    if quantize_delta:
        return dequantize(x.to_dense(), err)
    else:
        return x.to_dense()


def delta_compress_sparse(v2, v1, err=1e-4, name_map=None, quantize_delta=True):
    before = 0
    after = 0
    compressed = {}
    if name_map:
        convert = lambda x: name_map[x]
    else:
        convert = lambda x: x

    for name, v2_param in tqdm(v2.items()):

        v1_param = v1[convert(name)]
        if quantize_delta:
            delta = v2_param.cpu() - v1_param.cpu()
        else:
            delta = quantize(v2_param.cpu(), err=err) - quantize(
                v1_param.cpu(), err=err
            )
        if np.count_nonzero(delta) > 0:
            before += sys.getsizeof(v1_param.cpu().untyped_storage())
            x = compress_sparse(delta, err=err, quantize_delta=quantize_delta)
            after += (
                sys.getsizeof(x.coalesce().indices().untyped_storage())
                + sys.getsizeof(x.coalesce().values().untyped_storage())
                + 4 * len(list(x.coalesce().size()))
            )
            compressed[name] = x

    if after == 0:
        return compressed, -np.Inf
    else:
        return compressed, before / after


def delta_decompress_sparse(
    compressed, v1, err=1e-4, name_map=None, quantize_delta=True
):
    decompressed = {}
    if name_map:
        inv_map = {v: k for k, v in name_map.items()}
        convert = lambda x: inv_map[x]
    else:
        convert = lambda x: x

    for name in tqdm(v1.keys()):
        if convert(name) in compressed:
            x = compressed[convert(name)]
            delta = decompress_sparse(x, err=err, quantize_delta=quantize_delta)
            if quantize_delta:
                decompressed[convert(name)] = delta + v1[name].cpu()
            else:
                decompressed[convert(name)] = dequantize(
                    delta + quantize(v1[name].cpu(), err=err), err=err
                )

    return decompressed


def compress_sparse_dict(x, err=1e-4, quantize_delta=True):
    if quantize_delta:
        sparse = quantize(x, err).to_sparse()
    else:
        raise Exception(" uncontrolled behaviour for unquantized dict compression")
        #sparse = x.to_sparse()
    indices = sparse.coalesce().indices().numpy()
    max_index = np.max(indices)
    if max_index <= 255:
        indices = indices.astype(np.uint8)
    elif max_index <= 65535:
        indices = indices.astype(np.uint16)
    elif max_index <= 4294967295:
        indices = indices.astype(np.uint32)
    else:
        indices = indices.astype(int)

    values = sparse.coalesce().values().numpy()
    max_value = np.max(values)
    min_value = np.min(values)
    if max_value <= 127 and min_value >= -128:
        values = values.astype(np.int8)
    elif max_value <= 32767 and min_value >= -32768:
        values = values.astype(np.int16)
    elif max_value <= 2147483647 and min_value >= -2147483648:
        values = values.astype(np.int32)
    else:
        values = values.astype(int)

    size = sparse.coalesce().size()
    return indices, values, size


def decompress_sparse_dict(indices, values, size, err=1e-4, quantize_delta=True):
    if quantize_delta:
        return dequantize(
            torch.sparse_coo_tensor(indices.astype(int), values, size).to_dense(), err
        )
    else:
        raise Exception(" uncontrolled behaviour for unquantized sparse dict compression")
        #return torch.sparse_coo_tensor(indices.astype(int), values, size).to_dense()


def delta_compress_sparse_dict(v2, v1, err=1e-4, name_map=None, quantize_delta=True):
    before = 0
    after = 0
    compressed = {}
    if name_map:
        convert = lambda x: name_map[x]
    else:
        convert = lambda x: x

    for name, v2_param in tqdm(v2.items()):

        v1_param = v1[convert(name)]
        if quantize_delta:
            delta = v2_param.cpu() - v1_param.cpu()
        else:
            delta = quantize(v2_param.cpu(), err=err) - quantize(
                v1_param.cpu(), err=err
            )
        if np.count_nonzero(delta) > 0:
            before += sys.getsizeof(v1_param.cpu().untyped_storage())
            indices, values, size = compress_sparse_dict(
                delta, err=err, quantize_delta=quantize_delta
            )
            after += (
                indices.size * indices.itemsize
                + values.size * values.itemsize
                + 4 * len(list(size))
            )
            compressed[name] = (indices, values, size)
    if after == 0:
        return compressed, -np.Inf
    else:
        return compressed, before / after


def delta_decompress_sparse_dict(
    compressed, v1, err=1e-4, name_map=None, quantize_delta=True
):
    decompressed = {}
    if name_map:
        inv_map = {v: k for k, v in name_map.items()}
        convert = lambda x: inv_map[x]
    else:
        convert = lambda x: x

    for name in tqdm(v1.keys()):
        if convert(name) in compressed:
            indices, values, size = compressed[convert(name)]
            delta = decompress_sparse_dict(
                indices, values, size, err=err, quantize_delta=quantize_delta
            )
            if quantize_delta:
                decompressed[convert(name)] = delta + v1[name].cpu()
            else:
                decompressed[convert(name)] = dequantize(
                    delta + quantize(v1[name].cpu(), err=err), err=err
                )

    return decompressed


def vec_translate(a, my_dict):
    return np.vectorize(my_dict.__getitem__)(a)


def compress_dict(x, err=1e-4, quantize_delta=True):
    if quantize_delta:
        quantized = quantize(x, err)
        values = np.unique(quantized)
    else:
        raise Exception(" uncontrolled behaviour for unquantized dict compression")
        #values = np.unique(x)
    indices = len(values)
    max_value = np.max(values)
    min_value = np.min(values)

    if max_value <= 127 and min_value >= -128:
        values = values.astype(np.int8)
    elif max_value <= 32767 and min_value >= -32768:
        values = values.astype(np.int16)
    elif max_value <= 2147483647 and min_value >= -2147483648:
        values = values.astype(np.int32)
    else:
        values = values.astype(int)

    if indices <= 255:
        code_book = {k: np.uint8(v) for v, k in enumerate(values)}
        r_code_book = {k: v for v, k in code_book.items()}
    elif indices <= 65535:
        code_book = {k: np.uint16(v) for v, k in enumerate(values)}
        r_code_book = {k: v for v, k in code_book.items()}
    elif indices <= 4294967295:
        code_book = {k: np.uint32(v) for v, k in enumerate(values)}
        r_code_book = {k: v for v, k in code_book.items()}
    else:
        code_book = {k: int(v) for v, k in enumerate(values)}
        r_code_book = {k: v for v, k in code_book.items()}

    if quantize_delta:
        return vec_translate(quantized, code_book), r_code_book
    else:
        return vec_translate(x, code_book), r_code_book


def decompress_dict(coded, r_code_book, err=1e-4, quantize_delta=True):
    if quantize_delta:
        return torch.tensor(dequantize(vec_translate(coded, r_code_book), err))
    else:
        raise Exception(" uncontrolled behaviour for unquantized dict compression")
        #return torch.tensor(vec_translate(coded, r_code_book))


def delta_compress_dict(v2, v1, err=1e-4, name_map=None, quantize_delta=True):
    before = 0
    after = 0
    compressed = {}
    if name_map:
        convert = lambda x: name_map[x]
    else:
        convert = lambda x: x

    for name, v2_param in tqdm(v2.items()):

        v1_param = v1[convert(name)]
        if quantize_delta:
            delta = v2_param.cpu() - v1_param.cpu()
        else:
            delta = quantize(v2_param.cpu(), err=err) - quantize(
                v1_param.cpu(), err=err
            )
        if np.count_nonzero(delta) > 0:
            before += sys.getsizeof(v1_param.cpu().untyped_storage())
            coded, r_code_book = compress_dict(
                delta, err=err, quantize_delta=quantize_delta
            )
            after += (
                coded.size * coded.itemsize
                + len(r_code_book) * 4 * 2  # later one is a rough estimate
            )
            compressed[name] = (coded, r_code_book)
    if after == 0:
        return compressed, -np.Inf
    else:
        return compressed, before / after


def delta_decompress_dict(compressed, v1, err=1e-4, name_map=None, quantize_delta=True):
    decompressed = {}
    if name_map:
        inv_map = {v: k for k, v in name_map.items()}
        convert = lambda x: inv_map[x]
    else:
        convert = lambda x: x

    for name in tqdm(v1.keys()):
        if convert(name) in compressed:
            coded, r_code_book = compressed[convert(name)]
            delta = decompress_dict(
                coded, r_code_book, err=err, quantize_delta=quantize_delta
            )
            if quantize_delta:
                decompressed[convert(name)] = delta + v1[name].cpu()
            else:
                decompressed[convert(name)] = dequantize(
                    delta + quantize(v1[name].cpu(), err=err), err=err
                )

    return decompressed
