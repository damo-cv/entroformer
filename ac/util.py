import sys, os
import torch
import torchac
import numpy as np
from torch.utils.cpp_extension import load

python3 = sys.version_info.major >= 3
numpyAc_backend = load(
  name="numpyAc_backend",
  sources=["ac/backend/numpyAc_backend.cpp"],
  verbose=False)

PRECISION = 16 # DO NOT EDIT!


def convert_to_int_and_normalize(cdf_float, needs_normalization):
    """Convert floatingpoint CDF to integers. See README for more info.
    The idea is the following:
    When we get the cdf here, it is (assumed to be) between 0 and 1, i.e,
    cdf \in [0, 1)
    (note that 1 should not be included.)
    We now want to convert this to int16 but make sure we do not get
    the same value twice, as this would break the arithmetic coder
    (you need a strictly monotonically increasing function).
    So, if needs_normalization==True, we multiply the input CDF
    with 2**16 - (Lp - 1). This means that now,
    cdf \in [0, 2**16 - (Lp - 1)].
    Then, in a final step, we add an arange(Lp), which is just a line with
    slope one. This ensure that for sure, we will get unique, strictly
    monotonically increasing CDFs, which are \in [0, 2**16)
    """
    Lp = cdf_float.shape[-1]
    factor = torch.tensor(2, dtype=torch.float32, device=cdf_float.device).pow_(PRECISION)
    new_max_value = factor
    if needs_normalization:
        new_max_value = new_max_value - (Lp - 1)
    cdf_float = cdf_float.mul(new_max_value)
    cdf_float = cdf_float.round()
    cdf = cdf_float.to(dtype=torch.int16, non_blocking=True)
    if needs_normalization:
        r = torch.arange(Lp, dtype=torch.int16, device=cdf.device)
    cdf.add_(r)
    return cdf


class ArithmeticEncoder(object):
    def __init__(self, binfile):
        self.bitout = BitOutputStream(open(binfile, "wb"))

    def encode(self, cdf, sym):
        # assert pdf.shape[0]==sym.shape[0]
        assert cdf.ndim == sym.ndim + 1

        # self.sysNum = sym.shape[0]

        # cdf = pdf_to_cdf(pmf)

        # pdf = np.diff(cdfF)
        # print( -np.log2(pdf[range(0,self.sysNum),sym]).sum())

        byte_stream = torchac.encode_float_cdf(cdf, sym, check_input_bounds=True)
        real_bits = len(byte_stream) * 8

        self.bitout.output.write(byte_stream)

        return byte_stream, real_bits

    def write_int(self, values, numbits=32):
        for value in values:
            for i in reversed(range(numbits)):
                self.bitout.write((value >> i) & 1)    # Big endian

    def close(self):
        self.bitout.close()


class ArithmeticDecoder(object):
    """
    Decoding class
    byte_stream: the bin file stream.
    sysNum: the Number of symbols that you are going to decode. This value should be 
            saved in other ways.
    sysDim: the Number of the possible symbols.
    binfile: bin file path, if it is Not None, 'byte_stream' will read from this file
            and copy to Cpp backend Class 'InCacheString'
    """
    def __init__(self, binfile, L):
        self.L = L
        self.bitin = BitInputStream(open(binfile, "rb"))

    def construct(self, sym_num):
        self.sym_num = sym_num
        self.byte_stream = self.bitin.input.read()
        self.decoder = numpyAc_backend.decode(self.byte_stream, self.sym_num, self.L+1)

    def decode(self, cdf):
        assert cdf.size(-1) == self.L
        cdf_shape = list(cdf.size())[:-1]

        # cdfF = pdf_convert_to_cdf_and_normalize(pdf)
        cdf_int = convert_to_int_and_normalize(cdf, needs_normalization=True).numpy()
        cdf_int = cdf_int.astype(np.uint16)

        cdf_int = cdf_int.reshape(-1, self.L)
        sym_out = []
        for cdf_ in cdf_int:
            sym_out.append(self.decoder.decodeAsym(cdf_.tolist()))
        sym_out = np.array(sym_out).reshape(cdf_shape)
        return sym_out

    def read_head(self, num, numbits=32):
        buffer_list = []
        for _ in range(num):
            result = 0
            for _ in range(numbits):
                result = (result << 1) | self.bitin.read_no_eof()    # Big endian
            buffer_list.append(result)
        return buffer_list

    def close(self):
        self.bitin.close()


class BitInputStream(object):
    # Constructs a bit input stream based on the given byte input stream.
    def __init__(self, inp):
        # The underlying byte stream to read from
        self.input = inp
        # Either in the range [0x00, 0xFF] if bits are available, or -1 if end of stream is reached
        self.currentbyte = 0
        # Number of remaining bits in the current byte, always between 0 and 7 (inclusive)
        self.numbitsremaining = 0
      
    # Reads a bit from this stream. Returns 0 or 1 if a bit is available, or -1 if
    # the end of stream is reached. The end of stream always occurs on a byte boundary.
    def read(self):
        if self.currentbyte == -1:
           return -1
        if self.numbitsremaining == 0:
            temp = self.input.read(1)
            if len(temp) == 0:
                self.currentbyte = -1
                return -1
            self.currentbyte = temp[0] if python3 else ord(temp)
            self.numbitsremaining = 8
        assert self.numbitsremaining > 0
        self.numbitsremaining -= 1
        return (self.currentbyte >> self.numbitsremaining) & 1
    
    # Reads a bit from this stream. Returns 0 or 1 if a bit is available, or raises an EOFError
    # if the end of stream is reached. The end of stream always occurs on a byte boundary.
    def read_no_eof(self):
        result = self.read()
        if result != -1:
           return result
        else:
           raise EOFError()
    
    # Closes this stream and the underlying input stream.
    def close(self):
        self.input.close()
        self.currentbyte = -1
        self.numbitsremaining = 0


class BitOutputStream(object): 
    # Constructs a bit output stream based on the given byte output stream.
    def __init__(self, out):
        self.output = out  # The underlying byte stream to write to
        self.currentbyte = 0  # The accumulated bits for the current byte, always in the range [0x00, 0xFF]
        self.numbitsfilled = 0  # Number of accumulated bits in the current byte, always between 0 and 7 (inclusive)
      
    # Writes a bit to the stream. The given bit must be 0 or 1.
    def write(self, b):
        if b not in (0, 1):
            raise ValueError("Argument must be 0 or 1")
        self.currentbyte = (self.currentbyte << 1) | b
        self.numbitsfilled += 1
        if self.numbitsfilled == 8:
            towrite = bytes((self.currentbyte,)) if python3 else chr(self.currentbyte)
            self.output.write(towrite)
            self.currentbyte = 0
            self.numbitsfilled = 0
    
    # Closes this stream and the underlying output stream. If called when this
    # bit stream is not at a byte boundary, then the minimum number of "0" bits
    # (between 0 and 7 of them) are written as padding to reach the next byte boundary.
    def close(self):
        while self.numbitsfilled != 0:
            self.write(0)
        self.output.close()

