cdef enum EndianKey:
    # these are written as bytes so that they automatically get the
    # correct value on whatever system this is compiled on
    big = 0x01020304
    little = 0x04030201
    pairwise = 0x02010403
    system = 16909060