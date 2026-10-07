cdef enum CoordinateUnit:
    unknown = 0
    length      # (meters or feet, as the file has them)
    degrees     # decimal degrees: x is the longitude, y the latitude
