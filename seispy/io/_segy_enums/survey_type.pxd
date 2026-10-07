cdef enum SurveyType:
    # these are basically using a octal representation:
    # Group I (environment)
    land = 1                # 0b001
    marine = 2              # 0b010
    transition = 3          # 0b011
    downhole = 4            # 0b100

    # Group II (dimensionality)
    dim_1d = 8              # 0b0_001_000
    dim_2d = 16             # 0b0_010_000
    dim_3d = 24             # 0b0_011_000
    time_lapse = 32         # 0b0_100_000

    # Group III (Layout)
    parallel_lines = 128        # 0b00_010_000_000
    cross_spread = 256          # 0b00_100_000_000
    patches = 684               # 0b01_010_101_100
    towed_stream = 1024         # 0b10_000_000_000
    ocean_bottom = 1152         # 0b10_010_000_000
    pseudo_rand = 1280          # 0b10_100_000_000