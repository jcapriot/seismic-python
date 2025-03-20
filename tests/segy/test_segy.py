import pytest
from pathlib import Path
from seispy.io.segy_standard import SEGYCollection
import numpy as np

@pytest.mark.parametrize('endian', ['big', 'little', 'pairwise'])
@pytest.mark.parametrize('data_format', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16])
def test_segy_data_format_reading(endian, data_format):
    file = Path('test_files')/f"Format{data_format:02d}{endian}.segy"
    if data_format in [7, 15] and endian == 'pairwise':
        pytest.skip('3 byte pairwise numbers are not supported.')

    segy = SEGYCollection.from_file(file)
    trace = next(segy.__iter__())

    if np.issubdtype(trace.dtype, np.unsignedinteger):
        ref_data = np.arange(256, dtype=np.uint8)
    else:
        ref_data = np.arange(-128, 128, dtype=np.int8)

    np.testing.assert_array_equal(trace, ref_data)