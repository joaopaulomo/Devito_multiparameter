from examples.seismic.inversion.data_source.seismic_data_source import AbstractSeismicDataSource

class PrecomputedDataSource(AbstractSeismicDataSource):
    def __init__(self, geometry, src_positions, dobs_list: list):
        super().__init__()
        self._geometry = geometry
        self.src_positions = src_positions
        self._dobs_list = dobs_list
    
    def get_data(self, idx: int):
        return self._dobs_list[idx]
        