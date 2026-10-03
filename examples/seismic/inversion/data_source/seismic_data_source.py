import abc

import numpy as np

from examples.seismic import Receiver, AcquisitionGeometry

__all__ = ['AbstractSeismicDataSource']

class AbstractSeismicDataSource(abc.ABC):
    def __init__(self):
        self.src_positions = np.zeros((1,2))
        self._geometry = None

    def geometry(self) -> AcquisitionGeometry:
        return self._geometry
    
    @property
    def nshots(self) -> int:
        return self.src_positions.shape[0]
    
    @property
    def nrec(self) -> int:
        return self.geometry.rec_positions.shape[0]

    @property
    def f0(self) -> float:
        return self.geometry.f0
    
    @property
    def dt(self) -> float:
        return self.geometry.dt
    
    @property
    def nt(self) -> int:
        return self.geometry.nt

    @abc.abstractmethod
    def get_data(self, idx: int) -> Receiver:
        pass
