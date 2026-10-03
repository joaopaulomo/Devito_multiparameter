from typing import List, Dict, Tuple


class ModelNormalizationFilter():
    def __init__(self, parameter_ranges: Dict[str, Tuple[float, float]], denormalize_mode = False):
        self.parameter_ranges = parameter_ranges
        self.denormalize_mode = denormalize_mode
    

    def __call__(self, model_ndarray):
        if self.denormalize_mode:
            return self.denormalization(model_ndarray)
        else:
            return self.normalization(model_ndarray)


    def normalization(self, model_ndarray):
        model_ndarray.array = 1*model_ndarray.array
        for k, v in self.parameter_ranges.items():
            pmin, pmax = v
            model_ndarray[k] = (model_ndarray[k] - pmin)/(pmax - pmin)
        
        return model_ndarray
    

    def denormalization(self, model_ndarray):
        model_ndarray.array = 1*model_ndarray.array
        for k, v in self.parameter_ranges.items():
            pmin, pmax = v
            model_ndarray[k] = (pmax - pmin) * model_ndarray[k] + pmin

        return model_ndarray