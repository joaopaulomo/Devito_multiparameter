import numpy as np

class StepSizeCalculator:
    def update(self, model, grad):
        pass

    def step_size(self):
        pass


class BarzilaiBorweinCalculator(StepSizeCalculator):
    def __init__(self):
        super().__init__()
        self._last_model = None
        self._last_grad = None
        self._s  = None
        self._y  = None

    def update(self, model, grad):
        if not (self._last_model is None):
            self._s = model - self._last_model
            self._y = grad - self._last_grad

        self._last_model = 1*model
        self._last_grad = 1*grad
        

    def step_size(self) -> float:
        ss = np.dot(self._s.reshape(-1), self._s.reshape(-1))
        sy = np.dot(self._s.reshape(-1), self._y.reshape(-1))
        yy = np.dot(self._y.reshape(-1), self._y.reshape(-1))
        
        abb1 = ss / sy
        abb2 = sy / yy
        
        abb3 = abb2  / abb1

        return (abb3 > 0 and abb3 < 1) and abb2 or abb1


class LBFGSCalculator(StepSizeCalculator):
    def __init__(self, memory_length: int):
        super().__init__()
        self.memory_length = memory_length
        self._grads = []
        self._models = []
        

    def update(self, model, grad):

        self._models.insert(0, 1*model)
        self._grads.insert(0, 1*grad)

        if len(self._models) > self.memory_length:
            self._models.pop()
            self._grads.pop()
    
    def step_size(self):
        s = []
        y = []
        q = self._grads[0].reshape()

        for i in range(self.memory_length-1):
            si = (self._models[i] - self._models[i+1]).reshape(-1)
            yi = (self._grads[i] - self._grads[i+1]).reshape(-1)
            
            sigma_i = 1/(yi @ si)
            
            ei = sigma_i * si @ q

            # s.append(self._models[i] - self._models[i+1])
            # y.append(self._grads[i] - self._grads[i+1])
        
        for i in range(self.memory_length-1):
            pass