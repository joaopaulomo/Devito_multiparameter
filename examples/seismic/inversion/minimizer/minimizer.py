from typing import List, Tuple, Callable, Dict
import abc

from scipy.optimize import minimize
import numpy as np

from examples.seismic import Model
from devito import Constant, Function

__all__ = ["AbstractMinimizer", "ModelConverter", "NDArrayStructIntf"]


class AbstractMinimizer(abc.ABC):
    """
    An interface betweem SciPy's `minimize` function and Devito.

    Parameters
    ----------
    parameter_names : List[str]
        A list containing the names of the relevant parameters in the model.

    Attributes
    ----------
    model_ndarray_preprocess_pipeline : List[Callable[[np.ndarray], np.ndarray]]
        A list of callables that accepts a numpy.ndarray as input, representing the model
        and returns another numpy.ndarray, representing the model after some preprocessing
        has been done.

        These callbacks are called in order right after receiveing the model from `minimize()`
        and reshaping it and before converting it to a SeismicModel object.

    model_preprocess_pipeline : List[Callable[[Model], Model]]
        A list of callables that accepts a SeismicModel as input and returns another SeismicModel,
        representing the model after some preprocessing has been done.

        These callbacks are called in order right after converting the model received from `minimize()`
        to a SeismicModel and before sending it to the `loss_grad` procedure.

    grad_postprocess_pipeline : List[Callable[[dict], dict]]
        A list of callables that accepts a dictionary as input, containing the gradients for each
        parameter, and returns another dictionary containing the gradients after some postprocessing
        has been done.

        These callbacks are called in order right after receiving the gradients from `minimize()`
        to a SeismicModel and before sending it to the `loss_grad` procedure.

    grad_ndarray_postprocess_pipeline : List[Callable[[np.ndarray], np.ndarray]]
        A list of callables that accepts a numpy.ndarray as input, representing the gradient,
        and returns another dictionary numpy.ndarray after some postprocessing
        has been done.

        These callbacks are called in order right after receiving the gradients from `minimize()`
        to a SeismicModel and before sending it to the `loss_grad` procedure.
    """

    def __init__(self, parameter_names: List[str]):
        self.parameter_names = parameter_names
        self.loss_history = []

        self.model_ndarray_preprocess_pipeline: List[
            Callable[[NDArrayStructIntf], NDArrayStructIntf]
        ] = []
        self.model_preprocess_pipeline: List[Callable[[Model], Model]] = []
        self.grad_postprocess_pipeline: List[Callable[[dict], dict]] = []
        self.grad_ndarray_postprocess_pipeline: List[
            Callable[[NDArrayStructIntf], NDArrayStructIntf]
        ] = []

    @property
    def iteration(self):
        if hasattr(self, "_iteration"):
            return self._iteration
        return 0

    def start(
        self,
        model0: Model,
        params_range: Dict[str, Tuple[float, float]],
        maxiter: int,
        iteration_callback=lambda _: None,
        loss_grad_callback=lambda _: None,
        **kwargs,
    ):
        """
        Runs the inversion procedure.

        This method is a wrapper for the SciPy's minimize function. Its results are stored in the
        `result` attribute. The loss function value history is stored in the `loss_history`
        attribute.

        Parameters
        ----------
        model0 : SeismicModel
            The starting model.
        params_range : List[Tuple[float, float]]
            A list of pairs `(param_i_min, param_i_max)` where `param_min` is the
            minimum value of the i-th parameter, and `param_max`, its maximum value.
            The ordering of the parameters is the same as the order of the names passed
            in `parameter_names` to the object constructor.
        maxiter : int
            The maximum number of iterations.
        callback : Callable[[np.ndarray, AbstractMinimizer], None]
            A function called at the end of every iteration.
        """

        self.loss_history = []
        self.model_converter = ModelConverter(
            model0, self.parameter_names, params_range
        )

        self._iteration = 0
        self._loss_grad_callback = loss_grad_callback

        self.on_procedure_initialization(model0)

        model0_ndarray_struct = self.model_converter.model_to_ndarray(model0)

        self.result = minimize(
            self._loss_grad_wrapper,
            model0_ndarray_struct.array,
            method="L-BFGS-B",
            jac=True,
            callback=lambda x: self._iteration_finished_callback_wrapper(
                x, iteration_callback
            ),
            options={"maxiter": maxiter, "disp": True, "ftol": 1e-10},
            **kwargs,
        )

    def clip_model_values_filter(self, model_ndarray):
        """
        A filter to clip the model values to the region indicated by the
        `start()`'s `params_range` parameter. It must be used in the
        `model_ndarray_preprocess_pipeline`
        """
        return self.model_converter.clip_model_ndarray_values(model_ndarray)

    def on_procedure_initialization(self, model0: Model):
        """
        This method is called once by the `start()` method just before it calls SciPy's `minimize()`.

        Subclasses can use this method to setup the procedure based on the provided model before it
        starts.

        Parameters
        ----------
        model0 : SeismicModel
            The starting model.
        """
        pass

    @abc.abstractmethod
    def loss_grad(self, model0) -> Tuple[float, dict]:
        """
        Computes and returns the loss function and gradient.

        This method must be implemented by each subclass.

        Parameters
        ----------
        model0 : SeismicModel
            The current iteration model.

        Returns
        -------
        (loss, grad) : Tuple[float, dict]
            A tuple whose first element is the value of the objective function evaluated at `model0`
            and whose second element is a dictionary containing the gradients for each parameter and
            possibly metadata entries, which will be ignored when passing it to `minimize()`.

            The key for the gradient for each parameter must match its name as provided in the
            `parameter_names` in the constructor.

            The gradients are expected to be Devito's `Function` objects.
        """
        pass

    def _iteration_finished_callback_wrapper(self, xk, callback):
        self._iteration += 1
        callback(xk, self)

    def _loss_grad_wrapper(self, model_ndarray_unshaped):
        # model_ndarray = model_ndarray_unshaped.reshape(self.model_manager.params_shape)
        model_ndarray = NDArrayStructIntf(
            model_ndarray_unshaped, self.model_converter.array_fields
        )
        model_ndarray = self._preprocess_model_ndarray(model_ndarray)

        model0 = self.model_converter.ndarray_to_model(model_ndarray)
        model0 = self._preprocess_model(model0)

        loss, grad = self.loss_grad(model0)
        if len(self.loss_history) == self._iteration:
            self.loss_history.append(loss)
        else:
            self.loss_history[-1] = loss
        grad = self._postprocess_grad(grad)
        grad_ndarray = self.model_converter.dict_to_ndarray(grad)
        grad_ndarray = self._postprocess_grad_ndarray(grad_ndarray)

        self._loss_grad_callback(self)

        return loss, grad_ndarray.array

    def _preprocess_model_ndarray(self, model_ndarray):
        if len(self.model_ndarray_preprocess_pipeline) > 0:
            for F in self.model_ndarray_preprocess_pipeline:
                model_ndarray = F(model_ndarray)
        return model_ndarray

    def _preprocess_model(self, model):
        if len(self.model_preprocess_pipeline) > 0:
            for F in self.model_preprocess_pipeline:
                model = F(model)
        return model

    def _postprocess_grad(self, grad):
        if len(self.grad_postprocess_pipeline) > 0:
            for F in self.grad_postprocess_pipeline:
                grad = F(grad)
        return grad

    def _postprocess_grad_ndarray(self, grad_ndarray):
        if len(self.grad_ndarray_postprocess_pipeline) > 0:
            for F in self.grad_ndarray_postprocess_pipeline:
                grad_ndarray = F(grad_ndarray)
        return grad_ndarray


class NDArrayStructIntf:
    def __init__(self, array, fields: dict):
        self.array = array
        self.fields = fields

    def _get_field_data(self, field_name):
        if field_name in self.fields:
            offset = self.fields[field_name].get("offset", None)
            size = self.fields[field_name].get("size", None)
            shape = self.fields[field_name].get("shape", -1)

            if offset is None:
                raise ValueError(f"No offset value was found for field `{field_name}`.")
            elif size is None:
                raise ValueError(f"No size value was found for field `{field_name}`.")

            return offset, size, shape
        raise ValueError(
            f"Attempted to access non-existent field `{field_name}` in a structured array."
        )

    def __getitem__(self, field_name):
        offset, size, shape = self._get_field_data(field_name)
        v = self.array[offset : offset + size].reshape(shape)
        if size == 1:
            return v[0]
        return v

    def __setitem__(self, field_name, value):
        offset, size, _ = self._get_field_data(field_name)
        current = self.__getitem__(field_name)
        if hasattr(value, "shape"):
            if value.shape == current.shape:
                self.array[offset : offset + size] = value.reshape(-1)
        else:
            self.array[offset] = value

    def to_dict(self):
        d = {}
        for fname, _ in self.fields.items():
            d[fname] = self[fname]
        return d


class ModelConverter:
    """
    Manages the conversion betwen different representation of a model.

    This class handles 3 representations:

    - SeismicModel:

    This the usual model from Devito's `example.seimsic` package.

    - numpy.ndarray:

    This is a `nparams x model_shape` numpy array, where `nparams` is
    the number of model parameters.

    Each element of the first axis of this array represents the parameter
    with the same index according to `parameter_names`.

    - dict:

    This is similar to the output of SeismicModel.physical_params(),
    but contains only the properties indicated in `parameter_names`.

    Parameters
    ----------
    sample_model : SeimicModel
        A basis to create the reference model.
    parameter_names : List[str]
        A list containing the names of the relevant parameters in the model.
    parameter_ranges: List[Tuple[float, float]]
        A list of pairs `(param_i_min, param_i_max)` where `param_min` is the
        minimum value of the i-th parameter, and `param_max`, its maximum value.
        The ordering of the parameters is the same as the order of the names passed
        in `parameter_names` to the object constructor.

    Attributes
    ----------
    reference_model : SeismicModel
        A reference model.
    """

    def __init__(
        self,
        sample_model: Model,
        parameter_names: List[str],
        parameter_ranges: Dict[str, Tuple[float, float]],
    ):
        self.model_cls = type(sample_model)
        self.parameter_names = parameter_names
        self.parameter_ranges = parameter_ranges

        sample_parameters = sample_model.physical_params()
        sample_parameters.pop("damp")

        for pname in self.parameter_names:
            sample_parameters.pop(pname, None)

        self.fixed_parameters = {}
        for k, v in sample_parameters.items():
            if type(v) == Constant:
                self.fixed_parameters[k] = v.data
            elif type(v) == Function:
                self.fixed_parameters[k] = v.data[
                    [slice(sample_model.nbl, -sample_model.nbl)]
                    * len(sample_model.shape)
                ]
        self.damp_data = 1 * sample_model.damp.data

        size = 0
        fields = {}
        parameters = sample_model.physical_params()
        for pname in parameter_names:
            if type(parameters[pname]) == Constant:
                fields[pname] = {"offset": size, "size": 1, "shape": 1}
                size += 1
            else:
                prod = 1
                for i in range(len(sample_model.shape)):
                    prod *= sample_model.shape[i]

                fields[pname] = {
                    "offset": size,
                    "size": prod,
                    "shape": sample_model.shape,
                }

                size += prod
        self.array_fields = fields
        self.array_size = size

        self.reference_model = self._create_reference_model(
            sample_model, parameter_ranges, parameter_names
        )

    @property
    def model_shape(self):
        return self.reference_model.shape

    # @property
    # def nparams(self) -> int:
    #     return len(self.parameter_names)

    # @property
    # def params_shape(self) -> tuple:
    #     return (self.nparams, *self.reference_model.shape)

    @property
    def nbl_slice(self):
        return [slice(self.reference_model.nbl, -self.reference_model.nbl)] * len(
            self.reference_model.shape
        )

    @property
    def _empty_ndarray(self) -> NDArrayStructIntf:
        params = np.zeros(self.array_size, dtype=np.float32)
        params_struct = NDArrayStructIntf(params, self.array_fields)

        return params_struct

    def _create_reference_model(
        self, sample_model: Model, params_range, parameter_names
    ) -> Model:
        size = 0
        fields = {}
        params = np.zeros(self.array_size, dtype=np.float32)
        params_struct = NDArrayStructIntf(params, self.array_fields)

        for pname in parameter_names:
            param = params_struct[pname]
            prange = params_range[pname]
            if type(param) == np.ndarray:
                param[:] = prange[0] * np.ones(sample_model.shape)
                param[:, -1] = prange[1]
            else:
                params_struct[pname] = (prange[0] + prange[1]) / 2

        model_ref = self.model_cls(
            origin=sample_model.origin,
            spacing=sample_model.spacing,
            shape=sample_model.shape,
            space_order=sample_model.space_order,
            nbl=sample_model.nbl,
            grid=sample_model.grid,
            **params_struct.to_dict(),
            **self.fixed_parameters,
        )

        model_ref.damp.data[:] = self.damp_data

        return model_ref

    def clip_model_ndarray_values(self, model_ndarray: np.ndarray) -> np.ndarray:
        """
        Clips the values of a numpy.ndarray which represents a model to the ranges
        specified in `parameter_ranges`.

        This method is intended to be used as a filter through the `AbstractMinimizer.clip_model_values_filter()` method.

        Parameters
        ----------
        model_ndarray : ndarray
            A numpy.ndarray representing the model.

        Returns
        -------
        The clipped model.
        """
        for pname in self.parameter_names:
            v = model_ndarray[pname]
            v = np.clip(
                v, self.parameter_ranges[pname][0], self.parameter_ranges[pname][1]
            )
            model_ndarray[pname] = v
        return model_ndarray

    def model_to_dict(self, model: Model) -> dict:
        model_dict = model.physical_params()
        model_dict = {k: model_dict[k] for k in self.parameter_names}
        return model_dict

    def dict_to_model(self, model_dict: dict) -> Model:
        model = self.model_cls(
            origin=self.reference_model.origin,
            spacing=self.reference_model.spacing,
            shape=self.reference_model.shape,
            space_order=self.reference_model.space_order,
            nbl=self.reference_model.nbl,
            grid=self.reference_model.grid,
            **{
                k: (
                    model_dict[k].data[self.nbl_slice]
                    if type(model_dict[k]) != Constant
                    else model_dict[k].data
                )
                for k in self.parameter_names
            },
            **self.fixed_parameters,
        )

        model.damp.data[:] = self.damp_data

        return model

    def model_to_ndarray(self, model: Model) -> NDArrayStructIntf:
        array = self._empty_ndarray
        slc = self.nbl_slice
        parameters = model.physical_params()
        for pname in self.parameter_names:
            if type(parameters[pname]) == Constant:
                array[pname] = parameters[pname].data
            else:
                array[pname] = parameters[pname].data[slc]
        return array

    def ndarray_to_model(self, model_array: NDArrayStructIntf) -> Model:
        # model_array_struct = NDArrayStructIntf(model_array, self.array_fields)
        model = self.model_cls(
            origin=self.reference_model.origin,
            spacing=self.reference_model.spacing,
            shape=self.reference_model.shape,
            space_order=self.reference_model.space_order,
            nbl=self.reference_model.nbl,
            grid=self.reference_model.grid,
            **model_array.to_dict(),
            **self.fixed_parameters,
        )

        model.damp.data[:] = self.damp_data
        return model

    def dict_to_ndarray(self, model_dict: dict) -> NDArrayStructIntf:
        array = self._empty_ndarray
        slc = self.nbl_slice
        for pname in self.parameter_names:
            if type(model_dict[pname]) == Constant:
                array[pname] = model_dict[pname].data[:]
            else:
                array[pname] = model_dict[pname].data[slc]
        return array

    def ndarray_to_dict(self, model_array: NDArrayStructIntf) -> dict:
        model = self.ndarray_to_model(model_array)
        return self.model_to_dict(model)
