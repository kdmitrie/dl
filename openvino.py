import numpy as np
import openvino as ov
import openvino.properties as op
import timeit
import torch
from typing import Callable

class OVConverter:
    def __init__(self, half_precision: bool=False):
        """
        Init the converter
        """
        self.core = ov.Core()
        self.half = half_precision


    def compile_and_dump(self,
                         model: torch.nn.Module,
                         input_shape: list[int],
                         ov_fname: str):
        """
        Convert the model to OV format and save to the specified file
        """
        input = torch.zeros(input_shape)
        if self.half:
            input = input.half()
            model.half()

        openvino_model = ov.convert_model(model,
                                          input=input_shape,
                                          example_input=input)

        compiled_model = self.core.compile_model(openvino_model,
                                                 device_name="CPU",
                                                 config={
                                                    op.inference_num_threads(): 4,
                                                    op.hint.enable_hyper_threading(): True})

        with open(ov_fname, 'wb') as f:
            bytes_model = compiled_model.export_model()
            f.write(bytes_model.getvalue())


    def get_predictor(self, ov_fname: str) -> Callable:
        """
        Return a callable capable of processing the data
        """
        with open(ov_fname, 'rb') as f:
            compiled_model = self.core.import_model(f.read(), "CPU")
        return lambda x: self.__get_ov_prediction(compiled_model, x)


    def __get_ov_prediction(self, compiled_model, x) -> None:
        """
        Private function to actually return the result
        """
        infer_request = compiled_model.create_infer_request()
        infer_request.infer(inputs=[x.numpy()])
        return infer_request.get_output_tensor().data


    def calculate_error(self,
                        initial_model: torch.nn.Module,
                        ov_fname: str,
                        input_shape: list[int],
                        attempts: int=10) -> float:
        """
        Calculate the relative error between initial and optimized models predictions
        """
        ov_model = self.get_predictor(ov_fname)

        error = 0

        for n in range(attempts):
            input = torch.randn(input_shape)
            if self.half:
                input = input.half()
            with torch.no_grad():
                y1 = initial_model(input).cpu().numpy()
            y2 = ov_model(input)
            error += np.linalg.norm(y2 - y1) / np.linalg.norm(y1)
        return float(error) / attempts


    def execution_time(self,
                       model: torch.nn.Module | str,
                       input_shape: list[int],
                       attempts: int=10) -> float:
        """
        Calculate the execution time
        """
        if isinstance(model, str):
            model = self.get_predictor(ov_fname=model)

        def run():
            inp = torch.randn(input_shape)
            if self.half:
                inp = inp.half()
            model(inp)

        return timeit.timeit(run, number=attempts)
