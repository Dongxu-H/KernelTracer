import torch
from collections.abc import Mapping, Sequence
from execution.base import Execution

class HuggingFaceExecution(Execution):
    framework = "huggingface"

    def supports(self, feature: str) -> bool:
        return feature in {
            "torch_dispatch",
            "torch_profiler",
            "torch_inductor",
            "torch_dynamo_fx",
            "torch_dynamo_aten",
        }

    def __init__(self, *, device: str, mode: str):
        self.device = device
        self.mode = mode


    def execute(
        self,
        model,
        example_inputs=None
    ):
        if self.mode == "train":
            model.train()
            with torch.enable_grad():
                outputs = self._forward(model, example_inputs)
                loss = self._extract_loss(outputs)
                loss = self.custom_reduce_to_scalar_loss(loss)
                if isinstance(loss, torch.Tensor) and loss.requires_grad:
                    loss.backward()
                return outputs
        else:
            model.eval()
            with torch.no_grad():
                return self._forward(model, example_inputs)


    def _forward(self, model, example_inputs):
        if isinstance(example_inputs, dict):
            return model(**example_inputs)
        if isinstance(example_inputs, (list, tuple)):
            return model(*example_inputs)
        return model(example_inputs)


    def _extract_loss(self, outputs):
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss
        return outputs.sum() if torch.is_tensor(outputs) else torch.tensor(0.0)


    def custom_reduce_to_scalar_loss(self, outputs):
        """
        Reduce arbitrary model outputs to a scalar tensor suitable for backward().

        Semantics:
        - Prefer .loss if present
        - Otherwise sum/mean all tensor leaves
        """

        #if isinstance(output, dict):
        #    if "loss" in output:
        #        return output["loss"]
        #if hasattr(output, "loss"):
        #    return output.loss
        #raise RuntimeError(
        #    "Model output does not contain a reducible loss"
        #)

        if hasattr(outputs, "loss") and isinstance(outputs.loss, torch.Tensor):
            return outputs.loss

        tensors = []

        def collect(obj):
            if torch.is_tensor(obj):
                tensors.append(obj)
            elif isinstance(obj, Mapping):
                for v in obj.values():
                    collect(v)
            elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
                for v in obj:
                    collect(v)

        collect(outputs)

        if not tensors:
            raise RuntimeError(
                "custom_reduce_to_scalar_loss: no tensor outputs found"
            )

        # Ensure scalar
        loss = sum(t.mean() for t in tensors)

        if loss.dim() != 0:
            loss = loss.mean()

        return loss
