# https://github.com/pytorch/examples/blob/main/fx/module_tracer.py
import torch
import transformers.utils.fx as fx
from hashlib import sha256
from torch.fx._compatibility import compatibility
from typing import Any, Callable, Dict, Tuple
from deepdiff import DeepDiff
from deepdiff import DeepHash

__all__ = ["ModuleInfo", "ModulePathTracer", "start", "end"]
start = "start"
end = "end"


class ModuleInfo:
    def __init__(
        self,
        class_type="",
        qualified_path="",
        args=None,
        nn_attr_hashes=None,
        anchor=None,
    ):

        if args is None:
            args = {}
        if nn_attr_hashes is None:
            nn_attr_hashes = {}

        self.class_type = class_type
        self.qualified_path = qualified_path
        self.args = args
        self.nn_attr_hashes = nn_attr_hashes
        self.anchor = anchor

    def __str__(self):
        args = ", ".join([f"{k}={v}" for (k, v) in self.args.items()])
        return f"{self.class_type}({args})"

    def __eq__(self, other):
        if self.anchor == other.anchor == start or self.anchor == other.anchor == end:
            return True

        if (
            self.class_type == other.class_type
            and self.args == other.args
            and not DeepDiff(self.nn_attr_hashes, other.nn_attr_hashes)
        ):
            return True
        else:
            return False

    def __hash__(self):
        return hash(
            (
                self.class_type,
                self.qualified_path,
                DeepHash(self.args)[self.args],
                DeepHash(self.nn_attr_hashes)[self.nn_attr_hashes],
                self.anchor,
            )
        )


class ModulePathTracer(fx.HFTracer):
    """
    ModulePathTracer is an FX tracer that--for each operation--also records
    torch.nn.module level information of the Module from which the operation originated.
    """

    def __init__(self):
        super().__init__()
        self.submodule_info = []

    @compatibility(is_backward_compatible=True)
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """
        A method to specify whether a given ``nn.Module`` is a "leaf" module.
        Leaf modules are the atomic units that appear in
        the IR, referenced by ``call_module`` calls. By default,
        Modules in the PyTorch standard library namespace (torch.nn)
        and modules containing keyword 'activation' are leaf modules.
        """
        return (
            m.__module__.startswith("torch.nn")
            or "activation" in m.__module__.casefold()
        ) and not isinstance(m, torch.nn.Sequential)

    @compatibility(is_backward_compatible=True)
    def get_module_info(self, mod: torch.nn.Module) -> ModuleInfo:
        """ """
        nn_attrs = {
            k: v for k, v in vars(mod).items() if k.startswith("_") or k == "training"
        }
        nn_attr_hashes_ = {
            k: DeepHash(v)[v]
            for k, v in nn_attrs.items()
            if k not in ["_parameters", "_buffers"]
        }
        nn_para_hashes = {}
        for k, v in nn_attrs["_parameters"].items():
            if v is None:
                nn_para_hashes[k] = sha256(b"None")
            else:
                nn_para_hashes[k] = sha256(v.detach().numpy().tobytes()).hexdigest()
        nn_buf_hashes = {}
        for k, v in nn_attrs["_buffers"].items():
            if v is None:
                nn_buf_hashes[k] = sha256(b"None")
            else:
                nn_buf_hashes[k] = sha256(v.detach().numpy().tobytes()).hexdigest()

        nn_attr_hashes = {**nn_attr_hashes_, **nn_para_hashes, **nn_buf_hashes}

        # Prefer the O(1) algorithm
        if self.submodule_paths:
            path = self.submodule_paths.get(mod)
            if path is None:
                raise NameError("module is not installed as a submodule")
            assert isinstance(path, str)
            return ModuleInfo(
                str(type(mod))[8:-2],
                path,
                {
                    k: v
                    for k, v in vars(mod).items()
                    if not k.startswith("_") and k != "training"
                },
                nn_attr_hashes,
            )

        # O(N^2) fallback in the case that we didn't store the submodule
        # paths.

        else:
            for n, p in self.root.named_modules():
                if mod is p:
                    return ModuleInfo(
                        str(type(mod))[8:-2],
                        p,
                        {
                            k: v
                            for k, v in vars(mod).items()
                            if not k.startswith("_") and k != "training"
                        },
                        nn_attr_hashes,
                    )
            raise NameError("module is not installed as a submodule")

    def call_module(
        self,
        m: torch.nn.Module,
        forward: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        """
        Override of Tracer.call_module (see
        https://pytorch.org/docs/stable/fx.html#torch.fx.Tracer.call_module).

        This override:
        1) Delegates into the normal Tracer.call_module method
        2) Stores the qualified ModuleInfo of any leaf module
        """
        try:
            return super().call_module(m, forward, args, kwargs)
        finally:
            if self.is_leaf_module(m, ""):
                self.submodule_info.append(self.get_module_info(m))
