# https://github.com/pytorch/examples/blob/main/fx/module_tracer.py
import torch
import re
import copy
from hashlib import sha256
import transformers
import importlib
from transformers.utils.fx import HFTracer, _proxies_to_metas, _MANUAL_META_OVERRIDES
from torch.fx.graph import _Namespace, Graph
from torch.fx import Proxy
from torch.fx._compatibility import compatibility

from deepdiff import DeepDiff, DeepHash

from collections import OrderedDict, defaultdict
from typing import Any, Callable, Dict, Tuple, Optional

__all__ = ["ModuleInfo", "ModulePathTracer"]
layer_shape_info = [
    "embedding_dim",
    "num_embeddings",
    "in_features",
    "out_features",
    "normalized_shape",
    "adapter",
]


class ModuleInfo:
    def __init__(
        self,
        class_type="",
        args=None,
        nn_attr_hashes=None,
        matched=False,
        start=False,
    ):

        if args is None:
            args = {}
        if nn_attr_hashes is None:
            nn_attr_hashes = {}

        self.class_type = class_type
        self.args = args
        self.nn_attr_hashes = nn_attr_hashes
        self.matched = matched
        self.start = start

    def update_matched(self, matched):
        self.matched = matched

    def __str__(self):
        args = ", ".join([f"{k}={v}" for (k, v) in self.args.items()])
        return f"{self.class_type}({args})"

    def __eq__(self, other):

        if (
            self.class_type == other.class_type
            and self.args == other.args
            and not DeepDiff(self.nn_attr_hashes, other.nn_attr_hashes)
        ):
            return True
        else:
            return False

    def __hash__(self):
        content = {
            **{self.class_type: self.class_type},
            **self.args,
            **self.nn_attr_hashes,
        }
        return int(DeepHash(content)[content], 16)

    def class_hash(self):
        content = {**{self.class_type: self.class_type}, **self.args}
        return int(DeepHash(content)[content], 16)


class UpperModuleInfo(ModuleInfo):
    def __init__(
        self, class_type="", submoduleinfo_dict=None, matched=False, start=False
    ):
        super().__init__()
        self.matched = matched
        self.class_type = class_type
        self.nn_attr_hashes = {}
        self.start = start

        if submoduleinfo_dict:
            self.submoduleinfo_dict = copy.deepcopy(submoduleinfo_dict)
            self.nn_attr_hashes = {
                name: hash(submoduleinfo)
                for name, submoduleinfo in submoduleinfo_dict.items()
            }
        else:
            self.submoduleinfo_dict = {}

    def update_matched(self, matched):
        self.matched = matched

    # TODO
    def __eq__(self, other):
        if not DeepDiff(
            list(self.submoduleinfo_dict.values()),
            list(other.submoduleinfo_dict.values()),
        ):
            return True
        else:
            return False

    def __hash__(self):
        counter = defaultdict(int)
        hashes = []
        for _, moduleinfo in self.submoduleinfo_dict.items():
            hash_value = hash(moduleinfo)
            hashes.append((hash_value, counter[hash_value]))
            counter[hash_value] = counter[hash_value] + 1
        return int(DeepHash(hashes)[hashes], 16)

    def class_hash(self):
        counter = defaultdict(int)
        hashes = []
        for _, moduleinfo in self.submoduleinfo_dict.items():
            hash_value = moduleinfo.class_hash()
            hashes.append((hash_value, counter[hash_value]))
            counter[hash_value] = counter[hash_value] + 1
        return int(DeepHash(hashes)[hashes], 16)


class ModuleInfoNamespace(_Namespace):
    def __init__(self, obj_to_name=None, unassociated_names=None, used_names=None):

        if obj_to_name is None:
            obj_to_name = {}
        if unassociated_names is None:
            unassociated_names = set()
        if used_names is None:
            used_names = {}

        self._obj_to_name = copy.deepcopy(obj_to_name)
        self._unassociated_names = copy.deepcopy(unassociated_names)
        self._used_names = copy.deepcopy(used_names)

        self._illegal_char_regex = re.compile("[^0-9a-zA-Z_]+")
        self._name_suffix_regex = re.compile(r"(.*)_(\d+)$")
        if len(used_names) == 0:
            self._no_prior_names = True
        else:
            self._no_prior_names = False

    def used_names(self):
        return self._used_names

    def create_name(self, candidate: str, obj: Optional[Any]) -> str:
        """Create a unique name.
        Arguments:
            candidate: used as the basis for the unique name, relevant to the user.
            obj: If not None, an object that will be associated with the unique name.
        """
        if obj is not None and obj in self._obj_to_name:
            return self._obj_to_name[obj]

        # delete all characters that are illegal in a Python identifier
        if self._no_prior_names:
            candidate = self._illegal_char_regex.sub("_", candidate)
        else:
            candidate = "_" + self._illegal_char_regex.sub("_", candidate)

        if candidate[0].isdigit():
            candidate = f"_{candidate}"

        match = self._name_suffix_regex.match(candidate)
        if match is None:
            base = candidate
            num = None
        else:
            base, num_str = match.group(1, 2)
            num = int(num_str)

        candidate = base if num is None else f"{base}_{num}"
        num = num if num else 0

        while candidate in self._used_names or self._is_illegal_name(candidate, obj):
            num += 1
            candidate = f"{base}_{num}"

        self._used_names.setdefault(candidate, 0)
        if obj is None:
            self._unassociated_names.add(candidate)
        else:
            self._obj_to_name[obj] = candidate
        return candidate


class ModulePathTracer(HFTracer, Graph):
    """
    ModulePathTracer is an FX tracer that--for each operation--also records
    torch.nn.module level information of the Module from which the operation originated.
    """

    def __init__(self, used_names=None, tracing_module_pool=None):
        super().__init__()
        self.submodule_info = OrderedDict()
        self.namespace = ModuleInfoNamespace(used_names=used_names)
        self.tracing_module_pool = tracing_module_pool
        self.traced_module_info = OrderedDict()
        self.module_name_map = {}

    def create_proxy(
        self,
        kind,
        target,
        args,
        kwargs,
        name=None,
        type_expr=None,
        proxy_factory_fn=None,
    ):
        rv = super(HFTracer, self).create_proxy(
            kind, target, args, kwargs, name, type_expr, proxy_factory_fn
        )

        if kind == "placeholder" and target in self.meta_args:
            rv.install_metadata(self.meta_args[target])
            return rv

        if target in self.orig_fns:
            # NOTE: tensor constructors in PyTorch define the `device` argument as
            # *kwargs-only*. That is why this works. If you add methods to
            # _TORCH_METHODS_TO_PATCH that do not define `device` as kwarg-only,
            # this will break and you will likely see issues where we cannot infer
            # the size of the output.
            if "device" in kwargs:
                kwargs["device"] = "meta"

        try:
            args_metas = torch.fx.node.map_aggregate(args, _proxies_to_metas)
            kwargs_metas = torch.fx.node.map_aggregate(kwargs, _proxies_to_metas)

            if kind == "call_function":
                meta_target = _MANUAL_META_OVERRIDES.get(target, target)
                meta_out = meta_target(*args_metas, **kwargs_metas)
                if isinstance(meta_out, torch.Tensor):
                    meta_out = meta_out.to(device="meta")
            elif kind == "call_method":
                method = getattr(args_metas[0].__class__, target)
                meta_target = _MANUAL_META_OVERRIDES.get(method, method)
                meta_out = meta_target(*args_metas, **kwargs_metas)
            elif kind == "call_module":
                if not hasattr(self, "orig_forward"):
                    raise AttributeError(
                        f"{self} does not have an attribute called orig_forward"
                    )
                self._disable_module_getattr = True
                try:
                    mod = self.root.get_submodule(target)
                    mod_type = type(mod)
                    if mod_type in _MANUAL_META_OVERRIDES:
                        meta_out = _MANUAL_META_OVERRIDES[mod_type](
                            mod, *args_metas, **kwargs_metas
                        )
                    else:
                        meta_out = self.orig_forward(*args_metas, **kwargs_metas)
                finally:
                    self._disable_module_getattr = False
            elif kind == "get_attr":
                self._disable_module_getattr = True
                try:
                    attr_itr = self.root
                    atoms = target.split(".")
                    for atom in atoms:
                        attr_itr = getattr(attr_itr, atom)
                    if isinstance(attr_itr, torch.Tensor):
                        meta_out = attr_itr.to(device="meta")
                    else:
                        meta_out = attr_itr
                finally:
                    self._disable_module_getattr = False
            else:
                return rv

            if not isinstance(rv, Proxy):
                raise ValueError("Don't support composite output yet")
            rv.install_metadata(meta_out)
        except Exception as e:
            pass

        return rv

    # TODO: stop hardcode this
    def trace(self, module, concrete_args):
        exec("""fx.wrap('baddbmm')""", vars(torch))
        if 'mpt' in str(type(module)):
            _commit_hash = module.__module__.split('.')[-2]
            mpt = importlib.import_module(''.join([module.__module__.split(_commit_hash, 1)[0], _commit_hash, '.attention']))
            exec("""import torch.fx as fx; fx.wrap('rearrange')""", vars(mpt))
            if hasattr(module, 'transformer'):
                attn_bias_initialized = module.transformer._attn_bias_initialized
                attn_bias = module.transformer.attn_bias

        if 'falcon' in str(type(module)):
            falcon = importlib.import_module(module.__module__)
            exec("""def identity(seq_len, device, dtype):return (1, 'cpu', torch.float16)""",
                falcon.RotaryEmbedding.forward.__globals__,
            )
            exec("""import torch.fx as fx; fx.wrap('rotate_half')""", falcon.RotaryEmbedding.forward.__globals__,
            )

        if "llama" in str(type(module)):
            exec("""def check_inequality(a, b):return False""",
                transformers.models.llama.modeling_llama.LlamaAttention.forward.__globals__,
            )

        if "gptj" in str(type(module)):
            exec("""def len(e):return 5""", 
                 transformers.models.gptj.modeling_gptj.GPTJAttention._split_heads.__globals__
            )
            exec("""def len(e):return 5""", 
                 transformers.models.gptj.modeling_gptj.GPTJAttention._merge_heads.__globals__
            )

        if "distilbert" in str(type(module)):
            exec(
                """def len(e):return 1""",
                transformers.models.distilbert.modeling_distilbert.Transformer.forward.__globals__,
            )

        traced = super().trace(module, concrete_args)

        if "llama" in str(type(module)):
            exec("""def check_inequality(a, b):return a != b """,
                transformers.models.llama.modeling_llama.LlamaAttention.forward.__globals__,
            )

        if "gptj" in str(type(module)):
            exec("""import builtins; len = lambda e: builtins.len(e)""", 
                 transformers.models.gptj.modeling_gptj.GPTJAttention._split_heads.__globals__
            )
            exec("""import builtins; len = lambda e: builtins.len(e)""", 
                 transformers.models.gptj.modeling_gptj.GPTJAttention._merge_heads.__globals__
            )

        if "distilbert" in str(type(module)):
            exec(
                """import builtins; len = lambda e: builtins.len(e)""",
                transformers.models.distilbert.modeling_distilbert.Transformer.forward.__globals__,
            )

        if 'falcon' in str(type(module)):
            exec("""def identity(seq_len, device, dtype):return (seq_len, device, dtype)""",
                falcon.RotaryEmbedding.forward.__globals__,
            )

        if 'mpt' in str(type(module)) and hasattr(module, 'transformer'):
            module.transformer._attn_bias_initialized = attn_bias_initialized
            module.transformer.attn_bias = attn_bias
        
        return traced

    def get_submodule_info(self):
        if self.tracing_module_pool:
            return self.traced_module_info
        else:
            return self.submodule_info
    
    def get_module_name_map(self):
        return self.module_name_map

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
            or "Conv1D()" == str(m)
            or "LlamaRMSNorm()" == str(m)
            or str(m).startswith('LPLayerNorm')
            or (str(m).startswith("Linear") and m.__module__.startswith("transformers_modules"))
        ) and not isinstance(m, torch.nn.Sequential)

    @compatibility(is_backward_compatible=True)
    def path_of_module(self, mod: torch.nn.Module) -> str:
        """
        Helper method to find the qualified name of ``mod`` in the Module hierarchy
        of ``root``. For example, if ``root`` has a submodule named ``foo``, which has
        a submodule named ``bar``, passing ``bar`` into this function will return
        the string "foo.bar".
        Args:
            mod (str): The ``Module`` to retrieve the qualified name for.
        """
        # Prefer the O(1) algorithm

        # https://github.com/pytorch/pytorch/blob/91c5fc323b933516a46add626cfa402f73d7e474/torch/fx/graph.py#L728
        if "activation" in mod.__module__.casefold() and hasattr(mod, "act"):
            candidate = self._target_to_str(mod.act)
            return self.namespace.create_name(candidate, None), candidate

        if self.submodule_paths:
            path = self.submodule_paths.get(mod)
            if path is None:
                if str(mod) == "ReLU()":
                    path = f"torch.nn.relu"
                else:
                    raise NameError("module is not installed as a submodule")
            assert isinstance(path, str)
            candidate = self._target_to_str(path)
            return self.namespace.create_name(candidate, None), candidate

        # O(N^2) fallback in the case that we didn't store the submodule
        # paths.

        else:
            for n, p in self.root.named_modules():
                if mod is n:
                    assert isinstance(p, str)
                    candidate = self._target_to_str(p)
                    return self.namespace.create_name(candidate, None), candidate
            raise NameError("module is not installed as a submodule")

    @compatibility(is_backward_compatible=True)
    def module_info(self, mod: torch.nn.Module) -> ModuleInfo:
        """ """
        nn_attrs = {k: v for k, v in vars(mod).items() if k.startswith("_")}
        nn_attr_hashes_ = {
            k: DeepHash(v)[v]
            for k, v in nn_attrs.items()
            if k not in ["_parameters", "_buffers"]
        }
        nn_para_hashes = {}
        for k, v in nn_attrs["_parameters"].items():
            if v is None:
                nn_para_hashes[k] = sha256(b"None").hexdigest()
            else:
                nn_para_hashes[k] = sha256(
                    v.detach().cpu().numpy().tobytes()
                ).hexdigest()
        nn_buf_hashes = {}
        for k, v in nn_attrs["_buffers"].items():
            if v is None:
                nn_buf_hashes[k] = sha256(b"None").hexdigest()
            else:
                nn_buf_hashes[k] = sha256(v.detach().numpy().tobytes()).hexdigest()
        nn_attr_hashes = {**nn_attr_hashes_, **nn_para_hashes, **nn_buf_hashes}

        return ModuleInfo(
            str(type(mod))[8:-2],
            # {k: v for k, v in vars(mod).items() if not k.startswith('_') and k != 'training'},
            {k: v for k, v in vars(mod).items() if k in layer_shape_info},
            nn_attr_hashes,
        )

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
        module_qualified_name, module_name = self.path_of_module(m)

        self.orig_forward = forward
        if "seq_len" in kwargs:
            kwargs["seq_len"] = 1
        if "use_cache" in kwargs:
            kwargs["use_cache"] = False
        if "attn_bias" in kwargs:
            kwargs["attn_bias"] = None
        if "is_causal" in kwargs:
            kwargs["is_causal"] = False
        if "dropout_p" in kwargs:
            kwargs["dropout_p"] = 0.0
        if "needs_weights" in kwargs:
            kwargs["needs_weights"] = False
        
        if not self.is_leaf_module(m, ""):

            if self.tracing_module_pool:
                for modulename in self.tracing_module_pool:
                    if modulename in str(type(m)).casefold():
                        self.submodule_info[module_qualified_name] = OrderedDict()

            out = forward(*args, **kwargs)

            if self.tracing_module_pool:
                for modulename in self.tracing_module_pool:
                    if (
                        modulename in str(type(m)).casefold()
                        and not "group" in str(type(m)).casefold()
                    ):
                        self.traced_module_info[
                            module_qualified_name
                        ] = UpperModuleInfo(
                            str(type(m))[8:-2],
                            self.submodule_info[module_qualified_name],
                        )
                        self.submodule_info = OrderedDict()
                        self.module_name_map[module_qualified_name] = module_name

                        return self.create_proxy(
                            "call_module", module_qualified_name, args, kwargs
                        )

            return out

        elif self.tracing_module_pool:
            if len(self.submodule_info) == 0:
                self.traced_module_info[module_qualified_name] = self.module_info(m)
                self.module_name_map[module_qualified_name] = module_name
            else:
                self.submodule_info[list(self.submodule_info)[-1]][
                    module_qualified_name
                ] = self.module_info(m)
                self.module_name_map[module_qualified_name] = module_name
        else:
            self.submodule_info[module_qualified_name] = self.module_info(m)
            self.module_name_map[module_qualified_name] = module_name

        return self.create_proxy("call_module", module_qualified_name, args, kwargs)