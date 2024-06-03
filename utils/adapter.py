import logging
import torch
import torch.nn as nn
import copy
from transformers.activations import ACT2FN

__all__ = [
    "add_config",
    "add_adapters",
    "freeze_all_parameters",
    "unfreeze_adapters",
]
logging.basicConfig(level=logging.INFO)


class Adapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.down_project = torch.nn.Linear(config.hidden_size, config.adapter_size)
        nn.init.normal_(self.down_project.weight, std=config.adapter_initializer_range)
        nn.init.zeros_(self.down_project.bias)
        self.down_project.adapter = True

        if isinstance(config.adapter_act, str):
            self.activation = copy.copy(ACT2FN[config.adapter_act])
        else:
            self.activation = copy.copy(config.adapter_act)
        self.activation.adapter = True

        self.up_project = nn.Linear(config.adapter_size, config.hidden_size)
        nn.init.normal_(self.up_project.weight, std=config.adapter_initializer_range)
        nn.init.zeros_(self.up_project.bias)
        self.up_project.adapter = True

    def forward(self, hidden_states):
        down_projected = self.down_project(hidden_states)
        activated = self.activation(down_projected)
        up_projected = self.up_project(activated)

        return hidden_states + up_projected


class AdapterSelfOutput(nn.Module):
    def __init__(self, config, self_output):
        super().__init__()
        self.self_out = self_output
        self.adapter = Adapter(config=config)

    def forward(self, *args):
        if len(args) == 2:
            hidden_states, input_tensor = args
            hidden_states = self.self_out.dense(hidden_states)
            hidden_states = self.self_out.dropout(hidden_states)
            hidden_states = self.adapter(hidden_states)
            hidden_states = self.self_out.LayerNorm(hidden_states + input_tensor)

        else:
            input_tensor = args[0]
            ffn_output = self.self_out(input_tensor)
            hidden_states = self.adapter(ffn_output)

        return hidden_states


def adapter_self_output(config):
    return lambda self_output: AdapterSelfOutput(config, self_output)


def add_config(model, config):
    for attr, value in config.items():
        model.config.__setattr__(attr, value)


def add_adapters(model, config):
    for submodule in model.children():
        if hasattr(submodule, "encoder") or hasattr(submodule, "transformer"):
            try:
                if (
                    "albert" in model.config._name_or_path
                    or "albert" in model.config.architectures[0].casefold()
                ):
                    for alberta_layer_group in submodule.encoder.albert_layer_groups:
                        for layer in alberta_layer_group.albert_layers:
                            layer.attention.output_dropout = adapter_self_output(
                                config
                            )(layer.attention.output_dropout)
                            layer.ffn_output = adapter_self_output(config)(
                                layer.ffn_output
                            )
                    return
            except:
                pass

            if hasattr(submodule, "encoder"):
                layers = submodule.encoder.layer
            else:
                layers = submodule.transformer.layer

            for layer in layers:
                if hasattr(layer.attention, "out_lin"):
                    layer.attention.out_lin = adapter_self_output(config)(
                        layer.attention.out_lin
                    )
                else:
                    layer.attention.output = adapter_self_output(config)(
                        layer.attention.output
                    )
                if hasattr(layer, "ffn"):
                    layer.ffn.dropout = adapter_self_output(config)(layer.ffn.dropout)
                else:
                    layer.output = adapter_self_output(config)(layer.output)
            return


def freeze_all_parameters(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_all_parameters(model):
    for param in model.parameters():
        param.requires_grad = True


def unfreeze_adapters(model):
    for submodule in model.modules():
        # TODO: make sure if last submodule is always the head
        if isinstance(
            submodule, (Adapter, nn.LayerNorm, type(list(model.children())[-1]))
        ):
            for param_name, param in submodule.named_parameters():
                param.requires_grad = True
