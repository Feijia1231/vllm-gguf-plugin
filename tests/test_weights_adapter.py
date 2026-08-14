# SPDX-License-Identifier: Apache-2.0

import regex
from transformers import PretrainedConfig

from vllm_gguf_plugin.weights_adapter import (
    GGUFWeightsAdapter,
    OLMoEGGUFAdapter,
    get_weights_adapter,
)


def make_olmoe_config(num_hidden_layers: int = 2) -> PretrainedConfig:
    return PretrainedConfig(
        model_type="olmoe",
        num_hidden_layers=num_hidden_layers,
    )


def test_olmoe_uses_standalone_adapter():
    adapter = get_weights_adapter(make_olmoe_config())

    assert type(adapter) is OLMoEGGUFAdapter


def test_olmoe_model_specific_mapping():
    adapter = OLMoEGGUFAdapter(make_olmoe_config(num_hidden_layers=2))

    name_map, sideload_params = adapter._get_model_specific_mapping(adapter.config)

    assert name_map == {
        "blk.0.ffn_down_exps.weight": (
            "model.layers.0.mlp.experts.0.down_proj.weight"
        ),
        "blk.0.ffn_gate_exps.weight": (
            "model.layers.0.mlp.experts.0.gate_proj.weight"
        ),
        "blk.0.ffn_up_exps.weight": (
            "model.layers.0.mlp.experts.0.up_proj.weight"
        ),
        "blk.1.ffn_down_exps.weight": (
            "model.layers.1.mlp.experts.0.down_proj.weight"
        ),
        "blk.1.ffn_gate_exps.weight": (
            "model.layers.1.mlp.experts.0.gate_proj.weight"
        ),
        "blk.1.ffn_up_exps.weight": (
            "model.layers.1.mlp.experts.0.up_proj.weight"
        ),
    }
    assert len(sideload_params) == 4
    assert all(isinstance(pattern, regex.Pattern) for pattern in sideload_params)


def test_olmoe_sideload_patterns_match_expected_parameters():
    adapter = OLMoEGGUFAdapter(make_olmoe_config(num_hidden_layers=1))
    _, sideload_params = adapter._get_model_specific_mapping(adapter.config)

    expected = [
        "model.layers.0.mlp.experts.3.gate_proj.weight",
        "model.layers.0.mlp.experts.3.up_proj.weight",
        "model.layers.0.mlp.experts.3.down_proj.weight",
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
    ]
    assert all(
        any(regex.fullmatch(pattern, name) for pattern in sideload_params)
        for name in expected
    )


def test_non_olmoe_falls_back_to_default_adapter():
    adapter = get_weights_adapter(PretrainedConfig(model_type="llama"))

    assert type(adapter) is GGUFWeightsAdapter
