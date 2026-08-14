# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import regex

from .default import GGUFWeightsAdapter


class OLMoEGGUFAdapter(GGUFWeightsAdapter):
    """Adapter for OLMoE-specific GGUF parameter mappings."""

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type == "olmoe"

    def _get_model_specific_mapping(
        self,
        config,
    ) -> tuple[dict[str, str], list[regex.Pattern]]:
        name_map: dict[str, str] = {}
        sideload_params: list[regex.Pattern] = []
        for idx in range(config.num_hidden_layers):
            name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
            )
            name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
            )
            name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
            )
            sideload_params.extend(
                [
                    regex.compile(
                        f"model\\.layers\\.{idx}"
                        r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
                    ),
                    regex.compile(
                        f"model\\.layers\\.{idx}"
                        r"\.mlp\.experts\.(gate_up_proj|down_proj)"
                    ),
                ]
            )
        return name_map, sideload_params
