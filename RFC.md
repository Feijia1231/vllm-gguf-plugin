# [RFC] Generalize GGUF Model Adapters Beyond the Current Model Families

## Summary

PR #97 introduces the new `GGUFModelFiles` and `GGUFLoadPlan` based adapter
interface. PR #98 builds on that refactor to add Qwen3.5/3.6 multimodal GGUF
and MTP support, with dedicated adapters for dense, MoE, vision, and MTP
models.

This RFC proposes a follow-up step after those changes: make the adapter
design easier to extend without adding another large model-specific branch
for every new architecture.

A longer-term goal is to make it practical to support a broader range of
GGUF models published by Unsloth, including dense, MoE, multimodal, and
diffusion models:

https://huggingface.co/models?library=gguf&sort=trending&search=unsloth

## Motivation

The current refactor makes the loading lifecycle clearer, but model support
still depends on architecture-specific behavior inside adapters.

The model families being added now already require different combinations
of:

- QKV splitting or merging;
- merged-column splitting;
- MoE expert expansion;
- multimodal projector handling;
- MTP-specific fallback behavior;
- tensor-name transformations.

As more GGUF models are added, these operations may become increasingly
difficult to share between adapters. Diffusion loading also continues to
use a separate integration path.

## Proposed Direction

Keep the `GGUFModelFiles` and `GGUFLoadPlan` interfaces introduced by PR #97
as the foundation, and move the remaining model-specific behavior toward
reusable transformations:

```text
GGUFModelFiles
    → GGUFLoadPlan
    → name mapping
    → tensor transformations
    → model.load_weights()
```

Transformations such as QKV splitting, MoE expert expansion, and multimodal
projector mapping should be reusable across model adapters instead of being
implemented repeatedly in architecture-specific code.

The existing Transformers-based mapping can remain as a fallback. Explicit
architecture mappings can be added incrementally where they make model
support more predictable and reduce dependence on dummy Transformers models.

Diffusion loading should eventually use the same file, plan, and
transformation concepts, while keeping its current integration boundary
unchanged during the migration.

## Suggested Fix

After PR #97 and PR #98:

1. Identify transformations shared by the current dense, MoE, vision, and
   MTP adapters.
2. Extract those transformations into small reusable components.
3. Add a second model family using the shared components instead of adding
   another special-case implementation.
4. Add regression tests covering the relevant transformations, including
   tensor-parallel and sharded GGUF loading.
5. Use representative models from the Unsloth GGUF collection to guide the
   next adapter additions.

The goal is not to add a special-case adapter for every model in the
collection. A new model family should require a localized mapping and a
small number of model-specific transformations.

## References

- [PR #97: Refactor WeightsAdapter design](https://github.com/vllm-project/vllm-gguf-plugin/pull/97)
- [PR #98: Support Qwen3.5/3.6 multimodal GGUF and MTP](https://github.com/vllm-project/vllm-gguf-plugin/pull/98)
