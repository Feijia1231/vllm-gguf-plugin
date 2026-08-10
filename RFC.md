# [RFC] A Possible Next Step for GGUF Model Adapters

## Summary

PR #97 introduces the new `GGUFModelFiles` and `GGUFLoadPlan` based adapter
interface. PR #98 builds on that refactor to add Qwen3.5/3.6 multimodal GGUF
and MTP support, with dedicated adapters for dense, MoE, vision, and MTP
models.

While looking at the changes in these two PRs, I was wondering whether the
adapter design could be made a little easier to extend as more architectures
are added.

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

## Possible Direction

One possible direction would be to keep the `GGUFModelFiles` and
`GGUFLoadPlan` interfaces introduced by PR #97, and gradually move some of
the model-specific behavior toward reusable transformations:

```text
GGUFModelFiles
    → GGUFLoadPlan
    → name mapping
    → tensor transformations
    → model.load_weights()
```

For example, QKV splitting, MoE expert expansion, and multimodal projector
mapping could potentially be shared across adapters instead of being
implemented separately each time.

The existing Transformers-based mapping can remain as a fallback. Explicit
architecture mappings can be added incrementally where they make model
support more predictable and reduce dependence on dummy Transformers models.

It may also be useful for diffusion loading to share the same file, plan, and
transformation concepts in the future, without changing its current
integration boundary all at once.

## Possible Next Step

If this direction seems useful after PR #97 and PR #98, a first small step
could be:

1. Look for transformations shared by the current dense, MoE, vision, and
   MTP adapters.
2. Extract one or two of them into small reusable components.
3. Try the components with another model family.
4. Add regression tests for the relevant transformations, including
   tensor-parallel and sharded GGUF loading.

Ideally, adding another model family would mostly involve a localized name
mapping and a small number of model-specific transformations, rather than
another large branch in the generic adapter. If this direction makes sense,
I would be happy to help work on the first step.

## References

- [PR #97: Refactor WeightsAdapter design](https://github.com/vllm-project/vllm-gguf-plugin/pull/97)
- [PR #98: Support Qwen3.5/3.6 multimodal GGUF and MTP](https://github.com/vllm-project/vllm-gguf-plugin/pull/98)
