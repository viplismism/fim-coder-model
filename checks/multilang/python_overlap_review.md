# python overlap review (licence headers only removed)

Rule: only a leading licence/copyright comment block is removed; all other comments are kept.

| check | pool file | other file | J before | J after | lines left | shared lines after | verdict |
|---|---|---|---|---|---|---|---|
| pool_vs_benchmark | hiyouga/LlamaFactory/src/llamafactory/third_party/muon/__init__.py | huggingface/transformers:src/transformers/quantizers/__init__.py | 0.747 | 0.0 | 2 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_benchmark | hiyouga/LlamaFactory/src/llamafactory/train/dpo/__init__.py | huggingface/transformers:src/transformers/quantizers/__init__.py | 0.747 | 0.0 | 2 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_benchmark | hiyouga/LlamaFactory/src/llamafactory/train/kto/__init__.py | huggingface/transformers:src/transformers/quantizers/__init__.py | 0.747 | 0.0 | 2 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_benchmark | hiyouga/LlamaFactory/src/llamafactory/train/ppo/__init__.py | huggingface/transformers:src/transformers/quantizers/__init__.py | 0.747 | 0.0 | 2 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_benchmark | hiyouga/LlamaFactory/src/llamafactory/train/pt/__init__.py | huggingface/transformers:src/transformers/quantizers/__init__.py | 0.747 | 0.0 | 2 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_benchmark | hiyouga/LlamaFactory/src/llamafactory/train/rm/__init__.py | huggingface/transformers:src/transformers/quantizers/__init__.py | 0.747 | 0.0 | 2 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_benchmark | hiyouga/LlamaFactory/src/llamafactory/train/sft/__init__.py | huggingface/transformers:src/transformers/quantizers/__init__.py | 0.747 | 0.0 | 2 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_benchmark | hiyouga/LlamaFactory/src/llamafactory/v1/plugins/trainer_plugins/lr_scheduler.py | huggingface/transformers:src/transformers/quantizers/__init__.py | 0.727 | 0.0 | 3 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_benchmark | hiyouga/LlamaFactory/src/llamafactory/train/hyper_parallel/__init__.py | huggingface/transformers:src/transformers/quantizers/__init__.py | 0.718 | 0.0 | 2 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_benchmark | hiyouga/LlamaFactory/src/llamafactory/train/megatron_bridge/__init__.py | huggingface/transformers:src/transformers/quantizers/__init__.py | 0.718 | 0.0 | 2 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_benchmark | hiyouga/LlamaFactory/src/llamafactory/chat/__init__.py | huggingface/transformers:src/transformers/quantizers/__init__.py | 0.704 | 0.0 | 3 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_benchmark | hiyouga/LlamaFactory/src/llamafactory/train/mca/__init__.py | huggingface/transformers:src/transformers/quantizers/__init__.py | 0.691 | 0.0 | 2 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_benchmark | hiyouga/LlamaFactory/src/llamafactory/v1/utils/callbacks/__init__.py | huggingface/transformers:src/transformers/quantizers/__init__.py | 0.651 | 0.0 | 8 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_benchmark | hiyouga/LlamaFactory/src/train.py | huggingface/transformers:src/transformers/quantizers/__init__.py | 0.613 | 0.0 | 8 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_benchmark | hiyouga/LlamaFactory/tests/train/megatron_bridge/test_config_builder.py | huggingface/transformers:src/transformers/quantizers/__init__.py | 0.578 | 0.0 | 159 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_benchmark | hiyouga/LlamaFactory/src/llamafactory/webui/common.py | huggingface/transformers:src/transformers/quantizers/__init__.py | 0.564 | 0.0 | 216 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_benchmark | hiyouga/LlamaFactory/src/llamafactory/webui/components/__init__.py | huggingface/transformers:src/transformers/quantizers/__init__.py | 0.546 | 0.0 | 16 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_benchmark | hiyouga/LlamaFactory/src/llamafactory/v1/plugins/trainer_plugins/distributed/fsdp2.py | huggingface/transformers:src/transformers/quantizers/__init__.py | 0.542 | 0.0 | 608 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_benchmark | hiyouga/LlamaFactory/src/llamafactory/model/loader.py | huggingface/transformers:src/transformers/quantizers/__init__.py | 0.529 | 0.0 | 196 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_heldout | hiyouga/LlamaFactory/src/llamafactory/third_party/muon/__init__.py | RVC-Boss/GPT-SoVITS/GPT_SoVITS/text/zh_normalization/__init__.py | 0.755 | 0.0 | 2 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_heldout | hiyouga/LlamaFactory/src/llamafactory/train/dpo/__init__.py | RVC-Boss/GPT-SoVITS/GPT_SoVITS/text/zh_normalization/__init__.py | 0.755 | 0.0 | 2 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_heldout | hiyouga/LlamaFactory/src/llamafactory/train/kto/__init__.py | RVC-Boss/GPT-SoVITS/GPT_SoVITS/text/zh_normalization/__init__.py | 0.755 | 0.0 | 2 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_heldout | hiyouga/LlamaFactory/src/llamafactory/train/ppo/__init__.py | RVC-Boss/GPT-SoVITS/GPT_SoVITS/text/zh_normalization/__init__.py | 0.755 | 0.0 | 2 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_heldout | hiyouga/LlamaFactory/src/llamafactory/train/pt/__init__.py | RVC-Boss/GPT-SoVITS/GPT_SoVITS/text/zh_normalization/__init__.py | 0.755 | 0.0 | 2 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_heldout | hiyouga/LlamaFactory/src/llamafactory/train/rm/__init__.py | RVC-Boss/GPT-SoVITS/GPT_SoVITS/text/zh_normalization/__init__.py | 0.755 | 0.0 | 2 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_heldout | hiyouga/LlamaFactory/src/llamafactory/train/sft/__init__.py | RVC-Boss/GPT-SoVITS/GPT_SoVITS/text/zh_normalization/__init__.py | 0.755 | 0.0 | 2 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_heldout | hiyouga/LlamaFactory/src/llamafactory/v1/plugins/trainer_plugins/lr_scheduler.py | RVC-Boss/GPT-SoVITS/GPT_SoVITS/text/zh_normalization/__init__.py | 0.735 | 0.0 | 3 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_heldout | hiyouga/LlamaFactory/src/llamafactory/train/hyper_parallel/__init__.py | RVC-Boss/GPT-SoVITS/GPT_SoVITS/text/zh_normalization/__init__.py | 0.725 | 0.0 | 2 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_heldout | hiyouga/LlamaFactory/src/llamafactory/train/megatron_bridge/__init__.py | RVC-Boss/GPT-SoVITS/GPT_SoVITS/text/zh_normalization/__init__.py | 0.725 | 0.0 | 2 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_heldout | hiyouga/LlamaFactory/src/llamafactory/chat/__init__.py | RVC-Boss/GPT-SoVITS/GPT_SoVITS/text/zh_normalization/__init__.py | 0.712 | 0.0 | 3 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_heldout | hiyouga/LlamaFactory/src/llamafactory/train/mca/__init__.py | RVC-Boss/GPT-SoVITS/GPT_SoVITS/text/zh_normalization/__init__.py | 0.698 | 0.0 | 2 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_heldout | hiyouga/LlamaFactory/src/llamafactory/v1/utils/callbacks/__init__.py | RVC-Boss/GPT-SoVITS/GPT_SoVITS/text/zh_normalization/__init__.py | 0.657 | 0.0 | 8 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_heldout | hiyouga/LlamaFactory/src/train.py | RVC-Boss/GPT-SoVITS/GPT_SoVITS/text/zh_normalization/__init__.py | 0.627 | 0.0 | 8 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_heldout | hiyouga/LlamaFactory/tests/train/megatron_bridge/test_config_builder.py | RVC-Boss/GPT-SoVITS/GPT_SoVITS/text/zh_normalization/__init__.py | 0.59 | 0.0 | 159 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_heldout | hiyouga/LlamaFactory/src/llamafactory/webui/common.py | RVC-Boss/GPT-SoVITS/GPT_SoVITS/text/zh_normalization/__init__.py | 0.576 | 0.0 | 216 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_heldout | hiyouga/LlamaFactory/src/llamafactory/v1/plugins/trainer_plugins/distributed/fsdp2.py | RVC-Boss/GPT-SoVITS/GPT_SoVITS/text/zh_normalization/__init__.py | 0.553 | 0.0 | 608 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_heldout | hiyouga/LlamaFactory/src/llamafactory/webui/components/__init__.py | RVC-Boss/GPT-SoVITS/GPT_SoVITS/text/zh_normalization/__init__.py | 0.55 | 0.0 | 16 | (none) | boilerplate exception: matching text is the licence header |
| pool_vs_heldout | hiyouga/LlamaFactory/src/llamafactory/model/loader.py | RVC-Boss/GPT-SoVITS/GPT_SoVITS/text/zh_normalization/__init__.py | 0.539 | 0.0 | 196 | (none) | boilerplate exception: matching text is the licence header |

## Exact 5-line windows shared with the training pool

- benchmark: 66 shared; 66 contain licence wording; 0 other (listed below)
- heldout: 11 shared; 11 contain licence wording; 0 other (listed below)
