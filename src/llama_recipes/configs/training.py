# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class train_config:
    model_name: str="PATH/to/Model"
    tokenizer_name: str=None
    enable_fsdp: bool=False # shards model parameters, optimizer states and gradients across DDP ranks
    low_cpu_fsdp: bool=False # saves cpu memory by loading pretrained model on rank0 only
    run_validation: bool=True
    batch_size_training: int=4
    batching_strategy: str="padding" #alternative: padding or packing
    context_length: int=4096
    gradient_accumulation_steps: int=64
    gradient_clipping: bool = True
    gradient_clipping_threshold: float = 1.0
    num_epochs: int=2
    max_train_step: int=0
    max_eval_step: int=0
    num_workers_dataloader: int=1
    lr: float=2e-5
    weight_decay: float=0.0
    gamma: float=0.85 # multiplicatively decay the learning rate by gamma after each epoch
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=48
    dataset = "flores_dataset"
    subset_name = "wmt-da-17-22.csv"
    split = "test"
    peft_method: str = "lora" # None, llama_adapter (Caution: llama_adapter is currently not supported with FSDP)
    use_peft: bool=False # use parameter efficient fine tuning
    from_peft_checkpoint: str="" # if not empty and use_peft=True, will load the peft checkpoint and resume the fine-tuning on that checkpoint
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: str = None
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    use_wandb: bool = False # Enable wandb for experient tracking
    save_metrics: bool = False # saves training metrics to a json file for later plotting
    flop_counter: bool = False # Enable flop counter to measure model throughput, can not be used with pytorch profiler at the same time.
    flop_counter_start: int = 3 # The step to start profiling, default is 3, which means after 3 steps of warmup stage, the profiler will start to count flops.
    use_profiler: bool = False # Enable pytorch profiler, can not be used with flop counter at the same time.
    profiler_dir: str = "PATH/to/save/profiler/results" # will be used if using profiler
    lang_pairs: str = "en-de,en-zh,en-ru"
    preload_peft_dir: str = None
    beam_size: int = 1
    normalized_value_loss: bool = False
    preferred_finetune: bool = False
    z_value_loss: bool = False
    scaled_value_loss: bool = False
    listwise_loss: bool = False
    normalize_prob: bool = False
    list_size: int = 16
    bon_size: int = 16
    alpha: float = 0
    beta: float = 1
    gama: float = 0
    xpo_hyper: float = 15
    use_beam_search: bool = True
    num_return_samples: int = 10
    num_sampling_size: int = 1
    subset_size: int = -1
    metric: str = "kiwi-xxl.score"






