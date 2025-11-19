# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from warnings import warn

from llama_cookbook.configs import quantization_config as QUANT_CONFIG
from llama_cookbook.utils.config_utils import update_config
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    MllamaConfig,
    MllamaForConditionalGeneration,
)

from llama_cookbook.utils.action_model import LlamaForCausalLMWithActions


# Function to load the main model for text generation
def load_model(model_name, quantization, use_fast_kernels, **kwargs):
    if type(quantization) == type(True):
        warn(
            "Quantization (--quantization) is a boolean, please specify quantization as '4bit' or '8bit'. Defaulting to '8bit' but this might change in the future.",
            FutureWarning,
        )
        quantization = "8bit"

    bnb_config = None
    if quantization:
        quant_config = QUANT_CONFIG()
        update_config(quant_config, **kwargs)
        bnb_config = quant_config.create_bnb_config(quantization)

    print(f"use_fast_kernels{use_fast_kernels}")

    kwargs = {}
    if bnb_config:
        kwargs["quantization_config"] = bnb_config
    kwargs["device_map"] = "auto"
    kwargs["low_cpu_mem_usage"] = True
    kwargs["attn_implementation"] = "sdpa" if use_fast_kernels else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        **kwargs,
    )
    return model


# Function to load the PeftModel for performance optimization
def load_peft_model(model, peft_model):
    peft_model = PeftModel.from_pretrained(model, peft_model)
    return peft_model


# Loading the model from config to load FSDP checkpoints into that
def load_llama_from_config(config_path, **config_overrides):
    config = AutoConfig.from_pretrained(config_path)
    for key, value in config_overrides.items():
        setattr(config, key, value)
    if config.model_type == "mllama":
        model = MllamaForConditionalGeneration(config=config)
    elif config.model_type == "llama":
        use_action_head = bool(getattr(config, "use_action_head", False))
        if not use_action_head:
            for attr in ("action_head_output_dim", "action_head_hidden_dim", "action_head_num_layers"):
                if hasattr(config, attr):
                    use_action_head = True
                    break
        if use_action_head:
            model = LlamaForCausalLMWithActions(config=config)
        else:
            model = LlamaForCausalLM(config=config)
    else:
        raise ValueError(
            f"Unsupported model type: {config.model_type}, Please use llama or mllama model."
        )
    return model
