from typing import Any, cast

import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookedRootModule
from transformers import LlamaForCausalLM, AutoTokenizer,LlamaTokenizer

def load_model(
    model_class_name: str,
    model_name: str,
    device: str | torch.device | None = None,
    model_from_pretrained_kwargs: dict[str, Any] | None = None,
) -> HookedRootModule:
    model_from_pretrained_kwargs = model_from_pretrained_kwargs or {}

    if "n_devices" in model_from_pretrained_kwargs:
        n_devices = model_from_pretrained_kwargs["n_devices"]
        if n_devices > 1:
            print("MODEL LOADING:")
            print("Setting model device to cuda for d_devices")
            print(f"Will use cuda:0 to cuda:{n_devices-1}")
            device = "cuda"
            print("-------------")

    if model_class_name == "HookedTransformer":
        return HookedTransformer.from_pretrained(
            model_name=model_name, device=device, **model_from_pretrained_kwargs
        )
    elif model_class_name == "LlamaForCausalLM":
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        hf_model = LlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
        model = HookedTransformer.from_pretrained(
            model_name,
            hf_model=hf_model,
            device="cuda",
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            tokenizer=tokenizer,
            n_devices=4
        )
        # model = torch.nn.DataParallel(model)
        # model = model.to(device)
        return model
    elif model_class_name == "HookedMamba":
        try:
            from mamba_lens import HookedMamba
        except ImportError:  # pragma: no cover
            raise ValueError(
                "mamba-lens must be installed to work with mamba models. This can be added with `pip install sae-lens[mamba]`"
            )
        # HookedMamba has incorrect typing information, so we need to cast the type here
        return cast(
            HookedRootModule,
            HookedMamba.from_pretrained(
                model_name, device=cast(Any, device), **model_from_pretrained_kwargs
            ),
        )
    else:  # pragma: no cover
        raise ValueError(f"Unknown model class: {model_class_name}")
