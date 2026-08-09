from __future__ import annotations
import gradio as gr
import logging
import os
import re

import lora_patches
import unet_diffusers_map
import network
import network_lora
import network_glora
import network_hada
import network_ia3
import network_lokr
import network_full
import network_norm
import network_oft

import torch
from typing import Union

from modules import shared, devices, sd_models, errors, scripts, sd_hijack
import modules.textual_inversion.textual_inversion as textual_inversion
import modules.models.sd3.mmdit

from lora_logger import logger

module_types = [
    network_lora.ModuleTypeLora(),
    network_hada.ModuleTypeHada(),
    network_ia3.ModuleTypeIa3(),
    network_lokr.ModuleTypeLokr(),
    network_full.ModuleTypeFull(),
    network_norm.ModuleTypeNorm(),
    network_glora.ModuleTypeGLora(),
    network_oft.ModuleTypeOFT(),
]

_lora_loaded_log_fingerprint: tuple | None = None


def _is_clip_lora_sd_key(sd_key: str) -> bool:
    return "transformer" in sd_key[:20]


def _is_clip_lora_network_key(network_key: str) -> bool:
    k = network_key.replace(".", "_")
    if k.startswith("lora_te") or k.startswith("lycoris_te") or k.startswith("oft_te"):
        return True
    if "transformer" in k[:40]:
        return True
    return False


def _lora_apply_stats(net: network.Network):
    unet_loaded = 0
    clip_loaded = 0
    for mod in net.modules.values():
        if _is_clip_lora_sd_key(mod.sd_key):
            clip_loaded += 1
        else:
            unet_loaded += 1
    failed = getattr(net, "keys_failed_to_match", None) or []
    unet_skipped = sum(1 for k in failed if not _is_clip_lora_network_key(k))
    clip_skipped = sum(1 for k in failed if _is_clip_lora_network_key(k))
    return unet_loaded, clip_loaded, unet_skipped, clip_skipped


def _print_lora_loaded_line(filename, model_flag, target, loaded_count, weight, skipped_count, online_mode=False):
    if loaded_count == 0:
        return
    if skipped_count > 12:
        print(
            f"[LORA] Mismatch {filename} for {model_flag}-{target} with "
            f"{skipped_count} keys mismatched in {loaded_count} keys"
        )
    else:
        print(
            f"[LORA] Loaded {filename} for {model_flag}-{target} with {loaded_count} keys "
            f"at weight {weight} (skipped {skipped_count} keys) with on_the_fly = {online_mode}"
        )


def _log_lora_loaded_networks_if_changed():
    global _lora_loaded_log_fingerprint

    if not loaded_networks:
        _lora_loaded_log_fingerprint = ()
        return

    fingerprint = tuple(
        (net.network_on_disk.filename, net.unet_multiplier, net.te_multiplier, net.dyn_dim)
        for net in loaded_networks
    )
    if fingerprint == _lora_loaded_log_fingerprint:
        return

    _lora_loaded_log_fingerprint = fingerprint

    if hasattr(shared.sd_model, "model"):
        model_flag = type(shared.sd_model.model).__name__
    else:
        model_flag = "default"

    online_mode = False
    for net in loaded_networks:
        filename = net.network_on_disk.filename
        unet_loaded, clip_loaded, unet_skipped, clip_skipped = _lora_apply_stats(net)
        _print_lora_loaded_line(filename, model_flag, "UNet", unet_loaded, net.unet_multiplier, unet_skipped, online_mode)
        _print_lora_loaded_line(filename, model_flag, "CLIP", clip_loaded, net.te_multiplier, clip_skipped, online_mode)


re_digits = re.compile(r"\d+")
re_x_proj = re.compile(r"(.*)_([qkv]_proj)$")
re_compiled = {}

suffix_conversion = {
    "attentions": {},
    "resnets": {
        "conv1": "in_layers_2",
        "conv2": "out_layers_3",
        "norm1": "in_layers_0",
        "norm2": "out_layers_0",
        "time_emb_proj": "emb_layers_1",
        "conv_shortcut": "skip_connection",
    }
}

# Forge v1.7.4 (sd-webui-forge-classic-neo): SDXL CLIP-L supports both nested
# transformer.text_model.encoder.layers.* and flat transformer.encoder.layers.* (TF 5.x+).
_SDXL_CLIP_L_EMBEDDER_PREFIX = "0_"
_SDXL_CLIP_G_EMBEDDER_PREFIX = "1_"


def _sdxl_clip_l_layer_suffix_from_lora_key(key_network_without_network_parts: str) -> str | None:
    """Return layer suffix after lora_te1_/lora_te_text_model_ prefix, or None."""
    for prefix in ("lora_te1_text_model_encoder_layers_", "lora_te_text_model_encoder_layers_"):
        if key_network_without_network_parts.startswith(prefix):
            return key_network_without_network_parts[len(prefix):]
    return None


def _sdxl_clip_l_compvis_key_candidates(layer_suffix: str) -> list[str]:
    """CompVis module names for SDXL CLIP-L (embedder 0), nested and flat paths."""
    nested = f"{_SDXL_CLIP_L_EMBEDDER_PREFIX}transformer_text_model_encoder_layers_{layer_suffix}"
    flat = f"{_SDXL_CLIP_L_EMBEDDER_PREFIX}transformer_encoder_layers_{layer_suffix}"
    mapping = getattr(shared.sd_model, "network_layer_mapping", None)
    if mapping is not None and flat in mapping:
        return [flat, nested]
    return [nested, flat]


def _add_sdxl_clip_transformer_aliases(network_layer_mapping: dict) -> None:
    """Register alternate CLIP-L keys so lora_te1_* resolves for nested or flat transformers."""
    aliases = {}
    for network_name, module in list(network_layer_mapping.items()):
        if "transformer_text_model_encoder_layers" in network_name:
            alt = network_name.replace(
                "transformer_text_model_encoder_layers",
                "transformer_encoder_layers",
            )
            if alt not in network_layer_mapping:
                aliases[alt] = module
        elif (
            "transformer_encoder_layers" in network_name
            and "transformer_text_model_encoder_layers" not in network_name
        ):
            alt = network_name.replace(
                "transformer_encoder_layers",
                "transformer_text_model_encoder_layers",
            )
            if alt not in network_layer_mapping:
                aliases[alt] = module
    network_layer_mapping.update(aliases)


def _lookup_network_module(network_layer_mapping: dict, keys: list[str]):
    for key in keys:
        module = network_layer_mapping.get(key)
        if module is not None:
            return module
    return None


# Forge comfy/lora.py LORA_CLIP_MAP (sd-webui-forge-classic-neo v1.7.4)
LORA_CLIP_MAP = {
    "mlp.fc1": "mlp_fc1",
    "mlp.fc2": "mlp_fc2",
    "self_attn.k_proj": "self_attn_k_proj",
    "self_attn.q_proj": "self_attn_q_proj",
    "self_attn.v_proj": "self_attn_v_proj",
    "self_attn.out_proj": "self_attn_out_proj",
}
LORA_CLIP_SUFFIX_TO_HF = {v: k for k, v in LORA_CLIP_MAP.items()}

# OpenCLIP (SDXL CLIP-G) compvis suffix -> lora_te2_* suffix
_OPENCLIP_MLP_TO_LORA = {
    "mlp_c_fc": "mlp_fc1",
    "mlp_c_proj": "mlp_fc2",
}
_OPENCLIP_ATTN_TO_LORA = {
    "attn_q_proj": "self_attn_q_proj",
    "attn_k_proj": "self_attn_k_proj",
    "attn_v_proj": "self_attn_v_proj",
    "attn_out_proj": "self_attn_out_proj",
}

_sdxl_forge_lora_lookup: dict[str, tuple[str, torch.nn.Module]] | None = None

# SDXL UNet layout (Forge detection.py / Illustrious-class SDXL)
_SDXL_UNET_CONFIG_DEFAULT = {
    "num_res_blocks": [2, 2, 2],
    "channel_mult": [1, 2, 4],
    "transformer_depth": [0, 0, 2, 2, 10, 10],
    "transformer_depth_output": [0, 0, 0, 2, 2, 2, 10, 10, 10],
    "transformer_depth_middle": 10,
}


def _normalize_sdxl_unet_config(cfg: dict) -> dict:
    out = dict(cfg)
    num_blocks = len(out.get("channel_mult", _SDXL_UNET_CONFIG_DEFAULT["channel_mult"]))
    nrb = out.get("num_res_blocks", 2)
    if isinstance(nrb, int):
        out["num_res_blocks"] = [nrb] * num_blocks
    else:
        out["num_res_blocks"] = list(nrb)
    if "transformer_depth" in out:
        out["transformer_depth"] = list(out["transformer_depth"])
    if "transformer_depth_output" in out:
        out["transformer_depth_output"] = list(out["transformer_depth_output"])
    if "transformer_depth_middle" not in out:
        out["transformer_depth_middle"] = _SDXL_UNET_CONFIG_DEFAULT["transformer_depth_middle"]
    return out


def _get_sdxl_unet_config() -> dict:
    dm = getattr(getattr(shared.sd_model, "model", None), "diffusion_model", None)
    if dm is not None:
        raw = getattr(dm, "config", None)
        if isinstance(raw, dict) and "num_res_blocks" in raw:
            return _normalize_sdxl_unet_config(raw)
    return _normalize_sdxl_unet_config(_SDXL_UNET_CONFIG_DEFAULT)


def _compvis_suffix_to_network_key(compvis_suffix: str) -> str:
    return "diffusion_model_" + compvis_suffix.replace(".", "_")


def _diffusers_param_to_lora_suffix(diffusers_key: str) -> str:
    """Match Forge model_lora_keys_unet: strip .weight/.bias before underscore join."""
    if diffusers_key.endswith(".weight") or diffusers_key.endswith(".bias"):
        diffusers_key = diffusers_key[: -len(".weight")] if diffusers_key.endswith(".weight") else diffusers_key[: -len(".bias")]
    return diffusers_key.replace(".", "_")


def _register_diffusers_unet_aliases(
    lookup: dict[str, tuple[str, torch.nn.Module]],
    network_layer_mapping: dict,
    unet_config: dict,
) -> None:
    """Forge model_lora_keys_unet: map Diffusers-style lora_unet_* to compvis modules."""
    diffusers_map = unet_diffusers_map.unet_to_diffusers(unet_config)
    for diffusers_key, compvis_suffix in diffusers_map.items():
        compvis_key = _compvis_suffix_to_network_key(compvis_suffix)
        module = network_layer_mapping.get(compvis_key)
        if module is None:
            continue
        lora_key = _diffusers_param_to_lora_suffix(diffusers_key)
        _register_sdxl_lora_alias(lookup, f"lora_unet_{lora_key}", compvis_key, module)
        for prefix in ("lycoris_", "lycoris_unet_"):
            _register_sdxl_lora_alias(lookup, f"{prefix}{lora_key}", compvis_key, module)
        # Kohya / SimpleTuner diffusers-prefixed keys (Forge lora.py diffusers_lora_prefix loop)
        for p in ("", "unet."):
            diffusers_lora_key = f"{p}{diffusers_key[:-len('.weight')] if diffusers_key.endswith('.weight') else diffusers_key}"
            diffusers_lora_key = diffusers_lora_key.replace(".to_", ".processor.to_")
            if diffusers_lora_key.endswith(".to_out.0"):
                diffusers_lora_key = diffusers_lora_key[:-2]
            _register_sdxl_lora_alias(lookup, diffusers_lora_key, compvis_key, module)


def _invalidate_sdxl_forge_lora_lookup():
    global _sdxl_forge_lora_lookup
    _sdxl_forge_lora_lookup = None


def _is_openclip_attention(module) -> bool:
    if module is None:
        return False
    return (
        isinstance(getattr(module, "out_proj", None), torch.nn.Linear)
        and isinstance(getattr(module, "in_proj_weight", None), torch.nn.Parameter)
        and type(module).__name__ == "Attention"
        and str(type(module).__module__).startswith("open_clip")
    )


def _is_fused_qkv_attention_module(module) -> bool:
    return isinstance(module, torch.nn.MultiheadAttention) or _is_openclip_attention(module)


def _add_openclip_attention_virtual_keys(network_layer_mapping: dict) -> None:
    """Map lora_te2 self_attn_{q,k,v}_proj to parent OpenCLIP Attention (Forge parity)."""
    for network_name, module in list(network_layer_mapping.items()):
        if not _is_openclip_attention(module):
            continue
        for sfx in ("_q_proj", "_k_proj", "_v_proj", "_out_proj"):
            vk = f"{network_name}{sfx}"
            if vk not in network_layer_mapping:
                network_layer_mapping[vk] = module


def _register_sdxl_lora_alias(
    lookup: dict[str, tuple[str, torch.nn.Module]],
    lora_key: str,
    compvis_key: str,
    module: torch.nn.Module,
):
    if lora_key not in lookup:
        lookup[lora_key] = (compvis_key, module)


def _build_sdxl_forge_lora_lookup(network_layer_mapping: dict) -> dict[str, tuple[str, torch.nn.Module]]:
    lookup: dict[str, tuple[str, torch.nn.Module]] = {}

    for compvis_key, module in network_layer_mapping.items():
        if compvis_key.startswith("diffusion_model_"):
            suffix = compvis_key[len("diffusion_model_"):]
            _register_sdxl_lora_alias(lookup, f"lora_unet_{suffix}", compvis_key, module)
            _register_sdxl_lora_alias(lookup, compvis_key, compvis_key, module)

        clip_l_prefixes = (
            f"{_SDXL_CLIP_L_EMBEDDER_PREFIX}transformer_text_model_encoder_layers_",
            f"{_SDXL_CLIP_L_EMBEDDER_PREFIX}transformer_encoder_layers_",
        )
        for prefix in clip_l_prefixes:
            if not compvis_key.startswith(prefix):
                continue
            layer_rest = compvis_key[len(prefix):]
            m = re.match(r"(\d+)_(.+)", layer_rest)
            if not m:
                continue
            b, rest = m.group(1), m.group(2)
            _register_sdxl_lora_alias(
                lookup, f"lora_te1_text_model_encoder_layers_{b}_{rest}", compvis_key, module
            )
            _register_sdxl_lora_alias(
                lookup, f"lora_te_text_model_encoder_layers_{b}_{rest}", compvis_key, module
            )
            hf = LORA_CLIP_SUFFIX_TO_HF.get(rest)
            if hf is not None:
                _register_sdxl_lora_alias(
                    lookup, f"text_encoder.text_model.encoder.layers.{b}.{hf}", compvis_key, module
                )
                _register_sdxl_lora_alias(
                    lookup, f"text_encoder.encoder.layers.{b}.{hf}", compvis_key, module
                )

        g_prefix = f"{_SDXL_CLIP_G_EMBEDDER_PREFIX}model_transformer_resblocks_"
        if compvis_key.startswith(g_prefix):
            layer_rest = compvis_key[len(g_prefix):]
            m = re.match(r"(\d+)_(.+)", layer_rest)
            if not m:
                continue
            b, rest = m.group(1), m.group(2)
            lora_suffix = _OPENCLIP_MLP_TO_LORA.get(rest) or _OPENCLIP_ATTN_TO_LORA.get(rest)
            if lora_suffix is None:
                continue
            _register_sdxl_lora_alias(
                lookup, f"lora_te2_text_model_encoder_layers_{b}_{lora_suffix}", compvis_key, module
            )
            hf = LORA_CLIP_SUFFIX_TO_HF.get(lora_suffix)
            if hf is not None:
                _register_sdxl_lora_alias(
                    lookup, f"text_encoder_2.text_model.encoder.layers.{b}.{hf}", compvis_key, module
                )

    _register_diffusers_unet_aliases(lookup, network_layer_mapping, _get_sdxl_unet_config())

    return lookup


def _get_sdxl_forge_lora_lookup(network_layer_mapping: dict) -> dict[str, tuple[str, torch.nn.Module]]:
    global _sdxl_forge_lora_lookup
    if _sdxl_forge_lora_lookup is None:
        _sdxl_forge_lora_lookup = _build_sdxl_forge_lora_lookup(network_layer_mapping)
    return _sdxl_forge_lora_lookup


def _resolve_sdxl_lora_target(
    key_network_without_network_parts: str,
    network_layer_mapping: dict,
) -> tuple[str | None, torch.nn.Module | None]:
    hit = _get_sdxl_forge_lora_lookup(network_layer_mapping).get(key_network_without_network_parts)
    if hit is None:
        return None, None
    return hit[0], hit[1]


def convert_diffusers_name_to_compvis(key, is_sd2):
    def match(match_list, regex_text):
        regex = re_compiled.get(regex_text)
        if regex is None:
            regex = re.compile(regex_text)
            re_compiled[regex_text] = regex

        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, r"lora_unet_conv_in(.*)"):
        return f'diffusion_model_input_blocks_0_0{m[0]}'

    if match(m, r"lora_unet_conv_out(.*)"):
        return f'diffusion_model_out_2{m[0]}'

    if match(m, r"lora_unet_time_embedding_linear_(\d+)(.*)"):
        return f"diffusion_model_time_embed_{m[0] * 2 - 2}{m[1]}"

    if match(m, r"lora_unet_down_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_mid_block_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[0], {}).get(m[2], m[2])
        return f"diffusion_model_middle_block_{1 if m[0] == 'attentions' else m[1] * 2}_{suffix}"

    if match(m, r"lora_unet_up_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv"):
        return f"diffusion_model_input_blocks_{3 + m[0] * 3}_0_op"

    if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
        block_idx = 2 + m[0] * 3
        # SDXL: upsamplers always sit at type=2 (v2.1 fix). SD1.x/SD2: up_blocks_0 uses type=1.
        if shared.sd_model and getattr(shared.sd_model, "is_sdxl", False):
            type_idx = 2
        else:
            type_idx = 2 if m[0] > 0 else 1
        return f"diffusion_model_output_blocks_{block_idx}_{type_idx}_conv"

    if match(m, r"text_encoder\.text_model\.encoder\.layers\.(\d+)\.(.+)"):
        layer_suffix = f"{m[0]}_{m[1].replace('.', '_')}"
        if shared.sd_model and getattr(shared.sd_model, "is_sdxl", False):
            return f"{_SDXL_CLIP_L_EMBEDDER_PREFIX}transformer_text_model_encoder_layers_{layer_suffix}"
        return f"transformer_text_model_encoder_layers_{layer_suffix}"

    if match(m, r"text_encoder\.encoder\.layers\.(\d+)\.(.+)"):
        layer_suffix = f"{m[0]}_{m[1].replace('.', '_')}"
        if shared.sd_model and getattr(shared.sd_model, "is_sdxl", False):
            return f"{_SDXL_CLIP_L_EMBEDDER_PREFIX}transformer_encoder_layers_{layer_suffix}"
        return f"transformer_encoder_layers_{layer_suffix}"

    if match(m, r"text_encoder_2\.text_model\.encoder\.layers\.(\d+)\.(.+)"):
        layer_suffix = f"{m[0]}_{m[1].replace('.', '_')}"
        if shared.sd_model and getattr(shared.sd_model, "is_sdxl", False):
            return f"{_SDXL_CLIP_G_EMBEDDER_PREFIX}model_transformer_resblocks_{layer_suffix}"
        return f"model_transformer_resblocks_{layer_suffix}"

    if match(m, r"lora_te_text_model_encoder_layers_(\d+)_(.+)"):
        if is_sd2:
            if 'mlp_fc1' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
            elif 'mlp_fc2' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
            else:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

        layer_suffix = f"{m[0]}_{m[1]}"
        if shared.sd_model and getattr(shared.sd_model, "is_sdxl", False):
            return f"{_SDXL_CLIP_L_EMBEDDER_PREFIX}transformer_text_model_encoder_layers_{layer_suffix}"
        return f"transformer_text_model_encoder_layers_{layer_suffix}"

    if match(m, r"lora_te2_text_model_encoder_layers_(\d+)_(.+)"):
        if 'mlp_fc1' in m[1]:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
        elif 'mlp_fc2' in m[1]:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
        else:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

    return key


def assign_network_names_to_compvis_modules(sd_model):
    # Use the callback argument, not shared.sd_model: during weight reload /
    # VAE reload shared.sd_model can still be None while the passed model is valid.
    if sd_model is None:
        return

    network_layer_mapping = {}

    if sd_model.is_sdxl:
        for i, embedder in enumerate(sd_model.conditioner.embedders):
            if not hasattr(embedder, 'wrapped'):
                continue

            for name, module in embedder.wrapped.named_modules():
                network_name = f'{i}_{name.replace(".", "_")}'
                network_layer_mapping[network_name] = module
                module.network_layer_name = network_name

        _add_sdxl_clip_transformer_aliases(network_layer_mapping)
        _add_openclip_attention_virtual_keys(network_layer_mapping)
    else:
        cond_stage_model = getattr(sd_model.cond_stage_model, 'wrapped', sd_model.cond_stage_model)

        for name, module in cond_stage_model.named_modules():
            network_name = name.replace(".", "_")
            network_layer_mapping[network_name] = module
            module.network_layer_name = network_name

    for name, module in sd_model.model.named_modules():
        network_name = name.replace(".", "_")
        network_layer_mapping[network_name] = module
        module.network_layer_name = network_name

    sd_model.network_layer_mapping = network_layer_mapping
    _invalidate_sdxl_forge_lora_lookup()


class BundledTIHash(str):
    def __init__(self, hash_str):
        self.hash = hash_str

    def __str__(self):
        return self.hash if shared.opts.lora_bundled_ti_to_infotext else ''


def load_network(name, network_on_disk):
    net = network.Network(name, network_on_disk)
    net.mtime = os.path.getmtime(network_on_disk.filename)

    sd = sd_models.read_state_dict(network_on_disk.filename)

    # this should not be needed but is here as an emergency fix for an unknown error people are experiencing in 1.2.0
    if not hasattr(shared.sd_model, 'network_layer_mapping'):
        assign_network_names_to_compvis_modules(shared.sd_model)

    keys_failed_to_match = {}
    lora_bases_total: set[str] = set()
    lora_bases_failed: set[str] = set()
    is_sd2 = (
        not getattr(shared.sd_model, 'is_sdxl', False)
        and 'model_transformer_resblocks' in shared.sd_model.network_layer_mapping
    )
    lora_tensors_total = 0
    if hasattr(shared.sd_model, 'diffusers_weight_map'):
        diffusers_weight_map = shared.sd_model.diffusers_weight_map
    elif hasattr(shared.sd_model, 'diffusers_weight_mapping'):
        diffusers_weight_map = {}
        for k, v in shared.sd_model.diffusers_weight_mapping():
            diffusers_weight_map[k] = v
        shared.sd_model.diffusers_weight_map = diffusers_weight_map
    else:
        diffusers_weight_map = None

    matched_networks = {}
    bundle_embeddings = {}

    for key_network, weight in sd.items():

        if diffusers_weight_map:
            key_network_without_network_parts, network_name, network_weight = key_network.rsplit(".", 2)
            network_part = network_name + '.' + network_weight
        else:
            key_network_without_network_parts, _, network_part = key_network.partition(".")

        if key_network_without_network_parts == "bundle_emb":
            emb_name, vec_name = network_part.split(".", 1)
            emb_dict = bundle_embeddings.get(emb_name, {})
            if vec_name.split('.')[0] == 'string_to_param':
                _, k2 = vec_name.split('.', 1)
                emb_dict['string_to_param'] = {k2: weight}
            else:
                emb_dict[vec_name] = weight
            bundle_embeddings[emb_name] = emb_dict
            continue

        lora_tensors_total += 1
        lora_bases_total.add(key_network_without_network_parts)

        if diffusers_weight_map:
            key = diffusers_weight_map.get(key_network_without_network_parts, key_network_without_network_parts)
        else:
            key = convert_diffusers_name_to_compvis(key_network_without_network_parts, is_sd2)

        sd_module = None
        if getattr(shared.sd_model, "is_sdxl", False):
            forge_key, forge_module = _resolve_sdxl_lora_target(
                key_network_without_network_parts,
                shared.sd_model.network_layer_mapping,
            )
            if forge_module is not None:
                key = forge_key
                sd_module = forge_module

        if sd_module is None:
            sd_module = shared.sd_model.network_layer_mapping.get(key, None)

        if sd_module is None:
            m = re_x_proj.match(key)
            if m:
                sd_module = shared.sd_model.network_layer_mapping.get(m.group(1), None)

        if sd_module is None and "lora_unet" in key_network_without_network_parts:
            conv_key = convert_diffusers_name_to_compvis(key_network_without_network_parts, is_sd2)
            sd_module = shared.sd_model.network_layer_mapping.get(conv_key, None)
            if sd_module is not None:
                key = conv_key
        elif sd_module is None and "lora_te1_text_model" in key_network_without_network_parts:
            layer_suffix = _sdxl_clip_l_layer_suffix_from_lora_key(key_network_without_network_parts)
            if layer_suffix is not None:
                sd_module = _lookup_network_module(
                    shared.sd_model.network_layer_mapping,
                    _sdxl_clip_l_compvis_key_candidates(layer_suffix),
                )

            if sd_module is None:
                key = key_network_without_network_parts.replace("lora_te1_text_model", "0_transformer_text_model")
                sd_module = shared.sd_model.network_layer_mapping.get(key, None)

            # some SD1 Loras also have correct compvis keys
            if sd_module is None:
                key = key_network_without_network_parts.replace("lora_te1_text_model", "transformer_text_model")
                sd_module = shared.sd_model.network_layer_mapping.get(key, None)

        elif sd_module is None and "lora_te_text_model_encoder_layers_" in key_network_without_network_parts:
            layer_suffix = _sdxl_clip_l_layer_suffix_from_lora_key(key_network_without_network_parts)
            if layer_suffix is not None and getattr(shared.sd_model, "is_sdxl", False):
                sd_module = _lookup_network_module(
                    shared.sd_model.network_layer_mapping,
                    _sdxl_clip_l_compvis_key_candidates(layer_suffix),
                )

        # kohya_ss OFT module
        elif sd_module is None and "oft_unet" in key_network_without_network_parts:
            key = key_network_without_network_parts.replace("oft_unet", "diffusion_model")
            sd_module = shared.sd_model.network_layer_mapping.get(key, None)

        # KohakuBlueLeaf OFT module
        if sd_module is None and "oft_diag" in key_network_without_network_parts:
            oft_key = key_network_without_network_parts.replace("lora_unet", "diffusion_model")
            oft_key = oft_key.replace("lora_te1_text_model", "0_transformer_text_model")
            sd_module = shared.sd_model.network_layer_mapping.get(oft_key, None)
            if sd_module is not None:
                key = oft_key

        if sd_module is None:
            keys_failed_to_match[key_network] = key
            lora_bases_failed.add(key_network_without_network_parts)
            continue

        storage_key = key

        if storage_key not in matched_networks:
            matched_networks[storage_key] = network.NetworkWeights(network_key=key_network, sd_key=storage_key, w={}, sd_module=sd_module)

        matched_networks[storage_key].w[network_part] = weight

    if len(lora_bases_total) > 0 and len(lora_bases_failed) / len(lora_bases_total) > 0.5:
        model_flag = type(shared.sd_model.model).__name__ if hasattr(shared.sd_model, 'model') else 'default'
        logging.warning(
            f"[LORA] LoRA mismatch for {model_flag}: {network_on_disk.filename} "
            f"({len(lora_bases_failed)}/{len(lora_bases_total)} keys failed to match, skipping)"
        )
        skipped = network.Network(name, network_on_disk)
        skipped.mtime = net.mtime
        return skipped

    for key, weights in matched_networks.items():
        net_module = None
        for nettype in module_types:
            net_module = nettype.create_module(net, weights)
            if net_module is not None:
                break

        if net_module is None:
            raise AssertionError(f"Could not find a module type (out of {', '.join([x.__class__.__name__ for x in module_types])}) that would accept those keys: {', '.join(weights.w)}")

        net.modules[key] = net_module

    embeddings = {}
    for emb_name, data in bundle_embeddings.items():
        embedding = textual_inversion.create_embedding_from_data(data, emb_name, filename=network_on_disk.filename + "/" + emb_name)
        embedding.loaded = None
        embedding.shorthash = BundledTIHash(name)
        embeddings[emb_name] = embedding

    net.bundle_embeddings = embeddings

    if keys_failed_to_match:
        model_flag = type(shared.sd_model.model).__name__ if hasattr(shared.sd_model, 'model') else 'default'
        logging.warning(
            f"[LORA] Loading {network_on_disk.filename} for {model_flag} with "
            f"{len(keys_failed_to_match)}/{lora_tensors_total} unmatched keys"
        )
        logging.debug(f"Network {network_on_disk.filename} didn't match keys: {keys_failed_to_match}")

    net.keys_failed_to_match = keys_failed_to_match

    return net


def purge_networks_from_memory():
    while len(networks_in_memory) > shared.opts.lora_in_memory_limit and len(networks_in_memory) > 0:
        name = next(iter(networks_in_memory))
        networks_in_memory.pop(name, None)

    devices.torch_gc()


def load_networks(names, te_multipliers=None, unet_multipliers=None, dyn_dims=None):
    emb_db = sd_hijack.model_hijack.embedding_db
    already_loaded = {}

    for net in loaded_networks:
        if net.name in names:
            already_loaded[net.name] = net
        for emb_name, embedding in net.bundle_embeddings.items():
            if embedding.loaded:
                emb_db.register_embedding_by_name(None, shared.sd_model, emb_name)

    loaded_networks.clear()

    unavailable_networks = []
    for name in names:
        if name.lower() in forbidden_network_aliases and available_networks.get(name) is None:
            unavailable_networks.append(name)
        elif available_network_aliases.get(name) is None:
            unavailable_networks.append(name)

    if unavailable_networks:
        update_available_networks_by_names(unavailable_networks)

    networks_on_disk = [available_networks.get(name, None) if name.lower() in forbidden_network_aliases else available_network_aliases.get(name, None) for name in names]
    if any(x is None for x in networks_on_disk):
        list_available_networks()

        networks_on_disk = [available_networks.get(name, None) if name.lower() in forbidden_network_aliases else available_network_aliases.get(name, None) for name in names]

    failed_to_load_networks = []

    for i, (network_on_disk, name) in enumerate(zip(networks_on_disk, names)):
        net = already_loaded.get(name, None)

        if network_on_disk is not None:
            if net is None:
                net = networks_in_memory.get(name)

            if net is None or os.path.getmtime(network_on_disk.filename) > net.mtime:
                try:
                    net = load_network(name, network_on_disk)

                    networks_in_memory.pop(name, None)
                    networks_in_memory[name] = net
                except Exception as e:
                    errors.display(e, f"loading network {network_on_disk.filename}")
                    continue

            net.mentioned_name = name

            network_on_disk.read_hash()

        if net is None:
            failed_to_load_networks.append(name)
            logging.info(f"Couldn't find network with name {name}")
            continue

        net.te_multiplier = te_multipliers[i] if te_multipliers else 1.0
        net.unet_multiplier = unet_multipliers[i] if unet_multipliers else 1.0
        net.dyn_dim = dyn_dims[i] if dyn_dims else 1.0
        loaded_networks.append(net)

        for emb_name, embedding in net.bundle_embeddings.items():
            if embedding.loaded is None and emb_name in emb_db.word_embeddings:
                logger.warning(
                    f'Skip bundle embedding: "{emb_name}"'
                    ' as it was already loaded from embeddings folder'
                )
                continue

            embedding.loaded = False
            if emb_db.expected_shape == -1 or emb_db.expected_shape == embedding.shape:
                embedding.loaded = True
                emb_db.register_embedding(embedding, shared.sd_model)
            else:
                emb_db.skipped_embeddings[name] = embedding

    if failed_to_load_networks:
        lora_not_found_message = f'Lora not found: {", ".join(failed_to_load_networks)}'
        sd_hijack.model_hijack.comments.append(lora_not_found_message)
        if shared.opts.lora_not_found_warning_console:
            print(f'\n{lora_not_found_message}\n')
        if shared.opts.lora_not_found_gradio_warning:
            gr.Warning(lora_not_found_message)

    _log_lora_loaded_networks_if_changed()

    purge_networks_from_memory()


def allowed_layer_without_weight(layer):
    if isinstance(layer, torch.nn.LayerNorm) and not layer.elementwise_affine:
        return True

    if _is_openclip_attention(layer):
        return True

    return False


def store_weights_backup(weight):
    if weight is None:
        return None

    return weight.to(devices.cpu, copy=True)


def restore_weights_backup(obj, field, weight):
    if weight is None:
        setattr(obj, field, None)
        return

    getattr(obj, field).copy_(weight)


def network_restore_weights_from_backup(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.MultiheadAttention]):
    weights_backup = getattr(self, "network_weights_backup", None)
    bias_backup = getattr(self, "network_bias_backup", None)

    if weights_backup is None and bias_backup is None:
        return

    if weights_backup is not None:
        if _is_fused_qkv_attention_module(self):
            restore_weights_backup(self, 'in_proj_weight', weights_backup[0])
            restore_weights_backup(self.out_proj, 'weight', weights_backup[1])
        else:
            restore_weights_backup(self, 'weight', weights_backup)

    if _is_fused_qkv_attention_module(self):
        restore_weights_backup(self.out_proj, 'bias', bias_backup)
    else:
        restore_weights_backup(self, 'bias', bias_backup)


def network_apply_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.MultiheadAttention]):
    """
    Applies the currently selected set of networks to the weights of torch layer self.
    If weights already have this particular set of networks applied, does nothing.
    If not, restores original weights from backup and alters weights according to networks.
    """

    network_layer_name = getattr(self, 'network_layer_name', None)
    if network_layer_name is None:
        return

    current_names = getattr(self, "network_current_names", ())
    wanted_names = tuple((x.name, x.te_multiplier, x.unet_multiplier, x.dyn_dim) for x in loaded_networks)

    weights_backup = getattr(self, "network_weights_backup", None)
    if weights_backup is None and wanted_names != ():
        if current_names != () and not allowed_layer_without_weight(self):
            raise RuntimeError(f"{network_layer_name} - no backup weights found and current weights are not unchanged")

        if _is_fused_qkv_attention_module(self):
            weights_backup = (store_weights_backup(self.in_proj_weight), store_weights_backup(self.out_proj.weight))
        else:
            weights_backup = store_weights_backup(self.weight)

        self.network_weights_backup = weights_backup

    bias_backup = getattr(self, "network_bias_backup", None)
    if bias_backup is None and wanted_names != ():
        if _is_fused_qkv_attention_module(self) and self.out_proj.bias is not None:
            bias_backup = store_weights_backup(self.out_proj.bias)
        elif getattr(self, 'bias', None) is not None:
            bias_backup = store_weights_backup(self.bias)
        else:
            bias_backup = None

        # Unlike weight which always has value, some modules don't have bias.
        # Only report if bias is not None and current bias are not unchanged.
        if bias_backup is not None and current_names != ():
            raise RuntimeError("no backup bias found and current bias are not unchanged")

        self.network_bias_backup = bias_backup

    if current_names != wanted_names:
        network_restore_weights_from_backup(self)

        for net in loaded_networks:
            module = net.modules.get(network_layer_name, None)
            if module is not None and hasattr(self, 'weight') and not isinstance(module, modules.models.sd3.mmdit.QkvLinear):
                try:
                    with torch.no_grad():
                        if getattr(self, 'fp16_weight', None) is None:
                            weight = self.weight
                            bias = self.bias
                        else:
                            weight = self.fp16_weight.clone().to(self.weight.device)
                            bias = getattr(self, 'fp16_bias', None)
                            if bias is not None:
                                bias = bias.clone().to(self.bias.device)
                        updown, ex_bias = module.calc_updown(weight)

                        if len(weight.shape) == 4 and weight.shape[1] == 9:
                            # inpainting model. zero pad updown to make channel[1]  4 to 9
                            updown = torch.nn.functional.pad(updown, (0, 0, 0, 0, 0, 5))

                        self.weight.copy_((weight.to(dtype=updown.dtype) + updown).to(dtype=self.weight.dtype))
                        if ex_bias is not None and hasattr(self, 'bias'):
                            if self.bias is None:
                                self.bias = torch.nn.Parameter(ex_bias).to(self.weight.dtype)
                            else:
                                self.bias.copy_((bias + ex_bias).to(dtype=self.bias.dtype))
                except RuntimeError as e:
                    logging.debug(f"Network {net.name} layer {network_layer_name}: {e}")
                    extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1

                continue

            module_q = net.modules.get(network_layer_name + "_q_proj", None)
            module_k = net.modules.get(network_layer_name + "_k_proj", None)
            module_v = net.modules.get(network_layer_name + "_v_proj", None)
            module_out = net.modules.get(network_layer_name + "_out_proj", None)

            if _is_fused_qkv_attention_module(self) and (module_q or module_k or module_v or module_out):
                try:
                    with torch.no_grad():
                        if module_q or module_k or module_v:
                            qw, kw, vw = self.in_proj_weight.chunk(3, 0)
                            qkv_chunks = [qw, kw, vw]
                            updown_chunks = []
                            for idx, module in enumerate((module_q, module_k, module_v)):
                                if module is None:
                                    updown_chunks.append(torch.zeros_like(qkv_chunks[idx]))
                                else:
                                    updown, _ = module.calc_updown(qkv_chunks[idx])
                                    updown_chunks.append(updown)
                            del qw, kw, vw
                            self.in_proj_weight += torch.vstack(updown_chunks)

                        if module_out is not None:
                            updown_out, ex_bias = module_out.calc_updown(self.out_proj.weight)
                            self.out_proj.weight += updown_out
                            if ex_bias is not None:
                                if self.out_proj.bias is None:
                                    self.out_proj.bias = torch.nn.Parameter(ex_bias)
                                else:
                                    self.out_proj.bias += ex_bias

                except RuntimeError as e:
                    logging.debug(f"Network {net.name} layer {network_layer_name}: {e}")
                    extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1

                continue

            if isinstance(self, modules.models.sd3.mmdit.QkvLinear) and module_q and module_k and module_v:
                try:
                    with torch.no_grad():
                        # Send "real" orig_weight into MHA's lora module
                        qw, kw, vw = self.weight.chunk(3, 0)
                        updown_q, _ = module_q.calc_updown(qw)
                        updown_k, _ = module_k.calc_updown(kw)
                        updown_v, _ = module_v.calc_updown(vw)
                        del qw, kw, vw
                        updown_qkv = torch.vstack([updown_q, updown_k, updown_v])
                        self.weight += updown_qkv

                except RuntimeError as e:
                    logging.debug(f"Network {net.name} layer {network_layer_name}: {e}")
                    extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1

                continue

            if module is None:
                continue

            logging.debug(f"Network {net.name} layer {network_layer_name}: couldn't find supported operation")
            extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1

        self.network_current_names = wanted_names


def network_forward(org_module, input, original_forward):
    """
    Old way of applying Lora by executing operations during layer's forward.
    Stacking many loras this way results in big performance degradation.
    """

    if len(loaded_networks) == 0:
        return original_forward(org_module, input)

    input = devices.cond_cast_unet(input)

    network_restore_weights_from_backup(org_module)
    network_reset_cached_weight(org_module)

    y = original_forward(org_module, input)

    network_layer_name = getattr(org_module, 'network_layer_name', None)
    for lora in loaded_networks:
        module = lora.modules.get(network_layer_name, None)
        if module is None:
            continue

        y = module.forward(input, y)

    return y


def network_reset_cached_weight(self: Union[torch.nn.Conv2d, torch.nn.Linear]):
    self.network_current_names = ()
    self.network_weights_backup = None
    self.network_bias_backup = None


def network_Linear_forward(self, input):
    if shared.opts.lora_functional:
        return network_forward(self, input, originals.Linear_forward)

    network_apply_weights(self)

    return originals.Linear_forward(self, input)


def network_Linear_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    return originals.Linear_load_state_dict(self, *args, **kwargs)


def network_Conv2d_forward(self, input):
    if shared.opts.lora_functional:
        return network_forward(self, input, originals.Conv2d_forward)

    network_apply_weights(self)

    return originals.Conv2d_forward(self, input)


def network_Conv2d_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    return originals.Conv2d_load_state_dict(self, *args, **kwargs)


def network_GroupNorm_forward(self, input):
    if shared.opts.lora_functional:
        return network_forward(self, input, originals.GroupNorm_forward)

    network_apply_weights(self)

    return originals.GroupNorm_forward(self, input)


def network_GroupNorm_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    return originals.GroupNorm_load_state_dict(self, *args, **kwargs)


def network_LayerNorm_forward(self, input):
    if shared.opts.lora_functional:
        return network_forward(self, input, originals.LayerNorm_forward)

    network_apply_weights(self)

    return originals.LayerNorm_forward(self, input)


def network_LayerNorm_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    return originals.LayerNorm_load_state_dict(self, *args, **kwargs)


def network_MultiheadAttention_forward(self, *args, **kwargs):
    network_apply_weights(self)

    return originals.MultiheadAttention_forward(self, *args, **kwargs)


def network_MultiheadAttention_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    return originals.MultiheadAttention_load_state_dict(self, *args, **kwargs)


def network_OpenClipAttention_forward(self, *args, **kwargs):
    network_apply_weights(self)

    return originals.OpenClipAttention_forward(self, *args, **kwargs)


def network_OpenClipAttention_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    return originals.OpenClipAttention_load_state_dict(self, *args, **kwargs)


def process_network_files(names: list[str] | None = None):
    candidates = list(shared.walk_files(shared.cmd_opts.lora_dir, allowed_extensions=[".pt", ".ckpt", ".safetensors"]))
    candidates += list(shared.walk_files(shared.cmd_opts.lyco_dir_backcompat, allowed_extensions=[".pt", ".ckpt", ".safetensors"]))
    for filename in candidates:
        if os.path.isdir(filename):
            continue
        name = os.path.splitext(os.path.basename(filename))[0]
        # if names is provided, only load networks with names in the list
        if names and name not in names:
            continue
        try:
            entry = network.NetworkOnDisk(name, filename)
        except OSError:  # should catch FileNotFoundError and PermissionError etc.
            errors.report(f"Failed to load network {name} from {filename}", exc_info=True)
            continue

        available_networks[name] = entry

        if entry.alias in available_network_aliases:
            forbidden_network_aliases[entry.alias.lower()] = 1

        available_network_aliases[name] = entry
        available_network_aliases[entry.alias] = entry


def update_available_networks_by_names(names: list[str]):
    process_network_files(names)


def list_available_networks():
    available_networks.clear()
    available_network_aliases.clear()
    forbidden_network_aliases.clear()
    available_network_hash_lookup.clear()
    forbidden_network_aliases.update({"none": 1, "Addams": 1})

    os.makedirs(shared.cmd_opts.lora_dir, exist_ok=True)

    process_network_files()


re_network_name = re.compile(r"(.*)\s*\([0-9a-fA-F]+\)")


def infotext_pasted(infotext, params):
    if "AddNet Module 1" in [x[1] for x in scripts.scripts_txt2img.infotext_fields]:
        return  # if the other extension is active, it will handle those fields, no need to do anything

    added = []

    for k in params:
        if not k.startswith("AddNet Model "):
            continue

        num = k[13:]

        if params.get("AddNet Module " + num) != "LoRA":
            continue

        name = params.get("AddNet Model " + num)
        if name is None:
            continue

        m = re_network_name.match(name)
        if m:
            name = m.group(1)

        multiplier = params.get("AddNet Weight A " + num, "1.0")

        added.append(f"<lora:{name}:{multiplier}>")

    if added:
        params["Prompt"] += "\n" + "".join(added)


originals: lora_patches.LoraPatches = None

extra_network_lora = None

available_networks = {}
available_network_aliases = {}
loaded_networks = []
loaded_bundle_embeddings = {}
networks_in_memory = {}
available_network_hash_lookup = {}
forbidden_network_aliases = {}

list_available_networks()
