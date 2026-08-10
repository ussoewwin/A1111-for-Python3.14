"""
SDXL INT8 ConvRot + NVFP4 ConvRot support for A1111 (完全隔離型).

チェックポイントが comfy_quant + convrot マーカーを含む場合のみ動作。
int8 weight + weight_scale を float16 に逆量子化し、
ConvRot の Hadamard 回転を元に戻してから通常ロードパスに渡す。
NVFP4 (E2M1 packed) weight も comfy_kitchen で逆量子化する。

SDXL 以外、INT8/NVFP4 以外、ConvRot 以外のモデルには一切影響しない。
approach: offline dequantize + unrotate at load time → standard float model
"""

from __future__ import annotations

import json
import logging
import math

import torch

logger = logging.getLogger(__name__)

_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}

_LOG = "[INT8 ConvRot]"

# ---------------------------------------------------------------------------
# Hadamard matrix (same as native_convert_int8.py / comfy_kitchen)
# ---------------------------------------------------------------------------

def build_hadamard(size: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """Normalized regular Hadamard matrix (power-of-4)."""
    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1],
         [1, 1, -1, 1],
         [1, -1, 1, 1],
         [-1, 1, 1, 1]],
        dtype=dtype, device=device,
    )
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    h = h / (size ** 0.5)
    _HADAMARD_CACHE[cache_key] = h
    return h


def _valid_group_size(n: int, preferred: int = 256) -> int | None:
    """Return valid Hadamard group size (power of 4, divides n), or None."""
    if preferred is not None and preferred >= 4 and n % preferred == 0:
        if (preferred & (preferred - 1)) == 0 and math.log(preferred, 4) % 1 == 0:
            return preferred
    for gs in [256, 64, 16, 4]:
        if n % gs == 0:
            return gs
    return None


# ---------------------------------------------------------------------------
# Unrotate (inverse of ConvRot Hadamard rotation)
# H is symmetric + orthogonal: H^T = H, H^{-1} = H
# So unrotate = rotate = multiply by H along group dimension
# ---------------------------------------------------------------------------

def unrotate_weight(weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int) -> torch.Tensor:
    """Inverse Linear rotation: W = W_rot @ H.
    weight: (out_features, in_features)
    """
    out_features, in_features = weight.shape
    w = weight.reshape(out_features, in_features // group_size, group_size)
    w = torch.einsum("okg,gh->okh", w, h_matrix)
    return w.reshape(out_features, in_features)


def unrotate_weight_conv2d(weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int) -> torch.Tensor:
    """Inverse Conv2d rotation: unrotate along in_channels.
    weight: (O, I, kH, kW)
    """
    out_ch, in_ch, kH, kW = weight.shape
    flat = weight.reshape(out_ch, in_ch, kH * kW)
    flat = flat.permute(0, 2, 1).reshape(out_ch * kH * kW, in_ch)
    flat_un = unrotate_weight(flat, h_matrix, group_size)
    return flat_un.reshape(out_ch, kH * kW, in_ch).permute(0, 2, 1).reshape(out_ch, in_ch, kH, kW)


# ---------------------------------------------------------------------------
# comfy_quant metadata decode
# ---------------------------------------------------------------------------

def decode_comfy_quant_conf(raw) -> dict:
    """Decode a comfy_quant tensor (uint8 JSON bytes) into a dict."""
    if isinstance(raw, torch.Tensor):
        raw = raw.tolist()
    if isinstance(raw, (list, tuple)):
        raw = bytes(raw).decode("utf-8")
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, dict):
        return raw
    raise TypeError(f"comfy_quant config must be dict/str/bytes, got {type(raw).__name__}")


def _get_layer_conf(conf: dict, prefix: str) -> dict:
    """Extract per-layer config from comfy_quant conf."""
    if not isinstance(conf, dict):
        return {}
    if "format" in conf:
        return conf
    if "layers" in conf and isinstance(conf["layers"], dict):
        layer_conf = conf["layers"].get(prefix, {})
        if isinstance(layer_conf, dict) and "format" in layer_conf:
            return layer_conf
    return conf


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def state_dict_has_comfy_quant(state_dict: dict) -> bool:
    """Quick check: does state_dict have any .comfy_quant keys?"""
    for key in state_dict:
        if key.endswith(".comfy_quant"):
            return True
    return False


# ---------------------------------------------------------------------------
# INT8 dequantize + unrotate
# ---------------------------------------------------------------------------

def dequantize_int8_state_dict(
    state_dict: dict,
    target_dtype: torch.dtype = torch.float16,
) -> dict:
    """In-place: convert all int8_tensorwise (plain + convrot) weights to float.

    For each layer with .comfy_quant:
      - Read .weight (int8) and .weight_scale (float32)
      - Dequantize: w_float = weight.float() * weight_scale
      - If convrot: unrotate with Hadamard matrix
      - Convert to target_dtype
      - Replace state_dict[weight_key] = w_float
      - Remove .weight_scale, .comfy_quant, .input_scale, .weight_scale_2,
        .pre_quant_scale keys

    Non-int8 weights are silently skipped (already_float counter).
    Returns dict with stats for logging.
    """
    stats = {
        "plain": 0,
        "convrot_linear": 0,
        "convrot_conv2d": 0,
        "skipped": 0,
        "already_float": 0,
    }

    cq_keys = [k for k in list(state_dict.keys()) if k.endswith(".comfy_quant")]
    if not cq_keys:
        return stats

    for cq_key in cq_keys:
        prefix = cq_key[: -len(".comfy_quant")]
        weight_key = prefix + ".weight"
        scale_key = prefix + ".weight_scale"
        input_scale_key = prefix + ".input_scale"
        weight_scale_2_key = prefix + ".weight_scale_2"
        pre_quant_scale_key = prefix + ".pre_quant_scale"

        sidecar_keys = [cq_key, scale_key, input_scale_key, weight_scale_2_key, pre_quant_scale_key]

        if weight_key not in state_dict:
            stats["skipped"] += 1
            for k in sidecar_keys:
                state_dict.pop(k, None)
            continue

        weight = state_dict[weight_key]

        if weight.dtype not in (torch.int8, torch.uint8):
            stats["already_float"] += 1
            for k in sidecar_keys:
                state_dict.pop(k, None)
            continue

        scale = state_dict.get(scale_key)

        try:
            raw_conf = decode_comfy_quant_conf(state_dict[cq_key])
            conf = _get_layer_conf(raw_conf, prefix)
        except Exception:
            conf = {}

        fmt = conf.get("format", "")
        if fmt != "int8_tensorwise":
            stats["skipped"] += 1
            for k in sidecar_keys:
                state_dict.pop(k, None)
            continue

        is_convrot = conf.get("convrot", False)
        group_size = conf.get("convrot_groupsize", 256)

        w_float = weight.float()
        if scale is not None:
            w_float = w_float * scale.float()

        if is_convrot:
            if w_float.ndim == 2:
                gs = _valid_group_size(w_float.shape[1], group_size)
                if gs is not None:
                    H = build_hadamard(gs, device=w_float.device, dtype=torch.float32)
                    w_float = unrotate_weight(w_float, H, gs)
                    stats["convrot_linear"] += 1
                else:
                    stats["plain"] += 1
            elif w_float.ndim == 4:
                in_ch = w_float.shape[1]
                gs = _valid_group_size(in_ch, group_size)
                if gs is not None:
                    H = build_hadamard(gs, device=w_float.device, dtype=torch.float32)
                    w_float = unrotate_weight_conv2d(w_float, H, gs)
                    stats["convrot_conv2d"] += 1
                else:
                    stats["plain"] += 1
            else:
                stats["plain"] += 1
        else:
            stats["plain"] += 1

        state_dict[weight_key] = w_float.to(target_dtype)

        for k in sidecar_keys:
            state_dict.pop(k, None)

    return stats


# ---------------------------------------------------------------------------
# NVFP4 dequantize + unrotate
# ---------------------------------------------------------------------------

def _dequantize_nvfp4_layer(
    state_dict: dict,
    prefix: str,
    conf: dict,
    target_dtype: torch.dtype,
) -> str:
    """Dequantize a single NVFP4 Linear layer in-place.

    Returns "nvfp4_plain" or "nvfp4_convrot" on success, "skipped" on failure.
    """
    weight_key = prefix + ".weight"
    scale_key = prefix + ".weight_scale"        # block scales (float8_e4m3fn)
    scale_2_key = prefix + ".weight_scale_2"    # per-tensor scale (float32)
    input_scale_key = prefix + ".input_scale"
    pre_quant_scale_key = prefix + ".pre_quant_scale"

    weight = state_dict.get(weight_key)
    block_scale = state_dict.get(scale_key)
    tensor_scale = state_dict.get(scale_2_key)

    if weight is None or block_scale is None or tensor_scale is None:
        return "skipped"

    # Import comfy_kitchen for dequantize_nvfp4
    try:
        import comfy_kitchen as ck
    except ImportError:
        print(f"[NVFP4] ERROR: comfy_kitchen not available, cannot dequantize {prefix}")
        return "skipped"

    # Dequantize: packed uint8 → float (padded shape)
    # ck.dequantize_nvfp4(qx, per_tensor_scale, block_scales, output_type, hi_first)
    # hi_first defaults to True (matches nvfp4_conf.py / ck default)
    hi_first = conf.get("hi_first", True)
    output_type = torch.float32  # dequantize in fp32 for accuracy, convert later

    w_float = ck.dequantize_nvfp4(
        weight,
        tensor_scale.float(),
        block_scale,
        output_type=output_type,
        hi_first=hi_first,
    )

    # Crop from padded shape to logical orig_shape
    orig_shape = conf.get("orig_shape")
    if orig_shape and len(orig_shape) >= 2:
        out_f = int(orig_shape[0])
        in_f = int(orig_shape[1])
        if w_float.shape[0] > out_f or w_float.shape[1] > in_f:
            w_float = w_float[:out_f, :in_f]
    elif hasattr(w_float, "shape") and w_float.ndim == 2:
        # No orig_shape in conf — try in_features / out_features
        out_f = conf.get("out_features")
        in_f = conf.get("in_features")
        if out_f and in_f:
            w_float = w_float[:int(out_f), :int(in_f)]

    # ConvRot unrotate
    is_convrot = conf.get("convrot", False)
    group_size = conf.get("convrot_groupsize", 256)

    if is_convrot and w_float.ndim == 2:
        gs = _valid_group_size(w_float.shape[1], group_size)
        if gs is not None:
            H = build_hadamard(gs, device=w_float.device, dtype=torch.float32)
            w_float = unrotate_weight(w_float, H, gs)
            result = "nvfp4_convrot"
        else:
            result = "nvfp4_plain"
    else:
        result = "nvfp4_plain"

    # Store dequantized weight
    state_dict[weight_key] = w_float.to(target_dtype)

    # Remove all NVFP4 sidecar keys
    for k in [scale_key, scale_2_key, input_scale_key, pre_quant_scale_key,
              prefix + ".comfy_quant"]:
        state_dict.pop(k, None)

    return result


def dequantize_nvfp4_state_dict(
    state_dict: dict,
    target_dtype: torch.dtype = torch.float16,
) -> dict:
    """In-place: convert all NVFP4 (format=nvfp4) weights to float.

    Only processes layers where comfy_quant format == "nvfp4".
    INT8 layers in the same mixed-pack checkpoint are left for
    dequantize_int8_state_dict to handle.

    Returns dict with stats for logging.
    """
    stats = {
        "nvfp4_plain": 0,
        "nvfp4_convrot": 0,
        "skipped": 0,
    }

    cq_keys = [k for k in list(state_dict.keys()) if k.endswith(".comfy_quant")]
    if not cq_keys:
        return stats

    for cq_key in cq_keys:
        prefix = cq_key[: -len(".comfy_quant")]

        # Skip if comfy_quant already popped by int8 pass
        if cq_key not in state_dict:
            continue

        try:
            raw_conf = decode_comfy_quant_conf(state_dict[cq_key])
            conf = _get_layer_conf(raw_conf, prefix)
        except Exception:
            continue

        fmt = conf.get("format", "")
        if fmt != "nvfp4":
            continue  # Not NVFP4, leave for int8 or skip

        result = _dequantize_nvfp4_layer(state_dict, prefix, conf, target_dtype)
        stats[result] = stats.get(result, 0) + 1

    return stats


# ---------------------------------------------------------------------------
# Combined entry point (called from sd_models.py)
# ---------------------------------------------------------------------------

def dequantize_state_dict(
    state_dict: dict,
    target_dtype: torch.dtype = torch.float16,
) -> dict:
    """Combined dequantize: NVFP4 first, then INT8.

    NVFP4 Linear layers and INT8 Conv2d layers can coexist in the same
    checkpoint (mixed pack). Process NVFP4 first (it reads .comfy_quant),
    then INT8 (which also reads .comfy_quant — keys already popped by NVFP4
    are skipped automatically).

    Returns combined stats dict for logging.
    """
    # NVFP4 pass (Linear layers with format=nvfp4)
    nvfp4_stats = dequantize_nvfp4_state_dict(state_dict, target_dtype)

    # INT8 pass (Conv2d layers with format=int8_tensorwise, and any remaining int8)
    int8_stats = dequantize_int8_state_dict(state_dict, target_dtype)

    return {**nvfp4_stats, **int8_stats}
