from __future__ import annotations
import math
import psutil
import platform
import json
import os

import torch
from torch import einsum

from ldm.util import default
from einops import rearrange

from modules import shared, errors, devices, sub_quadratic_attention
from modules.hypernetworks import hypernetwork


def _should_upcast_attn(dtype):
    """Auto-upcast attention to float32 for SDXL models in float16.

    Forge/ComfyUI avoid brown-noise NaN by keeping attention math in
    float32 when the storage dtype is float16.  A1111 had this behind a
    manual toggle (Settings > upcast_attn, default OFF).  This helper
    makes it automatic for SDXL/Pony/Illustrious where fp16 overflow
    is the dominant failure mode.
    """
    if dtype != torch.float16:
        return False
    # Always upcast fp16 attention for SDXL-class models
    sd_model = getattr(shared, 'sd_model', None)
    if sd_model is not None and getattr(sd_model, 'is_sdxl', False):
        return True
    return False


import ldm.modules.attention
import ldm.modules.diffusionmodules.model

import sgm.modules.attention
import sgm.modules.diffusionmodules.model

diffusionmodules_model_AttnBlock_forward = ldm.modules.diffusionmodules.model.AttnBlock.forward
sgm_diffusionmodules_model_AttnBlock_forward = sgm.modules.diffusionmodules.model.AttnBlock.forward

FA3_LOGGED_THIS_GEN = False

# Flash-Attention direct import (without xformers) - Forge-compatible direct load
FLASH_ATTN_AVAILABLE = False
FLASH_ATTN_VERSION = None
FLASH_ATTN_TYPE = None  # "FA-3" or "FA-2"
_flash_attn_log_shown = False

# SDP は巨大な attention 行列を確保するため、シーケンス長がこれを超える場合は最初から sub_quad を使う（OOM 防止）
SDP_ATTNBLOCK_MAX_SEQ = 4096


def _module_file(mod) -> str:
    return str(getattr(mod, "__file__", None) or "?")


def _callable_file(fn) -> str:
    """Resolve the on-disk path of the callable's defining module (kernel proof)."""
    try:
        import importlib

        return _module_file(importlib.import_module(fn.__module__))
    except Exception:
        return "?"


def _module_version(mod, dist_name: str) -> str:
    ver = getattr(mod, "__version__", None)
    if ver:
        return str(ver)
    try:
        import importlib.metadata

        return str(importlib.metadata.version(dist_name))
    except Exception:
        return "?"


# Always try direct Flash-Attention import (Forge-compatible direct load)
try:
    import flash_attn
    from flash_attn import flash_attn_func  # noqa: F401
    FLASH_ATTN_AVAILABLE = True
    FLASH_ATTN_VERSION = _module_version(flash_attn, "flash-attn")
    
    # Determine FA-3 or FA-2 based on version
    major_version = 2
    if FLASH_ATTN_VERSION and FLASH_ATTN_VERSION != "unknown":
        try:
            major_version = int(str(FLASH_ATTN_VERSION).split('.')[0])
        except Exception:
            pass
    
    FLASH_ATTN_TYPE = "FA-3" if major_version >= 3 else "FA-2"
    print(f"[A1111] Flash-Attention direct import: {FLASH_ATTN_TYPE} version {FLASH_ATTN_VERSION}")
except ImportError as e:
    FLASH_ATTN_AVAILABLE = False
    if getattr(shared.cmd_opts, "flash_attention", False):
        print(f"[A1111] WARNING: --flash-attention specified but flash-attn is not installed: {e}")
        print("[A1111] Install with: pip install flash-attn --no-build-isolation")
except Exception as e:
    FLASH_ATTN_AVAILABLE = False
    if getattr(shared.cmd_opts, "flash_attention", False):
        print(f"[A1111] WARNING: --flash-attention specified but flash_attn failed to load: {e}")


def patch_xformers_memory_efficient_attention():
    """Patch xformers.ops.memory_efficient_attention to handle query_size > 256"""
    try:
        import xformers.ops
        
        # Store original before any modifications
        original_func = xformers.ops.memory_efficient_attention
        
        # Make sure we have the original
        if hasattr(original_func, '__wrapped__'):
            original_func = original_func.__wrapped__
        
        def patched_memory_efficient_attention(query, key, value, attn_bias=None, op=None, **kwargs):
            """Patched version with fallback chain: xformers → SDP → sub_quad"""
            try:
                # Try xformers first (FA-3→FA-2→Cutlass priority)
                query_size = query.shape[-1]
                if query_size > 256:
                    print(f"[A1111] xformers: query_size={query_size} > 256, attempting (FA-3→FA-2→Cutlass priority)")
                
                return original_func(query, key, value, attn_bias=attn_bias, op=op, **kwargs)
            except Exception as e:
                print(f"[A1111] xformers failed: {e}")
                # Fallback to SDP
                try:
                    print("[A1111] Fallback: torch.nn.functional.scaled_dot_product_attention")
                    return torch.nn.functional.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
                except Exception as e2:
                    print(f"[A1111] SDP failed: {e2}")
                    # Final fallback to sub_quad_attention
                    print("[A1111] Fallback: sub_quad_attention")
                    return sub_quad_attention(query, key, value)
        
        xformers.ops.memory_efficient_attention = patched_memory_efficient_attention
        print("[A1111] Applied xformers.ops.memory_efficient_attention patch (query_size > 256 → SDP, else → xformers FA)")
    except Exception as e:
        print(f"[A1111] Warning: Could not patch xformers.memory_efficient_attention: {e}")


def check_and_configure_fa3():
    """Check FA-3 availability and configure xformers accordingly"""
    try:
        import xformers.ops.fmha.dispatch
        import xformers.ops.fmha.flash3
        
        # Check if FA-3 is actually available
        fa3_available = xformers.ops.fmha.dispatch.fa3_available()
        
        if fa3_available:
            print("[A1111] FA-3 is available - enabling FA-3 priority")
            xformers.ops.fmha.dispatch._set_use_fa3(True)
        else:
            print("[A1111] FA-3 is not available - disabling FA-3 priority (FA-2 will be used)")
            xformers.ops.fmha.dispatch._set_use_fa3(False)
            
    except Exception as e:
        print(f"[A1111] Error checking FA-3 availability: {e}")
        # Default to False if we can't check
        try:
            import xformers.ops.fmha.dispatch
            xformers.ops.fmha.dispatch._set_use_fa3(False)
        except:
            pass
    
    # Patch memory_efficient_attention for query_size > 256
    patch_xformers_memory_efficient_attention()


def reset_fa3_log():
    global FA3_LOGGED_THIS_GEN
    FA3_LOGGED_THIS_GEN = False

def get_xformers_kernel_info():
    """Get information about the last used xformers kernel"""
    try:
        from xformers.ops.fmha import get_last_used_kernel
        kernel_name = get_last_used_kernel()
        if kernel_name:
            if 'fa2' in kernel_name.lower() or ('flash' in kernel_name.lower() and '2' in kernel_name):
                return "FA-2 (2nd priority)"
            elif 'fa3' in kernel_name.lower() or ('flash' in kernel_name.lower() and '3' in kernel_name):
                return "FA-3 (1st priority)"
            elif 'cutlass' in kernel_name.lower():
                return "Cutlass (3rd priority)"
            else:
                return f"{kernel_name} (unknown priority)"
    except ImportError:
        pass
    return None


def log_fa_operation(op_name, op_config, versions):
    global FA3_LOGGED_THIS_GEN
    if FA3_LOGGED_THIS_GEN:
        return
    display_name = op_config.get("display_name", op_name)
    v = versions[0] if isinstance(versions, tuple) else versions
    print(f"[A1111] {display_name} enabled via xformers (version: {v})")
    FA3_LOGGED_THIS_GEN = True

def log_fa_operation_runtime(q, k, v):
    """Best-effort runtime check if FA-3 was actually used for these inputs, with detailed debug info if not."""
    global FA3_LOGGED_THIS_GEN
    if FA3_LOGGED_THIS_GEN:
        return
    try:
        import xformers.ops
        op = getattr(xformers.ops, "MemoryEfficientAttentionFlashAttentionOp", None)
        if op is not None:
            fw, _ = op
            from xformers.ops.fmha import Inputs
            inputs = Inputs(query=q, key=k, value=v, attn_bias=None)
            supported = fw.supports(inputs)
            print("[A1111][FA-3 DEBUG] q.shape:", getattr(q, 'shape', None), "k.shape:", getattr(k, 'shape', None), "v.shape:", getattr(v, 'shape', None))
            print("[A1111][FA-3 DEBUG] q.dtype:", getattr(q, 'dtype', None), "k.dtype:", getattr(k, 'dtype', None), "v.dtype:", getattr(v, 'dtype', None))
            print("[A1111][FA-3 DEBUG] q.device:", getattr(q, 'device', None))
            if supported:
                print("[A1111] Flash Attention 3 (FA-3) is SUPPORTED for this input and was LIKELY used by xformers.")
            else:
                print("[A1111] Flash Attention 3 (FA-3) is NOT supported for this input. Fallback to other attention op.")
                if hasattr(fw, 'not_supported_reasons'):
                    reasons = fw.not_supported_reasons(inputs)
                    print("[A1111] FA-3 not supported reasons:")
                    for r in reasons:
                        print(f"  - {r}")
            FA3_LOGGED_THIS_GEN = True
            return
        print("[A1111] Could not import xformers.ops.MemoryEfficientAttentionFlashAttentionOp. Cannot check FA-3 usage.")
    except Exception as e:
        print(f"[A1111] Error during FA-3 runtime check: {e}")
    FA3_LOGGED_THIS_GEN = True

# Load flash_attention_ops.json
FLASH_ATTENTION_OPS = {}
try:
    with open(os.path.join(os.path.dirname(__file__), '../config/flash_attention_ops.json'), 'r', encoding='utf-8') as f:
        FLASH_ATTENTION_OPS = json.load(f)
except Exception:
    pass


class SdOptimization:
    name: str = None
    label: str | None = None
    cmd_opt: str | None = None
    priority: int = 0

    def title(self):
        if self.label is None:
            return self.name

        return f"{self.name} - {self.label}"

    def is_available(self):
        return True

    def apply(self):
        pass

    def undo(self):
        ldm.modules.attention.CrossAttention.forward = hypernetwork.attention_CrossAttention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = diffusionmodules_model_AttnBlock_forward

        sgm.modules.attention.CrossAttention.forward = hypernetwork.attention_CrossAttention_forward
        sgm.modules.diffusionmodules.model.AttnBlock.forward = sgm_diffusionmodules_model_AttnBlock_forward


class SdOptimizationFlashAttn(SdOptimization):
    name = "flash-attention"
    label = "Flash Attention 3/2 (direct)"
    cmd_opt = "flash_attention"
    priority = 110  # Highest priority (direct FA-2/FA-3 without xformers)

    def is_available(self):
        return FLASH_ATTN_AVAILABLE and torch.cuda.is_available() and (6, 0) <= torch.cuda.get_device_capability(shared.device)

    def apply(self):
        ldm.modules.attention.CrossAttention.forward = flash_attention_forward
        # Use SDP for AttnBlock since it doesn't have heads structure
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sdp_no_mem_attnblock_forward
        sgm.modules.attention.CrossAttention.forward = flash_attention_forward
        sgm.modules.diffusionmodules.model.AttnBlock.forward = sdp_no_mem_attnblock_forward


class SdOptimizationXformers(SdOptimization):
    name = "xformers"
    cmd_opt = "xformers"
    priority = 100

    def is_available(self):
        # Enable xformers if it's available and CUDA is available (no upper cap on compute capability)
        return shared.xformers_available and torch.cuda.is_available() and (6, 0) <= torch.cuda.get_device_capability(shared.device)

    def apply(self):
        ldm.modules.attention.CrossAttention.forward = xformers_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = xformers_attnblock_forward
        sgm.modules.attention.CrossAttention.forward = xformers_attention_forward
        sgm.modules.diffusionmodules.model.AttnBlock.forward = xformers_attnblock_forward


class SdOptimizationSdpNoMem(SdOptimization):
    name = "sdp-no-mem"
    label = "scaled dot product without memory efficient attention"
    cmd_opt = "opt_sdp_no_mem_attention"
    priority = 80

    def is_available(self):
        return hasattr(torch.nn.functional, "scaled_dot_product_attention") and callable(torch.nn.functional.scaled_dot_product_attention)

    def apply(self):
        ldm.modules.attention.CrossAttention.forward = scaled_dot_product_no_mem_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sdp_no_mem_attnblock_forward
        sgm.modules.attention.CrossAttention.forward = scaled_dot_product_no_mem_attention_forward
        sgm.modules.diffusionmodules.model.AttnBlock.forward = sdp_no_mem_attnblock_forward


class SdOptimizationSdp(SdOptimizationSdpNoMem):
    name = "sdp"
    label = "scaled dot product"
    cmd_opt = "opt_sdp_attention"
    priority = 70

    def apply(self):
        ldm.modules.attention.CrossAttention.forward = scaled_dot_product_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sdp_attnblock_forward
        sgm.modules.attention.CrossAttention.forward = scaled_dot_product_attention_forward
        sgm.modules.diffusionmodules.model.AttnBlock.forward = sdp_attnblock_forward


class SdOptimizationSubQuad(SdOptimization):
    name = "sub-quadratic"
    cmd_opt = "opt_sub_quad_attention"

    @property
    def priority(self):
        return 1000 if shared.device.type == 'mps' else 10

    def apply(self):
        ldm.modules.attention.CrossAttention.forward = sub_quad_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sub_quad_attnblock_forward
        sgm.modules.attention.CrossAttention.forward = sub_quad_attention_forward
        sgm.modules.diffusionmodules.model.AttnBlock.forward = sub_quad_attnblock_forward


class SdOptimizationV1(SdOptimization):
    name = "V1"
    label = "original v1"
    cmd_opt = "opt_split_attention_v1"
    priority = 10

    def apply(self):
        ldm.modules.attention.CrossAttention.forward = split_cross_attention_forward_v1
        sgm.modules.attention.CrossAttention.forward = split_cross_attention_forward_v1


class SdOptimizationInvokeAI(SdOptimization):
    name = "InvokeAI"
    cmd_opt = "opt_split_attention_invokeai"

    @property
    def priority(self):
        return 1000 if shared.device.type != 'mps' and not torch.cuda.is_available() else 10

    def apply(self):
        ldm.modules.attention.CrossAttention.forward = split_cross_attention_forward_invokeAI
        sgm.modules.attention.CrossAttention.forward = split_cross_attention_forward_invokeAI


class SdOptimizationDoggettx(SdOptimization):
    name = "Doggettx"
    cmd_opt = "opt_split_attention"
    priority = 90

    def apply(self):
        ldm.modules.attention.CrossAttention.forward = split_cross_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = cross_attention_attnblock_forward
        sgm.modules.attention.CrossAttention.forward = split_cross_attention_forward
        sgm.modules.diffusionmodules.model.AttnBlock.forward = cross_attention_attnblock_forward


def list_optimizers(res):
    res.extend([
        SdOptimizationFlashAttn(),  # Highest priority
        SdOptimizationXformers(),
        SdOptimizationSdpNoMem(),
        SdOptimizationSdp(),
        SdOptimizationSubQuad(),
        SdOptimizationV1(),
        SdOptimizationInvokeAI(),
        SdOptimizationDoggettx(),
    ])


# Always try to import xformers if available
try:
    import xformers.ops
    shared.xformers_available = True
    print("[A1111] xformers successfully imported")
except Exception as e:
    print(f"[A1111] Cannot import xformers: {e}")
    shared.xformers_available = False


def _try_clear_cuda_sticky_errors():
    """Surface asynchronous CUDA errors so mem queries can succeed again (OOM fallout)."""
    if shared.device.type != 'cuda':
        return
    try:
        torch.cuda.synchronize(device=shared.device)
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def get_available_vram():
    if shared.device.type == 'cuda':

        def from_stats_and_driver():
            stats = torch.cuda.memory_stats(shared.device)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_cuda, _ = torch.cuda.mem_get_info(shared.device)
            mem_free_torch = mem_reserved - mem_active
            return mem_free_cuda + mem_free_torch

        try:
            return from_stats_and_driver()
        except Exception as e:
            print(f"[A1111] get_available_vram failed ({e}); clearing CUDA sticky state and retrying")
            _try_clear_cuda_sticky_errors()
            try:
                return from_stats_and_driver()
            except Exception as e2:
                print(f"[A1111] get_available_vram retry failed ({e2}); using mem_get_info only")
                try:
                    mem_free_cuda, _ = torch.cuda.mem_get_info(shared.device)
                    return mem_free_cuda
                except Exception as e3:
                    print(f"[A1111] get_available_vram mem_get_info failed ({e3}), using conservative default")
                    return 256 * 1024 * 1024  # last resort — may underestimate; avoids throwing
    else:
        return psutil.virtual_memory().available


# see https://github.com/basujindal/stable-diffusion/pull/117 for discussion
def split_cross_attention_forward_v1(self, x, context=None, mask=None, **kwargs):
    h = self.heads

    q_in = self.to_q(x)
    context = default(context, x)

    context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)
    del context, context_k, context_v, x

    q, k, v = (rearrange(t, 'b n (h d) -> (b h) n d', h=h) for t in (q_in, k_in, v_in))
    del q_in, k_in, v_in

    dtype = q.dtype
    if shared.opts.upcast_attn or _should_upcast_attn(dtype):
        q, k, v = q.float(), k.float(), v.float()

    with devices.without_autocast(disable=not shared.opts.upcast_attn):
        r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
        for i in range(0, q.shape[0], 2):
            end = i + 2
            s1 = einsum('b i d, b j d -> b i j', q[i:end], k[i:end])
            s1 *= self.scale

            s2 = s1.softmax(dim=-1)
            del s1

            r1[i:end] = einsum('b i j, b j d -> b i d', s2, v[i:end])
            del s2
        del q, k, v

    r1 = r1.to(dtype)

    r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
    del r1

    return self.to_out(r2)


# taken from https://github.com/Doggettx/stable-diffusion and modified
def split_cross_attention_forward(self, x, context=None, mask=None, **kwargs):
    h = self.heads

    q_in = self.to_q(x)
    context = default(context, x)

    context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)

    dtype = q_in.dtype
    if shared.opts.upcast_attn or _should_upcast_attn(dtype):
        q_in, k_in, v_in = q_in.float(), k_in.float(), v_in if v_in.device.type == 'mps' else v_in.float()

    with devices.without_autocast(disable=not shared.opts.upcast_attn):
        k_in = k_in * self.scale

        del context, x

        q, k, v = (rearrange(t, 'b n (h d) -> (b h) n d', h=h) for t in (q_in, k_in, v_in))
        del q_in, k_in, v_in

        r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)

        mem_free_total = get_available_vram()

        gb = 1024 ** 3
        tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size()
        modifier = 3 if q.element_size() == 2 else 2.5
        mem_required = tensor_size * modifier
        steps = 1

        if mem_required > mem_free_total:
            steps = 2 ** (math.ceil(math.log(mem_required / mem_free_total, 2)))
            # print(f"Expected tensor size:{tensor_size/gb:0.1f}GB, cuda free:{mem_free_cuda/gb:0.1f}GB "
            #       f"torch free:{mem_free_torch/gb:0.1f} total:{mem_free_total/gb:0.1f} steps:{steps}")

        if steps > 64:
            max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
            raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                               f'Need: {mem_required / 64 / gb:0.1f}GB free, Have:{mem_free_total / gb:0.1f}GB free')

        slice_size = q.shape[1] // steps
        for i in range(0, q.shape[1], slice_size):
            end = min(i + slice_size, q.shape[1])
            s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k)

            s2 = s1.softmax(dim=-1, dtype=q.dtype)
            del s1

            r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)
            del s2

        del q, k, v

    r1 = r1.to(dtype)

    r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
    del r1

    return self.to_out(r2)


# -- Taken from https://github.com/invoke-ai/InvokeAI and modified --
mem_total_gb = psutil.virtual_memory().total // (1 << 30)


def einsum_op_compvis(q, k, v):
    s = einsum('b i d, b j d -> b i j', q, k)
    s = s.softmax(dim=-1, dtype=s.dtype)
    return einsum('b i j, b j d -> b i d', s, v)


def einsum_op_slice_0(q, k, v, slice_size):
    r = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
    for i in range(0, q.shape[0], slice_size):
        end = i + slice_size
        r[i:end] = einsum_op_compvis(q[i:end], k[i:end], v[i:end])
    return r


def einsum_op_slice_1(q, k, v, slice_size):
    r = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
    for i in range(0, q.shape[1], slice_size):
        end = i + slice_size
        r[:, i:end] = einsum_op_compvis(q[:, i:end], k, v)
    return r


def einsum_op_mps_v1(q, k, v):
    if q.shape[0] * q.shape[1] <= 2**16: # (512x512) max q.shape[1]: 4096
        return einsum_op_compvis(q, k, v)
    else:
        slice_size = math.floor(2**30 / (q.shape[0] * q.shape[1]))
        if slice_size % 4096 == 0:
            slice_size -= 1
        return einsum_op_slice_1(q, k, v, slice_size)


def einsum_op_mps_v2(q, k, v):
    if mem_total_gb > 8 and q.shape[0] * q.shape[1] <= 2**16:
        return einsum_op_compvis(q, k, v)
    else:
        return einsum_op_slice_0(q, k, v, 1)


def einsum_op_tensor_mem(q, k, v, max_tensor_mb):
    size_mb = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size() // (1 << 20)
    if size_mb <= max_tensor_mb:
        return einsum_op_compvis(q, k, v)
    div = 1 << int((size_mb - 1) / max_tensor_mb).bit_length()
    if div <= q.shape[0]:
        return einsum_op_slice_0(q, k, v, q.shape[0] // div)
    return einsum_op_slice_1(q, k, v, max(q.shape[1] // div, 1))


def einsum_op_cuda(q, k, v):
    stats = torch.cuda.memory_stats(q.device)
    mem_active = stats['active_bytes.all.current']
    mem_reserved = stats['reserved_bytes.all.current']
    mem_free_cuda, _ = torch.cuda.mem_get_info(q.device)
    mem_free_torch = mem_reserved - mem_active
    mem_free_total = mem_free_cuda + mem_free_torch
    # Divide factor of safety as there's copying and fragmentation
    return einsum_op_tensor_mem(q, k, v, mem_free_total / 3.3 / (1 << 20))


def einsum_op(q, k, v):
    if q.device.type == 'cuda':
        return einsum_op_cuda(q, k, v)

    if q.device.type == 'mps':
        if mem_total_gb >= 32 and q.shape[0] % 32 != 0 and q.shape[0] * q.shape[1] < 2**18:
            return einsum_op_mps_v1(q, k, v)
        return einsum_op_mps_v2(q, k, v)

    # Smaller slices are faster due to L2/L3/SLC caches.
    # Tested on i7 with 8MB L3 cache.
    return einsum_op_tensor_mem(q, k, v, 32)


def split_cross_attention_forward_invokeAI(self, x, context=None, mask=None, **kwargs):
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)

    context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    k = self.to_k(context_k)
    v = self.to_v(context_v)
    del context, context_k, context_v, x

    dtype = q.dtype
    if shared.opts.upcast_attn or _should_upcast_attn(dtype):
        q, k, v = q.float(), k.float(), v if v.device.type == 'mps' else v.float()

    with devices.without_autocast(disable=not shared.opts.upcast_attn):
        k = k * self.scale

        q, k, v = (rearrange(t, 'b n (h d) -> (b h) n d', h=h) for t in (q, k, v))
        r = einsum_op(q, k, v)
    r = r.to(dtype)
    return self.to_out(rearrange(r, '(b h) n d -> b n (h d)', h=h))

# -- End of code from https://github.com/invoke-ai/InvokeAI --


# Based on Birch-san's modified implementation of sub-quadratic attention from https://github.com/Birch-san/diffusers/pull/1
# The sub_quad_attention_forward function is under the MIT License listed under Memory Efficient Attention in the Licenses section of the web UI interface
def sub_quad_attention_forward(self, x, context=None, mask=None, **kwargs):
    assert mask is None, "attention-mask not currently implemented for SubQuadraticCrossAttnProcessor."

    h = self.heads

    q = self.to_q(x)
    context = default(context, x)

    context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    k = self.to_k(context_k)
    v = self.to_v(context_v)
    del context, context_k, context_v, x

    q = q.unflatten(-1, (h, -1)).transpose(1,2).flatten(end_dim=1)
    k = k.unflatten(-1, (h, -1)).transpose(1,2).flatten(end_dim=1)
    v = v.unflatten(-1, (h, -1)).transpose(1,2).flatten(end_dim=1)

    if q.device.type == 'mps':
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    dtype = q.dtype
    if shared.opts.upcast_attn or _should_upcast_attn(dtype):
        q, k = q.float(), k.float()

    x = sub_quad_attention(q, k, v, q_chunk_size=shared.cmd_opts.sub_quad_q_chunk_size, kv_chunk_size=shared.cmd_opts.sub_quad_kv_chunk_size, chunk_threshold=shared.cmd_opts.sub_quad_chunk_threshold, use_checkpoint=self.training)

    x = x.to(dtype)

    x = x.unflatten(0, (-1, h)).transpose(1,2).flatten(start_dim=2)

    out_proj, dropout = self.to_out
    x = out_proj(x)
    x = dropout(x)

    return x


def sub_quad_attention(q, k, v, q_chunk_size=1024, kv_chunk_size=None, kv_chunk_size_min=None, chunk_threshold=None, use_checkpoint=True):
    bytes_per_token = torch.finfo(q.dtype).bits//8
    batch_x_heads, q_tokens, _ = q.shape
    _, k_tokens, _ = k.shape
    qk_matmul_size_bytes = batch_x_heads * bytes_per_token * q_tokens * k_tokens

    if chunk_threshold is None:
        if q.device.type == 'mps':
            chunk_threshold_bytes = 268435456 * (2 if platform.processor() == 'i386' else bytes_per_token)
        else:
            chunk_threshold_bytes = int(get_available_vram() * 0.7)
    elif chunk_threshold == 0:
        chunk_threshold_bytes = None
    else:
        chunk_threshold_bytes = int(0.01 * chunk_threshold * get_available_vram())

    if kv_chunk_size_min is None and chunk_threshold_bytes is not None:
        kv_chunk_size_min = chunk_threshold_bytes // (batch_x_heads * bytes_per_token * (k.shape[2] + v.shape[2]))
    elif kv_chunk_size_min == 0:
        kv_chunk_size_min = None

    if chunk_threshold_bytes is not None and qk_matmul_size_bytes <= chunk_threshold_bytes:
        # the big matmul fits into our memory limit; do everything in 1 chunk,
        # i.e. send it down the unchunked fast-path
        kv_chunk_size = k_tokens

    with devices.without_autocast(disable=q.dtype == v.dtype):
        return sub_quadratic_attention.efficient_dot_product_attention(
            q,
            k,
            v,
            query_chunk_size=q_chunk_size,
            kv_chunk_size=kv_chunk_size,
            kv_chunk_size_min = kv_chunk_size_min,
            use_checkpoint=use_checkpoint,
        )


def get_xformers_flash_attention_op(q, k, v):
    # Forgeの実装を参考に、より積極的にFA-2を優先
    try:
        # FA-2を優先的に試す
        flash_attention_op = xformers.ops.MemoryEfficientAttentionFlashAttentionOp
        fw, bw = flash_attention_op
        if fw.supports(xformers.ops.fmha.Inputs(query=q, key=k, value=v, attn_bias=None)):
            return flash_attention_op
    except Exception as e:
        # FA-2が使用できない場合はNoneを返してxformersに自動選択させる
        pass

    return None


def get_xformers_kernel_name():
    try:
        import xformers
        # xformers.ops.memory_efficient_attention._last_kernel is set internally after call
        kernel = getattr(xformers.ops.memory_efficient_attention, "_last_kernel", None)
        if kernel is not None:
            print(f"[A1111][xformers] Actually used kernel: {kernel}")
        else:
            print("[A1111][xformers] Could not determine which kernel was used.")
    except Exception as e:
        print(f"[A1111][xformers] Error while checking used kernel: {e}")


def flash_attention_forward(self, x, context=None, mask=None, **kwargs):
    """Direct Flash-Attention 3/2 implementation without xformers"""
    global _flash_attn_log_shown
    kwargs.pop('additional_tokens', None)
    
    try:
        from flash_attn import flash_attn_func
        
        h = self.heads
        q_in = self.to_q(x)
        context = context if context is not None else x
        k_in = self.to_k(context)
        v_in = self.to_v(context)
        
        # Flash-Attention requires (batch, seqlen, nheads, headdim) format
        q = q_in.reshape(q_in.shape[0], q_in.shape[1], h, -1).contiguous()
        k = k_in.reshape(k_in.shape[0], k_in.shape[1], h, -1).contiguous()
        v = v_in.reshape(v_in.shape[0], v_in.shape[1], h, -1).contiguous()
        
        del q_in, k_in, v_in
        
        # Flash-Attention requires float16 or bfloat16
        original_dtype = x.dtype
        if q.dtype not in [torch.float16, torch.bfloat16]:
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)
        
        # Call flash_attn_func directly
        out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
        
        # Log only after real kernel returns once per generation (matching Forge attention_backend_info.py)
        if not _flash_attn_log_shown:
            import flash_attn as flash_attn_mod
            ver = _module_version(flash_attn_mod, "flash-attn") or FLASH_ATTN_VERSION
            path = _callable_file(flash_attn_func)
            if path == "?":
                path = _module_file(flash_attn_mod)
            label = FLASH_ATTN_TYPE or "FA-2"
            print(f"[A1111] {label} kernel ran: {flash_attn_func.__module__}.{flash_attn_func.__name__} ver={ver} file={path}")
            _flash_attn_log_shown = True
        
        # Convert back to original dtype if needed
        if out.dtype != original_dtype:
            out = out.to(original_dtype)
        
        # Reshape to (b, s, h*d)
        out = out.reshape(out.shape[0], out.shape[1], h * out.shape[3])
        
        return self.to_out(out)
        
    except Exception as e:
        label = FLASH_ATTN_TYPE or "FA-2"
        print(f"[A1111] {label} kernel NOT used — flash_attn_func failed ({e}); fallback pytorch SDPA")
        # Fallback to SDPA
        try:
            h = self.heads
            q_in = self.to_q(x)
            context = context if context is not None else x
            k_in = self.to_k(context)
            v_in = self.to_v(context)
            q, k, v = (t.reshape(t.shape[0], t.shape[1], h, -1) for t in (q_in, k_in, v_in))
            dtype = q.dtype
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
            out = out.to(dtype)
            b, n, h, d = out.shape
            out = out.reshape(b, n, h * d)
            return self.to_out(out)
        except Exception as e2:
            print(f"[A1111] SDPA failed: {e2}")
            print("[A1111] Fallback: sub_quad_attention")
            try:
                # Final fallback to sub_quad for memory efficiency
                h = self.heads
                q_in = self.to_q(x)
                context = context if context is not None else x
                k_in = self.to_k(context)
                v_in = self.to_v(context)
                q, k, v = (t.reshape(t.shape[0], t.shape[1], h, -1) for t in (q_in, k_in, v_in))
                out = sub_quad_attention(q.reshape(q.shape[0], q.shape[1], -1), 
                                        k.reshape(k.shape[0], k.shape[1], -1), 
                                        v.reshape(v.shape[0], v.shape[1], -1))
                b, s, c = out.shape
                out = out.reshape(b, s, h, -1)
                out = out.reshape(b, s, h * out.shape[3])
                return self.to_out(out)
            except Exception as e3:
                print(f"[A1111] sub_quad_attention also failed: {e3}")
                if hasattr(self, "_old_attention_forward"):
                    return self._old_attention_forward(x, context, mask)
                raise


def flash_attn_attnblock_forward(self, x):
    """Flash-Attention 3/2 for AttnBlock without xformers"""
    global _flash_attn_log_shown
    try:
        from flash_attn import flash_attn_func
        
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        
        # Reshape to (batch, seqlen, channels)
        q = q.reshape(b, c, h * w).permute(0, 2, 1).contiguous()
        k = k.reshape(b, c, h * w).permute(0, 2, 1).contiguous()
        v = v.reshape(b, c, h * w).permute(0, 2, 1).contiguous()
        
        # Flash-Attention requires float16 or bfloat16
        original_dtype = x.dtype
        if q.dtype not in [torch.float16, torch.bfloat16]:
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)
        
        # Call flash_attn_func directly
        out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
        
        # Log only after real kernel returns once per generation (matching Forge attention_backend_info.py)
        if not _flash_attn_log_shown:
            import flash_attn as flash_attn_mod
            ver = _module_version(flash_attn_mod, "flash-attn") or FLASH_ATTN_VERSION
            path = _callable_file(flash_attn_func)
            if path == "?":
                path = _module_file(flash_attn_mod)
            label = FLASH_ATTN_TYPE or "FA-2"
            print(f"[A1111] {label} kernel ran: {flash_attn_func.__module__}.{flash_attn_func.__name__} ver={ver} file={path}")
            _flash_attn_log_shown = True
        
        # Convert back to original dtype if needed
        if out.dtype != original_dtype:
            out = out.to(original_dtype)
        
        torch.cuda.empty_cache()  # CRITICAL: Clear memory after FA to prevent fragmentation
        
        # Reshape back to (batch, channels, height, width)
        out = out.permute(0, 2, 1).reshape(b, c, h, w)
        out = self.proj_out(out)
        
        return x + out
        
    except Exception as e:
        label = FLASH_ATTN_TYPE or "FA-2"
        print(f"[A1111] {label} kernel NOT used (AttnBlock) — flash_attn_func failed ({e}); fallback pytorch SDPA")
        try:
            return sdp_no_mem_attnblock_forward(self, x)
        except Exception as e2:
            print(f"[A1111] SDPA fallback also failed: {e2}")
            if hasattr(self, "_old_attnblock_forward"):
                return self._old_attnblock_forward(x)
            raise


def reset_flash_attn_log():
    """Reset the log flag for Flash-Attention"""
    global _flash_attn_log_shown
    _flash_attn_log_shown = False


def xformers_attention_forward(self, x, context=None, mask=None, **kwargs):
    global FA3_LOGGED_THIS_GEN
    # Remove unsupported kwargs for xformers/SDP
    kwargs.pop('additional_tokens', None)
    try:
        import xformers.ops
        h = self.heads
        q_in = self.to_q(x)
        context = context if context is not None else x
        k_in = self.to_k(context)
        v_in = self.to_v(context)
        
        q, k, v = (t.reshape(t.shape[0], t.shape[1], h, -1) for t in (q_in, k_in, v_in))
        
        del q_in, k_in, v_in

        # Check if query_size > 256 (xformers 0.0.33 limitation)
        query_size = q.shape[-1]
        if query_size > 256:
            print(f"[A1111] Query size {query_size} > 256 detected, using SDP instead of xformers")
            # Skip xformers and use SDP directly
            raise NotImplementedError("Query size too large for xformers")

        # 出力dtypeは入力xに合わせる
        dtype = x.dtype
        if shared.opts.upcast_attn or _should_upcast_attn(q.dtype):
            q, k, v = q.float(), k.float(), v.float()
        else:
            # Forgeに寄せて、FAカーネルを使うためfloat32なら半精度で実行→出力は元dtypeへ戻す
            if q.dtype == torch.float32:
                q = q.half(); k = k.half(); v = v.half()

        # Forgeのシンプルな実装を参考に、カスタムopを指定
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask, op=get_xformers_flash_attention_op(q, k, v))
        
        if not FA3_LOGGED_THIS_GEN:
            # 使用されたカーネルを確認
            kernel_info = get_xformers_kernel_info()
            if kernel_info:
                print(f"[A1111] xformers.memory_efficient_attention called - {kernel_info} kernel detected")
            else:
                print("[A1111] xformers.memory_efficient_attention called (FA-3→FA-2→Cutlass priority order)")
            FA3_LOGGED_THIS_GEN = True
        
        out = out.to(dtype)
        b, n, h, d = out.shape
        out = out.reshape(b, n, h * d)
        return self.to_out(out)
    except Exception as e:
        print(f"[A1111] xformers.memory_efficient_attention failed, falling back to SDP. Exception: {e}")
        # Try PyTorch's scaled_dot_product_attention (SDP)
        try:
            h = self.heads
            q_in = self.to_q(x)
            context = context if context is not None else x
            k_in = self.to_k(context)
            v_in = self.to_v(context)
            q, k, v = (t.reshape(t.shape[0], t.shape[1], h, -1) for t in (q_in, k_in, v_in))
            dtype = q.dtype
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
            out = out.to(dtype)
            b, n, h, d = out.shape
            out = out.reshape(b, n, h * d)
            print("[A1111] Fallback: torch.nn.functional.scaled_dot_product_attention (SDP) used.")
            return self.to_out(out)
        except Exception as e2:
            print(f"[A1111] SDP also failed, falling back to legacy. Exception: {e2}")
            if hasattr(self, "_old_attention_forward"):
                return self._old_attention_forward(x, context, mask)
            raise


# Based on Diffusers usage of scaled dot product attention from https://github.com/huggingface/diffusers/blob/c7da8fd23359a22d0df2741688b5b4f33c26df21/src/diffusers/models/cross_attention.py
# The scaled_dot_product_attention_forward function contains parts of code under Apache-2.0 license listed under Scaled Dot Product Attention in the Licenses section of the web UI interface
def scaled_dot_product_attention_forward(self, x, context=None, mask=None, **kwargs):
    batch_size, sequence_length, inner_dim = x.shape

    if mask is not None:
        mask = self.prepare_attention_mask(mask, sequence_length, batch_size)
        mask = mask.view(batch_size, self.heads, -1, mask.shape[-1])

    h = self.heads
    q_in = self.to_q(x)
    context = default(context, x)

    context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)

    head_dim = inner_dim // h
    q = q_in.view(batch_size, -1, h, head_dim).transpose(1, 2)
    k = k_in.view(batch_size, -1, h, head_dim).transpose(1, 2)
    v = v_in.view(batch_size, -1, h, head_dim).transpose(1, 2)

    del q_in, k_in, v_in

    dtype = q.dtype
    if shared.opts.upcast_attn or _should_upcast_attn(dtype):
        q, k, v = q.float(), k.float(), v.float()

    # the output of sdp = (batch, num_heads, seq_len, head_dim)
    hidden_states = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
    )

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, h * head_dim)
    hidden_states = hidden_states.to(dtype)

    # linear proj
    hidden_states = self.to_out[0](hidden_states)
    # dropout
    hidden_states = self.to_out[1](hidden_states)
    return hidden_states


def scaled_dot_product_no_mem_attention_forward(self, x, context=None, mask=None, **kwargs):
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False):
        return scaled_dot_product_attention_forward(self, x, context, mask)


def cross_attention_attnblock_forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q1 = self.q(h_)
        k1 = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q1.shape

        q2 = q1.reshape(b, c, h*w)
        del q1

        q = q2.permute(0, 2, 1)   # b,hw,c
        del q2

        k = k1.reshape(b, c, h*w) # b,c,hw
        del k1

        h_ = torch.zeros_like(k, device=q.device)

        mem_free_total = get_available_vram()

        tensor_size = q.shape[0] * q.shape[1] * k.shape[2] * q.element_size()
        mem_required = tensor_size * 2.5
        steps = 1

        if mem_required > mem_free_total:
            steps = 2**(math.ceil(math.log(mem_required / mem_free_total, 2)))

        slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
        for i in range(0, q.shape[1], slice_size):
            end = i + slice_size

            w1 = torch.bmm(q[:, i:end], k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
            w2 = w1 * (int(c)**(-0.5))
            del w1
            w3 = torch.nn.functional.softmax(w2, dim=2, dtype=q.dtype)
            del w2

            # attend to values
            v1 = v.reshape(b, c, h*w)
            w4 = w3.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
            del w3

            h_[:, :, i:end] = torch.bmm(v1, w4)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
            del v1, w4

        h2 = h_.reshape(b, c, h, w)
        del h_

        h3 = self.proj_out(h2)
        del h2

        h3 += x

        return h3


def xformers_attnblock_forward(self, x):
    global FA3_LOGGED_THIS_GEN
    try:
        import xformers.ops
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q, k, v = (t.reshape(t.shape[0], t.shape[1], -1) for t in (q, k, v))
        
        # Check if query_size > 256 (xformers 0.0.33 limitation)
        query_size = q.shape[-1]
        if query_size > 256:
            print(f"[A1111] AttnBlock: Query size {query_size} > 256 detected, using SDP instead of xformers")
            return sdp_attnblock_forward(self, x)
        
        # Align dtypes (prefer float16 if available)
        dtypes = [q.dtype, k.dtype, v.dtype]
        if torch.float16 in dtypes:
            target_dtype = torch.float16
        else:
            target_dtype = q.dtype
        q = q.to(target_dtype)
        k = k.to(target_dtype)
        v = v.to(target_dtype)
        out = xformers.ops.memory_efficient_attention(q, k, v)
        
        if not FA3_LOGGED_THIS_GEN:
            # 使用されたカーネルを確認
            kernel_info = get_xformers_kernel_info()
            if kernel_info:
                print(f"[A1111] xformers.memory_efficient_attention called - {kernel_info} kernel detected")
            else:
                print("[A1111] xformers.memory_efficient_attention called (FA-3→FA-2→Cutlass priority order)")
            FA3_LOGGED_THIS_GEN = True
        out = out.to(x.dtype)
        out = out.reshape(b, c, h, w)
        out = self.proj_out(out)
        return x + out
    except Exception as e:
        print(f"[A1111] xformers.memory_efficient_attention failed in attnblock, falling back. Exception: {e}")
        if hasattr(self, "_old_attnblock_forward"):
            return self._old_attnblock_forward(x)
        raise


def sdp_attnblock_forward(self, x):
    h_ = x
    h_ = self.norm(h_)
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)
    b, c, h, w = q.shape
    seq_len = h * w
    q, k, v = (rearrange(t, 'b c h w -> b (h w) c') for t in (q, k, v))
    dtype = q.dtype
    if shared.opts.upcast_attn or _should_upcast_attn(dtype):
        q, k, v = q.float(), k.float(), v.float()
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Large seq_len → never attempt SDP here (attention materialization cost explodes near this scale).
    # Use ">=" so seq_len == threshold (e.g. 64*64 == 4096) does not slip into SDP/sub_quad heuristic bugs.
    use_sub_quad_direct = seq_len >= SDP_ATTNBLOCK_MAX_SEQ
    # chunk_threshold=0 disables sub_quad_attention's "fits VRAM → kv_chunk=k_tokens" shortcut
    # that routes to _get_attention_scores_no_kv_chunking and OOMs on large seq_len.
    _sub_quad_attnblock_kw = dict(
        q_chunk_size=shared.cmd_opts.sub_quad_q_chunk_size,
        kv_chunk_size=shared.cmd_opts.sub_quad_kv_chunk_size,
        chunk_threshold=0,
        use_checkpoint=self.training,
    )
    if use_sub_quad_direct:
        out = sub_quad_attention(q, k, v, **_sub_quad_attnblock_kw)
        out = out.to(dtype)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h)
        out = self.proj_out(out)
        return x + out

    try:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.to(dtype)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h)
        out = self.proj_out(out)
        return x + out
    except Exception as e:
        print(f"[A1111] SDP attnblock failed: {e}")
        print("[A1111] Fallback: sub_quad_attention")
        if q.device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        out = sub_quad_attention(q, k, v, **_sub_quad_attnblock_kw)
        out = out.to(dtype)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h)
        out = self.proj_out(out)
        return x + out


def sdp_no_mem_attnblock_forward(self, x):
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False):
        return sdp_attnblock_forward(self, x)


def sub_quad_attnblock_forward(self, x):
    h_ = x
    h_ = self.norm(h_)
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)
    b, c, h, w = q.shape
    q, k, v = (rearrange(t, 'b c h w -> b (h w) c') for t in (q, k, v))
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = sub_quad_attention(q, k, v, q_chunk_size=shared.cmd_opts.sub_quad_q_chunk_size, kv_chunk_size=shared.cmd_opts.sub_quad_kv_chunk_size, chunk_threshold=shared.cmd_opts.sub_quad_chunk_threshold, use_checkpoint=self.training)
    out = rearrange(out, 'b (h w) c -> b c h w', h=h)
    out = self.proj_out(out)
    return x + out
