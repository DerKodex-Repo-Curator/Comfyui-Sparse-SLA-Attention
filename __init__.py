"""Comfyui-Sparse-Attention.

One node, H3 SLA Attention, which gives MiniMax-H3 the block-sparse attention
backend that ComfyUI does not otherwise have -- the inference path the lightx2v
SLA turbo LoRA was distilled against, and the reason that LoRA produces no
speedup on its own. See README.md.

Standalone extraction of the ComfyUI-H3-SLA-Attention pack from
ComfyUI-PlagueKind-Nodes (MIT, (c) 2026 PlagueKind):

    https://github.com/PlagueKind/ComfyUI-PlagueKind-Nodes

No other nodes from that repository are needed; this folder is the whole pack.

Registration is deliberately defensive: if anything in this package fails to
import -- a ComfyUI build without the V3 API, a missing dependency -- we log
the traceback and expose no node, so ComfyUI skips the pack instead of failing
to start. A broken node must never stop ComfyUI from booting.

Two registration paths, exactly one of which ends up defined:

1. ComfyUI's V3 extension API (``comfy_api.latest``, ``comfy_entrypoint``),
   preferred by current ComfyUI builds.
2. The classic V1 mappings (``NODE_CLASS_MAPPINGS`` /
   ``NODE_DISPLAY_NAME_MAPPINGS``) for builds without ``comfy_api``, and as a
   recovery path if the V3 node cannot be imported.

Note we must NOT define ``NODE_CLASS_MAPPINGS`` alongside
``comfy_entrypoint``, not even as an empty dict. ComfyUI's loader (nodes.py,
"V1 node definition") checks it with ``is not None`` and returns early when it
exists -- so an empty mapping would shadow ``comfy_entrypoint`` and silently
register zero nodes.

To remove the pack entirely, delete this folder.
"""

import logging

log = logging.getLogger("H3Utils")

__all__: list = []


def _register_v3():
    """Register through ``comfy_entrypoint`` (ComfyUI's V3 extension API)."""
    from comfy_api.latest import ComfyExtension

    from .sla_node import H3SLAAttention

    class H3SLAExtension(ComfyExtension):
        async def get_node_list(self):
            return [H3SLAAttention]

    async def comfy_entrypoint():
        return H3SLAExtension()

    globals()["comfy_entrypoint"] = comfy_entrypoint
    __all__.append("comfy_entrypoint")


def _register_v1():
    """Register through the classic ``NODE_CLASS_MAPPINGS`` interface."""
    from .v1_node import H3SLAAttentionV1

    globals()["NODE_CLASS_MAPPINGS"] = {"H3SLAAttention": H3SLAAttentionV1}
    globals()["NODE_DISPLAY_NAME_MAPPINGS"] = {
        "H3SLAAttention": "H3 SLA Attention"}
    __all__.extend(["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"])


try:
    from comfy_api.latest import ComfyExtension  # noqa: F401
    _HAS_V3 = True
except ImportError:
    _HAS_V3 = False
except Exception:  # noqa: BLE001 - never block ComfyUI startup
    log.exception("[H3Utils] comfy_api present but broken; "
                  "falling back to V1 registration.")
    _HAS_V3 = False

if _HAS_V3:
    try:
        _register_v3()
    except Exception:  # noqa: BLE001 - never block ComfyUI startup
        log.exception("[H3Utils] V3 registration failed; trying V1 mappings.")
        try:
            _register_v1()
        except Exception:  # noqa: BLE001
            log.exception("[H3Utils] SLA failed to load; the node will not be "
                          "available. ComfyUI startup is unaffected.")
else:
    try:
        _register_v1()
    except Exception:  # noqa: BLE001 - never block ComfyUI startup
        log.exception("[H3Utils] SLA failed to load; the node will not be "
                      "available. ComfyUI startup is unaffected.")
