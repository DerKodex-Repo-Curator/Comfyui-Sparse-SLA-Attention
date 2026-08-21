# Comfyui-Sparse-Attention

This is a standalone version of the `ComfyUI-H3-SLA-Attention` node from
[ComfyUI-PlagueKind-Nodes](https://github.com/PlagueKind/ComfyUI-PlagueKind-Nodes) pack, all credit goes to the original authors listed below.

## Important note

By default this node won't trigger at all on lower resolutions/lengths (ie: 512x512) because those generations fall outside the default 8192 sequence length. You can of course lower this limit, but results might be worse.

## Requirements

- triton

## Credits

- The node and wiring are (c) 2026
  [PlagueKind](https://github.com/PlagueKind), extracted from
  [ComfyUI-PlagueKind-Nodes](https://github.com/PlagueKind/ComfyUI-PlagueKind-Nodes)
  under the MIT license -- see [LICENSE](LICENSE).
- `sla/block_map.py` and `sla/kernel.py` contain code vendored from
  [LightX2V](https://github.com/ModelTC/LightX2V) (Apache-2.0), with their
  original attribution headers intact -- see
  [LICENSE-APACHE-2.0.txt](LICENSE-APACHE-2.0.txt).
