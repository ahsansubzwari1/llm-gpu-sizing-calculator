# LLM-GPU Sizing Calculator

A single-file, browser-based tool for estimating GPU memory requirements when deploying large language models (LLMs) for inference. Input your model architecture, precision format, concurrency target, and context window — get a step-by-step memory breakdown and a recommendation matrix across a library of GPU options.

**[Launch the calculator →]((https://ahsansubzwari1.github.io/llm-gpu-sizing-calculator/LLM-GPU-Sizing%20Calculator.html))**  
*(Update this link after enabling GitHub Pages — see setup below)*

---

## What it does

The calculator solves a common infrastructure sizing problem: given a specific LLM and a target deployment configuration, how much GPU VRAM is required, and which GPU options can support it?

It computes three components of total VRAM demand:

| Component | What it represents |
|---|---|
| **Model weights** | Parameters × bytes per precision format (BF16, FP8, INT4) |
| **KV cache** | Per-token memory × context length × concurrent users × utilization |
| **Serving overhead** | Framework buffers (vLLM, TensorRT-LLM), CUDA kernels, activation workspace |

It then maps the total against a GPU library at 1×, 2×, and 4× GPU counts, with color-coded fit ratings (Comfortable / Tight / Insufficient) and a live headroom indicator.

---

## How the KV cache math works

The KV cache stores Key and Value vectors for every token in the active context window. The formula is:

```
Bytes per token = 2 × num_layers × num_KV_heads × head_dim × bytes_per_element
```

For a model like Nemotron-3-Super-120B with GQA (8 KV heads), at FP8:

```
Per token = 2 × 96 layers × 8 KV heads × 128 head_dim × 1 byte (FP8)
          = 196,608 bytes = 0.1875 MB

Per session (128K context) = 0.1875 MB × 131,072 tokens = 24.6 GB (maximum)
At 60% average utilization = 14.8 GB/session × 5 users = 74 GB KV cache
```

The calculator shows this breakdown step by step so users can audit the logic.

**Key concepts:**

- **GQA (Grouped Query Attention):** Most modern models share Key/Value heads across groups of Query heads, dramatically reducing KV cache size. The KV heads field captures this — not the full query head count.
- **Average vs. worst-case:** Sessions build context incrementally. The utilization slider accounts for the realistic average across concurrent users; the worst-case figure assumes every user simultaneously fills their entire context window.
- **MoE models:** For Mixture-of-Experts architectures (e.g., Nemotron 120B, Mixtral), only a fraction of parameters are active per forward pass, but *all* parameters must reside in VRAM. The full parameter count is used for weight memory estimation.

---

## GPU library

The tool includes the following GPUs:

| GPU | VRAM | Memory BW | MIG | NVLink |
|---|---|---|---|---|
| H200 SXM | 141 GB | 4.8 TB/s | Yes | Yes |
| H200 NVL | 141 GB | 4.8 TB/s | Yes | Yes |
| H100 SXM 80GB | 80 GB | 3.35 TB/s | Yes | Yes |
| H100 PCIe 80GB | 80 GB | 2.0 TB/s | Yes | No |
| A100 SXM 80GB | 80 GB | 2.0 TB/s | Yes | Yes |
| A100 PCIe 80GB | 80 GB | 1.94 TB/s | Yes | No |
| RTX PRO 6000 Blackwell | 96 GB | 1.7 TB/s | Yes | Limited |
| RTX PRO 6000D Blackwell | 84 GB | 1.5 TB/s | Yes | Limited |
| RTX 6000 Ada | 48 GB | 960 GB/s | No | No |
| L40S | 48 GB | 864 GB/s | No | No |
| L40 | 48 GB | 864 GB/s | No | No |
| A40 | 48 GB | 696 GB/s | No | No |
| RTX 4090 | 24 GB | 1.0 TB/s | No | No |

---

## Model preset library

| Model | Params | Class | Notes |
|---|---|---|---|
| Nemotron-3-Super-120B-A12B | 120B | MoE | 8 KV heads (GQA) |
| gpt-oss-120b | 120B | MoE | 8 KV heads (GQA) |
| Llama 3.1 405B | 405B | Dense | 8 KV heads (GQA) |
| Llama 3.3 70B Instruct | 70B | Dense | 8 KV heads (GQA) |
| Qwen2.5-Coder-72B Instruct | 72B | Dense | 8 KV heads (GQA) |
| Mixtral 8×22B | 141B | MoE | 8 KV heads (GQA) |
| Qwen2.5-32B Instruct | 32B | Dense | 8 KV heads (GQA) |
| Llama 3.2 90B Vision | 90B | Dense | 8 KV heads (GQA) |
| Devstral-Small-2-24B | 24B | Dense | 8 KV heads (GQA) |
| Mistral-Small-22B | 22B | Dense | 8 KV heads (GQA) |
| Mixtral 8×7B | 46.7B | MoE | 8 KV heads (GQA) |
| Llama 3.1 8B Instruct | 8B | Dense | 8 KV heads (GQA) |
| Qwen2.5-7B Instruct | 7B | Dense | 4 KV heads (GQA) |
| Mistral 7B v0.3 | 7B | Dense | 8 KV heads (GQA) |

For models not in the preset list, select **custom / manual entry** and fill in the architecture fields directly.

---

## Setup and hosting

### Option A — Open locally (no setup required)

1. Clone or download this repository
2. Open `index.html` in any modern browser
3. No server, no install, no dependencies

```bash
git clone https://github.com/your-org/llm-gpu-sizing-calculator.git
cd llm-gpu-sizing-calculator
open index.html   # macOS
# or double-click index.html in Windows Explorer / Linux file manager
```

### Option B — Host on GitHub Pages (recommended)

1. Push this repository to GitHub
2. Go to **Settings → Pages**
3. Under **Source**, select **Deploy from a branch**
4. Set branch to `main` and folder to `/ (root)`
5. Click **Save**
6. Your tool will be live at `https://your-org.github.io/llm-gpu-sizing-calculator` within a few minutes
7. Update the link at the top of this README

---

## Adding new GPUs or model presets

All GPU and model data is defined in the `<script>` block near the bottom of `index.html`.

**To add a GPU**, add an entry to the `GPUS` array:

```javascript
{ name:'H100 NVL', vram:94, bw:'3.35 TB/s', mig:'Yes', nvlink:'Yes' },
```

**To add a model preset**, add an entry to the `PRESETS` object and a matching `<option>` in the select element:

```javascript
// In PRESETS object:
mymodel: { layers:80, kvheads:8, headdim:128, params:70, label:'My Model 70B' },

// In the <select id="preset"> element:
<option value="mymodel">My Model 70B</option>
```

Fields for model presets:

| Field | Description |
|---|---|
| `layers` | Number of transformer layers |
| `kvheads` | Number of KV heads after GQA |
| `headdim` | Attention head dimension (usually 64 or 128) |
| `params` | Total parameter count in billions |

---

## Limitations and caveats

- **Estimates only.** Results are engineering approximations based on published architecture specs and standard KV cache arithmetic. Actual VRAM usage varies by serving framework, batch size, sampling parameters, and quantization implementation.
- **Validate before ordering.** Always profile with your actual serving stack (vLLM, TensorRT-LLM, etc.) before finalizing hardware procurement.
- **Multi-GPU assumes tensor parallelism.** The 2× and 4× GPU columns assume total VRAM is pooled via NVLink or equivalent. For GPUs without NVLink (e.g., L40S), multi-GPU serving requires pipeline parallelism, which has different efficiency characteristics.
- **Activation memory not included.** Forward pass activation buffers are small relative to weights and KV cache for typical batch sizes, but may be material for very large batches.

---

## License

MIT — free to use, modify, and redistribute.
