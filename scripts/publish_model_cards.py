"""Build the Hugging Face model cards from the published benchmark artefacts.

Downloads eval/eval_results.json, eval/predictions.parquet and training_summary.json from the adapter repo,
prints routing accuracy split by label provenance (synthetic = gold labels, ai4privacy = keyword-labeler labels),
writes both model cards to model_cards/, and uploads them with --push.

    pip install pandas pyarrow huggingface_hub
    python scripts/publish_model_cards.py            # preview: writes model_cards/*.md
    hf auth login                                     # write-scope token, once
    python scripts/publish_model_cards.py --push     # upload both READMEs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

HF_USER = "Zain9989"
LORA_REPO = f"{HF_USER}/pii-masker-qwen3-4b-lora"
GGUF_REPO = f"{HF_USER}/pii-masker-qwen3-4b-GGUF"
DATASET_REPO = f"{HF_USER}/enterprise-pii-ticket-sft"
GITHUB = "https://github.com/zainabbas14/slm-pii-masker"
WANDB = "https://wandb.ai/za488778-zelulabs/slm-pii-masker/runs/zt81un21"
GGUF_FILE = "pii-masker-qwen3-4b-Q4_K_M.gguf"
BASE, FT = "Base Qwen3-4B", "Fine-tuned (QLoRA)"

ROWS = [
    ("Strict JSON (raw output is pure JSON)", "strict_json_rate"),
    ("Schema adherence (Pydantic)", "schema_adherence"),
    ("Text fidelity (no rephrasing)", "text_fidelity"),
    ("PII precision (typed, exact span)", "pii_precision"),
    ("PII recall (typed, exact span)", "pii_recall"),
    ("PII F1 (typed, exact span)", "pii_f1"),
    ("**Redaction recall (PII values not leaked)**", "redaction_recall"),
    ("Department accuracy", "category_accuracy"),
    ("Department macro-F1", "category_macro_f1"),
    ("Urgency accuracy", "urgency_accuracy"),
]


def fetch(local_dir: Path | None) -> tuple[dict, pd.DataFrame, dict]:
    if local_dir:
        return (json.loads((local_dir / "eval_results.json").read_text()),
                pd.read_parquet(local_dir / "predictions.parquet"),
                json.loads((local_dir / "training_summary.json").read_text()))
    get = lambda f: hf_hub_download(LORA_REPO, f)  # noqa: E731
    return (json.loads(Path(get("eval/eval_results.json")).read_text()),
            pd.read_parquet(get("eval/predictions.parquet")),
            json.loads(Path(get("training_summary.json")).read_text()))


def provenance_table(preds: pd.DataFrame) -> pd.DataFrame:
    g = preds.groupby(["model", "origin"])
    out = g.agg(n=("category_correct", "size"), department_acc=("category_correct", "mean"),
                urgency_acc=("urgency_correct", "mean"), schema_valid=("schema_valid", "mean"))
    red = g[["values_redacted", "gold_values"]].sum()
    out["redaction_recall"] = red["values_redacted"] / red["gold_values"].clip(lower=1)
    return out


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def benchmark_md(results: dict) -> str:
    q, sp = results["quality"], results["speed"]
    lines = ["| Metric | Base Qwen3-4B | Fine-tuned (QLoRA) |", "|---|---:|---:|"]
    lines += [f"| {name} | {pct(q[BASE][k])} | **{pct(q[FT][k])}** |" for name, k in ROWS]
    lines.append(f"| Tokens/sec (single request, T4, transformers) | {sp[BASE]['tokens_per_sec_mean']} | "
                 f"{sp[FT]['tokens_per_sec_mean']} |")
    lines.append(f"| Latency p50 / p95 | {sp[BASE]['latency_p50_s']} s / {sp[BASE]['latency_p95_s']} s | "
                 f"{sp[FT]['latency_p50_s']} s / {sp[FT]['latency_p95_s']} s |")
    return "\n".join(lines)


def provenance_md(prov: pd.DataFrame) -> str:
    lines = ["| Rows | n | Department acc. (base → fine-tuned) | Urgency acc. (base → fine-tuned) |",
             "|---|---:|---:|---:|"]
    labels = {"synthetic": "Synthetic (gold labels)", "ai4privacy": "ai4privacy (keyword-labeler labels)"}
    for origin, label in labels.items():
        if (FT, origin) not in prov.index:
            continue
        b, f = prov.loc[(BASE, origin)], prov.loc[(FT, origin)]
        lines.append(f"| {label} | {int(f['n'])} | {pct(b['department_acc'])} → **{pct(f['department_acc'])}** | "
                     f"{pct(b['urgency_acc'])} → **{pct(f['urgency_acc'])}** |")
    return "\n".join(lines)


def per_type_md(results: dict) -> str:
    pt = results.get("per_type_recall", {})
    if not pt:
        return ""
    tags = list(pt[FT].keys())
    lines = ["| Entity | Base recall | Fine-tuned recall |", "|---|---:|---:|"]
    for t in tags:
        b, f = pt[BASE].get(t), pt[FT].get(t)
        if f is None:
            continue
        lines.append(f"| `[{t}]` | {pct(b) if b is not None else '–'} | **{pct(f)}** |")
    return "\n".join(lines)


def lora_card(results: dict, prov: pd.DataFrame, ts: dict) -> str:
    q = results["quality"]
    leak_base, leak_ft = 1 - q[BASE]["redaction_recall"], 1 - q[FT]["redaction_recall"]
    lora = ts.get("lora", {})
    return f"""---
license: apache-2.0
language: [en]
base_model: unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit
datasets: [{DATASET_REPO}]
library_name: peft
pipeline_tag: text-generation
tags: [pii, redaction, data-privacy, ticket-classification, json, qlora, unsloth, trl, qwen3]
---

# PII Masker & Ticket Router: Qwen3-4B QLoRA adapter

LoRA adapter that turns **Qwen3-4B-Instruct-2507** into an enterprise **PII masker and ticket router**. Given a raw
support ticket or HR/legal snippet, it returns one strict JSON object:

```json
{{"masked_text": "Hi, I'm [PERSON]. My card [CREDIT_CARD] was charged twice, refund ASAP. [EMAIL]",
 "detected_pii_types": ["PERSON", "EMAIL", "CREDIT_CARD"],
 "ticket_category": "Billing_and_Payments",
 "urgency_level": "High"}}
```

Tags: `[PERSON]` `[EMAIL]` `[PHONE]` `[SSN]` `[CREDIT_CARD]` `[IP_ADDRESS]` `[API_KEY]`.
Departments: `Billing_and_Payments`, `Technical_Support`, `Security_and_Access`, `Legal_and_Compliance`,
`General_Inquiry`. Urgency: `Low`, `Medium`, `High`.

> The untuned base model leaked **{pct(leak_base)}** of PII values; this adapter leaks **{pct(leak_ft)}**.

**Code, API server, tests:** [{GITHUB}]({GITHUB}) · **CPU / Ollama build:** [{GGUF_REPO}](https://huggingface.co/{GGUF_REPO})
· **Dataset:** [{DATASET_REPO}](https://huggingface.co/datasets/{DATASET_REPO}) · **Training run:** [W&B]({WANDB})

## Benchmark

{results['n_eval']} held-out test tickets, identical system prompt, greedy decoding, 4-bit weights, Tesla T4.
PII is scored strictly: a tag only counts if it replaces exactly the right characters under the right type, and
outputs that rephrase the ticket get no PII credit.

{benchmark_md(results)}

### Recall by entity type

{per_type_md(results)}

### Routing accuracy by label provenance

Real ai4privacy rows have no routing labels, so their department/urgency labels come from a keyword labeler;
synthetic rows carry gold labels. Accuracy on real rows therefore partly measures agreement with the labeler.

{provenance_md(prov)}

Synthetic test and training tickets share templates, so the synthetic row measures template recall more than generalisation.

Raw per-ticket predictions: [`eval/predictions.parquet`](https://huggingface.co/{LORA_REPO}/blob/main/eval/predictions.parquet).

## Usage

The system prompt is in [`prompt_config.json`](https://huggingface.co/{LORA_REPO}/blob/main/prompt_config.json);
use it verbatim and wrap the ticket in the `user_template`.

```python
import json
from huggingface_hub import hf_hub_download
from unsloth import FastLanguageModel

cfg = json.load(open(hf_hub_download("{LORA_REPO}", "prompt_config.json")))
model, tok = FastLanguageModel.from_pretrained("{LORA_REPO}", max_seq_length=2048, load_in_4bit=True)
FastLanguageModel.for_inference(model)

ticket = "Hi, I'm Jane Doe, card 4111 1111 1111 1111 was charged twice, refund ASAP."
msgs = [{{"role": "system", "content": cfg["system_prompt"]}},
        {{"role": "user", "content": cfg["user_template"].format(ticket=ticket)}}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
out = model.generate(ids, max_new_tokens=512, do_sample=False)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```

For production use, put it behind the FastAPI service in the GitHub repo: it validates every answer with
Pydantic, re-scans the output with deterministic regex, and falls back to rule-based masking if the model fails.

## Training

| | |
|---|---|
| Base | `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0), Unsloth 4-bit build |
| Method | QLoRA r={lora.get('r', 16)}, α={lora.get('alpha', 32)}, all attention + MLP projections; loss on assistant tokens only |
| Data | {ts.get('train_rows', 3400):,} training examples; {ts.get('epochs', 2)} epochs, {ts.get('global_steps', 850)} steps, effective batch {ts.get('effective_batch_size', 8)}, LR {ts.get('learning_rate', 2e-4)} |
| Hardware | Free Colab Tesla T4: {ts.get('train_minutes_this_session', '–')} min, {ts.get('peak_vram_gb', '–')} GB peak VRAM |
| Validation loss | best {ts.get('best_eval_loss', '–')} |

Chat format: plain ChatML (the Qwen3 hybrid-thinking template was replaced so no `<think>` block is ever emitted).

## Limitations

- Typed redaction, not anonymisation: addresses, dates and job titles are not masked; two people both become `[PERSON]`.
- English only. Training data is largely synthetic (ai4privacy + templates); expect lower accuracy on real tickets
  until evaluated in-domain.
- Routing labels on real rows are weak labels (see the provenance table).
"""


def gguf_card(results: dict) -> str:
    q = results["quality"]
    return f"""---
license: apache-2.0
language: [en]
base_model: {LORA_REPO}
tags: [gguf, llama.cpp, ollama, pii, redaction, ticket-classification, qwen3]
---

# PII Masker & Ticket Router: GGUF (Q4_K_M)

[`{LORA_REPO}`](https://huggingface.co/{LORA_REPO}) merged into Qwen3-4B-Instruct-2507 and quantized to
**Q4_K_M (2.3 GB)** for CPU or GPU inference with Ollama or llama.cpp.

On {results['n_eval']} held-out tickets the fine-tuned model reached **{pct(q[FT]['pii_f1'])} PII F1**,
**{pct(q[FT]['redaction_recall'])} redaction recall** and **{pct(q[FT]['schema_adherence'])} schema adherence**
(full benchmark on the [adapter card](https://huggingface.co/{LORA_REPO})). Measured on a laptop CPU through
Ollama: ~10–11 s per ticket once loaded.

## Run with Ollama

Use the repo's `Modelfile`: it sets the ChatML template, the exact system prompt, `temperature 0` and
`repeat_penalty 1.0` (the default 1.1 penalises the verbatim copying this task depends on).

```bash
git clone {GITHUB} && cd slm-pii-masker
hf download {GGUF_REPO} {GGUF_FILE} --local-dir models
ollama create pii-masker -f Modelfile
uvicorn app.inference_server:app --port 8000     # FastAPI + validation + regex guard
```

## Run with llama.cpp

```bash
llama-server -m models/{GGUF_FILE} --port 8080 --temp 0 --repeat-penalty 1.0 --ctx-size 4096
```

Send the system prompt from [`prompt_config.json`](https://huggingface.co/{GGUF_REPO}/blob/main/prompt_config.json)
and wrap the ticket in its `user_template`, or run the GitHub repo's API with `BACKEND=llamacpp`.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="upload both README.md files to the Hub")
    ap.add_argument("--local-dir", type=Path, help="read artefacts from a folder instead of the Hub (testing)")
    args = ap.parse_args()

    results, preds, ts = fetch(args.local_dir)
    prov = provenance_table(preds)
    pd.options.display.float_format = "{:.3f}".format
    print("Routing accuracy by label provenance:\n")
    print(prov.to_string(), "\n")
    print(provenance_md(prov), "\n")

    out = Path("model_cards")
    out.mkdir(exist_ok=True)
    cards = {LORA_REPO: lora_card(results, prov, ts), GGUF_REPO: gguf_card(results)}
    for repo, text in cards.items():
        path = out / f"{repo.split('/')[1]}.md"
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path} ({len(text):,} chars)")

    if args.push:
        api = HfApi()
        for repo, text in cards.items():
            api.upload_file(path_or_fileobj=text.encode(), path_in_repo="README.md", repo_id=repo,
                            commit_message="Model card with benchmark results")
            print(f"✅ https://huggingface.co/{repo}")
    else:
        print("\nPreview only. Re-run with --push to upload.")


if __name__ == "__main__":
    main()
