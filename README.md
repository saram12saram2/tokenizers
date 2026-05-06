# tokencost

Compare LLM token cost and context stress across languages and model families — side by side, in the terminal.

**Zero dependencies. Python 3.8+.**

```bash
python3 tokencost.py --all-models ko
python3 tokencost.py --compare ko en --model llama3
python3 tokencost.py --language hi --model o200k --requests 500000
```

---

## The problem

LLM pricing is per token. Tokenizers are mostly trained on English data. The same meaning in Korean, Hindi, or Arabic requires 2–5× more tokens than in English — and you pay for every one of them.

This is not a property of the languages. The academic consensus is consistent: **insufficient training data representation** — either because corpora deprioritized a language, or because it has less text on the internet. Korean's Hangul has only ~2,350 commonly used syllable blocks. Any tokenizer with a 100k+ vocabulary *could* cover it near-perfectly. The gap is a choice.

This tool puts that gap in dollar terms.

---

## Install

```bash
# No install needed — single file, no dependencies
python3 tokencost.py --help
```

---

## Commands

| What you want | Command |
|---|---|
| All model families for one language | `--all-models LANG` |
| Two languages side by side | `--compare LANG_A LANG_B --model MODEL` |
| One language + one model, detailed | `--language LANG --model MODEL` |
| Interactive menu | *(no arguments)* |
| List language codes | `--list-languages` |
| List model keys | `--list-models` |

<details>
<summary>All flags</summary>

| Flag | Short | Default | Description |
|---|---|---|---|
| `--language` | `-l` | — | Language code (`ko`, `ar`, `hi` …) |
| `--model` | `-m` | — | Model key (`llama3`, `o200k` …) |
| `--compare` | `-c` | — | Two language codes |
| `--all-models` | `-a` | — | All model families for a language |
| `--requests` | `-r` | `100000` | Monthly request volume |
| `--input-tokens` | `-i` | `600` | Avg input tokens/request (English baseline) |
| `--output-tokens` | `-o` | `250` | Avg output tokens/request |
| `--list-languages` | — | — | All language codes |
| `--list-models` | — | — | All model keys |

</details>

---

## Coverage

<details>
<summary>19 languages</summary>

| Code | Language | Code | Language |
|---|---|---|---|
| `en` | English | `ko` | Korean |
| `fr` | French | `ar` | Arabic |
| `de` | German | `hi` | Hindi |
| `es` | Spanish | `th` | Thai |
| `pt` | Portuguese | `bn` | Bengali |
| `ru` | Russian | `tr` | Turkish |
| `ja` | Japanese | `vi` | Vietnamese |
| `zh` | Chinese | `sw` | Swahili |
| `pl` | Polish | `id` | Indonesian |
| `uk` | Ukrainian | | |

</details>

<details>
<summary>6 model families</summary>

| Key | Model | Note |
|---|---|---|
| `gpt2` | GPT-2 legacy | Reference only |
| `cl100k` | OpenAI cl100k | GPT-3.5-turbo, GPT-4-turbo |
| `o200k` | OpenAI o200k | GPT-4o, ChatGPT 5.x |
| `mistral` | Mistral family | Mistral-7B, Mixtral |
| `llama3` | Llama 3 family | Best Korean RTC |
| `qwen25` | Qwen 2.5 | Near-parity on Chinese; weaker on Korean |

</details>

---

## What the numbers mean

**RTC** (Relative Token Cost) = `language_tokens / english_tokens` for the same meaning.
`1.0×` is parity. `2.0×` means twice the input cost.

**Context loss** = `(1 − 1/RTC) × 100`.
At RTC 2.35× (Korean on GPT-4o), 57% of a 128k window is overhead — only 54k tokens of actual content fit.

**Risk levels:** `low` < 1.3× · `moderate` 1.3–2.0× · `high` 2.0–3.5× · `severe` > 3.5×

<details>
<summary>Cost formula</summary>

```
monthly_input_cost = requests × (baseline_input_tokens × RTC) × price_per_million / 1,000,000
```

Output token cost is not multiplied by RTC — output length is determined by the model response, not the input language. This makes cost estimates a conservative lower bound.

</details>

---

## Tests

Run these in order. Each takes under a second.

**1. Sanity check — list languages**
```bash
python3 tokencost.py --list-languages
```
Korean (`ko`) should show `2.35×`. English should show `1.00×`.

---

**2. Korean across all models**
```bash
python3 tokencost.py --all-models ko
```
Llama 3 should appear first (lowest RTC ~1.88×). Qwen 2.5 should show ~3.05× despite being cheapest per token. GPT-2 should show ~5.2× and ~80% context loss.

---

**3. Side-by-side: Korean vs English**
```bash
python3 tokencost.py --compare ko en --model llama3
```
Korean: RTC 1.88×, context loss ~47%, effective context ~68k.
English: RTC 1.00×, context loss 0%, effective context 128k.

---

**4. The Qwen anomaly**
```bash
python3 tokencost.py --compare zh ko --model qwen25
```
Chinese on Qwen: ~1.02× — near English parity.
Korean on Qwen: ~3.05× — same vendor, same tokenizer family, 3× penalty.
This is the core insight: strong CJK coverage does not transfer to Hangul.

---

**5. Custom traffic**
```bash
python3 tokencost.py --language hi --model o200k --requests 1000000 --input-tokens 1000 --output-tokens 500
```
Expected monthly input cost: ~$4,050. vs English baseline: ~+$1,550 (+21%).

---

**6. Latin script sanity check**
```bash
python3 tokencost.py --language fr --model o200k
```
French RTC should be ~1.11×, risk level `low`, cost delta ~4% above English. Confirms Latin-script languages get near-equitable treatment.

---

**7. Error handling**
```bash
python3 tokencost.py --language xx --model llama3    # → Unknown language: xx
python3 tokencost.py --language ko --model gpt5      # → Unknown model: gpt5
```
Both should exit with a non-zero code.

---

**8. Cross-model Korean comparison**
```bash
python3 tokencost.py --language zh --model qwen25   # ~1.02×
python3 tokencost.py --language ko --model qwen25   # ~3.05×
python3 tokencost.py --language ko --model llama3   # ~1.88×
```
Qwen at $0.08/M with 3.05× RTC vs Llama3 at $0.20/M with 1.88× — Llama3 produces a lower monthly bill for Korean-primary products at most traffic levels.

---

## Extending

<details>
<summary>Add a language</summary>

Add to `RTC_DATA` and `LANGUAGE_NAMES` in the script:

```python
"ta": { "gpt2": 5.50, "cl100k": 3.20, "o200k": 2.80, "mistral": 2.60, "llama3": 2.45, "qwen25": 3.00 },
# LANGUAGE_NAMES
"ta": "Tamil",
```

Use FLORES-200 aligned sentence pairs for RTC values. Do not mix aligned and naturalistic text — they produce different results.

</details>

<details>
<summary>Add a model family</summary>

```python
MODEL_FAMILIES["gemini"] = {
    "label": "Google Gemini", "key": "gemini",
    "context_k": 128, "input_per_m": 0.075, "output_per_m": 0.30,
    "note": "Gemini 1.5 Flash family.",
}
# Then add "gemini": <rtc_value> to each language in RTC_DATA
```

</details>

<details>
<summary>Update pricing</summary>

Find `MODEL_FAMILIES` at the top of `tokencost.py` and update `input_per_m` / `output_per_m` (USD per million tokens).

</details>

---

## Data source

RTC values from the FLORES-200 aligned multilingual benchmark.

- Petrov et al. (2023) — [`aleksandarpetrov/tokenization-fairness`](https://github.com/aleksandarpetrov/tokenization-fairness)
- *Tokenization Disparities as Infrastructure Bias* — arXiv:2510.12389
- *The Token Tax: Systematic Bias in Multilingual Tokenization* — arXiv:2509.05486
- FLORES-200 — [`haoranxu/FLORES-200`](https://huggingface.co/datasets/haoranxu/FLORES-200)

---

MIT license.