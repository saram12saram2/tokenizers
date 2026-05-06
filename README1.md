# tokencost

> Compare LLM token cost and context stress across languages and model families.

A zero-dependency CLI tool that surfaces the tokenization tax paid by non-English languages — and translates it into monthly API cost and effective context window loss, side by side.

---

## Why this exists

LLM pricing is per token. Tokenizers are trained mostly on English data. The result: the same meaning expressed in Korean, Hindi, or Arabic requires 2–5× more tokens than in English — and you pay for every one of them.

The academic consensus on root cause is consistent across multiple studies: **insufficient language representation in training data** — either because training corpora deprioritized a language, or because the language simply has less text on the internet. This is not a property of the languages themselves. Korean's Hangul script, for example, has only ~2,350 commonly used syllable blocks and could theoretically be covered near-perfectly by any tokenizer with a 100k+ vocabulary. The gap is a data prioritization decision.

This tool makes that gap visible in production terms: dollars per month and effective context window per request.

**Data source:** RTC values are derived from the FLORES-200 aligned multilingual benchmark, consistent with:
- Petrov et al. (2023) *"Language Model Tokenizers Introduce Unfairness Between Languages"*
- [`aleksandarpetrov/tokenization-fairness`](https://github.com/aleksandarpetrov/tokenization-fairness)

---

## Requirements

- Python 3.8+
- No dependencies — stdlib only

```bash
python3 --version   # must be 3.8 or higher
```

---

## Installation

```bash
# Clone or download — no package install needed
curl -O https://raw.githubusercontent.com/your-repo/tokencost/main/tokencost.py

# Make executable (optional)
chmod +x tokencost.py
```

---

## Quick start

```bash
# Interactive menu
python3 tokencost.py

# Korean across all 6 model families
python3 tokencost.py --all-models ko

# Korean vs English, side by side, on Llama 3
python3 tokencost.py --compare ko en --model llama3

# Detail view: Korean on GPT-4o with custom traffic
python3 tokencost.py --language ko --model o200k --requests 500000 --input-tokens 800
```

---

## Usage reference

```
usage: tokencost [-h]
                 [--language LANG] [--model MODEL]
                 [--compare LANG_A LANG_B]
                 [--all-models LANG]
                 [--requests N] [--input-tokens N] [--output-tokens N]
                 [--list-languages] [--list-models]
```

| Flag | Short | Default | Description |
|---|---|---|---|
| `--language` | `-l` | — | Language code (e.g. `ko`, `ar`, `hi`) |
| `--model` | `-m` | — | Model family key (e.g. `llama3`, `o200k`) |
| `--compare` | `-c` | — | Two language codes to compare side by side |
| `--all-models` | `-a` | — | Show all model families for one language |
| `--requests` | `-r` | `100000` | Monthly request volume |
| `--input-tokens` | `-i` | `600` | Avg input tokens per request (English baseline) |
| `--output-tokens` | `-o` | `250` | Avg output tokens per request |
| `--list-languages` | — | — | Print all supported language codes |
| `--list-models` | — | — | Print all model family keys |

---

## Supported languages

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

```bash
python3 tokencost.py --list-languages   # see RTC overview for all languages
```

---

## Supported model families

| Key | Label | Notes |
|---|---|---|
| `gpt2` | GPT-2 legacy | Baseline reference; not for production |
| `cl100k` | OpenAI cl100k | GPT-3.5-turbo, GPT-4-turbo |
| `o200k` | OpenAI o200k | GPT-4o, ChatGPT 5.x |
| `mistral` | Mistral family | Mistral-7B, Mixtral |
| `llama3` | Llama 3 family | Meta Llama 3/3.1/3.3. Best Korean RTC |
| `qwen25` | Qwen 2.5 family | Near-parity on Chinese; weaker on Korean |

```bash
python3 tokencost.py --list-models
```

---

## What the output means

### RTC — Relative Token Cost
```
RTC = language_tokens / english_tokens   (same meaning, aligned text)
```
- `1.0×` — parity with English
- `2.0×` — twice the tokens, twice the input cost for the same content
- `5.0×` — five times the tokens; a 128k window delivers ~26k meaningful tokens

### Context loss %
```
context_loss = (1 - 1/RTC) × 100
```
At `RTC 2.35×` (Korean on GPT-4o): **57% of the context window is consumed by tokenization overhead**, not by actual content. A 128k window effectively delivers 54k tokens of meaning.

### Cost calculation
```
monthly_input_cost = requests × (baseline_input_tokens × RTC) × price_per_million / 1,000,000
```
Output token cost is not multiplied by RTC (output tokens are fixed by the model response, not the input language).

### Risk levels
| Level | RTC range | Meaning |
|---|---|---|
| `low` | < 1.3× | Minimal overhead vs English |
| `moderate` | 1.3–2.0× | Noticeable but manageable |
| `high` | 2.0–3.5× | Significant cost and context impact |
| `severe` | > 3.5× | Avoid for production at scale |

---

## Test instructions

### Test 1 — Verify basic output

```bash
python3 tokencost.py --list-languages
```

**Expected:** A table of 19 languages with their `o200k` RTC values. Korean (`ko`) should show `2.35×`. English should show `1.00×`.

---

### Test 2 — Korean across all model families

```bash
python3 tokencost.py --all-models ko
```

**Expected output shape:**
```
All model families · Korean (KO)
Traffic: 100,000 req/mo  · 600 input tok  · 250 output tok

MODEL FAMILY          RTC   CONTEXT LOSS   EFF. CTX   MONTHLY COST   vs EN
──────────────────────────────────────────────────────────────────────────────
Llama 3 family       1.88×        46.8%        68k          $XX.XX     +62%
Mistral family       2.15×        53.5%        60k          $XX.XX     +XX%
...
```

**What to verify:**
- Llama 3 should have the **lowest RTC** (~1.88×) and appear first (table is sorted by RTC ascending)
- Qwen 2.5 should show **highest RTC** (~3.05×) despite being cheap on Chinese
- GPT-2 should show **~5.2×** and ~80% context loss
- All `vs EN` values should be positive (Korean always costs more than English)

---

### Test 3 — Side-by-side language comparison

```bash
python3 tokencost.py --compare ko en --model llama3
```

**Expected:**
- Korean RTC: `1.88×`, English: `1.00×`
- Korean context loss: ~`46.8%`, English: `0.0%`
- Korean effective context: ~`68k`, English: `128k`
- Summary line: `English is cheaper by $X.XX / month (62% more expensive for Korean)`

---

### Test 4 — Same language, different model — verify Qwen anomaly

```bash
python3 tokencost.py --compare zh ko --model qwen25
```

**Expected:**
- Chinese (`zh`) RTC on Qwen 2.5: `~1.02×` — near English parity
- Korean (`ko`) RTC on Qwen 2.5: `~3.05×` — high cost despite same vendor
- Chinese should be dramatically cheaper than Korean on this model
- This is the Qwen anomaly: strong CJK coverage does not transfer to Hangul

---

### Test 5 — Custom traffic parameters

```bash
python3 tokencost.py --language hi --model o200k --requests 1000000 --input-tokens 1000 --output-tokens 500
```

**Expected:**
- Hindi on o200k: RTC `~1.62×`
- Effective input tokens: `~1,620` per request
- Monthly input cost should be roughly `1,000,000 × 1,620 × $2.50 / 1,000,000 = $4,050`
- vs English baseline should show `+$1,550` approximately (`+62%`)

---

### Test 6 — Compare high-cost vs low-cost language

```bash
python3 tokencost.py --compare hi sw --model llama3
```

**Expected:**
- Hindi RTC on llama3: `~2.50×`
- Swahili RTC on llama3: `~2.00×`
- Swahili should be cheaper and have less context loss
- Both should show significantly worse than English

---

### Test 7 — Latin script language (sanity check)

```bash
python3 tokencost.py --language fr --model o200k
```

**Expected:**
- French RTC: `~1.11×` — close to English
- Context loss: `~10%`
- Risk level: `low`
- Cost delta vs English: small (`~10%`)

This confirms Latin-script languages are treated near-equitably by modern tokenizers.

---

### Test 8 — Unknown language / model error handling

```bash
python3 tokencost.py --language xx --model llama3
```

**Expected:** `Unknown language: xx` error message, non-zero exit code.

```bash
python3 tokencost.py --language ko --model gpt5
```

**Expected:** `Unknown model: gpt5` error message, non-zero exit code.

---

### Test 9 — Chinese vs Korean on Qwen vs Llama3 (full anomaly test)

```bash
# Chinese on Qwen (should be best-in-class)
python3 tokencost.py --language zh --model qwen25

# Korean on Qwen (should be surprisingly expensive)
python3 tokencost.py --language ko --model qwen25

# Korean on Llama3 (should be cheaper than Korean on Qwen)
python3 tokencost.py --language ko --model llama3
```

**Expected pattern:**
- Chinese on Qwen: RTC `~1.02×` — near English parity
- Korean on Qwen: RTC `~3.05×` — 3× penalty
- Korean on Llama3: RTC `~1.88×` — better for Korean than Qwen despite Qwen being cheaper per token

**Interpretation:** The cheapest-per-token model is not always the cheapest model for your language. Qwen's low per-token price (`$0.08/M`) is offset by its 3× RTC on Korean. Llama3 at `$0.20/M` with 1.88× RTC may produce a lower monthly bill depending on traffic mix.

---

### Test 10 — Help text

```bash
python3 tokencost.py --help
```

**Expected:** Full usage reference with examples and data source citation.

---

## Interpreting results for your team

### Choosing a model for Korean-primary products

Run `--all-models ko` and look at **monthly total** column, not just RTC. Qwen 2.5 often appears cheapest in absolute terms (low per-token price) but the high Korean RTC can flip this at scale.

**Decision framework:**
1. Sort by RTC to find the most efficient tokenizer for your language
2. Check context loss — if above 50%, budget your prompts accordingly
3. Multiply by your actual traffic to get real monthly delta vs English baseline
4. Factor in generation quality — RTC and quality are correlated but not identical

### Context budget planning

At `RTC 2.35×` (Korean, GPT-4o), a 128k context window effectively holds:
```
128,000 / 2.35 ≈ 54,500 tokens of Korean meaning
```

Practical implications:
- Retrieval chunks should be sized at `chunk_size / RTC` to hit your intended semantic density
- System prompts in Korean consume 2.35× the context budget of the same prompt in English
- Conversation history truncation should account for RTC, not raw token count

---

## Data notes and limitations

- **RTC values are medians** across FLORES-200 aligned sentence pairs, not from a single sentence. Real text will vary.
- **Output token RTC is not applied** in cost calculations. In practice, output RTC depends on the task; for generation tasks it applies proportionally, for extraction tasks less so. This makes cost estimates conservative (lower bound).
- **Model pricing** reflects approximate current rates and will drift over time. Update `MODEL_FAMILIES` in the script to adjust.
- **Six tokenizer families** are covered. Models using tokenizers outside this set (e.g. Gemini, Claude) would require adding their RTC data.

---

## Extending the tool

### Adding a language

Add an entry to `RTC_DATA` and `LANGUAGE_NAMES`:

```python
# RTC_DATA
"ta": { "gpt2": 5.50, "cl100k": 3.20, "o200k": 2.80, "mistral": 2.60, "llama3": 2.45, "qwen25": 3.00 },

# LANGUAGE_NAMES
"ta": "Tamil",
```

RTC values should come from FLORES-200 benchmark runs using aligned sentence pairs. Do not mix aligned and naturalistic text — they give different results.

### Adding a model family

Add an entry to `MODEL_FAMILIES` and a column to `RTC_DATA`:

```python
MODEL_FAMILIES["gemini"] = {
    "label":        "Google Gemini",
    "key":          "gemini",
    "context_k":    128,
    "input_per_m":  0.075,
    "output_per_m": 0.30,
    "note":         "Gemini 1.5 Flash family.",
}

# Then add "gemini": <rtc_value> to each language in RTC_DATA
```

### Updating pricing

Locate `MODEL_FAMILIES` at the top of the script and update `input_per_m` and `output_per_m` values (USD per million tokens).

---

## References

- Petrov et al. (2023). *Language Model Tokenizers Introduce Unfairness Between Languages.* [`aleksandarpetrov/tokenization-fairness`](https://github.com/aleksandarpetrov/tokenization-fairness)
- *Tokenization Disparities as Infrastructure Bias.* arXiv:2510.12389
- *The Token Tax: Systematic Bias in Multilingual Tokenization.* arXiv:2509.05486
- FLORES-200 benchmark: [`haoranxu/FLORES-200`](https://huggingface.co/datasets/haoranxu/FLORES-200)

---

## License

MIT. Use freely, cite the data sources if you publish findings.