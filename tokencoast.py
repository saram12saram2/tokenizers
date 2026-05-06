#!/usr/bin/env python3
"""
tokencost — compare LLM token cost and context stress across languages and model families.

RTC (Relative Token Cost) data is derived from FLORES-200 aligned benchmark research,
consistent with: Petrov et al. (2023) "Language Model Tokenizers Introduce Unfairness
Between Languages" and the tokenization-fairness study (aleksandarpetrov/tokenization-fairness).

Academic consensus on the root cause: insufficient language representation in training
data — either because training corpora deprioritized the language, or because the language
simply has less text on the internet.

Usage:
    python tokencost.py
    python tokencost.py --language ko --model llama-3 --requests 100000 --input-tokens 600 --output-tokens 250
    python tokencost.py --list-languages
    python tokencost.py --list-models
    python tokencost.py --compare ko en --model o200k
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# DATA: RTC values per language per tokenizer family
# Source: FLORES-200 aligned benchmark, consistent with Petrov et al. (2023)
# and aleksandarpetrov/tokenization-fairness
# RTC = source_tokens / english_tokens for same meaning
# 1.0 = parity with English. 2.0 = twice the tokens.
# ─────────────────────────────────────────────────────────────────────────────

RTC_DATA: dict[str, dict[str, float]] = {
    #           gpt2   cl100k  o200k  mistral  llama3  qwen25
    "en": {     "gpt2": 1.00, "cl100k": 1.00, "o200k": 1.00, "mistral": 1.00, "llama3": 1.00, "qwen25": 1.00 },
    "fr": {     "gpt2": 1.22, "cl100k": 1.14, "o200k": 1.11, "mistral": 1.13, "llama3": 1.10, "qwen25": 1.12 },
    "de": {     "gpt2": 1.30, "cl100k": 1.20, "o200k": 1.17, "mistral": 1.18, "llama3": 1.15, "qwen25": 1.17 },
    "es": {     "gpt2": 1.25, "cl100k": 1.15, "o200k": 1.12, "mistral": 1.14, "llama3": 1.11, "qwen25": 1.13 },
    "pt": {     "gpt2": 1.28, "cl100k": 1.17, "o200k": 1.13, "mistral": 1.16, "llama3": 1.12, "qwen25": 1.15 },
    "ru": {     "gpt2": 2.20, "cl100k": 1.55, "o200k": 1.48, "mistral": 1.42, "llama3": 1.40, "qwen25": 1.50 },
    "ja": {     "gpt2": 2.80, "cl100k": 1.72, "o200k": 1.60, "mistral": 1.55, "llama3": 1.52, "qwen25": 1.15 },
    "zh": {     "gpt2": 2.90, "cl100k": 1.80, "o200k": 1.23, "mistral": 1.45, "llama3": 1.60, "qwen25": 1.02 },
    "ko": {     "gpt2": 5.20, "cl100k": 2.65, "o200k": 2.35, "mistral": 2.15, "llama3": 1.88, "qwen25": 3.05 },
    "ar": {     "gpt2": 4.20, "cl100k": 2.40, "o200k": 2.48, "mistral": 2.30, "llama3": 2.70, "qwen25": 2.60 },
    "hi": {     "gpt2": 7.13, "cl100k": 3.80, "o200k": 1.62, "mistral": 2.90, "llama3": 2.50, "qwen25": 4.50 },
    "tr": {     "gpt2": 2.80, "cl100k": 1.90, "o200k": 1.75, "mistral": 1.70, "llama3": 1.65, "qwen25": 1.80 },
    "vi": {     "gpt2": 3.10, "cl100k": 2.10, "o200k": 1.85, "mistral": 1.80, "llama3": 1.75, "qwen25": 1.90 },
    "th": {     "gpt2": 4.50, "cl100k": 2.80, "o200k": 2.60, "mistral": 2.40, "llama3": 2.35, "qwen25": 2.20 },
    "sw": {     "gpt2": 3.60, "cl100k": 2.50, "o200k": 2.20, "mistral": 2.10, "llama3": 2.00, "qwen25": 2.30 },
    "bn": {     "gpt2": 5.80, "cl100k": 3.50, "o200k": 2.90, "mistral": 2.70, "llama3": 2.60, "qwen25": 3.20 },
    "id": {     "gpt2": 1.55, "cl100k": 1.30, "o200k": 1.22, "mistral": 1.20, "llama3": 1.18, "qwen25": 1.25 },
    "pl": {     "gpt2": 1.75, "cl100k": 1.40, "o200k": 1.32, "mistral": 1.30, "llama3": 1.28, "qwen25": 1.35 },
    "uk": {     "gpt2": 2.40, "cl100k": 1.60, "o200k": 1.52, "mistral": 1.48, "llama3": 1.45, "qwen25": 1.55 },
}

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",   "fr": "French",     "de": "German",
    "es": "Spanish",   "pt": "Portuguese", "ru": "Russian",
    "ja": "Japanese",  "zh": "Chinese",    "ko": "Korean",
    "ar": "Arabic",    "hi": "Hindi",      "tr": "Turkish",
    "vi": "Vietnamese","th": "Thai",       "sw": "Swahili",
    "bn": "Bengali",   "id": "Indonesian", "pl": "Polish",
    "uk": "Ukrainian",
}

MODEL_FAMILIES: dict[str, dict] = {
    "gpt2": {
        "label":        "GPT-2 legacy",
        "key":          "gpt2",
        "context_k":    4,
        "input_per_m":  0.50,
        "output_per_m": 1.50,
        "note":         "Legacy. Rarely used in production today.",
    },
    "cl100k": {
        "label":        "OpenAI cl100k (GPT-3.5/4)",
        "key":          "cl100k",
        "context_k":    128,
        "input_per_m":  1.00,
        "output_per_m": 2.00,
        "note":         "GPT-3.5-turbo, GPT-4-turbo family.",
    },
    "o200k": {
        "label":        "OpenAI o200k (GPT-4o)",
        "key":          "o200k",
        "context_k":    128,
        "input_per_m":  2.50,
        "output_per_m": 10.00,
        "note":         "GPT-4o, ChatGPT 5.x family.",
    },
    "mistral": {
        "label":        "Mistral family",
        "key":          "mistral",
        "context_k":    128,
        "input_per_m":  0.25,
        "output_per_m": 0.25,
        "note":         "Mistral-7B, Mixtral family.",
    },
    "llama3": {
        "label":        "Llama 3 family",
        "key":          "llama3",
        "context_k":    128,
        "input_per_m":  0.20,
        "output_per_m": 0.20,
        "note":         "Meta Llama 3 / 3.1 / 3.3. Best Korean RTC.",
    },
    "qwen25": {
        "label":        "Qwen 2.5 family",
        "key":          "qwen25",
        "context_k":    128,
        "input_per_m":  0.08,
        "output_per_m": 0.29,
        "note":         "Alibaba Qwen 2.5. Near-parity on Chinese; weaker on Korean.",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL COLOURS
# ─────────────────────────────────────────────────────────────────────────────

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    GREEN  = "\033[92m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    GREY   = "\033[90m"
    BG_DARK = "\033[40m"

def no_color(text: str) -> str:
    import re
    return re.sub(r'\033\[[0-9;]*m', '', text)

USE_COLOR = sys.stdout.isatty()

def c(color: str, text: str) -> str:
    return f"{color}{text}{C.RESET}" if USE_COLOR else text

# ─────────────────────────────────────────────────────────────────────────────
# CORE CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioResult:
    language:        str
    lang_name:       str
    model_key:       str
    model_label:     str
    rtc:             float
    monthly_requests:int
    base_input_tok:  int
    base_output_tok: int
    input_per_m:     float
    output_per_m:    float
    context_k:       int

    @property
    def effective_input_tokens(self) -> int:
        return max(1, round(self.base_input_tok * self.rtc))

    @property
    def effective_output_tokens(self) -> int:
        return self.base_output_tok  # output RTC effect is smaller; conservative estimate

    @property
    def monthly_input_cost(self) -> float:
        return self.monthly_requests * self.effective_input_tokens * self.input_per_m / 1_000_000

    @property
    def monthly_output_cost(self) -> float:
        return self.monthly_requests * self.effective_output_tokens * self.output_per_m / 1_000_000

    @property
    def monthly_total_cost(self) -> float:
        return self.monthly_input_cost + self.monthly_output_cost

    @property
    def context_loss_pct(self) -> float:
        return max(0.0, (1.0 - 1.0 / self.rtc) * 100.0)

    @property
    def effective_context_k(self) -> float:
        return self.context_k / self.rtc

    @property
    def risk_level(self) -> str:
        if self.rtc < 1.3:   return "low"
        if self.rtc < 2.0:   return "moderate"
        if self.rtc < 3.5:   return "high"
        return "severe"

    @property
    def risk_color(self) -> str:
        return {
            "low":      C.GREEN,
            "moderate": C.YELLOW,
            "high":     C.RED,
            "severe":   C.RED + C.BOLD,
        }[self.risk_level]


def compute(
    language: str,
    model_key: str,
    monthly_requests: int,
    base_input_tok: int,
    base_output_tok: int,
) -> ScenarioResult:
    rtc = RTC_DATA[language][model_key]
    model = MODEL_FAMILIES[model_key]
    return ScenarioResult(
        language=language,
        lang_name=LANGUAGE_NAMES[language],
        model_key=model_key,
        model_label=model["label"],
        rtc=rtc,
        monthly_requests=monthly_requests,
        base_input_tok=base_input_tok,
        base_output_tok=base_output_tok,
        input_per_m=model["input_per_m"],
        output_per_m=model["output_per_m"],
        context_k=model["context_k"],
    )

# ─────────────────────────────────────────────────────────────────────────────
# RENDERING
# ─────────────────────────────────────────────────────────────────────────────

def bar(value: float, max_value: float, width: int = 28, fill: str = "█", empty: str = "░") -> str:
    filled = round((value / max_value) * width) if max_value > 0 else 0
    filled = min(filled, width)
    return fill * filled + empty * (width - filled)

def rtc_bar(rtc: float) -> str:
    max_rtc = 6.0
    pct = min(rtc / max_rtc, 1.0)
    filled = round(pct * 20)
    color = C.GREEN if rtc < 1.5 else C.YELLOW if rtc < 2.5 else C.RED
    b = c(color, "█" * filled) + c(C.GREY, "░" * (20 - filled))
    return b

def context_bar(loss_pct: float) -> str:
    lost   = round(loss_pct / 100 * 20)
    remain = 20 - lost
    return c(C.RED, "▓" * lost) + c(C.GREEN, "░" * remain)

def fmt_cost(val: float) -> str:
    if val >= 1000:
        return f"${val:,.0f}"
    if val >= 10:
        return f"${val:.2f}"
    return f"${val:.4f}"

def print_header(title: str) -> None:
    width = 70
    print()
    print(c(C.BOLD + C.WHITE, "─" * width))
    print(c(C.BOLD + C.WHITE, f"  {title}"))
    print(c(C.BOLD + C.WHITE, "─" * width))

def print_result_card(r: ScenarioResult, en_result: Optional[ScenarioResult] = None) -> None:
    risk_label = c(r.risk_color, r.risk_level.upper())
    print()
    print(c(C.BOLD, f"  {r.lang_name} ({r.language.upper()})  ·  {r.model_label}"))
    print(c(C.GREY, f"  {'─' * 52}"))

    # RTC
    rtc_str  = c(r.risk_color + C.BOLD, f"{r.rtc:.2f}×")
    rtc_note = c(C.GREY, f"  ← relative token cost vs English")
    print(f"  {'RTC':<22} {rtc_str}  {rtc_bar(r.rtc)}  {risk_label}")

    # Effective tokens
    eff = c(C.CYAN, f"{r.effective_input_tokens:,}")
    base = c(C.GREY, f"(baseline {r.base_input_tok:,})")
    print(f"  {'Effective input tokens':<22} {eff} / request   {base}")

    # Context loss
    cl  = c(C.RED if r.context_loss_pct > 40 else C.YELLOW, f"{r.context_loss_pct:.1f}%")
    ek  = c(C.CYAN, f"{r.effective_context_k:.0f}k")
    print(f"  {'Context loss':<22} {cl} lost  [{context_bar(r.context_loss_pct)}]")
    print(f"  {'Effective context':<22} {ek} meaningful tokens  (of {r.context_k}k window)")

    # Cost
    print(c(C.GREY, f"  {'─' * 52}"))
    print(f"  {'Monthly input cost':<22} {c(C.BOLD, fmt_cost(r.monthly_input_cost))}")
    print(f"  {'Monthly output cost':<22} {c(C.BOLD, fmt_cost(r.monthly_output_cost))}")
    total_str = c(C.BOLD + C.WHITE, fmt_cost(r.monthly_total_cost))
    print(f"  {'Monthly total':<22} {total_str}")

    # Delta vs English
    if en_result and r.language != "en":
        delta_cost = r.monthly_total_cost - en_result.monthly_total_cost
        delta_pct  = (delta_cost / en_result.monthly_total_cost * 100) if en_result.monthly_total_cost else 0
        delta_str  = c(C.RED, f"+{fmt_cost(delta_cost)}  (+{delta_pct:.0f}%)")
        print(f"  {'vs English baseline':<22} {delta_str}")

    print()


def print_all_models_table(
    language: str,
    monthly_requests: int,
    base_input_tok: int,
    base_output_tok: int,
) -> None:
    lang_name = LANGUAGE_NAMES[language]
    print_header(f"All model families · {lang_name} ({language.upper()})")
    print(f"\n  {c(C.GREY, 'Traffic:')} {monthly_requests:,} req/mo  "
          f"· {base_input_tok} input tok  · {base_output_tok} output tok\n")

    en_results = {mk: compute("en", mk, monthly_requests, base_input_tok, base_output_tok)
                  for mk in MODEL_FAMILIES}

    col = f"{'MODEL FAMILY':<28}{'RTC':>6}  {'CONTEXT LOSS':>13}  {'EFF. CTX':>9}  {'MONTHLY COST':>13}  {'vs EN':>10}"
    print(c(C.GREY, f"  {col}"))
    print(c(C.GREY, f"  {'─' * 90}"))

    results = []
    for mk in MODEL_FAMILIES:
        r = compute(language, mk, monthly_requests, base_input_tok, base_output_tok)
        results.append(r)

    results.sort(key=lambda x: x.rtc)

    for r in results:
        rtc_s    = c(r.risk_color, f"{r.rtc:.2f}×")
        cl_s     = c(C.RED if r.context_loss_pct > 40 else C.YELLOW if r.context_loss_pct > 20 else C.GREEN,
                     f"{r.context_loss_pct:.1f}%")
        eff_ctx  = f"{r.effective_context_k:.0f}k"
        cost_s   = fmt_cost(r.monthly_total_cost)
        en_r     = en_results[r.model_key]
        delta    = r.monthly_total_cost - en_r.monthly_total_cost
        delta_pct= (delta / en_r.monthly_total_cost * 100) if en_r.monthly_total_cost else 0
        delta_s  = c(C.RED, f"+{delta_pct:.0f}%") if delta > 0 else c(C.GREEN, "—")

        row = (f"  {r.model_label:<28}"
               f"{rtc_s:>6}  "
               f"{cl_s:>13}  "
               f"{eff_ctx:>9}  "
               f"{cost_s:>13}  "
               f"{delta_s:>10}")
        print(row)

    print()
    print(c(C.GREY, "  RTC source: FLORES-200 aligned benchmark · Petrov et al. (2023) / tokenization-fairness"))
    print(c(C.GREY, "  Root cause: training data representation — language volume on the internet + corpus prioritization"))
    print()


def print_comparison(
    lang_a: str,
    lang_b: str,
    model_key: str,
    monthly_requests: int,
    base_input_tok: int,
    base_output_tok: int,
) -> None:
    model  = MODEL_FAMILIES[model_key]
    r_a    = compute(lang_a, model_key, monthly_requests, base_input_tok, base_output_tok)
    r_b    = compute(lang_b, model_key, monthly_requests, base_input_tok, base_output_tok)
    en_r   = compute("en",   model_key, monthly_requests, base_input_tok, base_output_tok) \
             if lang_a != "en" and lang_b != "en" else None

    print_header(f"Side-by-side · {r_a.lang_name} vs {r_b.lang_name}  ·  {model['label']}")
    print(f"\n  {c(C.GREY, 'Traffic:')} {monthly_requests:,} req/mo  "
          f"· {base_input_tok} input tok  · {base_output_tok} output tok\n")

    col_w = 28
    h_a   = c(C.BOLD + C.CYAN,  f"{r_a.lang_name} ({lang_a.upper()})")
    h_b   = c(C.BOLD + C.WHITE, f"{r_b.lang_name} ({lang_b.upper()})")
    print(f"  {'':30}{h_a:<38}{h_b}")
    print(c(C.GREY, f"  {'─' * 80}"))

    rows = [
        ("RTC vs English",         f"{r_a.rtc:.2f}×",                     f"{r_b.rtc:.2f}×"),
        ("Effective input tokens",  f"{r_a.effective_input_tokens:,}",     f"{r_b.effective_input_tokens:,}"),
        ("Context loss",           f"{r_a.context_loss_pct:.1f}%",         f"{r_b.context_loss_pct:.1f}%"),
        ("Effective context",      f"{r_a.effective_context_k:.0f}k tokens", f"{r_b.effective_context_k:.0f}k tokens"),
        ("Monthly input cost",     fmt_cost(r_a.monthly_input_cost),       fmt_cost(r_b.monthly_input_cost)),
        ("Monthly output cost",    fmt_cost(r_a.monthly_output_cost),      fmt_cost(r_b.monthly_output_cost)),
        ("Monthly total",          fmt_cost(r_a.monthly_total_cost),       fmt_cost(r_b.monthly_total_cost)),
    ]

    for label, val_a, val_b in rows:
        print(f"  {c(C.GREY, label):<30}{val_a:<30}{val_b}")

    # winner
    print(c(C.GREY, f"\n  {'─' * 80}"))
    if r_a.monthly_total_cost < r_b.monthly_total_cost:
        cheaper, more_ex = r_a, r_b
    else:
        cheaper, more_ex = r_b, r_a

    delta     = abs(more_ex.monthly_total_cost - cheaper.monthly_total_cost)
    delta_pct = (delta / cheaper.monthly_total_cost * 100) if cheaper.monthly_total_cost else 0
    print(f"\n  {c(C.BOLD, cheaper.lang_name)} is cheaper by "
          f"{c(C.GREEN, fmt_cost(delta))} / month  ({delta_pct:.0f}% more expensive for {more_ex.lang_name})")

    ctx_winner = r_a if r_a.context_loss_pct < r_b.context_loss_pct else r_b
    ctx_loser  = r_b if ctx_winner == r_a else r_a
    ctx_gap    = abs(r_a.context_loss_pct - r_b.context_loss_pct)
    print(f"  {c(C.BOLD, ctx_winner.lang_name)} has {ctx_gap:.1f}pp less context loss "
          f"({ctx_winner.effective_context_k:.0f}k vs {ctx_loser.effective_context_k:.0f}k effective tokens)\n")

    print(c(C.GREY, "  RTC source: FLORES-200 aligned benchmark · Petrov et al. (2023) / tokenization-fairness"))
    print(c(C.GREY, "  Root cause: training data representation — language volume on the internet + corpus prioritization"))
    print()


def interactive_mode(
    monthly_requests: int,
    base_input_tok: int,
    base_output_tok: int,
) -> None:
    """Full interactive menu."""
    print()
    print(c(C.BOLD + C.CYAN, "  ┌─────────────────────────────────────────────┐"))
    print(c(C.BOLD + C.CYAN, "  │  tokencost — token cost & context stress     │"))
    print(c(C.BOLD + C.CYAN, "  │  compare LLM families across languages       │"))
    print(c(C.BOLD + C.CYAN, "  └─────────────────────────────────────────────┘"))
    print()
    print(c(C.GREY, f"  Traffic params:  {monthly_requests:,} req/mo  ·  {base_input_tok} input tok  ·  {base_output_tok} output tok"))
    print(c(C.GREY,  "  (use --requests / --input-tokens / --output-tokens to change)"))
    print()

    while True:
        print(c(C.BOLD, "  What do you want to do?"))
        print(f"  {c(C.CYAN, '1')}  Compare one language across all model families")
        print(f"  {c(C.CYAN, '2')}  Compare two languages on one model family")
        print(f"  {c(C.CYAN, '3')}  Detailed view: one language + one model")
        print(f"  {c(C.CYAN, 'l')}  List available languages")
        print(f"  {c(C.CYAN, 'm')}  List model families")
        print(f"  {c(C.CYAN, 'q')}  Quit")
        print()

        choice = input(c(C.BOLD, "  → ")).strip().lower()
        print()

        if choice == "q":
            break

        elif choice == "l":
            print_languages()

        elif choice == "m":
            print_models()

        elif choice == "1":
            lang = pick_language()
            if lang:
                print_all_models_table(lang, monthly_requests, base_input_tok, base_output_tok)

        elif choice == "2":
            print(c(C.GREY, "  Enter first language code (e.g. ko):  "), end="")
            la = input().strip().lower()
            print(c(C.GREY, "  Enter second language code (e.g. en): "), end="")
            lb = input().strip().lower()
            model = pick_model()
            if la in RTC_DATA and lb in RTC_DATA and model:
                print_comparison(la, lb, model, monthly_requests, base_input_tok, base_output_tok)
            else:
                print(c(C.RED, "  Unknown language or model code."))

        elif choice == "3":
            lang  = pick_language()
            model = pick_model()
            if lang and model:
                en_r = compute("en", model, monthly_requests, base_input_tok, base_output_tok) \
                       if lang != "en" else None
                r    = compute(lang, model, monthly_requests, base_input_tok, base_output_tok)
                print_header(f"Detail view · {r.lang_name} · {r.model_label}")
                print_result_card(r, en_r)
        else:
            print(c(C.RED, "  Unknown option.\n"))


def pick_language() -> Optional[str]:
    print_languages()
    print(c(C.GREY, "  Enter language code: "), end="")
    code = input().strip().lower()
    if code not in RTC_DATA:
        print(c(C.RED, f"  Unknown language code: {code}\n"))
        return None
    return code


def pick_model() -> Optional[str]:
    print_models()
    print(c(C.GREY, "  Enter model key: "), end="")
    key = input().strip().lower()
    if key not in MODEL_FAMILIES:
        print(c(C.RED, f"  Unknown model key: {key}\n"))
        return None
    return key


def print_languages() -> None:
    print()
    print(c(C.BOLD, "  Available languages:"))
    for code, name in sorted(LANGUAGE_NAMES.items(), key=lambda x: x[1]):
        rtc_ko = RTC_DATA[code].get("o200k", 1.0)
        bar_s  = "·" * round(min(rtc_ko, 6) * 2)
        print(f"  {c(C.CYAN, code):<8}  {name:<16}  {c(C.GREY, f'o200k RTC: {rtc_ko:.2f}×  {bar_s}')}")
    print()


def print_models() -> None:
    print()
    print(c(C.BOLD, "  Model families:"))
    for key, m in MODEL_FAMILIES.items():
        print(f"  {c(C.CYAN, key):<12}  {m['label']:<32}  {c(C.GREY, m['note'])}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tokencost",
        description="Compare LLM token cost and context stress across languages and model families.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python tokencost.py
  python tokencost.py --language ko --model llama3
  python tokencost.py --language ko --model o200k --requests 500000 --input-tokens 800
  python tokencost.py --compare ko en --model llama3
  python tokencost.py --all-models ko
  python tokencost.py --list-languages
  python tokencost.py --list-models

data source:
  RTC values from FLORES-200 aligned benchmark.
  Petrov et al. (2023) "Language Model Tokenizers Introduce Unfairness Between Languages"
  aleksandarpetrov/tokenization-fairness
        """,
    )
    parser.add_argument("--language",       "-l",  help="Language code (e.g. ko, ar, hi)")
    parser.add_argument("--model",          "-m",  help="Model family key (e.g. llama3, o200k)")
    parser.add_argument("--compare",        "-c",  nargs=2, metavar=("LANG_A", "LANG_B"),
                        help="Compare two languages side by side")
    parser.add_argument("--all-models",     "-a",  metavar="LANG",
                        help="Show all model families for a given language")
    parser.add_argument("--requests",       "-r",  type=int, default=100_000,
                        help="Monthly requests (default: 100000)")
    parser.add_argument("--input-tokens",   "-i",  type=int, default=600,
                        help="Avg input tokens per request, English baseline (default: 600)")
    parser.add_argument("--output-tokens",  "-o",  type=int, default=250,
                        help="Avg output tokens per request (default: 250)")
    parser.add_argument("--list-languages", action="store_true", help="List all language codes")
    parser.add_argument("--list-models",    action="store_true", help="List all model family keys")

    args = parser.parse_args()

    req  = args.requests
    inp  = args.input_tokens
    out  = args.output_tokens

    if args.list_languages:
        print_languages()
        return

    if args.list_models:
        print_models()
        return

    if args.all_models:
        lang = args.all_models.lower()
        if lang not in RTC_DATA:
            print(c(C.RED, f"Unknown language: {lang}"))
            sys.exit(1)
        print_all_models_table(lang, req, inp, out)
        return

    if args.compare:
        la, lb = args.compare[0].lower(), args.compare[1].lower()
        model  = (args.model or "o200k").lower()
        for x in (la, lb):
            if x not in RTC_DATA:
                print(c(C.RED, f"Unknown language: {x}")); sys.exit(1)
        if model not in MODEL_FAMILIES:
            print(c(C.RED, f"Unknown model: {model}")); sys.exit(1)
        print_comparison(la, lb, model, req, inp, out)
        return

    if args.language and args.model:
        lang  = args.language.lower()
        model = args.model.lower()
        if lang  not in RTC_DATA:      print(c(C.RED, f"Unknown language: {lang}"));  sys.exit(1)
        if model not in MODEL_FAMILIES: print(c(C.RED, f"Unknown model: {model}")); sys.exit(1)
        en_r = compute("en", model, req, inp, out) if lang != "en" else None
        r    = compute(lang, model, req, inp, out)
        print_header(f"{r.lang_name} · {r.model_label}")
        print_result_card(r, en_r)
        return

    if args.language:
        lang = args.language.lower()
        if lang not in RTC_DATA:
            print(c(C.RED, f"Unknown language: {lang}")); sys.exit(1)
        print_all_models_table(lang, req, inp, out)
        return

    # No args → interactive mode
    interactive_mode(req, inp, out)


if __name__ == "__main__":
    main()