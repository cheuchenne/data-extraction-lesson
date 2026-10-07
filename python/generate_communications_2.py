"""LLM-enriched communications (v2).

Run from the project root after installing the optional extras
(torch, transformers, accelerate):

    python python/generate_communications_2.py

Reads template drafts from data/raw/telco/customer_communications.csv and
rewrites them with a local Hugging Face model. ENRICHMENT_PROMPT constraints
are always applied.
An optional realism-noise instruction (tone jitter + higher temperature) reduces
overly label-predictive wording without inventing facts.

Progress: logs after each customer milestone (default every 20 customers;
set PROGRESS_EVERY_CUSTOMERS=1000 when running a large cohort).

Runtime note:
  Local TinyLlama on CPU is ~60–90s/text. Full 7043 customers (~25k texts)
  would need hundreds of hours. Use N_CUSTOMERS for a stratified subsample.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd

PYTHON_DIR = Path(__file__).resolve().parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from telco_paths import COMMS_FILE, COMMS_LLM_FILE, PROFILES_FILE, TELCO_RESULTS  # noqa: E402

BENCHMARK_FILE = COMMS_FILE
_OUTPUT = os.environ.get("OUTPUT_FILE", "").strip()
OUTPUT_FILE = Path(_OUTPUT) if _OUTPUT else COMMS_LLM_FILE
_PROGRESS = os.environ.get("PROGRESS_FILE", "").strip()
PROGRESS_FILE = Path(_PROGRESS) if _PROGRESS else TELCO_RESULTS / "llm_enrichment_progress.json"

_MAX = os.environ.get("MAX_TEXTS", "").strip()
MAX_TEXTS = int(_MAX) if _MAX else None
_NC = os.environ.get("N_CUSTOMERS", "").strip()
N_CUSTOMERS = int(_NC) if _NC else 120  # stratified subsample; None/0 → all
PROGRESS_EVERY_CUSTOMERS = int(os.environ.get("PROGRESS_EVERY_CUSTOMERS", "20"))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))

HF_BACKEND = os.environ.get("HF_BACKEND", "local").strip().lower()
HF_MODEL = os.environ.get("HF_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
HF_TOKEN = (
    os.environ.get("HF_TOKEN", "").strip()
    or os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()
)
REQUEST_PAUSE_SEC = float(os.environ.get("REQUEST_PAUSE_SEC", "0"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "120"))
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "1.2"))
TOP_P = float(os.environ.get("LLM_TOP_P", "0.95"))
TONE_NOISE_PROB = float(os.environ.get("TONE_NOISE_PROB", "0.45"))
# Hard cap so a single generation cannot hang the whole overnight job.
GENERATION_TIMEOUT_SEC = int(os.environ.get("GENERATION_TIMEOUT_SEC", "180"))

# ---------------------------------------------------------------------------
# >>> ENRICHMENT PROMPT (must be respected) <<<
# ---------------------------------------------------------------------------

FORMAT_RULES_BY_TYPE = {
    "crm_note": (
        "FORMAT (mandatory): Keep this as a CRM note only — a short internal "
        "prose note written by a company employee into a CRM system. "
        "Do NOT turn it into an email, letter, chat, or Agent/Customer dialog. "
        "No Subject line, no greetings like Hello/Hi there, no sign-off. "
        "Prefer compact factual sentences (similar length to the draft)."
    ),
    "billing_renewal": (
        "FORMAT (mandatory): Keep this as a company billing/renewal email "
        "to the customer. Clear company sender (e.g. Billing Team). "
        "Do NOT make it a dialog."
    ),
    "promotion": (
        "FORMAT (mandatory): Keep this as a company promotional email "
        "to the customer. Clear company sender. Do NOT make it a dialog. "
        "Do NOT write it as if the customer authored it."
    ),
    "customer_service": (
        "FORMAT: You may keep an Agent/Customer dialog OR rewrite as a single "
        "email/message, but speaker roles must stay coherent. "
        "If you keep a dialog, label turns clearly as **Agent:** / **Customer:**. "
        "If you convert to one email/message, state clearly who writes it "
        "(company agent OR customer), and that writer must stay in role: "
        "a company agent may help/confirm services; a customer may ask questions "
        "or report issues — a company must NOT write a customer complaint as if "
        "it were the company, and a customer must NOT offer help/services."
    ),
    "support": (
        "FORMAT: You may keep an Agent/Customer dialog OR rewrite as a single "
        "email/message, but speaker roles must stay coherent. "
        "If you keep a dialog, label turns clearly as **Agent:** / **Customer:**. "
        "If you convert to one email/message, state clearly who writes it "
        "(company support OR customer). Company support may offer help and "
        "updates; the customer may describe problems — never reverse those roles."
    ),
    "complaint": (
        "FORMAT: You may keep an Agent/Customer dialog OR rewrite as messages, "
        "but roles must stay coherent. The complaint itself must come from the "
        "customer (or be clearly quoted as the customer's words). "
        "The company/agent may apologize and help; the company must NOT author "
        "the complaint as if the company were complaining."
    ),
}

ENRICHMENT_PROMPT = """\
You rewrite a DRAFT telco customer communication to make it more natural \
and varied, while staying faithful to the facts.

Communication type: {communication_type}

{format_rules}

Customer profile (source of truth — do not contradict it):
{customer_profile}

DRAFT text to enrich:
{template_text}

Rules:
1. Rewrite in different words — do NOT copy the draft almost verbatim. \
Vary tone and wording so it feels less templated, but respect the FORMAT rules above.
2. Keep all hard facts consistent with the profile (services, contract, \
payment method, tenure, exact dollar amounts when present in the draft).
3. Do NOT invent new services, prices, tenure, or personal details that are \
absent from the profile / draft.
4. Do NOT mention churn, retention risk, or that the customer will/will not leave.
5. Keep the same communication type and general situation as the draft.
6. Stay reasonably short (do not write a long essay). For crm_note, stay short.
7. Return ONLY the final communication text itself. \
Do NOT add any meta commentary, titles, or prefaces such as \
"here is an enriched version", "here's a rewritten draft", \
"enriched customer communication", "sure,", or similar. \
The output must look like a real communication, not like an LLM explanation.

{noise_block}
"""

# Extra block AFTER the mandatory rules (does not replace them).
NOISE_BLOCK_BASE = """\
Realism / ambiguity (still obey Rules 1–7 above):
- Add mild natural ambiguity so the message is not a perfect predictor of future loyalty.
- Prefer ordinary business language; avoid extreme wording.
- Do not invent facts; do not mention churn/leaving/retention.
"""

TONE_JITTER_OPTIONS = [
    "Tone jitter: prefer a slightly calmer / more neutral tone than the draft.",
    "Tone jitter: prefer a slightly more impatient customer tone than the draft "
    "(still polite; no insults).",
    "Tone jitter: prefer a warmer, more appreciative tone than the draft.",
    "Tone jitter: keep the situation but understate emotional intensity.",
]


def build_enrichment_prompt(
    *,
    communication_type: str,
    customer_profile: str,
    template_text: str,
    rng: random.Random,
) -> str:
    format_rules = FORMAT_RULES_BY_TYPE.get(
        communication_type,
        (
            "FORMAT: Preserve a coherent genre. If converting a dialog into one "
            "message, make the writer identity explicit and keep roles consistent."
        ),
    )
    noise = NOISE_BLOCK_BASE
    if rng.random() < TONE_NOISE_PROB:
        noise += "- " + rng.choice(TONE_JITTER_OPTIONS) + "\n"
    return ENRICHMENT_PROMPT.format(
        communication_type=communication_type,
        format_rules=format_rules,
        customer_profile=customer_profile,
        template_text=template_text,
        noise_block=noise,
    )


def _row_key(source_row, text_index, communication_type) -> tuple:
    return (int(source_row), int(text_index), str(communication_type))


def load_completed_keys(output_path: Path) -> set[tuple]:
    if not output_path.exists():
        return set()
    existing = pd.read_csv(output_path)
    if existing.empty:
        return set()
    return {
        _row_key(r.source_row, r.text_index, r.communication_type)
        for r in existing.itertuples(index=False)
    }


def append_result_row(output_path: Path, row: dict) -> None:
    frame = pd.DataFrame([row])
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    frame.to_csv(output_path, mode="a", header=write_header, index=False)


def clean_model_output(text: str, template_text: str) -> str:
    enriched = (text or "").strip()
    meta_prefixes = (
        "enriched text:",
        "here is the rewritten text:",
        "here's the rewritten text:",
        "here is an enriched version",
        "here's an enriched version",
        "here is an enriched version of the customer communication:",
        "here's an enriched version of the customer communication:",
        "enriched version of the customer communication:",
        "enriched customer communication:",
        "rewritten draft:",
        "sure,",
        "certainly,",
        "of course,",
    )
    lowered = enriched.lower()
    for prefix in meta_prefixes:
        if lowered.startswith(prefix):
            enriched = enriched[len(prefix) :].lstrip(" \n:-")
            lowered = enriched.lower()
            break
    lines = enriched.splitlines()
    if lines:
        first = lines[0].strip().lower()
        if (
            "enriched version" in first
            or first.startswith("here is")
            or first.startswith("here's")
        ) and ("enrich" in first or "rewrite" in first or "version" in first):
            enriched = "\n".join(lines[1:]).lstrip(" \n-")
    enriched = enriched.strip().strip("---").strip()
    if not enriched:
        return template_text
    if "churn" in enriched.lower():
        return template_text
    return enriched


def select_customer_ids(bench: pd.DataFrame, profiles: pd.DataFrame, n: int | None) -> list:
    """Stratified sample of customerIDs by Churn (if n set)."""
    ids = bench["customerID"].drop_duplicates().tolist()
    if not n or n <= 0 or n >= len(ids):
        return ids
    meta = profiles.loc[profiles["customerID"].isin(ids), ["customerID", "Churn"]].drop_duplicates(
        "customerID"
    )
    rng = random.Random(RANDOM_SEED)
    parts = []
    for _, g in meta.groupby(meta["Churn"].astype(str).str.strip()):
        take = max(1, int(round(n * len(g) / len(meta))))
        chosen = g["customerID"].tolist()
        rng.shuffle(chosen)
        parts.extend(chosen[:take])
    rng.shuffle(parts)
    return parts[:n]


class LocalHFEnricher:
    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading local HF model: {model_id} …", flush=True)
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if not torch.cuda.is_available():
            self.model = self.model.to("cpu")
        self.model.eval()
        print("Local model ready.", flush=True)

    def enrich(self, prompt: str) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = prompt

        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=self.tokenizer.eos_token_id,
                max_time=float(GENERATION_TIMEOUT_SEC),
            )
        new_tokens = output_ids[0, input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class ApiHFEnricher:
    def __init__(self, model_id: str, token: str):
        from huggingface_hub import InferenceClient

        if not token:
            raise RuntimeError("HF_BACKEND=api requires HF_TOKEN.")
        self.model_id = model_id
        self.client = InferenceClient(model=model_id, token=token)

    def enrich(self, prompt: str) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
            )
            return (completion.choices[0].message.content or "").strip()
        except Exception:
            return self.client.text_generation(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
            ).strip()


def build_enricher():
    if HF_BACKEND == "api":
        return ApiHFEnricher(HF_MODEL, HF_TOKEN)
    return LocalHFEnricher(HF_MODEL)


def enrich_text_with_hf(
    *,
    template_text: str,
    communication_type: str,
    customer_profile: str,
    enricher,
    rng: random.Random,
) -> str:
    prompt = build_enrichment_prompt(
        communication_type=communication_type,
        customer_profile=customer_profile,
        template_text=template_text,
        rng=rng,
    )
    raw = enricher.enrich(prompt)
    return clean_model_output(raw, template_text)


def write_progress(payload: dict) -> None:
    path = Path(PROGRESS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    t_start = time.time()
    benchmark_path = Path(BENCHMARK_FILE)
    profiles_path = Path(PROFILES_FILE)
    output_path = Path(OUTPUT_FILE)

    if not benchmark_path.exists():
        raise FileNotFoundError(f"Missing {BENCHMARK_FILE}")
    if not profiles_path.exists():
        raise FileNotFoundError(f"Missing {PROFILES_FILE}")

    bench = pd.read_csv(benchmark_path)
    profiles = pd.read_csv(profiles_path)

    selected_ids = set(select_customer_ids(bench, profiles, N_CUSTOMERS))
    bench = bench[bench["customerID"].isin(selected_ids)].copy()
    # stable order by customer then text_index
    bench = bench.sort_values(["customerID", "text_index"]).reset_index(drop=True)

    completed = load_completed_keys(output_path)
    pending_rows = []
    for idx, row in bench.iterrows():
        key = _row_key(row["source_row"], row["text_index"], row["communication_type"])
        if key not in completed:
            pending_rows.append((idx, row))
    if MAX_TEXTS is not None:
        pending_rows = pending_rows[:MAX_TEXTS]

    n_customers_target = bench["customerID"].nunique()
    print(f"Benchmark file: {BENCHMARK_FILE}")
    print(f"Output file:    {OUTPUT_FILE}")
    print(f"HF model:       {HF_MODEL} (temp={TEMPERATURE}, max_new_tokens={MAX_NEW_TOKENS})")
    print(f"Customers in run: {n_customers_target} (N_CUSTOMERS={N_CUSTOMERS})")
    print(f"Texts in scope:   {len(bench)}")
    print(f"Already done:     {len(completed)}")
    print(f"Still to do:      {len(pending_rows)}")
    print(
        "NOTE: full 7043-customer LLM enrichment is too slow on CPU for a <10h budget; "
        "this run uses a stratified subsample.",
        flush=True,
    )

    if not pending_rows:
        print("Nothing left to enrich.")
        return

    enricher = build_enricher()
    rng = random.Random(RANDOM_SEED)
    newly_done = 0
    customers_finished = set()
    # customers that already have all their pending texts done before this run
    done_before = {
        cid
        for cid, g in bench.groupby("customerID")
        if all(
            _row_key(r.source_row, r.text_index, r.communication_type) in completed
            for r in g.itertuples(index=False)
        )
    }
    customers_finished |= done_before
    last_milestone = (len(customers_finished) // PROGRESS_EVERY_CUSTOMERS) * PROGRESS_EVERY_CUSTOMERS

    for pending_i, (idx, row) in enumerate(pending_rows):
        source_row = int(row["source_row"])
        cid = row["customerID"]
        profile = str(profiles.iloc[source_row]["CustomerProfile"])
        template_text = str(row["text"])
        communication_type = str(row["communication_type"])

        print(
            f"[{pending_i + 1}/{len(pending_rows)}] "
            f"customer={cid} type={communication_type} …",
            flush=True,
        )
        t0 = time.time()
        enriched = enrich_text_with_hf(
            template_text=template_text,
            communication_type=communication_type,
            customer_profile=profile,
            enricher=enricher,
            rng=rng,
        )
        dt = time.time() - t0
        if not enriched.strip():
            print("  empty model output → keep template text", flush=True)
            enriched = template_text

        append_result_row(
            output_path,
            {
                "source_row": source_row,
                "customerID": cid,
                "n_texts": row["n_texts"],
                "text_index": row["text_index"],
                "communication_type": communication_type,
                "template_text": template_text,
                "text": enriched,
                "model": HF_MODEL,
                "backend": HF_BACKEND,
                "temperature": TEMPERATURE,
            },
        )
        newly_done += 1
        completed.add(_row_key(source_row, row["text_index"], communication_type))

        # Mark customer finished when all their texts in scope are done
        cust_rows = bench[bench["customerID"] == cid]
        if all(
            _row_key(r.source_row, r.text_index, r.communication_type) in completed
            for r in cust_rows.itertuples(index=False)
        ):
            customers_finished.add(cid)

        elapsed = time.time() - t_start
        rate = newly_done / max(elapsed, 1e-6)
        remaining = len(pending_rows) - (pending_i + 1)
        eta_s = remaining / max(rate, 1e-9)

        print(
            f"  saved ({dt:.1f}s) | texts_done_this_run={newly_done} "
            f"| customers_done={len(customers_finished)}/{n_customers_target} "
            f"| ETA≈{eta_s/3600:.2f}h",
            flush=True,
        )

        milestone = (len(customers_finished) // PROGRESS_EVERY_CUSTOMERS) * PROGRESS_EVERY_CUSTOMERS
        if milestone > last_milestone and milestone > 0:
            last_milestone = milestone
            payload = {
                "customers_done": len(customers_finished),
                "customers_target": n_customers_target,
                "texts_done_this_run": newly_done,
                "texts_pending_remaining": remaining,
                "elapsed_hours": round(elapsed / 3600, 3),
                "eta_hours": round(eta_s / 3600, 3),
                "output_file": OUTPUT_FILE,
                "model": HF_MODEL,
            }
            write_progress(payload)
            print(
                f"\n===== PROGRESS MILESTONE: {len(customers_finished)} customers done "
                f"({newly_done} texts this run, ETA≈{eta_s/3600:.2f}h) =====\n",
                flush=True,
            )

        if REQUEST_PAUSE_SEC > 0 and pending_i < len(pending_rows) - 1:
            time.sleep(REQUEST_PAUSE_SEC)

    write_progress(
        {
            "customers_done": len(customers_finished),
            "customers_target": n_customers_target,
            "texts_done_this_run": newly_done,
            "texts_pending_remaining": 0,
            "elapsed_hours": round((time.time() - t_start) / 3600, 3),
            "eta_hours": 0.0,
            "output_file": OUTPUT_FILE,
            "model": HF_MODEL,
            "status": "finished",
        }
    )
    print(f"\nFinished. Wrote {newly_done} enriched text(s) → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
