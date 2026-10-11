"""WTQ Build Track - Utility Bill Decoder, Level 1 (extraction).

Reads K-Electric / LESCO / IESCO bill images with gemini-3.5-flash-lite and fills level1.csv.

Pipeline per bill:
  image prep -> N model samples (structured output) -> normalise -> per-field majority vote
  -> guide-specific rules -> consistency checks -> optional repair call -> CSV

Usage (defaults match the layout: bills/, csv/level1.csv, output/level1.csv):
  python decoder.py
  python decoder.py --bills test/bills --template test/csv/level1.csv --output output/level1.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from PIL import Image, ImageOps
from pydantic import BaseModel

log = logging.getLogger("decoder")

ALLOWED_MODEL = "gemini-3.5-flash-lite"
IMAGE_EXTS = (".png", ".jpg", ".jpeg")

# =====================================================================================
# Schema (section 5.1). No default values: the Gemini API rejects "default" in schemas.
# =====================================================================================


class ChargeType(str, Enum):
    energy = "energy"
    fixed = "fixed"
    fpa = "fpa"
    quarterly_adjustment = "quarterly_adjustment"
    surcharge = "surcharge"
    meter_rent = "meter_rent"
    subsidy = "subsidy"
    other = "other"


class TaxType(str, Enum):
    gst = "gst"
    electricity_duty = "electricity_duty"
    income_tax = "income_tax"
    municipal_tax = "municipal_tax"
    other_tax = "other_tax"


class ChargeItem(BaseModel):
    type: ChargeType
    amount: float


class TaxItem(BaseModel):
    type: TaxType
    amount: float


class BillExtractionSchema(BaseModel):
    provider: Optional[str]
    tariff: Optional[str]
    sanctioned_load_kw: Optional[float]
    bill_month: Optional[str]
    reading_date: Optional[str]
    issue_date: Optional[str]
    due_date: Optional[str]
    previous_reading: Optional[float]
    current_reading: Optional[float]
    units_consumed: Optional[float]
    charges: list[ChargeItem]
    total_charges: Optional[float]
    taxes: list[TaxItem]
    total_taxes: Optional[float]
    current_bill: Optional[float]
    arrears: Optional[float]
    payable_within_due_date: Optional[float]
    payable_after_due_date: Optional[float]


class RegisterRow(BaseModel):
    previous_reading: Optional[float]
    current_reading: Optional[float]
    units: Optional[float]


class BillExtractionWithRegisters(BillExtractionSchema):
    """Used only when MULTI_REGISTER_MODE=matched; meter_rows is internal and never written to the CSV."""
    meter_rows: list[RegisterRow]


FIELD_ORDER = list(BillExtractionSchema.model_fields)
LIST_FIELDS = ("charges", "taxes")
SCALAR_FIELDS = [f for f in FIELD_ORDER if f not in LIST_FIELDS]
NUMERIC_FIELDS = [
    "sanctioned_load_kw", "previous_reading", "current_reading", "units_consumed", "total_charges",
    "total_taxes", "current_bill", "arrears", "payable_within_due_date", "payable_after_due_date",
]
CHARGE_TYPES = {t.value for t in ChargeType}
TAX_TYPES = {t.value for t in TaxType}
PREFIX_TO_PROVIDER = {"KESC": "KE", "KE": "KE", "LESCO": "LESCO", "IESCO": "IESCO"}

# =====================================================================================
# Prompts
# =====================================================================================

SYSTEM_PROMPT = """You are a meticulous data-extraction engine for Pakistani electricity bills \
(K-Electric "KE", LESCO, IESCO). The image may be a scan, screenshot or tilted phone photo, \
partly in Urdu, with names/addresses/ID numbers hidden by grey boxes. Read the English printed \
text and numbers only. Return ONE JSON object that follows the response schema exactly.

## GENERAL RULES
- Copy values AS PRINTED. Never recalculate, round, correct or "fix" a printed amount.
- Never guess. If a value is not printed, blank, hidden by a grey box or unreadable -> null.
  A correct null scores as correct; an invented value is penalised.
- All amounts are plain numbers in PKR: remove "Rs.", commas, spaces. "Rs. 19,964" -> 19964.
- Credits are NEGATIVE: a trailing "CR" means credit -> negative ("54770CR" -> -54770,
  "38209CR" -> -38209). A printed minus sign stays negative ("Rs. -0.53" -> -0.53).
- Dates: YYYY-MM-DD ("22nd Sep 2026", "22 SEP 26", "22-Sep-26" -> 2026-09-22).
  bill_month: YYYY-MM ("Apr-26", "SEP 26", "MAR 2026" -> 2026-04 / 2026-09 / 2026-03). Two-digit years are 20xx.
- Do NOT output any personal identifier (name, address, CNIC, account/reference/consumer/contract/
  invoice/meter numbers). The schema has no fields for them.

## PHOTOS ARE OFTEN TILTED
In tilted/photographed bills the numbers can sit one row above or below their label. Match each number
to its label by reading the row/column layout carefully and by arithmetic sense:
  current_bill = (net charges) + (taxes)            e.g. 17367.77 + 3358.23 = 20726
  payable_within_due_date = current_bill + arrears   e.g. 20726 + (-75496) = -54770
Use such relations ONLY to decide which printed number belongs to which label - still output the
printed numbers, never computed ones.

## FIELD GUIDE
provider: "KE" (K-Electric; "Electricity You Have Used / Bill Calculation"), "LESCO", or "IESCO".
tariff: tariff code exactly as printed (e.g. "A1-R", "A-1a(01)", "A-2a(04)", "A-1b(03)T"). KE: "Tariff" in the
  Customer Information box. LESCO/IESCO: the "TARIFF" cell (NOT "TARIFF CATEGORY", which is Domestic/Commercial).
sanctioned_load_kw: number next to "Sanc Load" / "S.Load" / "San Load" (NOT "Conn Load", NOT MDI).
bill_month, reading_date, issue_date: as printed. KE prints "Bill Month" and "Issue Date" top-left and the
  "Reading Date" inside the Bill Calculation box. LESCO/IESCO print them in the top-right box.
due_date: the "Due Date" = last date to pay WITHOUT late surcharge. NOT the "after due date" dates.
previous_reading / current_reading: printed meter readings. If several register rows exist
  (net metering, peak/off-peak, import/export), use the FIRST row.
units_consumed: units billed for the current month as printed. KE: the number in the "Current Month ... units"
  card (e.g. "901 units" -> 901). LESCO/IESCO: the "UNITS" value in Meter Info (first row; 1,017 -> 1017).

charges: one entry per charge LINE in the charges section, in printed order.
  - Skip blank lines and lines printed as 0. Keep SEPARATE entries when a type repeats (never merge or add lines).
  - KE "Bill Calculation" lines between the header and the "Electricity Charges" subtotal:
      Fixed Charges -> fixed; Variable Upto.../Variable Off Peak/Variable Peak/Cost of Electricity -> energy;
      Uniform Quarterly Adjustment / QTA -> quarterly_adjustment; FCA / FPA -> fpa;
      Additional Surcharge (PHL) / F.C Surcharge -> surcharge; meter/service rent -> meter_rent;
      subsidy / relief lines -> subsidy (negative); anything else -> other.
    Negative printed amounts stay negative (e.g. a Uniform Quarterly Adjustment of -1,442.74).
    Electricity Duty / Sales Tax / MUCT are TAXES, not charges. "Net Metering Benefit", "Opening Balance",
    "Previous Dues", "Outstanding Balance" are NOT charge lines.
  - LESCO/IESCO "Bill Charges Breakdown" prints only summary rows (Total Electricity Charges, Subsidies,
    Net Electricity Charges, Taxes, Current Bill, Total FPA ...). Those total rows are NOT charge lines.
    Only create entries for: a non-zero "Subsidies" amount -> {"subsidy", NEGATIVE amount}; a non-zero
    "Total FPA" -> {"fpa", amount}. If neither exists, charges = [].
total_charges: the PRINTED charges subtotal. KE: the "Electricity Charges" subtotal row.
  LESCO/IESCO: the printed "Net Electricity Charges" value (after subsidies). Never compute it. null if not printed.

taxes: one entry per line in the Taxes/Government section ("Taxes and Duties" block on KE), classified by
  SECTION not name: Electricity Duty -> electricity_duty; Sales Tax / GST (incl. GST on FPA) -> gst;
  Income Tax -> income_tax; MUCT (KMC)/municipal -> municipal_tax; extra/further tax, TV fee -> other_tax.
  LESCO/IESCO print ONE combined "Taxes" number: then taxes = [] and that number goes in total_taxes.
total_taxes: the printed tax subtotal ("Taxes and Duties" on KE, "Taxes" on LESCO/IESCO), else null.

current_bill: the printed amount for THIS month's bill. KE: "Electricity Charges for current month" /
  the "Current Month" card amount. LESCO/IESCO: the "Current Bill" row.
arrears: previous dues carried forward, keep printed sign. KE: "Previous Dues". LESCO/IESCO: "Arrears"
  (a printed "39411 / 2" means 39411). null if blank. A printed 0 is 0.
payable_within_due_date: amount payable by the due date (KE "Amount Due"/"Amount Payable within Due Date";
  LESCO/IESCO "Payable within due date" - or "Grand Total" when that is the amount shown). Apply the CR rule.
payable_after_due_date: amount payable after the due date. If several late-payment amounts are shown
  (e.g. "Till 12-Oct-26: 3674" and "After 12-Oct-26: 3812"), return the HIGHEST. If the bill says
  "NOT TO BE PAID" or prints nothing usable -> null. Do not confuse "LP Surcharge" amounts with payable amounts.

## WORKED EXAMPLE (K-Electric bill, Apr-26) - for mapping only
Printed: Fixed Charges 900.00; Variable Upto 100 Units 1,054.00; Variable Upto 200 Units 663.51;
Uniform Quarterly Adjustment 52.91; FCA Feb-26 125.27; Additional Surcharge (PHL) 64.93; Electricity Charges 2,860.62;
Electricity Duty 29.41; Sales Tax 520.21; MUCT (KMC) 20.00; Taxes and Duties 569.62; Electricity Charges for current month
3,430.24; Previous Dues -0.53; Amount Payable within Due Date 3,430; Till 22-Apr 3,574; After 24-Apr 3,716.
JSON: {"provider":"KE","tariff":"A1-R","sanctioned_load_kw":3,"bill_month":"2026-04","reading_date":"2026-04-03",
"issue_date":"2026-04-07","due_date":"2026-04-21","previous_reading":8819,"current_reading":8970,"units_consumed":151,
"charges":[{"type":"fixed","amount":900},{"type":"energy","amount":1054},{"type":"energy","amount":663.51},
{"type":"quarterly_adjustment","amount":52.91},{"type":"fpa","amount":125.27},{"type":"surcharge","amount":64.93}],
"total_charges":2860.62,"taxes":[{"type":"electricity_duty","amount":29.41},{"type":"gst","amount":520.21},
{"type":"municipal_tax","amount":20}],"total_taxes":569.62,"current_bill":3430.24,"arrears":-0.53,
"payable_within_due_date":3430,"payable_after_due_date":3716}
"""

USER_PROMPT = (
    "Extract the Level 1 JSON from this electricity bill image. Provider (from the file name): {provider}. "
    "Double-check every digit of every amount against the image before answering."
)

REPAIR_PROMPT = """Here is a draft extraction of this same bill:

{draft}

Automatic consistency checks flagged these problems:
{problems}

Look at the image again. For each flagged field, re-read the PRINTED value carefully (watch for tilted rows where \
numbers sit next to the wrong label, misread digits, missed CR suffixes, merged or missing lines). Correct a field \
ONLY if the printed value differs from the draft; if the draft matches the print, keep it (some bills legitimately \
do not add up). Return the full corrected JSON."""

# =====================================================================================
# Normalisation and guide-specific rules
# =====================================================================================


def provider_from_bill_id(bill_id: str) -> Optional[str]:
    return PREFIX_TO_PROVIDER.get(bill_id.split("_")[0].upper())


def to_number(value: Any) -> int | float | None:
    """'1,017' -> 1017, 'Rs. 19,964' -> 19964, '54770CR' -> -54770, '39411 / 2' -> 39411, junk -> None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        num = float(value)
    else:
        s = str(value).strip()
        if not s or s.lower() in {"null", "none", "n/a", "na", "-"}:
            return None
        negative = bool(re.search(r"CR\.?\s*$", s, re.I)) or (s.startswith("(") and s.endswith(")"))
        if re.fullmatch(r"-?[\d,\.]+\s*/\s*\d+", s):
            s = s.split("/")[0]
        s = re.sub(r"[A-Za-z]+\.?", " ", s)
        s = re.sub(r"[^\d.\-]", "", s.replace(",", "")).strip(".")
        if s in {"", "-", ".", "-."}:
            return None
        try:
            num = float(s)
        except ValueError:
            return None
        if negative and num > 0:
            num = -num
    num = round(num, 2)
    return int(num) if num == int(num) else num


def normalize_date(value: Any, month_only: bool = False) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not re.fullmatch(r"\d{4}-\d{2}" if month_only else r"\d{4}-\d{2}-\d{2}", s):
        return None
    try:
        date(int(s[:4]), int(s[5:7]), 1 if month_only else int(s[8:10]))
    except ValueError:
        return None
    return s


def _clean_lines(items: Any, allowed: set[str], fallback: str) -> list[dict]:
    out: list[dict] = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        amount = to_number(it.get("amount"))
        if amount is None or amount == 0:  # guide: skip blank / zero lines
            continue
        t = str(it.get("type", "")).strip().lower()
        t = t if t in allowed else fallback
        if t == "subsidy" and amount > 0:  # guide: subsidies are deductions
            amount = -amount
        out.append({"type": t, "amount": amount})
    return out


def match_registers(rows: Any, units: Optional[float]) -> Optional[tuple[Any, Any]]:
    """Multi-register bills (net metering, peak/off-peak): find the smallest group of register rows (in printed order)
    whose units add up to the printed billed units (+-1) and return (sum of previous, sum of current) readings.
    Returns None when nothing matches, so the caller keeps the first row."""
    if units is None:
        return None
    regs = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        prev, cur, u = to_number(r.get("previous_reading")), to_number(r.get("current_reading")), to_number(r.get("units"))
        if prev is None or cur is None:
            continue
        regs.append((prev, cur, u if u is not None else round(cur - prev, 2)))
    for k in range(1, len(regs) + 1):
        for combo in itertools.combinations(regs, k):
            if abs(sum(c[2] for c in combo) - units) <= 1.0:
                return to_number(sum(c[0] for c in combo)), to_number(sum(c[1] for c in combo))
    return None


def normalize(raw: dict[str, Any], bill_id: str, register_mode: str = "first") -> dict[str, Any]:
    """Every field present, correct types, guide's field order. Printed values are never recalculated."""
    r: dict[str, Any] = {f: to_number(raw.get(f)) for f in NUMERIC_FIELDS}
    r["provider"] = provider_from_bill_id(bill_id) or (
        raw.get("provider") if raw.get("provider") in {"KE", "LESCO", "IESCO"} else None
    )
    tariff = raw.get("tariff")
    r["tariff"] = str(tariff).strip() if tariff not in (None, "") else None
    r["bill_month"] = normalize_date(raw.get("bill_month"), month_only=True)
    for f in ("reading_date", "issue_date", "due_date"):
        r[f] = normalize_date(raw.get(f))
    r["charges"] = _clean_lines(raw.get("charges"), CHARGE_TYPES, "other")
    r["taxes"] = _clean_lines(raw.get("taxes"), TAX_TYPES, "other_tax")
    if register_mode == "matched":
        matched = match_registers(raw.get("meter_rows"), r["units_consumed"])
        if matched:
            r["previous_reading"], r["current_reading"] = matched
    return apply_rules({k: r.get(k) for k in FIELD_ORDER})


def _close(a: float, b: float, tol: float = 0.011) -> bool:
    return abs(a - b) <= tol


def apply_rules(r: dict[str, Any]) -> dict[str, Any]:
    """Deterministic fixes for model mistakes that the guide's rules rule out."""
    charges, taxes = r["charges"], r["taxes"]
    tc, tt = r["total_charges"], r["total_taxes"]

    if r["provider"] in ("LESCO", "IESCO"):
        # These bills print only ONE combined tax figure -> taxes = [] and the figure is total_taxes.
        if taxes:
            if tt is None:
                tt = round(sum(t["amount"] for t in taxes), 2)
            taxes = []
        # Summary rows (Total/Net Electricity Charges) are not charge lines: drop entries equal to them.
        if charges and tc is not None:
            subsidy = sum(c["amount"] for c in charges if c["type"] == "subsidy")
            summary_values = [tc, round(tc - subsidy, 2)]  # net, and gross (net + subsidy given)
            charges = [c for c in charges
                       if c["type"] == "subsidy" or not any(_close(c["amount"], v) for v in summary_values)]
    elif r["provider"] == "KE" and charges and tc is not None:
        # "Net Metering Benefit" etc. are not charge lines: if the lines already add up to the printed
        # subtotal without the subsidy entries, those entries were wrongly added.
        total = sum(c["amount"] for c in charges)
        without = [c for c in charges if c["type"] != "subsidy"]
        if len(without) != len(charges) and not _close(total, tc, 1.5) \
                and _close(sum(c["amount"] for c in without), tc, 1.5):
            charges = without

    r["charges"], r["taxes"], r["total_taxes"] = charges, taxes, tt
    return r


def check_consistency(r: dict[str, Any]) -> list[str]:
    """Problems used to pick between samples / trigger a repair call. Never edits values."""
    p: list[str] = []
    tc, tt, cb = r["total_charges"], r["total_taxes"], r["current_bill"]
    if None not in (tc, tt, cb) and abs(tc + tt - cb) > 2.5:
        p.append(f"total_charges ({tc}) + total_taxes ({tt}) != current_bill ({cb})")
    if r["provider"] == "KE" and r["charges"] and tc is not None:
        s = sum(c["amount"] for c in r["charges"])
        if abs(s - tc) > 1.5:
            p.append(f"sum of charge lines ({s:.2f}) != total_charges ({tc}); lines may be merged/missing/misread")
    if r["taxes"] and tt is not None:
        s = sum(t["amount"] for t in r["taxes"])
        if abs(s - tt) > 1.5:
            p.append(f"sum of tax lines ({s:.2f}) != total_taxes ({tt})")
    ar, pw, pa = r["arrears"], r["payable_within_due_date"], r["payable_after_due_date"]
    fpa = sum(c["amount"] for c in r["charges"] if c["type"] == "fpa") if r["provider"] == "IESCO" else 0
    if None not in (cb, ar, pw) and abs(cb + ar - pw) > 3 and abs(cb + ar + fpa - pw) > 3:  # IESCO grand total adds FPA
        p.append(f"current_bill ({cb}) + arrears ({ar}) != payable_within_due_date ({pw})")
    if pa is not None and pw is not None and pw >= 0 and pa < pw:
        p.append(f"payable_after_due_date ({pa}) < payable_within_due_date ({pw})")
    d = [r["reading_date"], r["issue_date"], r["due_date"]]
    if all(d) and not (d[0] <= d[1] <= d[2]):
        p.append(f"dates not in order reading<=issue<=due: {d}")
    if None not in (r["previous_reading"], r["current_reading"]) and r["current_reading"] < r["previous_reading"]:
        p.append("current_reading < previous_reading")
    missing = [f for f in ("tariff", "bill_month", "due_date", "current_bill", "units_consumed") if r[f] is None]
    if missing:
        p.append(f"required fields are null: {missing} (re-check whether they are really not printed)")
    return p


def merge_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-field majority vote (null is a vote). Ties go to the sample whose numbers add up best."""
    if len(samples) == 1:
        return samples[0]
    ranked = sorted(samples, key=lambda s: len(check_consistency(s)))

    def vote(fields, keyf) -> dict[str, Any]:
        out = {}
        for f in fields:
            votes = Counter(keyf(s[f]) for s in samples)
            top = max(votes.values())
            winners = {k for k, c in votes.items() if c == top}
            out[f] = next(s[f] for s in ranked if keyf(s[f]) in winners)
        return out

    merged = vote(SCALAR_FIELDS, lambda v: round(v, 2) if isinstance(v, float) else v)
    merged.update(vote(LIST_FIELDS, lambda v: tuple((x["type"], round(x["amount"], 2)) for x in v)))
    merged = {f: merged[f] for f in FIELD_ORDER}
    return apply_rules(merged)


def empty_result(bill_id: str) -> dict[str, Any]:
    return normalize({}, bill_id)


# =====================================================================================
# Image preparation
# =====================================================================================


def load_bill_image(path: Path) -> tuple[bytes, str]:
    """EXIF-rotate, RGB, upscale small photos (<1600px) up to 2x, cap at 2600px. Returns PNG bytes."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        long_side = max(im.size)
        scale = min(2.0, 1600 / long_side) if long_side < 1600 else (2600 / long_side if long_side > 2600 else 1.0)
        if abs(scale - 1.0) > 0.01:
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), "image/png"


# =====================================================================================
# Gemini extractor
# =====================================================================================


class Extractor:
    def __init__(self, api_key: str, model: str, n_samples: int, rpm: int, max_retries: int,
                 temperature: Optional[float], register_mode: str = "first"):
        self.register_mode = register_mode if register_mode in ("first", "matched") else "first"
        self.schema = BillExtractionWithRegisters if self.register_mode == "matched" else BillExtractionSchema
        self.client = genai.Client(api_key=api_key)
        self.model, self.n_samples, self.max_retries, self.temperature = model, n_samples, max_retries, temperature
        self.min_gap, self._last = 60.0 / rpm, 0.0
        self._use_schema = True

    def _config(self, schema: Any = BillExtractionSchema, system: str = SYSTEM_PROMPT) -> types.GenerateContentConfig:
        kw: dict[str, Any] = dict(
            system_instruction=system,
            response_mime_type="application/json",
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
        )
        if self._use_schema:
            kw["response_schema"] = schema
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        return types.GenerateContentConfig(**kw)

    def _call(self, parts: list[Any], schema: Any = BillExtractionSchema,
              system: str = SYSTEM_PROMPT) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            wait = self._last + self.min_gap - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                resp = self.client.models.generate_content(model=self.model, contents=parts,
                                                          config=self._config(schema, system))
                return self._parse(resp.text or "")
            except errors.ClientError as e:
                last = e
                if e.code == 400 and self._use_schema:
                    log.warning("Schema rejected (%s); falling back to prompt-only JSON.", e)
                    self._use_schema = False
                    continue
                if e.code not in (408, 429):
                    raise
                self._backoff(attempt, str(e))
            except (errors.ServerError, ValueError, TimeoutError, ConnectionError) as e:  # incl. JSONDecodeError
                last = e
                self._backoff(attempt)
        raise RuntimeError(f"model call failed after {self.max_retries} attempts: {last}")

    @staticmethod
    def _parse(text: str) -> dict[str, Any]:
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
        data = json.loads(text)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        if not isinstance(data, dict):
            raise ValueError("model did not return a JSON object")
        return data

    @staticmethod
    def _backoff(attempt: int, hint: str = "") -> None:
        m = re.search(r"retry in ([\d.]+)s", hint, re.I)
        delay = float(m.group(1)) + 1 if m else min(60.0, 2.0 ** attempt) + random.random()
        log.warning("Transient error; retrying in %.1fs", delay)
        time.sleep(delay)

    def extract(self, image_path: Path, bill_id: str) -> dict[str, Any]:
        data, mime = load_bill_image(image_path)
        image = types.Part.from_bytes(data=data, mime_type=mime)
        user_text = USER_PROMPT.format(provider=provider_from_bill_id(bill_id) or "unknown")
        if self.register_mode == "matched":
            user_text += (" Also fill meter_rows with EVERY register line of the meter-reading table, one entry per "
                          "line, each with its own previous_reading, current_reading and units as printed.")

        samples = []
        for i in range(self.n_samples):
            try:
                samples.append(normalize(self._call([image, user_text], schema=self.schema), bill_id, self.register_mode))
            except Exception as e:
                log.error("%s: sample %d failed: %s", bill_id, i + 1, e)
        if not samples:
            log.error("%s: all samples failed -> all-null record", bill_id)
            return empty_result(bill_id)

        merged = merge_samples(samples)
        problems = check_consistency(merged)
        if problems:
            log.info("%s: consistency issues -> repair pass: %s", bill_id, problems)
            try:
                text = REPAIR_PROMPT.format(draft=json.dumps(merged, ensure_ascii=False),
                                            problems="\n".join(f"- {p}" for p in problems))
                fixed = normalize(self._call([image, text], schema=self.schema), bill_id, self.register_mode)
                if len(check_consistency(fixed)) < len(problems):
                    merged = fixed
            except Exception as e:
                log.error("%s: repair pass failed: %s", bill_id, e)
        return merged


# =====================================================================================
# CSV + CLI
# =====================================================================================


def find_images(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTS}


def process_level1(extractor: Extractor, bills_dir: Path, template_csv: Optional[Path], output_csv: Path) -> None:
    images = find_images(bills_dir)
    if template_csv:
        with open(template_csv, encoding="utf-8-sig", newline="") as f:
            bill_ids = [row["bill_id"] for row in csv.DictReader(f)]
    else:
        bill_ids = list(images)

    results: dict[str, dict] = {}
    for i, bill_id in enumerate(bill_ids, 1):
        print(f"[{i}/{len(bill_ids)}] {bill_id}", flush=True)
        path = images.get(bill_id)
        try:
            if path is None:
                raise FileNotFoundError(f"no image named {bill_id}.png/.jpg in {bills_dir}")
            results[bill_id] = extractor.extract(path, bill_id)
        except Exception as e:  # one bad bill must never drop a row or crash the run
            print(f"  ! {bill_id} failed: {e} -> writing all-null record", file=sys.stderr)
            results[bill_id] = empty_result(bill_id)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writerow(["bill_id", "json"])
        for bill_id in bill_ids:  # template order, one row per template row
            w.writerow([bill_id, json.dumps(results[bill_id], ensure_ascii=False, separators=(",", ":"))])
    print(f"Level 1 completed: {len(bill_ids)} rows saved to {output_csv}")


def main() -> int:
    ap = argparse.ArgumentParser(description="WTQ Level 1: extract electricity-bill details into level1.csv")
    ap.add_argument("--bills", type=Path, default=Path("bills"))
    ap.add_argument("--template", type=Path, default=Path("csv/level1.csv"))
    ap.add_argument("--output", type=Path, default=Path("output/level1.csv"))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")

    load_dotenv()                                                          # .env in the working directory
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")           # or one folder above this file
    api_key = (os.getenv("API_KEY") or "").strip()
    model = os.getenv("MODEL_NAME", ALLOWED_MODEL).strip()
    if model != ALLOWED_MODEL:
        raise SystemExit(f"MODEL_NAME must be {ALLOWED_MODEL!r} (got {model!r}); other models disqualify you.")
    if not api_key or api_key == "your-api-key-here":
        raise SystemExit("API_KEY missing: put your Google AI Studio key in .env")
    if not args.bills.is_dir():
        raise SystemExit(f"Bills folder not found: {args.bills}")
    template = args.template if args.template.exists() else None
    if template is None:
        print(f"Template {args.template} not found; writing one row per image instead.", file=sys.stderr)

    temp = os.getenv("TEMPERATURE", "").strip()
    extractor = Extractor(
        api_key=api_key, model=model,
        n_samples=max(1, int(os.getenv("N_SAMPLES", "3"))),
        rpm=max(1, int(os.getenv("REQUESTS_PER_MINUTE", "12"))),
        max_retries=max(1, int(os.getenv("MAX_RETRIES", "5"))),
        temperature=float(temp) if temp else None,
        register_mode=os.getenv("MULTI_REGISTER_MODE", "first").strip().lower(),
    )
    process_level1(extractor, args.bills, template, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())