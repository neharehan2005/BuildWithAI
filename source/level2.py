"""WTQ Build Track - Utility Bill Decoder, Level 2 (customer questions).

Fills csv/level2.csv (bill_id, question_id, question, answer) for any set of bills and questions.

Per bill:
  1. Level 1 data  : read from output/level1.csv (keeps both levels consistent) or extracted with decoder.py
  2. Bill facts    : Gemini reads the supporting tables (usage/billing history, meter rows, rate lines,
                     printed % split, late-payment amounts, message board ...), 3 samples, majority vote
  3. Derived values: Python computes averages, counts, shares, differences, late fee, days ... (no LLM maths)
  4. Answers       : Gemini answers ALL of the bill's questions in one call from image + facts + derived values
  5. Review        : a second call audits every number/claim against the bill and fixes mistakes
  6. Safety net    : any missing/empty answer is re-asked alone; never leaves an empty cell

Usage (defaults: bills/, csv/level2.csv, output/level1.csv -> output/level2.csv):
  python level2.py
  python level2.py --bills test/bills --template test/csv/level2.csv --level1 output/level1.csv --output output/level2.csv
  python level2.py --bills train/bills --output output/level2_train.csv      # no template -> built-in sample questions
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from google.genai import types
from pydantic import BaseModel

import decoder as d

log = logging.getLogger("level2")

# =====================================================================================
# Schemas (no defaults: the Gemini API rejects "default" in response schemas)
# =====================================================================================


class HistoryRow(BaseModel):
    month: Optional[str]          # YYYY-MM
    units: Optional[float]
    bill_amount: Optional[float]
    payment_amount: Optional[float]
    pay_date: Optional[str]       # YYYY-MM-DD
    status: Optional[str]


class MeterRow(BaseModel):
    register_label: Optional[str]
    previous_reading: Optional[float]
    current_reading: Optional[float]
    units: Optional[float]
    mdi: Optional[float]


class CalcLine(BaseModel):
    label: str
    units: Optional[float]
    rate: Optional[float]
    amount: Optional[float]


class LabelValue(BaseModel):
    label: str
    value: str


class BillFacts(BaseModel):
    history: list[HistoryRow]
    meter_rows: list[MeterRow]
    calculation_lines: list[CalcLine]
    printed_energy_percent: Optional[float]
    printed_tax_percent: Optional[float]
    payable_after_options: list[LabelValue]
    late_surcharge_values: list[LabelValue]
    other_values: list[LabelValue]
    message_board: Optional[str]


class AnswerItem(BaseModel):
    question_id: str
    answer: str


class AnswerSet(BaseModel):
    answers: list[AnswerItem]


# =====================================================================================
# Prompts
# =====================================================================================

FACTS_SYSTEM = """You read Pakistani electricity bills (K-Electric "KE", LESCO, IESCO): scans, screenshots or tilted \
phone photos, partly Urdu, personal data hidden by grey boxes. Extract the SUPPORTING DATA that customer questions \
are asked about. Copy values AS PRINTED, never calculate or guess; unreadable/not printed -> null. Amounts are plain \
numbers in PKR (no Rs./commas); a trailing "CR" or a minus sign means negative. Never output names, addresses, CNIC, \
account/reference/consumer/contract/invoice/meter numbers.

history: ONE row per month of usage/billing history shown on the bill, month as YYYY-MM.
 - KE: "13 Month Usage History" bar chart = units per month (the LAST bar is the current bill month; the bars go back
   one month at a time; the year headings, e.g. 2025 ... 2026, tell you where the year changes). The "Billing & Payment
   History" table (MM/YY, Billed Amount, Pay-Date, Payment) gives bill_amount, pay_date and payment_amount for the
   months it lists (e.g. 03/26 -> 2026-03). Merge both sources into one row per month.
 - LESCO / IESCO: the "Bill History" table: MONTH (e.g. "SEP 25", "Sep25" -> 2025-09), STATUS, UNITS, BILL (RS.),
   PAYMENT (RS.). It is split into a left and a right block of 6 rows; read the left block top to bottom, then the right.
   These rows are the 12 months BEFORE the current bill month. Negative units/bills (net metering) stay negative.
   A blank payment cell -> null. status = the printed status text (e.g. "EX") or null.
 - Check each row's month: months in one table are consecutive. Do not invent rows that are not printed.
meter_rows: one row per meter register line in the meter/reading section (net metering and peak/off-peak bills have
  several): register_label if printed, previous_reading, current_reading, units, mdi (null if not shown).
calculation_lines: every line of the KE "Bill Calculation" table (fixed charges, variable slabs, adjustments, FCA,
  surcharge, subtotals, taxes, current-month total, opening balance, previous dues, net metering benefit,
  outstanding balance ...) with label, units, rate, amount as printed. For LESCO/IESCO use the Bill Charges Breakdown rows
  (Total Electricity Charges, Subsidies, Net Electricity Charges, Taxes, Current Bill, Total FPA, Arrears, Installment,
  Adjustments, Grand Total ...) with only the amount filled.
printed_energy_percent / printed_tax_percent: the printed percentage split (KE: e.g. "83.39 %" next to Energy Charges
  and "16.61 %" next to Taxes and Duties; IESCO: "Net Electricity Charges 84.76 %", "Taxes 15.24 %"). null if not printed.
payable_after_options: every printed late-payment/after-due-date amount with its label/date range
  (e.g. label "Till 12-Oct-26", value "3674"). late_surcharge_values: printed late-payment SURCHARGE amounts
  (e.g. "LP Surcharge Upto 03/08/26" -> "0"). other_values: any other printed fact a customer might ask about
  (Connected Load, MDI, Security Deposit, Tariff Category, Bill Charge Mode, No. of months, Installation Date,
  Payments/Adjustments, Opening Balance, Net Metering Benefit, Outstanding Balance, Deferred Amount, Outstanding
  Installments, Meter status, any note about FPA/FCA/subsidy ...) as label/value strings. Skip personal identifiers.
message_board: the English text of the Message Board / printed notices (e.g. the FPA/FCA note), or null.
"""

ANSWER_SYSTEM = """You are a friendly, precise electricity-bill assistant for customers of K-Electric, LESCO and IESCO. \
You answer customer questions about ONE bill, using ONLY what is printed on the bill image and the verified data given to you \
(BILL_DATA = extracted fields, BILL_FACTS = supporting tables, DERIVED = values already computed by code).

RULES
1. Numbers must be exactly right. Use DERIVED values for averages, counts, differences, percentages, shares and late fees \
instead of calculating yourself. If the image clearly disagrees with the data, trust the image.
2. Ground every statement in the bill. Never use outside tariff, rate or regulation information. If the bill does not show \
something needed, say clearly that the bill does not show it (and answer whatever part you can).
3. Label estimates. If an answer is an estimate or projection, say "estimate" and show how you calculated it (inputs and \
arithmetic), based only on rates/shares printed on this bill; mention what the estimate leaves out (e.g. fixed charges, \
slab changes, taxes, FPA) when relevant.
4. If a question has more than one reasonable meaning (e.g. history with or without the current month, "total bill" as \
current_bill vs amount payable), state the meaning you used or answer both briefly.
5. Write in plain English for a non-expert, even though parts of the bill are in Urdu. Amounts as "PKR 3,430.24". Dates as \
"21 Apr 2026". Months as "July 2025". A credit/negative amount: say it is a credit and give the amount.
6. Be direct: answer in the first sentence, then give the supporting numbers (typically 2-5 sentences). Name the specific \
lines/amounts from the bill. You may briefly explain what a bill term means in general words, but amounts come from the bill.
7. Never mention names, addresses, CNIC, or account/reference/consumer/meter numbers. No markdown, bullets or headings: \
plain sentences only.
8. Net-metering/credit bills: a negative bill or "CR" amount means the customer has a credit; explain it that way and note \
when "NOT TO BE PAID" is printed.

STYLE EXAMPLES (different bill):
Q: How much of my bill is taxes?
A: Taxes and duties total PKR 569.62 out of your current bill of PKR 3,430.24, or about 16.6%. The bill shows a split of \
83.39% energy charges and 16.61% taxes. The tax lines are sales tax (PKR 520.21), electricity duty (PKR 29.41), and MUCT (KMC) (PKR 20.00).
Q: How many months in my history went above 200 units?
A: Two of the 12 previous months exceeded 200 units: July 2025 (239) and August 2025 (203). The current month used 151 units, \
so the count is two whether or not the current month is included.

OUTPUT: JSON {"answers":[{"question_id":"Q1","answer":"..."}, ...]} with exactly one entry per requested question_id.
"""

REVIEW_SYSTEM = ANSWER_SYSTEM + """
You are now the REVIEWER. You receive draft answers. Check every number, month, name of a charge and claim against the bill \
image, BILL_DATA, BILL_FACTS and DERIVED. Fix wrong or unsupported numbers, wrong months, missing estimate labels, missing \
statements about ambiguity, outside-knowledge claims, personal identifiers, markdown. Keep answers that are already correct \
unchanged. Return ALL answers (same question_ids).
"""

DEFAULT_QUESTIONS = [
    "How much of my bill is taxes?",
    "How many months in my history went above 200 units?",
    "What is my average monthly consumption over the history shown on the bill?",
    "Which month in my history had the highest bill, and how much was it?",
    "How much extra will I pay if I pay after the due date?",
    "What are the biggest charges on my bill?",
    "How does this month's consumption compare with last month?",
    "What is my effective cost per unit this month?",
    "What is my arrears or previous balance, and why is my payable amount what it is?",
    "If I use 50 more units next month, roughly how much more would I pay?",
    "How many days do I have to pay after the bill was issued?",
    "What are my tariff and sanctioned load, and what do they mean?",
]

# =====================================================================================
# Cleaning + merging the extracted facts
# =====================================================================================

NUM = d.to_number


def clean_facts(raw: dict[str, Any]) -> dict[str, Any]:
    hist: dict[str, dict] = {}
    for h in raw.get("history") or []:
        if not isinstance(h, dict):
            continue
        month = d.normalize_date(h.get("month"), month_only=True)
        if not month:
            continue
        row = {"month": month, "units": NUM(h.get("units")), "bill_amount": NUM(h.get("bill_amount")),
               "payment_amount": NUM(h.get("payment_amount")), "pay_date": d.normalize_date(h.get("pay_date")),
               "status": (str(h["status"]).strip() or None) if h.get("status") else None}
        old = hist.get(month)
        if old:  # same month twice (e.g. chart + table): fill gaps
            for k, v in row.items():
                if old.get(k) is None:
                    old[k] = v
        else:
            hist[month] = row
    meter = [{"register_label": (str(m.get("register_label")).strip() or None) if m.get("register_label") else None,
              "previous_reading": NUM(m.get("previous_reading")), "current_reading": NUM(m.get("current_reading")),
              "units": NUM(m.get("units")), "mdi": NUM(m.get("mdi"))}
             for m in raw.get("meter_rows") or [] if isinstance(m, dict)]
    calc = [{"label": str(c.get("label", "")).strip(), "units": NUM(c.get("units")), "rate": NUM(c.get("rate")),
             "amount": NUM(c.get("amount"))}
            for c in raw.get("calculation_lines") or [] if isinstance(c, dict) and str(c.get("label", "")).strip()]

    def lv(items):
        return [{"label": str(i.get("label", "")).strip(), "value": str(i.get("value", "")).strip()}
                for i in items or [] if isinstance(i, dict) and str(i.get("label", "")).strip()]

    mb = raw.get("message_board")
    return {"history": sorted(hist.values(), key=lambda r: r["month"]), "meter_rows": meter,
            "calculation_lines": calc,
            "printed_energy_percent": NUM(raw.get("printed_energy_percent")),
            "printed_tax_percent": NUM(raw.get("printed_tax_percent")),
            "payable_after_options": lv(raw.get("payable_after_options")),
            "late_surcharge_values": lv(raw.get("late_surcharge_values")),
            "other_values": lv(raw.get("other_values")),
            "message_board": str(mb).strip() if mb else None}


def _majority(values: list[Any]) -> Any:
    """Most common non-null value (ties -> earliest); None only if every value is None."""
    non_null = [v for v in values if v is not None]
    if not non_null:
        return None
    key = lambda v: round(v, 2) if isinstance(v, float) else v  # noqa: E731
    counts = Counter(key(v) for v in non_null)
    top = max(counts.values())
    return next(v for v in non_null if counts[key(v)] == top)


def _most_common_list(lists: list[list]) -> list:
    sigs = [json.dumps(l, sort_keys=True) for l in lists]
    counts = Counter(sigs)
    top = max(counts.values())
    cands = [l for l, s in zip(lists, sigs) if counts[s] == top]
    return max(cands, key=len)  # tie -> the more complete one


def merge_facts(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if len(samples) == 1:
        return samples[0]
    need = len(samples) // 2 + 1 if len(samples) >= 3 else 1
    month_count = Counter(r["month"] for s in samples for r in s["history"])
    history = []
    for month in sorted(m for m, c in month_count.items() if c >= need):
        rows = [r for s in samples for r in s["history"] if r["month"] == month]
        history.append({k: (month if k == "month" else _majority([r[k] for r in rows]))
                        for k in ("month", "units", "bill_amount", "payment_amount", "pay_date", "status")})
    merged = {"history": history}
    for k in ("meter_rows", "calculation_lines", "payable_after_options", "late_surcharge_values", "other_values"):
        merged[k] = _most_common_list([s[k] for s in samples])
    for k in ("printed_energy_percent", "printed_tax_percent", "message_board"):
        merged[k] = _majority([s[k] for s in samples])
    return merged


# =====================================================================================
# Derived values (all arithmetic is done here, not by the LLM)
# =====================================================================================

UNIT_THRESHOLDS = (100, 200, 300, 400, 500, 1000)
BILL_THRESHOLDS = (1000, 2000, 5000, 10000, 20000, 50000)


def _r(x: Optional[float], n: int = 2) -> Optional[float]:
    return None if x is None else round(x, n)


def _series_stats(pairs: list[tuple[str, float]], thresholds: tuple[int, ...]) -> Optional[dict[str, Any]]:
    if not pairs:
        return None
    vals = [v for _, v in pairs]
    hi, lo = max(pairs, key=lambda p: p[1]), min(pairs, key=lambda p: p[1])
    return {
        "n_months": len(pairs),
        "values_by_month": {m: v for m, v in pairs},
        "sum": _r(sum(vals)), "average": _r(statistics.mean(vals)), "median": _r(statistics.median(vals)),
        "highest": {"month": hi[0], "value": hi[1]}, "lowest": {"month": lo[0], "value": lo[1]},
        "count_strictly_above": {str(t): sum(1 for v in vals if v > t) for t in thresholds},
        "months_strictly_above": {str(t): [m for m, v in pairs if v > t] for t in thresholds},
    }


def _pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if a is None or not b else round(a / b * 100, 2)


def _days(a: Optional[str], b: Optional[str]) -> Optional[int]:
    if not a or not b:
        return None
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def compute_derived(l1: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    cb, tc, tt = l1.get("current_bill"), l1.get("total_charges"), l1.get("total_taxes")
    units, pw, pa = l1.get("units_consumed"), l1.get("payable_within_due_date"), l1.get("payable_after_due_date")
    cur_month = l1.get("bill_month")
    out: dict[str, Any] = {"notes": [
        "'above X' means strictly greater than X.",
        "history_previous_months = months shown in the history excluding the current bill month; "
        "history_including_current adds this month's own figures.",
    ]}

    # --- shares and per-unit values
    out["shares"] = {
        "taxes_percent_of_current_bill": _pct(tt, cb),
        "charges_percent_of_current_bill": _pct(tc, cb),
        "taxes_percent_of_charges": _pct(tt, tc),
        "printed_energy_percent": facts.get("printed_energy_percent"),
        "printed_tax_percent": facts.get("printed_tax_percent"),
    }
    out["per_unit"] = {
        "current_bill_per_unit": _r(cb / units) if cb is not None and units else None,
        "charges_per_unit": _r(tc / units) if tc is not None and units else None,
        "taxes_per_unit": _r(tt / units) if tt is not None and units else None,
    }
    energy = [c["amount"] for c in l1.get("charges", []) if c["type"] == "energy"]
    out["per_unit"]["energy_charges_per_unit"] = _r(sum(energy) / units) if energy and units else None
    rates = [c for c in facts.get("calculation_lines", [])
             if c.get("rate") and c.get("units") and any(w in c["label"].lower() for w in ("variable", "energy", "peak"))]
    if rates:
        out["per_unit"]["printed_energy_rates"] = [{"label": c["label"], "units": c["units"], "rate": c["rate"]}
                                                    for c in rates]
        out["per_unit"]["highest_printed_energy_rate"] = max(c["rate"] for c in rates)

    # --- line items
    def lines(items, base):
        return [{"type": i["type"], "amount": i["amount"], "percent_of_current_bill": _pct(i["amount"], base)}
                for i in items]

    out["charge_lines"] = lines(l1.get("charges", []), cb)
    out["tax_lines"] = lines(l1.get("taxes", []), cb)
    by_type: dict[str, float] = {}
    for c in l1.get("charges", []):
        by_type[c["type"]] = round(by_type.get(c["type"], 0) + c["amount"], 2)
    out["charge_totals_by_type"] = by_type
    pos = [c for c in l1.get("charges", []) if c["amount"] > 0]
    out["largest_charge_line"] = max(pos, key=lambda c: c["amount"]) if pos else None
    out["largest_tax_line"] = max(l1["taxes"], key=lambda c: c["amount"]) if l1.get("taxes") else None

    # --- payment / dues
    out["payment"] = {
        "current_bill": cb, "arrears": l1.get("arrears"), "payable_within_due_date": pw,
        "payable_after_due_date": pa,
        "extra_if_paid_after_due_date": _r(pa - pw) if pa is not None and pw is not None else None,
        "extra_after_due_percent": _pct(pa - pw, pw) if pa is not None and pw else None,
        "days_issue_to_due": _days(l1.get("issue_date"), l1.get("due_date")),
        "days_reading_to_issue": _days(l1.get("reading_date"), l1.get("issue_date")),
        "days_reading_to_due": _days(l1.get("reading_date"), l1.get("due_date")),
        "is_credit_balance": bool(pw is not None and pw < 0),
        "current_bill_plus_arrears": _r(cb + l1["arrears"]) if cb is not None and l1.get("arrears") is not None else None,
    }

    # --- history
    hist = facts.get("history", [])
    prev = [h for h in hist if h["month"] != cur_month]
    unit_prev = [(h["month"], h["units"]) for h in prev if h["units"] is not None]
    bill_prev = [(h["month"], h["bill_amount"]) for h in prev if h["bill_amount"] is not None]
    pay_prev = [(h["month"], h["payment_amount"]) for h in prev if h["payment_amount"] is not None]
    cur_row = next((h for h in hist if h["month"] == cur_month), None)
    unit_all = unit_prev + [(cur_month, units if units is not None else (cur_row or {}).get("units"))] \
        if cur_month and (units is not None or (cur_row or {}).get("units") is not None) else list(unit_prev)
    bill_all = bill_prev + ([(cur_month, cb)] if cur_month and cb is not None else [])
    out["history_rows_as_read"] = hist
    out["units_history_previous_months"] = _series_stats(unit_prev, UNIT_THRESHOLDS)
    out["units_history_including_current"] = _series_stats(sorted(unit_all), UNIT_THRESHOLDS)
    out["bill_history_previous_months"] = _series_stats(bill_prev, BILL_THRESHOLDS)
    out["bill_history_including_current"] = _series_stats(sorted(bill_all), BILL_THRESHOLDS)
    out["payments_history"] = {
        "n_months_with_payment_shown": len(pay_prev),
        "total_paid": _r(sum(v for _, v in pay_prev)) if pay_prev else None,
        "months_billed_but_no_payment_shown": [h["month"] for h in prev
                                               if h["bill_amount"] and h["bill_amount"] > 0
                                               and h["payment_amount"] in (None, 0)],
    } if prev else None

    # --- this month vs history
    if units is not None and unit_prev:
        last_m, last_u = unit_prev[-1]
        avg = statistics.mean(v for _, v in unit_prev)
        out["units_vs_history"] = {
            "latest_previous_month": last_m, "latest_previous_units": last_u,
            "change_vs_latest_previous": _r(units - last_u),
            "change_vs_latest_previous_percent": _pct(units - last_u, last_u) if last_u else None,
            "average_previous_months": _r(avg), "change_vs_average": _r(units - avg),
            "change_vs_average_percent": _pct(units - avg, avg) if avg else None,
        }
    if cb is not None and bill_prev:
        last_m, last_b = bill_prev[-1]
        out["bill_vs_history"] = {
            "latest_previous_month": last_m, "latest_previous_bill": last_b,
            "change_vs_latest_previous": _r(cb - last_b),
            "change_vs_latest_previous_percent": _pct(cb - last_b, last_b) if last_b else None,
            "average_previous_bills": _r(statistics.mean(v for _, v in bill_prev)),
        }

    # --- comparisons required by the supplied Level 2 question template
    # History months are sorted chronologically by clean_facts(). Do not treat the current bill as a prior month.
    month_map = {h["month"]: h for h in hist if h.get("month")}
    def shift_month(ym: str, delta: int) -> Optional[str]:
        try:
            year, month = map(int, ym.split("-"))
            index = year * 12 + (month - 1) + delta
            return f"{index // 12:04d}-{index % 12 + 1:02d}"
        except (ValueError, AttributeError):
            return None

    # Same calendar month last year, when the bill history actually contains it.
    same_month_last_year = shift_month(cur_month, -12) if cur_month else None
    yoy_row = month_map.get(same_month_last_year) if same_month_last_year else None
    out["same_month_last_year"] = {
        "month": same_month_last_year,
        "units": (yoy_row or {}).get("units"),
        "current_units": units,
        "difference_units": _r(units - yoy_row["units"]) if units is not None and yoy_row and yoy_row.get("units") is not None else None,
        "difference_percent": _pct(units - yoy_row["units"], yoy_row["units"]) if units is not None and yoy_row and yoy_row.get("units") not in (None, 0) else None,
        "available": bool(yoy_row and yoy_row.get("units") is not None),
    }

    # Previous 3 calendar months, plus direction across the available readings.
    last3 = []
    for offset in (3, 2, 1):
        m = shift_month(cur_month, -offset) if cur_month else None
        row = month_map.get(m) if m else None
        if row and row.get("units") is not None:
            last3.append({"month": m, "units": row["units"]})
    vals3 = [x["units"] for x in last3]
    out["last_3_months_trend"] = {
        "months": last3,
        "direction": ("increased" if len(vals3) >= 2 and vals3[-1] > vals3[0] else
                      "decreased" if len(vals3) >= 2 and vals3[-1] < vals3[0] else
                      "unchanged" if len(vals3) >= 2 and vals3[-1] == vals3[0] else "insufficient data"),
        "change_units": _r(vals3[-1] - vals3[0]) if len(vals3) >= 2 else None,
    }

    # Seasonal comparison: use only months present in the bill's history, and compare averages as well as totals.
    def seasonal(month_nums: set[int]) -> dict[str, Any]:
        selected = [(h["month"], h["units"]) for h in hist
                    if h.get("month") and int(h["month"].split("-")[1]) in month_nums and h.get("units") is not None
                    and h["month"] != cur_month]
        return {"months": [{"month": m, "units": u} for m, u in selected],
                "n_months": len(selected), "total_units": _r(sum(u for _, u in selected)) if selected else None,
                "average_units": _r(statistics.mean(u for _, u in selected)) if selected else None}
    summer = seasonal({6, 7, 8})
    winter = seasonal({12, 1, 2})
    out["summer_vs_winter"] = {
        "summer_jun_aug": summer, "winter_dec_feb": winter,
        "difference_total_units": _r(summer["total_units"] - winter["total_units"])
            if summer["total_units"] is not None and winter["total_units"] is not None else None,
        "difference_average_units": _r(summer["average_units"] - winter["average_units"])
            if summer["average_units"] is not None and winter["average_units"] is not None else None,
        "note": "Compare only the summer/winter months actually shown; totals are not directly comparable if month counts differ."
    }

    # A simple forecast uses the mean of available prior history, not a fabricated trend model.
    forecast_pairs = unit_prev
    out["next_month_units_estimate"] = {
        "estimate_units": _r(statistics.mean(v for _, v in forecast_pairs)) if forecast_pairs else None,
        "based_on_months": len(forecast_pairs),
        "method": "arithmetic mean of available previous-month usage values shown on the bill",
    }

    # Average monthly bill set-aside using current bill plus historical bill amounts that are printed.
    budget_pairs = bill_prev + ([(cur_month, cb)] if cur_month and cb is not None else [])
    out["monthly_bill_budget_estimate"] = {
        "average_pkr": _r(statistics.mean(v for _, v in budget_pairs)) if budget_pairs else None,
        "months_used": len(budget_pairs),
        "method": "average of printed bill amounts available, including credit/negative amounts as printed",
    }

    # Estimate bill at last month's usage using this month's all-in average per-unit cost.
    latest_units = unit_prev[-1][1] if unit_prev else None
    out["same_as_last_month_bill_estimate"] = {
        "last_month": unit_prev[-1][0] if unit_prev else None,
        "last_month_units": latest_units,
        "estimate_pkr": _r((cb / units) * latest_units) if cb is not None and units not in (None, 0) and latest_units is not None else None,
        "method": "last month's units multiplied by current bill divided by current units; rough estimate only, assumes the all-in average rate stays constant",
    }

    # --- helper for "what if I use N more units" estimates (the model must label them estimates)
    out["estimate_helper"] = {
        "avg_energy_charge_per_unit_this_bill": out["per_unit"]["energy_charges_per_unit"],
        "highest_printed_energy_rate": out["per_unit"].get("highest_printed_energy_rate"),
        "taxes_percent_of_charges": out["shares"]["taxes_percent_of_charges"],
        "all_in_bill_per_unit": out["per_unit"]["current_bill_per_unit"],
        "how_to": ("extra_units x (highest printed energy rate or average energy rate) = extra energy charge; add taxes at "
                   "taxes_percent_of_charges; state clearly that this is an estimate and what is not included."),
    }
    return out


# =====================================================================================
# Gemini calls
# =====================================================================================


class Level2Engine(d.Extractor):
    def __init__(self, *args: Any, facts_samples: int = 3, review: bool = True, **kw: Any):
        super().__init__(*args, **kw)
        self.facts_samples, self.review = facts_samples, review

    # ---- facts
    def get_facts(self, image: Any, bill_id: str, l1: dict[str, Any]) -> dict[str, Any]:
        hint = (f"Provider: {l1.get('provider')}. Current bill month: {l1.get('bill_month')} "
                "(use it to place months in the right year). Extract the supporting data as instructed.")
        samples = []
        for i in range(self.facts_samples):
            try:
                samples.append(clean_facts(self._call([image, hint], schema=BillFacts, system=FACTS_SYSTEM)))
            except Exception as e:
                log.error("%s: facts sample %d failed: %s", bill_id, i + 1, e)
        if not samples:
            log.error("%s: no facts could be extracted; answering from Level 1 data and the image only", bill_id)
            return clean_facts({})
        return merge_facts(samples)

    # ---- answers
    @staticmethod
    def _context(l1: dict[str, Any], facts: dict[str, Any], derived: dict[str, Any]) -> str:
        return ("BILL_DATA:\n" + json.dumps(l1, ensure_ascii=False) +
                "\n\nBILL_FACTS:\n" + json.dumps(facts, ensure_ascii=False) +
                "\n\nDERIVED:\n" + json.dumps(derived, ensure_ascii=False))

    def _ask(self, image: Any, context: str, questions: list[dict[str, str]], system: str,
             drafts: Optional[dict[str, str]] = None) -> dict[str, str]:
        qtext = "\n".join(f'{q["question_id"]}: {q["question"]}' for q in questions)
        text = f"{context}\n\nQUESTIONS:\n{qtext}\n"
        if drafts:
            text += "\nDRAFT ANSWERS TO REVIEW:\n" + json.dumps(
                {"answers": [{"question_id": k, "answer": v} for k, v in drafts.items()]}, ensure_ascii=False)
        data = self._call([image, text], schema=AnswerSet, system=system)
        wanted = {q["question_id"] for q in questions}
        out: dict[str, str] = {}
        for a in data.get("answers", []):
            if isinstance(a, dict) and a.get("question_id") in wanted and str(a.get("answer", "")).strip():
                out[a["question_id"]] = " ".join(str(a["answer"]).split())  # one clean paragraph
        return out

    def answer_bill(self, bill_id: str, image_path: Path, l1: dict[str, Any],
                    questions: list[dict[str, str]]) -> dict[str, str]:
        data, mime = d.load_bill_image(image_path)
        image = types.Part.from_bytes(data=data, mime_type=mime)
        facts = self.get_facts(image, bill_id, l1)
        derived = compute_derived(l1, facts)
        ctx = self._context(l1, facts, derived)

        answers: dict[str, str] = {}
        try:
            answers = self._ask(image, ctx, questions, ANSWER_SYSTEM)
        except Exception as e:
            log.error("%s: batch answering failed: %s", bill_id, e)

        if self.review and answers:
            try:
                reviewed = self._ask(image, ctx, [q for q in questions if q["question_id"] in answers],
                                     REVIEW_SYSTEM, drafts=answers)
                answers.update(reviewed)  # keep draft where the reviewer returned nothing
            except Exception as e:
                log.error("%s: review pass failed (keeping drafts): %s", bill_id, e)

        for q in questions:  # safety net: ask any missing question on its own
            if q["question_id"] not in answers:
                try:
                    answers.update(self._ask(image, ctx, [q], ANSWER_SYSTEM))
                except Exception as e:
                    log.error("%s/%s: single-question retry failed: %s", bill_id, q["question_id"], e)
        for q in questions:
            answers.setdefault(q["question_id"], FALLBACK_ANSWER)
        return answers


FALLBACK_ANSWER = ("I could not read enough of this bill to answer this question reliably, so I cannot give a "
                   "figure for it. Please check the printed bill or contact your electricity provider.")

# =====================================================================================
# Level 1 data, CSV and CLI
# =====================================================================================


def load_level1(path: Optional[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if path and path.exists():
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    out[row["bill_id"]] = json.loads(row["json"])
                except (ValueError, KeyError, TypeError):
                    pass
    return out


def read_questions(template: Optional[Path], bill_ids_fallback: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    """Return (fieldnames, rows). Rows keep template order; without a template build default questions."""
    if template and template.exists():
        with open(template, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader.fieldnames or []), [dict(r) for r in reader]
    print(f"Template {template} not found: using {len(DEFAULT_QUESTIONS)} built-in sample questions per bill.",
          file=sys.stderr)
    rows = [{"bill_id": b, "question_id": f"Q{i}", "question": q, "answer": ""}
            for b in bill_ids_fallback for i, q in enumerate(DEFAULT_QUESTIONS, 1)]
    return ["bill_id", "question_id", "question", "answer"], rows


def process_level2(engine: Level2Engine, bills_dir: Path, template: Optional[Path], level1_csv: Optional[Path],
                   output_csv: Path, only: Optional[str] = None) -> None:
    images = d.find_images(bills_dir)
    fieldnames, rows = read_questions(template, list(images))
    if only:
        rows = [r for r in rows if r["bill_id"] == only]
    level1 = load_level1(level1_csv)

    order: list[str] = []
    for r in rows:
        if r["bill_id"] not in order:
            order.append(r["bill_id"])
    all_answers: dict[str, dict[str, str]] = {}

    for i, bill_id in enumerate(order, 1):
        qs = [{"question_id": r["question_id"], "question": r["question"]} for r in rows if r["bill_id"] == bill_id]
        print(f"[{i}/{len(order)}] {bill_id}: {len(qs)} questions", flush=True)
        path = images.get(bill_id)
        try:
            if path is None:
                raise FileNotFoundError(f"no image named {bill_id}.png/.jpg in {bills_dir}")
            l1 = level1.get(bill_id)
            if not l1 or l1.get("current_bill") is None:
                print("  (no usable Level 1 data for this bill: extracting it now)", flush=True)
                l1 = engine.extract(path, bill_id)
            all_answers[bill_id] = engine.answer_bill(bill_id, path, l1, qs)
        except Exception as e:  # one bad bill must never drop rows or crash the run
            print(f"  ! {bill_id} failed: {e} -> fallback answers", file=sys.stderr)
            all_answers[bill_id] = {q["question_id"]: FALLBACK_ANSWER for q in qs}

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writeheader()
        for r in rows:  # template order, bill_id/question_id/question untouched
            r = dict(r)
            r["answer"] = all_answers[r["bill_id"]].get(r["question_id"], FALLBACK_ANSWER)
            w.writerow(r)
    print(f"Level 2 completed: {len(rows)} rows saved to {output_csv}")


def main() -> int:
    ap = argparse.ArgumentParser(description="WTQ Level 2: answer customer questions about electricity bills")
    ap.add_argument("--bills", type=Path, default=Path("bills"))
    ap.add_argument("--template", type=Path, default=Path("csv/level2.csv"))
    ap.add_argument("--level1", type=Path, default=Path("output/level1.csv"),
                    help="Level 1 results to reuse (missing bills are extracted automatically)")
    ap.add_argument("--output", type=Path, default=Path("output/level2.csv"))
    ap.add_argument("--only", help="process just one bill_id (quick test)")
    ap.add_argument("--no-review", action="store_true", help="skip the review pass (faster, slightly less accurate)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")

    load_dotenv()
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    api_key = (os.getenv("API_KEY") or "").strip()
    model = os.getenv("MODEL_NAME", d.ALLOWED_MODEL).strip()
    if model != d.ALLOWED_MODEL:
        raise SystemExit(f"MODEL_NAME must be {d.ALLOWED_MODEL!r} (got {model!r}); other models disqualify you.")
    if not api_key or api_key == "your-api-key-here":
        raise SystemExit("API_KEY missing: put your Google AI Studio key in .env")
    if not args.bills.is_dir():
        raise SystemExit(f"Bills folder not found: {args.bills}")

    temp = os.getenv("TEMPERATURE", "").strip()
    engine = Level2Engine(
        api_key, model, n_samples=max(1, int(os.getenv("N_SAMPLES", "3"))),
        rpm=max(1, int(os.getenv("REQUESTS_PER_MINUTE", "12"))),
        max_retries=max(1, int(os.getenv("MAX_RETRIES", "5"))),
        temperature=float(temp) if temp else None,
        facts_samples=max(1, int(os.getenv("FACTS_SAMPLES", "3"))),
        review=not args.no_review,
        register_mode=os.getenv("MULTI_REGISTER_MODE", "first").strip().lower(),
    )
    process_level2(engine, args.bills, args.template, args.level1, args.output, args.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())