"""Rule-based lease field extraction with provenance.

Leases are regular legal prose: labelled clauses, a few date and money formats. Patterns fail
loudly (field comes back empty and gets flagged) instead of guessing, which is the property
that matters when a wrong rent or date costs someone money. Every value carries the page,
box and quote it was read from. An optional LlmExtractor can later fill only the fields
this module left empty; see README.
"""
from __future__ import annotations

import re
from datetime import date

from .domain import ExtractedField, Source
from .ingest import Document

FIELD_NAMES = [
    "landlord_name", "tenant_name", "landlord_signed", "tenant_signed", "unit_ref",
    "commencement_date", "expiry_date", "term_months", "monthly_rent", "rent_frequency",
    "annual_rent", "deposit_amount", "escalation_clause", "escalation_defined",
    "renewal_terms", "termination_terms",
]

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

DATE = (r"(\d{4}-\d{2}-\d{2}|\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9},?\s+\d{4}"
        r"|[A-Za-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}|\d{1,2}[/.-]\d{1,2}[/.-]\d{4})")
CUR = r"(?:QAR|QR|Qatari\s+Riyals?)"
# A period is accepted as a thousands separator too: OCR often reads the comma in "9,500" as "9.500".
# QAR amounts have two decimals at most, so exactly three digits after a separator is a thousands group.
NUM = r"(\d{1,3}(?:[,.]\d{3})+(?!\d)(?:[.,]\d{1,2}(?!\d))?|\d+(?:\.\d{1,2})?(?!\d))"


def money_patterns(label: str) -> list[str]:
    """A bare number counts only straight after a colon; a looser gap needs a currency token,
    so '5%' in an escalation clause is never read as a rent."""
    return [rf"{label}\s*[:\-–]\s*{CUR}?\s*{NUM}", rf"{label}\b[^\d]{{0,60}}?{CUR}\s*{NUM}"]

# "5. RENT" and the "5.RENT" that OCR often produces (no space) are both headings; "12.5% rent" is not.
HEADING = re.compile(r"^\s*(\d+[\.\)](?:\s|(?=[A-Z]))|[A-Z][A-Z &/-]{3,}$)")


def parse_date(text: str) -> date | None:
    s = text.strip().rstrip(".,")
    try:
        if m := re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s):
            return date(int(m[1]), int(m[2]), int(m[3]))
        if m := re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+),?\s+(\d{4})", s):
            return date(int(m[3]), _MONTHS[m[2][:3].lower()], int(m[1]))
        if m := re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", s):
            return date(int(m[3]), _MONTHS[m[1][:3].lower()], int(m[2]))
        if m := re.fullmatch(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})", s):  # day first
            return date(int(m[3]), int(m[2]), int(m[1]))
    except (ValueError, KeyError):
        return None
    return None


def _iso_date(m: re.Match) -> str | None:
    parsed = parse_date(m[1])
    return parsed.isoformat() if parsed else None


def _money(s: str) -> float:
    """'9,500', '9.500' (OCR), '9,500.50' and '12.50' -> 9500.0, 9500.0, 9500.5, 12.5."""
    m = re.fullmatch(r"(\d{1,3}(?:[,.]\d{3})+)(?:[.,](\d{1,2}))?", s)
    if m:
        return float(re.sub(r"[,.]", "", m[1]) + (f".{m[2]}" if m[2] else ""))
    return float(s.replace(",", ""))


class _Reader:
    def __init__(self, doc: Document):
        self.doc = doc
        self.fields: dict[str, ExtractedField] = {n: ExtractedField(name=n) for n in FIELD_NAMES}

    def _set(self, name, value, source: Source | None, confidence: float, note: str | None = None):
        f = self.fields[name]
        f.value, f.original_value, f.source, f.confidence, f.note = value, value, source, confidence, note

    def find(self, patterns: list[str], convert, name: str, confidence: float = 0.9):
        """Collect every match of every pattern; first wins, disagreement is recorded."""
        hits: list[tuple[object, Source | None]] = []
        for pat in patterns:
            for m in re.finditer(pat, self.doc.text, re.I):
                try:
                    value = convert(m)
                except (ValueError, KeyError):
                    value = None
                if value is not None:
                    hits.append((value, self.doc.locate(m.start(), m.end())))
        if not hits:
            return
        distinct = []
        for v, _ in hits:
            if v not in distinct:
                distinct.append(v)
        if len(distinct) > 1:
            self._set(name, hits[0][0], hits[0][1], confidence * 0.6,
                      f"CONFLICT: the lease gives different values: {', '.join(map(str, distinct))}")
        else:
            self._set(name, hits[0][0], hits[0][1], confidence)

    def check_separator(self, name: str):
        """A period where a thousands comma belongs ('9.500') was read as 9,500: say so, lower the confidence,
        and let a person confirm it against the page. OCR makes this misread; a person must see it."""
        f = self.fields[name]
        if f.value is None or f.source is None or f.note:
            return
        m = re.search(r"\d\.\d{3}(?!\d)", f.source.quote)
        if m:
            f.confidence = min(f.confidence, 0.6)
            f.note = (f"CHECK: the page shows '{m.group(0)}', a period where a thousands comma is expected. "
                      f"Read as {f.value:,.0f}. Confirm against the page.")

    def clause(self, name: str, keywords: str, prefer: str | None = None, max_lines: int = 3):
        """Capture the clause around a keyword: its heading's body, or the line plus followers."""
        lines = self.doc.lines
        idx = [i for i, ln in enumerate(lines) if re.search(keywords, ln.text, re.I)]
        if prefer:
            narrowed = [i for i in idx if re.search(prefer, lines[i].text, re.I)]
            idx = narrowed or idx
        if not idx:
            return None
        i = idx[0]
        start = i + 1 if HEADING.match(lines[i].text) and i + 1 < len(lines) else i
        end = start
        while end + 1 < len(lines) and end - start + 1 < max_lines and not HEADING.match(lines[end + 1].text):
            end += 1
        text = " ".join(l.text for l in lines[start:end + 1])
        self._set(name, text, self.doc.locate_lines(start, end), 0.8)
        return text


def extract_fields(doc: Document) -> dict[str, ExtractedField]:
    r = _Reader(doc)

    # Parties
    party = r"[:\-]\s*([^\n(,]+?)\s*(?:[,(\n]|$)"
    r.find([rf"(?:landlord|lessor)(?:\s+name)?\s*{party}",
            r"between\s+([^\n(]+?)\s*\((?:the\s+)?[\"“']?(?:landlord|lessor)"],
           lambda m: m[1].strip(), "landlord_name")
    r.find([rf"(?:tenant|lessee)(?:\s+name)?\s*{party}",
            r"and\s+([^\n(]+?)\s*\((?:the\s+)?[\"“']?(?:tenant|lessee)"],
           lambda m: m[1].strip(), "tenant_name")

    # Signatures: filled means letters after the label, once the date part is removed.
    def signed(m):
        body = re.sub(r"date\s*[:\-].*", "", m[1], flags=re.I)
        return bool(re.search(r"[A-Za-z]{2,}", body))

    for who, key in (("landlord", "landlord_signed"), ("tenant", "tenant_signed")):
        r.find([rf"{who}\s+signature\s*[:\-]\s*([^\n]*)"], signed, key, confidence=0.6)
        if r.fields[key].value is not None:
            r.fields[key].note = "Text layer only. A handwritten signature on a scan is not verified."

    # Unit: a full unit id wins, otherwise the apartment number and tower.
    r.find([r"\b([A-Z]{2,4}-[A-Z]-\d{3,4})\b"], lambda m: m[1].upper(), "unit_ref")
    if r.fields["unit_ref"].value is None:
        r.find([r"\b((?:apartment|apt|unit|flat)\s+(?:no\.?\s*)?\d{3,4}(?:,?\s*tower\s+[A-Z])?)\b"],
               lambda m: m[1].strip(), "unit_ref", confidence=0.7)

    # Dates and term
    r.find([rf"(?:commencement|start(?:ing)?|effective)\s+date\b[^\d]{{0,30}}{DATE}",
            rf"commenc\w+\s+on\s+{DATE}"], _iso_date, "commencement_date")
    r.find([rf"(?:expiry|expiration|end)\s+date\b[^\d]{{0,30}}{DATE}",
            rf"(?:expires?|terminates?|ending)\s+on\s+{DATE}"], _iso_date, "expiry_date")

    def term(m):
        n = int(m[1])
        return n * 12 if m[2].lower().startswith("year") else n

    r.find([r"term\s+of\s+(?:[a-z\-]+\s*)?\(?(\d{1,3})\)?\s*\(?\s*(month|year)s?",
            r"lease\s+term\b[^\d\n]{0,10}(\d{1,3})\s*(month|year)s?"], term, "term_months")

    # Money
    money = lambda m: _money(m[1])
    r.find(money_patterns(r"monthly\s+rent") +
           [rf"rent\s+of\s+{CUR}?\s*{NUM}\s*(?:\([^)]*\)\s*)?(?:per|a|each)\s+month"], money, "monthly_rent")
    if r.fields["monthly_rent"].value is not None:
        r._set("rent_frequency", "monthly", r.fields["monthly_rent"].source, 0.9)
    r.find(money_patterns(r"(?:annual|yearly)\s+rent"), money, "annual_rent")
    # "Securty Deposit" (an OCR dropout) and a bare "Deposit:" are read too, but only straight after a colon.
    r.find(money_patterns(r"security\s+deposit") + [rf"(?:secu\w*\s+)?deposit\s*[:\-–]\s*{CUR}?\s*{NUM}"],
           money, "deposit_amount")
    for amount in ("monthly_rent", "annual_rent", "deposit_amount"):
        r.check_separator(amount)
    if r.fields["deposit_amount"].value is None and r.fields["monthly_rent"].value is not None:
        words = {"one": 1, "two": 2, "three": 3, "1": 1, "2": 2, "3": 3}
        m = re.search(r"security\s+deposit[^.\n]{0,60}?\b(one|two|three|1|2|3)\s*(?:\(\d\)\s*)?months?", doc.text, re.I)
        if m:
            rent = r.fields["monthly_rent"].value
            r._set("deposit_amount", words[m[1].lower()] * rent, doc.locate(m.start(), m.end()), 0.6,
                   f"Stated as {m[1]} month(s) of rent, converted using monthly rent {rent:,.0f}.")

    # Clauses
    text = r.clause("escalation_clause",
                    r"escalat|rent\s+(?:increase|review)|increase\s+in\s+(?:the\s+)?rent|annual\s+increase")
    if text is not None:
        mechanism = bool(re.search(r"\d+(?:\.\d+)?\s*%|per\s*cent|\bCPI\b|consumer\s+price|\bindex", text, re.I))
        vague = bool(re.search(r"mutually\s+agreed|to\s+be\s+agreed|as\s+agreed|negotiat", text, re.I))
        has_percent = bool(re.search(r"\d+(?:\.\d+)?\s*%", text))
        r._set("escalation_defined", mechanism and (not vague or has_percent),
               r.fields["escalation_clause"].source, 0.8)
    r.clause("renewal_terms", r"\brenew|option\s+to\s+extend")
    r.clause("termination_terms", r"terminat|notice", prefer=r"notice|early\s+terminat")

    return r.fields
