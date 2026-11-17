"""SHARED: category -> subcategory taxonomy and rule-based priority (ARCHITECTURE.md Section 6)."""
from __future__ import annotations

import json
import re
from collections import Counter

from app import config

DEFAULT_TAXONOMY: dict[str, list[str]] = {
    "Access/Identity": ["Password Reset", "Account Lockout", "MFA Issue", "Access Request"],
    "Network": ["VPN", "WiFi", "DNS/Connectivity", "Proxy/Firewall"],
    "Hardware": ["Laptop", "Printer", "Peripheral", "Mobile Device"],
    "Software/Application": ["Email/Outlook", "SAP", "Software Installation", "Application Crash",
                             "OneDrive/SharePoint Sync", "Web Application Error"],
    "Database": ["DB Performance", "DB Connection", "Backup/Restore"],
    "Security": ["Phishing", "Malware/Antivirus"],
    "Infrastructure": ["Server/Storage"],
    "Inquiry": ["How-To", "License Request"],
}


def build_from_labels(pairs: list[tuple[str, str]]) -> dict[str, list[str]]:
    """Taxonomy = the label set found in the data."""
    tax: dict[str, list[str]] = {}
    for cat, sub in sorted(set(pairs)):
        tax.setdefault(cat, []).append(sub)
    return tax


def save(tax: dict[str, list[str]]) -> None:
    config.TAXONOMY_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.TAXONOMY_JSON.write_text(json.dumps(tax, indent=2), encoding="utf-8")


def load() -> dict[str, list[str]]:
    if config.TAXONOMY_JSON.exists():
        return json.loads(config.TAXONOMY_JSON.read_text(encoding="utf-8"))
    return DEFAULT_TAXONOMY


def sub_to_cat(tax: dict[str, list[str]]) -> dict[str, str]:
    return {s: c for c, subs in tax.items() for s in subs}


def canonical(tax: dict[str, list[str]], category: str | None, subcategory: str | None) -> tuple[str | None, str | None]:
    """Map possibly mis-cased / mismatched labels onto the taxonomy. Returns (None, None) if invalid."""
    cats = {c.lower(): c for c in tax}
    subs = {s.lower(): s for s in sub_to_cat(tax)}
    s = subs.get((subcategory or "").strip().lower())
    if s:                                   # subcategory decides the category
        return sub_to_cat(tax)[s], s
    c = cats.get((category or "").strip().lower())
    return (c, None) if c else (None, None)


# ---- priority rules ------------------------------------------------------
_P1 = re.compile(
    r"\b(outage|all users|everyone|entire (company|office|site|floor|department)|whole (company|office|site)|"
    r"production (is )?down|prod (is )?down|system down|completely down|security incident|breach|ransomware|"
    r"malware|data loss|nobody can|no one can|company-wide|site-wide)\b", re.I)
_P2 = re.compile(
    r"\b(multiple users|several users|many users|whole team|entire team|the team|our team|department|"
    r"blocked|blocking|cannot work|can't work|unable to work|deadline|month-end|payroll|customer-facing|"
    r"urgent|critical|asap)\b", re.I)
_P4 = re.compile(
    r"\b(how (do|can|to)|is it possible|would like to know|question about|request(ing)? (a|an|for)|"
    r"need (a|an) [\w ]{0,25}licen[cs]e|requesting|please provide|information on|guide|documentation)\b", re.I)

ORDER = ["P1", "P2", "P3", "P4"]


def rule_priority(text: str, environment: str | None = None) -> str:
    """Outage / all users / prod down / security -> P1; many users / blocked -> P2;
    how-to / request -> P4; otherwise a single user -> P3."""
    t = text or ""
    if _P1.search(t) or (environment == "prod" and re.search(r"\bdown\b", t, re.I)):
        return "P1"
    if _P2.search(t):
        return "P2"
    if _P4.search(t):
        return "P4"
    return "P3"


def more_urgent(a: str, b: str) -> str:
    return a if ORDER.index(a) <= ORDER.index(b) else b


def majority(items: list[str]) -> str | None:
    items = [i for i in items if i]
    return Counter(items).most_common(1)[0][0] if items else None
