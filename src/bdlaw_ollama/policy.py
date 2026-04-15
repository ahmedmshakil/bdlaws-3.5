from __future__ import annotations

import re
from dataclasses import dataclass

from .config import AppConfig
from .utils import compact_whitespace, contains_bangla, normalize_digits


GREETING_HINTS = (
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "salam",
    "assalamu alaikum",
    "assalamualaikum",
)

INTRO_HINTS = (
    "introduce yourself",
    "who are you",
    "what are you",
    "tell me about yourself",
    "who developed you",
    "developed by",
    "developer",
    "নিজের পরিচয়",
    "পরিচয় দাও",
    "তুমি কে",
    "কে তৈরি",
    "কে develop",
)

LAW_HINTS = (
    "bangladesh law",
    "bangladeshi law",
    "bd law",
    "act",
    "acts",
    "ordinance",
    "order",
    "code",
    "constitution",
    "legal",
    "legislation",
    "statute",
    "section",
    "article",
    "chapter",
    "evidence act",
    "penal code",
    "ধারা",
    "অ্যাক্ট",
    "অনুচ্ছেদ",
    "অধ্যায়",
    "বিধি",
    "অর্ডিন্যান্স",
    "সংবিধান",
    "আইনপাঠ",
    "দণ্ডবিধি",
    "সাক্ষ্য আইন",
    "বাংলাদেশের আইন",
    "বাংলাদেশি আইন",
)

OFFTOPIC_HINTS = (
    "weather",
    "temperature",
    "fahrenheit",
    "celsius",
    "math",
    "calculate",
    "calculation",
    "sum",
    "plus",
    "minus",
    "multiply",
    "division",
    "translate",
    "translation",
    "poem",
    "story",
    "joke",
    "recipe",
    "song",
    "movie",
    "football",
    "cricket",
    "code",
    "python",
    "programming",
    "আবহাওয়া",
    "তাপমাত্রা",
    "গণিত",
    "যোগ",
    "বিয়োগ",
    "গল্প",
    "কবিতা",
    "রেসিপি",
)

ROMANIZED_BANGLA_HINTS = (
    "amake",
    "bolo",
    "bolen",
    "koro",
    "koren",
    "dhara",
    "niye",
    "ain",
    "shongkhepe",
    "shoja",
)

BROAD_PATTERNS = (
    r"\bbangladesh(i)? laws?\b",
    r"\bbd laws?\b",
    r"বাংলাদেশি আইন",
    r"বাংলাদেশের আইন",
    r"আইন নিয়ে বল",
    r"আইন সম্পর্কে বল",
    r"\b\d{4}\s*(er|এর)\s*act\b",
    r"\bact niye bolo\b",
)

SPECIFIC_PATTERNS = (
    r"\bsection\b",
    r"\barticle\b",
    r"\bchapter\b",
    r"\bordinance\b",
    r"\border\b",
    r"\bcode\b",
    r"\bconstitution\b",
    r"\bact no\b",
    r"ধারা",
    r"অনুচ্ছেদ",
    r"অধ্যায়",
    r"বিধি",
    r"\b\d+\s*(section|article|chapter)\b",
    r"[০-৯]+\s*ধারা",
)


@dataclass(slots=True)
class PromptRoute:
    intent: str
    language: str


def _normalize(text: str) -> str:
    return compact_whitespace(normalize_digits(text)).lower()


def _contains_any(text: str, hints: tuple[str, ...]) -> bool:
    for hint in hints:
        if re.fullmatch(r"[a-z0-9 ]+", hint):
            if re.search(rf"(?<![a-z0-9]){re.escape(hint)}(?![a-z0-9])", text):
                return True
            continue
        if hint in text:
            return True
    return False


def resolve_response_language(question: str, config: AppConfig) -> str:
    normalized = _normalize(question)
    if contains_bangla(question):
        return "bangla"
    if _contains_any(normalized, ROMANIZED_BANGLA_HINTS):
        return "bangla"
    default_language = str(config.project["assistant"].get("default_response_language", "bangla")).lower()
    return "english" if re.search(r"[a-z]", normalized) else default_language


def route_user_message(question: str, config: AppConfig) -> PromptRoute:
    normalized = _normalize(question)
    language = resolve_response_language(question, config)
    greeting = _contains_any(normalized, GREETING_HINTS)
    intro = _contains_any(normalized, INTRO_HINTS)
    law = _contains_any(normalized, LAW_HINTS) or bool(re.search(r"\b(18|19|20)\d{2}\b", normalized))
    obvious_offtopic = _contains_any(normalized, OFFTOPIC_HINTS) or bool(re.search(r"\b\d+\s*[\+\-\*/]\s*\d+\b", normalized))

    if (greeting or intro) and not law:
        return PromptRoute(intent="greeting", language=language)

    if law:
        broad = any(re.search(pattern, normalized) for pattern in BROAD_PATTERNS)
        specific = any(re.search(pattern, normalized) for pattern in SPECIFIC_PATTERNS)
        if broad and not specific:
            return PromptRoute(intent="law_broad", language=language)
        return PromptRoute(intent="law", language=language)

    if obvious_offtopic or normalized:
        return PromptRoute(intent="offtopic", language=language)

    return PromptRoute(intent="offtopic", language=language)


def _assistant_text(config: AppConfig, key: str, language: str) -> str:
    suffix = "bn" if language == "bangla" else "en"
    return str(config.project["assistant"][f"{key}_{suffix}"]).strip()


def build_guardrail_payload(question: str, config: AppConfig) -> dict | None:
    route = route_user_message(question, config)
    if route.intent == "law":
        return None

    if route.intent == "greeting":
        answer = _assistant_text(config, "intro_message", route.language)
    elif route.intent == "law_broad":
        answer = _assistant_text(config, "clarification_message", route.language)
    else:
        answer = _assistant_text(config, "offtopic_refusal", route.language)

    return {
        "answer": answer,
        "citations": [],
        "confidence": "policy",
        "grounded": False,
        "intent": route.intent,
    }
