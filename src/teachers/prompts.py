"""Teacher prompts.

Until the four LoRA teachers of §4.1 are trained, the same-tokenizer group is
Qwen2.5-7B-Instruct with four PROMPT variants that mirror the planned LoRA data mixes
(balanced, dialect-heavy, finance/PII-heavy, transliteration-heavy). Each variant also
sees the few-shot examples in a different order. When LoRA adapters exist, set
`adapter:` per variant in the config; the prompt stays the same.

All example PII below is synthetic (dummy numbers, example.com, test card number).
"""
from src.teachers.output_schema import serialize_with_offsets

TASK = """You are a privacy sanitizer for Romanized, code-mixed and dialectal Bangla (Banglish) prompts.
Rewrite the user's prompt into clear English and mask personal identifiers. Return ONLY one JSON object with exactly these keys:
- "sanitized_prompt": the prompt rewritten in clear English. Replace every PII value with a typed placeholder: <PHONE_1>, <NID_1>, <TXN_ID_1>, <ACCOUNT_1>, <CARD_1>, <OTP_1>, <EMAIL_1>, <NAME_1>, <ADDRESS_1>, or <ID_NUMBER_1> when the identifier type is unclear. Number placeholders per type (<PHONE_1>, <PHONE_2>, ...).
- "pii": list of {"type", "span", "placeholder"}. "type" is one of PHONE, NID, ID_NUMBER, TXN_ID, ACCOUNT, CARD, OTP, EMAIL, NAME, ADDRESS. "span" is the exact substring from the input.
- "preserved_entities": non-private facts the request needs, as {"type", "value"}, e.g. AMOUNT "500 BDT", SERVICE "bKash", DATE.
- "ambiguous_spans": input spans whose meaning is genuinely unclear, as {"span", "types", "candidates"}. "types" uses TRANSLITERATION, NUMERIC_AMBIGUITY, PII_BOUNDARY, DIALECT_MEANING, INTENT. "candidates" are the plausible readings.
Rules:
- A digit string in an identity or account context is PII. Use type ID_NUMBER if the exact type is unclear. It is NUMERIC_AMBIGUITY only when a currency or quantity marker is present and the magnitude is unclear.
- A regional dialect word that changes the meaning is DIALECT_MEANING. A standard word whose Romanized spelling maps to several words is TRANSLITERATION.
- Never copy a PII value into sanitized_prompt.
- Use empty lists when there is nothing to report. Output nothing except the JSON object."""

VARIANT_FOCUS = {
    "balanced": "",
    "dialect": "Pay special attention to regional dialect words (Chittagonian, Sylheti, Noakhali, Barishal and others) and what they mean in standard Bangla.",
    "finance_pii": "Pay special attention to mobile-wallet requests (bKash, Nagad, Rocket), amounts, transaction IDs, OTPs, and account, card, NID and phone numbers.",
    "transliteration": "Pay special attention to Romanized spellings that map to more than one Bangla word (for example kal = yesterday or tomorrow) and to digit-letter swaps such as 5oo.",
}

EXAMPLES = [
    ("bkash e 5oo tk send korsi trans id 9X87KQ2M, taka pay nai. amar number 01700000000",
     {"sanitized_prompt": "I sent 500 BDT via bKash (transaction ID <TXN_ID_1>), but the money was not received. My number is <PHONE_1>.",
      "pii": [{"type": "TXN_ID", "span": "9X87KQ2M", "placeholder": "<TXN_ID_1>"},
              {"type": "PHONE", "span": "01700000000", "placeholder": "<PHONE_1>"}],
      "preserved_entities": [{"type": "AMOUNT", "value": "500 BDT"}, {"type": "SERVICE", "value": "bKash"}],
      "ambiguous_spans": [{"span": "5oo tk", "types": ["NUMERIC_AMBIGUITY"], "candidates": ["500 BDT", "5000 BDT"]}]}),
    ("vai kal er meeting ta niye ekta email likhe dao, amar mail rahim.test@example.com, ar 1987000000 eta o add koro",
     {"sanitized_prompt": "Brother, please write an email about the meeting from yesterday/tomorrow. My email is <EMAIL_1>, and also add <ID_NUMBER_1>.",
      "pii": [{"type": "EMAIL", "span": "rahim.test@example.com", "placeholder": "<EMAIL_1>"},
              {"type": "ID_NUMBER", "span": "1987000000", "placeholder": "<ID_NUMBER_1>"}],
      "preserved_entities": [],
      "ambiguous_spans": [{"span": "kal er meeting", "types": ["TRANSLITERATION"], "candidates": ["yesterday's meeting", "tomorrow's meeting"]},
                          {"span": "1987000000", "types": ["PII_BOUNDARY"], "candidates": ["NID", "account number"]}]}),
    ("hetere koiyo amar NID 1990000000123 diye form ta fill kori dite, ar eta thik koro",
     {"sanitized_prompt": "Tell him/her to fill in the form using my NID <NID_1>, and fix this.",
      "pii": [{"type": "NID", "span": "1990000000123", "placeholder": "<NID_1>"}],
      "preserved_entities": [],
      "ambiguous_spans": [{"span": "hetere", "types": ["DIALECT_MEANING"], "candidates": ["him", "her"]},
                          {"span": "eta thik koro", "types": ["INTENT"], "candidates": ["fix the form", "fix this message"]}]}),
    ("amar laptop onek slow hoye geche, ki korbo?",
     {"sanitized_prompt": "My laptop has become very slow. What should I do?",
      "pii": [], "preserved_entities": [], "ambiguous_spans": []}),
]
VARIANT_ORDER = {"balanced": [0, 1, 2, 3], "dialect": [2, 3, 0, 1],
                 "finance_pii": [0, 3, 1, 2], "transliteration": [1, 0, 3, 2]}


def build_messages(variant, text):
    if variant not in VARIANT_FOCUS:
        raise ValueError(f"unknown variant {variant}; choose from {list(VARIANT_FOCUS)}")
    system = TASK + ("\n" + VARIANT_FOCUS[variant] if VARIANT_FOCUS[variant] else "")
    msgs = [{"role": "system", "content": system}]
    for i in VARIANT_ORDER[variant]:
        inp, out = EXAMPLES[i]
        msgs.append({"role": "user", "content": inp})
        msgs.append({"role": "assistant", "content": serialize_with_offsets(out)[0]})
    msgs.append({"role": "user", "content": text})
    return msgs
