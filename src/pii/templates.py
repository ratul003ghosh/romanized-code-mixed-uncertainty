"""Banglish templates with typed slots (proposal v2, section 3.2 step 3).

Slots in braces are filled by build.py: PII slots use the schema v0.2 type names,
{AMOUNT} and {SERVICE} become preserved entities.

These are hand-written starting points. Native speakers on the team should review
them, add dialect variants, and extend the list; record changes in docs/decisions.md.
"""

# Full finance prompts: send money, cash-out, recharge, failed transfer, verification, OTP.
FINANCE_TEMPLATES = [
    # send money / failed transfer
    "{SERVICE} e {AMOUNT} send korsi {PHONE} number e, kintu taka jay nai",
    "ami kal {PHONE} e {AMOUNT} {SERVICE} korsi, TrxID {TXN_ID}, ekhono pay nai",
    "bhai {SERVICE} theke {AMOUNT} pathaisi trans id {TXN_ID}, receiver bolche ashe nai",
    "{AMOUNT} send money korechi {PHONE} te, taka kete nise kintu deliver hoy nai",
    "bhul number e {AMOUNT} chole gese, number ta {PHONE}, ferot pabo kivabe?",
    "amar {SERVICE} account theke {AMOUNT} kete nise, TrxID {TXN_ID}, ami kisu kori nai",
    # cash-out
    "agent er kach theke {AMOUNT} cash out korsi, TrxID {TXN_ID}, balance update hoy nai",
    "{SERVICE} cash out charge koto? {AMOUNT} tulbo {PHONE} number theke",
    # recharge
    "{PHONE} number e {AMOUNT} recharge dilam kintu ashe nai, ki korbo",
    "mobile recharge {AMOUNT} korsi {SERVICE} diye, TrxID {TXN_ID}, balance ashe nai",
    # verification / identity (NID, account, card)
    "amar NID {NID}, {SERVICE} account verify hocche na keno?",
    "NID number {NID} diye account khulte chaisi, error dekhacche",
    "bank account {ACCOUNT} e {AMOUNT} deposit korsi, statement e dekhay na",
    "card {CARD} theke {AMOUNT} kete nise, ami kono payment kori nai",
    "amar card number {CARD}, eta block korte chai",
    "account no {ACCOUNT} er balance check korte parchi na",
    # OTP (scam / login)
    "ekjon call kore OTP chaise, OTP ta {OTP}, dewa ki thik hobe?",
    "{SERVICE} login korte OTP {OTP} ashche kintu kaj korche na",
    # names, e-mail, address
    "amar naam {NAME}, {PHONE} number er {SERVICE} account lock hoye gese",
    "{NAME} ke {AMOUNT} pathabo {PHONE} e, charge koto katbe?",
    "refund er jonno mail korsi {EMAIL} theke, kono reply nai",
    "parcel {ADDRESS} e deliver hobar kotha, {AMOUNT} COD dite hobe?",
    # bare identifier with no type cue (PII_BOUNDARY case -> ID_NUMBER)
    "{SERVICE} e {AMOUNT} send korsi, taka pay nai. {ID_NUMBER} eta check koren",
    "ei number ta {ID_NUMBER} diye kisu ber kora jabe?",
]

# Short PII clauses attached before or after carrier sentences from real corpora.
PII_CLAUSES = [
    "amar number {PHONE}",
    "call dio {PHONE} e",
    "NID {NID}",
    "amar NID number {NID}",
    "TrxID {TXN_ID}",
    "OTP {OTP}",
    "account {ACCOUNT}",
    "card {CARD}",
    "mail {EMAIL}",
    "ami {NAME}",
    "thikana {ADDRESS}",
    "{ID_NUMBER} eta dekhen",
]

SERVICES = ["bKash", "bkash", "Bkash", "bikash", "Nagad", "nagad", "Rocket", "rocket", "upay"]
AMOUNT_UNITS = ["tk", "taka", "tk.", "BDT", "Tk"]
AMOUNT_VALUES = [20, 50, 100, 150, 200, 300, 500, 700, 1000, 1200, 1500, 2000, 2500,
                 3000, 5000, 7000, 10000, 15000, 20000, 25000, 50000]
