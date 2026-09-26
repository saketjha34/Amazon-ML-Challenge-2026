"""
Text Normalization - Script-agnostic canonical forms shared by blocking and features.

Produces, for every record:
  name_t     : transliterated (any script -> ASCII), accent-folded, lowercased name
  name_core  : name_t with legal forms / honorifics removed
  name_key   : sorted unique tokens of name_core (word-order invariant)
  name_skel  : phonetic consonant skeleton of name_core (bridges transliteration noise)
  addr_t     : transliterated, abbreviation- and state-canonicalized address
  house_c    : house number with leading zeros stripped
  is_native  : 1 if the raw name contains non-Latin script characters

Country is treated as an open label set: nothing here filters on specific countries.
"""

import os
import re
import unicodedata
from multiprocessing import Pool
from pathlib import Path
from typing import List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from anyascii import anyascii

# Bump when normalization changes: invalidates the normalized parquet and TF-IDF caches.
NORM_VERSION = 2

NORM_COLUMNS = [
    "entity_id", "country_normalized", "state_code", "city",
    "name_t", "name_core", "name_key", "name_skel", "addr_t", "house_c", "is_native",
]

# Legal forms / honorifics across US, India (incl. transliterated native-script variants) and France.
LEGAL_TOKENS = {
    # US / generic
    "llc", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
    "pllc", "pc", "llp", "lp", "plc", "the", "dba",
    # India
    "pvt", "private", "opc", "praivet", "praibhet", "piraivet", "praivrr", "prayvet", "privet",
    "limitet", "limirrd", "limited", "elelpi", "pra", "li", "m", "s", "ms", "sri", "shri", "smt",
    # France
    "sarl", "sas", "sasu", "sa", "sci", "eurl", "snc", "scp", "selarl", "cie", "ets", "ei",
    # Honorifics and web noise ("www example com")
    "mr", "mrs", "dr", "prof", "www", "com", "net", "org", "http", "https",
}
# Digits used as letters inside words: "m0dern", "mc8ride", "c1ub"
_LEET = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"}
_LEET_RE = re.compile(r"(?<=[a-z])[0134578](?=[a-z])")
# "id 17650" suffixes and long digit runs (phone numbers / registration ids) inside names
_NAME_ID_RE = re.compile(r"\bid \d+\b|\b\d{5,}\b")
_LEGAL_RE = re.compile(r"^(p(i)?ra[iy]?(v|bh|w)[ae]?(t+|rr?)|limi(t|r)+[ae]?(d+|t|rr?)?)$")

# Multi-token legal sequences collapsed before tokenization.
_MULTI_LEGAL = [
    (re.compile(r"\bl l c\b"), " llc "),
    (re.compile(r"\bp c\b"), " pc "),
    (re.compile(r"\bl l p\b"), " llp "),
    (re.compile(r"\bm s\b"), " ms "),
    (re.compile(r"\bet cie\b"), " cie "),
    (re.compile(r"l l p"), " llp "),
]

ADDR_ABBREV = {
    "road": "rd", "street": "st", "avenue": "ave", "av": "ave", "drive": "dr", "lane": "ln",
    "boulevard": "blvd", "bd": "blvd", "court": "ct", "place": "pl", "highway": "hwy",
    "suite": "ste", "apartment": "apt", "floor": "fl", "building": "bldg", "bld": "bldg",
    "north": "n", "south": "s", "east": "e", "west": "w", "circle": "cir", "parkway": "pkwy",
    "terrace": "ter", "square": "sq", "number": "no", "nagar": "ngr", "marg": "mg",
    "r": "rue", "chem": "chemin", "rte": "route", "imp": "impasse", "all": "allee",
    "sector": "sec", "near": "nr", "opposite": "opp", "opp": "opp", "house": "h", "hno": "h no",
    "null": "", "nan": "", "none": "",
}

STATE_NAMES = {
    # US
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd",
    "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
    # India
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "telangana": "ts", "tripura": "tr", "uttar pradesh": "up",
    "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl", "jammu and kashmir": "jk",
    "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
    # common transliterated / misspelled forms seen in native-script addresses
    "dilli": "dl", "new delhi": "dl", "maharastra": "mh", "mharastr": "mh", "tamilnadu": "tn",
    "tmilnatu": "tn", "karnatak": "ka", "gujrat": "gj", "hriyana": "hr", "pashchim bangal": "wb",
}
# Longest names first so "west virginia" wins over "virginia".
_STATE_RE = re.compile(
    r"\b(" + "|".join(sorted((re.escape(k) for k in STATE_NAMES), key=len, reverse=True)) + r")\b"
)

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")
_NUM_LEAD0 = re.compile(r"\b0+(\d)")
_DIGITS = re.compile(r"\d+")


def fold(s: str) -> str:
    """NFKD accent folding + lowercase (Latin scripts only; other scripts untouched)."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()


def has_non_latin(s: str) -> bool:
    for c in s:
        if ord(c) > 0x24F and c.isalpha():
            return True
    return False


def to_ascii(s: str) -> str:
    """Transliterate any script to ASCII, lowercase, keep only [a-z0-9 ]."""
    if not s:
        return ""
    s = anyascii(fold(s)).lower()
    s = _NON_ALNUM.sub(" ", s)
    return _SPACES.sub(" ", s).strip()


def _prep_name(raw: str) -> str:
    s = anyascii(fold(raw)).lower().replace("&", " and ") if raw else ""
    s = _NON_ALNUM.sub(" ", s)
    s = _LEET_RE.sub(lambda m: _LEET[m.group(0)], s)
    s = _NAME_ID_RE.sub(" ", s)
    s = _SPACES.sub(" ", s).strip()
    for rx, rep in _MULTI_LEGAL:
        s = rx.sub(rep, s)
    return _SPACES.sub(" ", s).strip()


def core_name(name_t: str) -> str:
    toks = [t for t in name_t.split() if t not in LEGAL_TOKENS and not _LEGAL_RE.match(t)]
    if not toks:  # name was only legal words: keep it as-is
        return name_t
    return " ".join(toks)


_SKEL_SUBS = [
    ("ph", "f"), ("bh", "b"), ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"),
    ("ch", "c"), ("sh", "s"), ("ck", "k"), ("q", "k"), ("x", "ks"), ("z", "j"),
    ("w", "v"), ("n", "m"),
    # voicing merges (Tamil/Malayalam transliteration has no voiced/unvoiced distinction)
    ("b", "p"), ("g", "k"), ("d", "t"), ("c", "s"),
]


def skeleton(core: str) -> str:
    """Phonetic consonant skeleton per token: bridges 'marketing'/'marketimg', 'solutions'/'solyusms'."""
    out = []
    for tok in core.split():
        if tok.isdigit():
            out.append(tok)
            continue
        t = tok
        for a, b in _SKEL_SUBS:
            t = t.replace(a, b)
        head, tail = t[0], re.sub(r"[aeiouy]", "", t[1:])
        t = re.sub(r"(.)\1+", r"\1", head + tail)
        if t:
            out.append(t)
    return " ".join(out)


def canon_addr(raw: str) -> str:
    s = to_ascii(raw)
    if not s:
        return ""
    s = _STATE_RE.sub(lambda m: STATE_NAMES[m.group(1)], s)
    toks = []
    for t in s.split():
        t = ADDR_ABBREV.get(t, t)
        if t:
            toks.append(t)
    s = " ".join(toks)
    return _NUM_LEAD0.sub(r"\1", s)


def canon_house(h: str) -> str:
    if not h:
        return ""
    h = h.lower().strip()
    return _NUM_LEAD0.sub(r"\1", h)


def numbers_of(addr_t: str) -> frozenset:
    return frozenset(_DIGITS.findall(addr_t))


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    names = df["business_name"].fillna("").astype(str).tolist()
    addrs = df["business_address"].fillna("").astype(str).tolist()
    houses = df["house_number"].fillna("").astype(str).tolist()

    name_t = [_prep_name(n) for n in names]
    name_core = [core_name(n) for n in name_t]
    name_key = [" ".join(sorted(set(c.split()))) for c in name_core]
    name_skel = [skeleton(c) for c in name_core]
    addr_t = [canon_addr(a) for a in addrs]
    house_c = [canon_house(h) for h in houses]
    is_native = [1 if has_non_latin(n) else 0 for n in names]

    return pd.DataFrame({
        "entity_id": df["entity_id"].values,
        "country_normalized": df["country_normalized"].fillna("").astype(str).values,
        "state_code": df["state_code"].fillna("").astype(str).str.upper().values,
        "city": df["city"].fillna("").astype(str).str.lower().values,
        "name_t": name_t,
        "name_core": name_core,
        "name_key": name_key,
        "name_skel": name_skel,
        "addr_t": addr_t,
        "house_c": house_c,
        "is_native": pd.array(is_native, dtype="int8"),
    })


_SRC_COLS = ["entity_id", "business_name", "business_address", "house_number",
             "country_normalized", "state_code", "city"]


def norm_path_for(src_path: Path) -> Path:
    src_path = Path(src_path)
    return src_path.parent / "norm" / src_path.name.replace("_final.parquet", f"_norm_v{NORM_VERSION}.parquet")


def ensure_norm(src_path: Path, n_procs: int = None, batch_rows: int = 200_000) -> Path:
    """Build (once) and return the cached normalized parquet for a source file."""
    dst = norm_path_for(src_path)
    if dst.is_file():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    from src.resources import cpu_threads
    n_procs = n_procs or cpu_threads()
    print(f"  [Norm] Building normalized cache {dst.name} ({n_procs} procs)...")

    pf = pq.ParquetFile(str(src_path))
    writer = None
    batch_iter = pf.iter_batches(batch_size=batch_rows, columns=_SRC_COLS)
    with Pool(n_procs) as pool:
        while True:
            # Bounded window: at most n_procs batches in memory at once
            window = []
            for b in batch_iter:
                window.append(b.to_pandas())
                if len(window) >= n_procs:
                    break
            if not window:
                break
            for out in pool.map(normalize_frame, window):
                table = pa.Table.from_pandas(out, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(str(tmp), table.schema)
                writer.write_table(table)
            del window
    if writer is not None:
        writer.close()
    os.replace(tmp, dst)
    print(f"  [Norm] Saved {dst}")
    return dst
