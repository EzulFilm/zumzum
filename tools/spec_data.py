"""Test recordings for `fix --spec` : 10 per rule set, public domain / CC0 / CC BY / CC BY-SA, from
Wikimedia Commons. Licence, author and page come from each file's Commons metadata; SHA-256 pinned at first download.

    python tools/spec_data.py --make     # resolve the titles below on Commons and write examples/spec_sources.json
    python tools/spec_data.py            # download what is missing into cache/spec_audio/, checking each hash
"""

import hashlib
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "examples" / "spec_sources.json"
CACHE = ROOT / "cache" / "spec_audio"
UA = {"User-Agent": "Zumzum/0.1 (delivery-rule test data fetch; mail@ezul.ai)"}
MAX_BYTES = 40 * 1024 * 1024
OK_LICENCES = ("Public domain", "CC0", "CC BY ", "CC BY-SA ", "PDM")

TITLES = {
    "podcast": ["JFK inaugural address.ogg", "Harry Truman Announcing Surrender Of Germany.ogg",
                "Gerald Ford's comments at his Swearing in Ceremony to be 38th President of the United States.ogg",
                "Booker T. Washington, Speech, 1895.ogg", "Marcus Garvey, speech, 1921.ogg",
                "An address by Opposition Leader Anthony Albanese.ogg",
                "Isar Ul Qasmi's Speech.ogg", "Speech by Siarhiej Dubaviec.ogg",
                "Nixon resignation audio with buzz removed.ogg", "A J Cook Speech from Lansbury's Labour Weekly.ogg"],
    "audiobook": ["6ann 08 samuelpepys bg 64kb.ogg", "Bookofmarcopolo1 chap22 pisa 64kb.ogg",
                  "Brüder Grimm - Rotkäppchen.ogg", "Journallewisandclarke 06 lewis-clark 64kb.ogg",
                  "La Papallona capítol 09 àudio.ogg", "La punyalada capítol 12 àudio.oga",
                  "Farewelladdresses U.S President John Adams64kb.ogg",
                  "El Donador de Almas de Amado Nervo, De los Autores y sus Obras.ogg",
                  "6ann 60 treatysixnations law 64kb.ogg", "Interestingnarrative 02 equiano 64kb.ogg"],
    "broadcast": ["Antonin Dvorak - carnival overture, op. 92.ogg",
                  "Alexander Borodin - In The Steppes Of Central Asia.ogg",
                  "A Midsummer Night's Dream Op. 61 Wedding March (Mendelssohn) European Archive.ogg",
                  "Ballade no. 1 - Op. 23.mp3", "Bach, Johann Sebastian - Suite No.2 in B Minor - X. Badinerie.ogg",
                  "Harry Truman Announcing Surrender Of Japan.ogg",
                  "John F Kennedy Address on the Buildup of Arms in Cuba.ogg",
                  "Emil Hácha - nové státoprávní uspořádání 1939.ogg", "Ronald Reagan First Inaugural.ogg",
                  "Bryan - The Ideal Republic.ogg"],
}


def _get(url, limit=None):
    import time
    for wait in (0, 30, 90, 240):  # Commons answers 429 to bursts: back off and retry
        time.sleep(wait)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                body = r.read((limit or 8 * 1024 * 1024) + 1)
            break
        except urllib.error.HTTPError as e:
            if e.code != 429 or wait == 240:
                raise
    if limit and len(body) > limit:
        raise ValueError(f"{url}: over {limit} bytes")
    return body


def resolve_all(titles):
    q = urllib.parse.urlencode({"action": "query", "titles": "|".join("File:" + t for t in titles),
                                "prop": "imageinfo", "format": "json", "iiprop": "url|size|extmetadata"})
    d = json.loads(_get("https://commons.wikimedia.org/w/api.php?" + q))["query"]
    names = {n["to"]: n["from"] for n in d.get("normalized", [])}
    pages = {names.get(pg["title"], pg["title"])[5:]: pg for pg in d["pages"].values()}
    return {t: resolve(t, pages.get(t, {})) for t in titles}


def resolve(title, pg):
    if "imageinfo" not in pg:
        raise KeyError(f"not on Commons: {title}")
    ii, em = pg["imageinfo"][0], pg["imageinfo"][0]["extmetadata"]
    lic = em.get("LicenseShortName", {}).get("value", "")
    if not lic.startswith(OK_LICENCES):
        raise ValueError(f"{title}: licence {lic!r} is not public domain / CC0 / CC BY / CC BY-SA")
    return {"title": title, "page": ii["descriptionurl"], "url": ii["url"].split("?")[0], "bytes": ii["size"],
            "licence": lic, "author": re.sub(r"<[^>]+>|\s+", " ", em.get("Artist", {}).get("value", "")).strip()[:160]}


def key(title):
    return re.sub(r"[^a-z0-9]+", "_", title.rsplit(".", 1)[0].lower()).strip("_")[:48]


def make():
    out = {"about": "Test recordings for fix --spec, 10 per rule set; public domain / CC0 / CC BY / CC BY-SA from "
                    "Wikimedia Commons (licence and author from each file's Commons metadata, 2026-11-10). Audio is "
                    "never committed: tools/spec_data.py downloads it into cache/spec_audio and checks SHA-256.",
           "sets": {}}
    for rule, titles in TITLES.items():
        out["sets"][rule] = {key(t): e for t, e in resolve_all(titles).items()}
    MANIFEST.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def path_of(k, e):
    return CACHE / (k + Path(e["url"]).suffix.lower())


def fetch():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    CACHE.mkdir(parents=True, exist_ok=True)
    changed, files = False, {}
    for rule, entries in data["sets"].items():
        for k, e in entries.items():
            p = path_of(k, e)
            if not p.exists():
                import time
                time.sleep(8)  # one file at a time, spaced (Wikimedia's rate limit)
                p.write_bytes(_get(e["url"], MAX_BYTES))
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            if e.get("sha256") is None:
                e["sha256"], changed = h, True
            elif e["sha256"] != h:
                raise ValueError(f"{p}: SHA-256 {h} is not the pinned {e['sha256']}")
            files.setdefault(rule, {})[k] = p
    if changed:
        MANIFEST.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return files, data


if __name__ == "__main__":
    if "--make" in sys.argv:
        make()
    files, _ = fetch()
    for rule, fs in files.items():
        print(rule, len(fs), "files")
