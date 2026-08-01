#!/usr/bin/env python3
"""Deterministic synthetic English-like text corpus generator.

Both the weight trainer and the judge's prompt sampler need text with no
network dependency, and they must agree exactly. This script generates it from
a fixed word bank + sentence templates using Python's stdlib RNG with a pinned
seed, so the corpus is byte-reproducible everywhere: same seed -> same SHA256.
The judge imports the generator as a library at a nonce-derived seed, which is
why this module's hash is pinned in MANIFEST.json alongside the model assets.

Output: corpus/train.txt (~4.0 MB), corpus/val.txt (~0.26 MB), corpus/META.json
"""

import hashlib
import json
import os
import random

SEED = 42
TRAIN_CHARS = 4_000_000
VAL_CHARS = 260_000

NOUNS = """model gradient loss batch epoch layer neuron tensor matrix vector dataset corpus token
experiment result paper idea method system machine network signal noise pattern feature label
teacher student engineer researcher scientist writer farmer doctor pilot artist musician judge
mountain river forest valley ocean island desert meadow garden bridge tower castle village city
morning evening winter summer autumn spring shadow sunlight thunder rainfall breeze horizon
table chair window door kitchen library school station market street corner building room floor
book letter journal story poem chapter page sentence word language question answer argument
coffee bread apple honey butter garlic pepper dinner breakfast supper basket bottle kettle
dog cat horse sparrow falcon rabbit tortoise salmon spider beetle donkey goat lantern candle
clock mirror ladder hammer needle thread blanket pillow curtain carpet engine wheel compass map
friend neighbor stranger crowd family child parent brother sister cousin captain sailor guard
storm cloud puddle pebble boulder cliff stream harbor lighthouse meadowlark orchard vineyard""".split()

VERBS_T = """observed measured trained evaluated improved reduced increased computed predicted
described explained examined compared repeated recorded reported checked tested debugged fixed
carried lifted opened closed painted cleaned washed folded planted watered gathered collected
borrowed returned offered promised showed taught told asked answered followed guided welcomed
finished started continued stopped remembered forgot noticed ignored trusted doubted admired""".split()

VERBS_I = """converged diverged improved failed succeeded vanished appeared slept walked ran
wandered rested waited listened smiled laughed frowned nodded shrugged hesitated stumbled
arrived departed returned remained stayed traveled drifted floated settled paused blinked""".split()

ADJS = """small large tiny enormous quiet loud bright dark gentle fierce careful careless quick
slow steady shaky warm cold damp dry smooth rough simple complex clever foolish patient restless
ancient modern narrow wide shallow deep heavy light empty full sturdy fragile curious cautious
honest stubborn cheerful gloomy tidy messy distant nearby familiar strange ordinary peculiar
reliable noisy silent golden silver crimson pale vivid faded hollow solid brittle tender crisp""".split()

ADVS = """quickly slowly carefully carelessly quietly loudly gently firmly barely nearly almost
often rarely sometimes usually finally suddenly eventually gradually steadily reluctantly
eagerly calmly nervously patiently precisely roughly smoothly awkwardly deliberately happily""".split()

NAMES = """Alice Omar Priya Chen Miguel Sofia Ravi Hana Tariq Lena Kofi Ingrid Mateo Yuki Amara
Dmitri Farah Liam Noor Pablo Greta Anders Bina Carlos Devi Emil Freya Gustav Hiro Iris Jonas""".split()

PLACES = """Aldergrove Brimwick Cedarholm Dunmore Eastvale Fernbridge Grimsby Hollowell Ivorydale
Juniper Kestrelton Larkspur Millbrook Northgate Oakhurst Pinecrest Quarryville Rosewood""".split()

PREPS = "near beside beyond across behind under over along toward past within around".split()
CONJS = ["because", "although", "while", "when", "after", "before", "unless", "since"]


def make_sentence(rng: random.Random) -> str:
    def noun_phrase():
        det = rng.choice(["the", "a", "the", "the", "one", "every", "each", "that"])
        if rng.random() < 0.12:
            return rng.choice(NAMES)
        adj = (rng.choice(ADJS) + " ") if rng.random() < 0.55 else ""
        adj2 = (rng.choice(ADJS) + " ") if rng.random() < 0.10 else ""
        return f"{det} {adj2}{adj}{rng.choice(NOUNS)}"

    def clause():
        subj = noun_phrase()
        if rng.random() < 0.45:
            verb, obj = rng.choice(VERBS_T), " " + noun_phrase()
        else:
            verb, obj = rng.choice(VERBS_I), ""
        adv = (" " + rng.choice(ADVS)) if rng.random() < 0.35 else ""
        pp = (f" {rng.choice(PREPS)} {noun_phrase()}") if rng.random() < 0.30 else ""
        return f"{subj} {verb}{adv}{obj}{pp}"

    r = rng.random()
    if r < 0.08:  # numeric / ML-flavored sentence
        s = rng.choice([
            f"experiment {rng.randint(1, 999)} converged after {rng.randint(2, 90) * 100} steps with loss {rng.uniform(0.4, 4.0):.2f}",
            f"the {rng.choice(NOUNS)} required {rng.randint(2, 480)} minutes and {rng.randint(1, 64)} retries",
            f"run {rng.randint(1, 99)} reached {rng.uniform(50.0, 99.9):.1f} percent accuracy on {rng.randint(3, 400) * 1000} examples",
            f"{rng.choice(NAMES)} counted {rng.randint(12, 9000)} tokens in chapter {rng.randint(1, 40)}",
        ])
    elif r < 0.16:  # subordinate clause first
        s = f"{rng.choice(CONJS)} {clause()}, {clause()}"
    elif r < 0.28:  # coordination
        s = f"{clause()}, {rng.choice(['and', 'but', 'so', 'yet'])} {clause()}"
    elif r < 0.33:  # quoted speech
        s = f'"{clause()}," said {rng.choice(NAMES)} {rng.choice(ADVS)}'
    elif r < 0.38:  # place-anchored
        s = f"in {rng.choice(PLACES)}, {clause()}"
    else:
        s = clause()

    end = "?" if rng.random() < 0.05 else ("!" if rng.random() < 0.03 else ".")
    return s[0].upper() + s[1:] + end


def make_paragraph(rng: random.Random) -> str:
    return " ".join(make_sentence(rng) for _ in range(rng.randint(3, 8)))


def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(SEED)

    val_parts, train_parts = [], []
    val_n = train_n = 0
    while val_n < VAL_CHARS:
        p = make_paragraph(rng)
        val_parts.append(p)
        val_n += len(p) + 2
    while train_n < TRAIN_CHARS:
        p = make_paragraph(rng)
        train_parts.append(p)
        train_n += len(p) + 2

    meta = {"seed": SEED, "generator": "gen_corpus.py"}
    for name, parts in [("train", train_parts), ("val", val_parts)]:
        text = "\n\n".join(parts) + "\n"
        path = os.path.join(out_dir, f"{name}.txt")
        with open(path, "w", encoding="ascii") as f:
            f.write(text)
        meta[name] = {
            "chars": len(text),
            "sha256": hashlib.sha256(text.encode("ascii")).hexdigest(),
            "vocab_size": len(set(text)),
        }
        print(f"{name}: {len(text):,} chars  vocab={len(set(text))}  sha256={meta[name]['sha256'][:16]}...")

    with open(os.path.join(out_dir, "META.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
