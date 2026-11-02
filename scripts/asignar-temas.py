#!/usr/bin/env python3
"""Assign final-project topics to groups from their ranked preferences.

Usage:
  python3 scripts/asignar-temas.py preferencias.csv [--semilla N]

The CSV has a header row and one row per group. The first column is the group
(names), and the next three are the 1st, 2nd and 3rd preference. Any extra
leading columns, such as the timestamp of a Google Forms export, are skipped:
the group is the first column whose header contains "grupo" (or "integrantes"),
and the preferences are the next three columns. A preference cell may be a
topic number or text that starts with it ("7. Ray tracing"), so a form can use
the topic list as dropdown options.

Each topic goes to at most one group. The assignment minimises the total cost
(1st choice 0, 2nd 1, 3rd 4, a topic not listed 100), so nobody falls to an
unlisted topic if it can be avoided, and the fewest groups get their 3rd. When
several assignments tie, one is drawn at random with the seed, which is printed
so the draw can be repeated (e.g. pick the seed in class). Do not commit the
CSV: it holds student names.
"""
import argparse
import csv
import random
import re
import sys

TOPICS = {
    1: "Ecuación del calor 2D",
    2: "Ecuación de onda 2D",
    3: "Reacción-difusión de Gray-Scott",
    4: "Ecuación de Poisson 2D",
    5: "Modelo de Ising 2D",
    6: "Ecuación de Schrödinger 2D",
    7: "Ray tracing",
    8: "Fluidos con Lattice Boltzmann",
    9: "Dinámica molecular de Lennard-Jones",
}
RANK_COST = [0, 1, 4]
UNLISTED_COST = 100


def topic_number(cell, where):
    m = re.match(r"\s*(\d+)", cell)
    if not m or int(m.group(1)) not in TOPICS:
        sys.exit(f"{where}: not a topic number: {cell!r}")
    return int(m.group(1))


def read_groups(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f) if any(c.strip() for c in r)]
    header, rows = rows[0], rows[1:]
    col = next((i for i, h in enumerate(header)
                if "grupo" in h.lower() or "integrantes" in h.lower()), 0)
    groups = []
    for n, row in enumerate(rows, start=2):
        prefs = [topic_number(c, f"{path}:{n}") for c in row[col + 1:col + 4]]
        if len(prefs) != 3 or len(set(prefs)) != 3:
            sys.exit(f"{path}:{n}: need three different preferences: {row}")
        groups.append((row[col].strip(), prefs))
    return groups


def best_assignments(groups):
    """All assignments of minimum cost, by depth-first search with pruning."""
    costs = [{t: (RANK_COST[p.index(t)] if t in p else UNLISTED_COST)
              for t in TOPICS} for _, p in groups]
    cheapest = [min(c.values()) for c in costs]
    best, found = float("inf"), []

    def search(g, used, total, chosen):
        nonlocal best, found
        if total + sum(cheapest[g:]) > best:
            return
        if g == len(groups):
            if total < best:
                best, found = total, []
            found.append(list(chosen))
            return
        for t in sorted(TOPICS, key=costs[g].get):
            if t not in used:
                used.add(t)
                chosen.append(t)
                search(g + 1, used, total + costs[g][t], chosen)
                chosen.pop()
                used.discard(t)

    search(0, set(), 0, [])
    return best, found


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv")
    ap.add_argument("--semilla", type=int,
                    help="seed for breaking ties (random if omitted)")
    args = ap.parse_args()

    groups = read_groups(args.csv)
    if len(groups) > len(TOPICS):
        sys.exit(f"{len(groups)} groups but only {len(TOPICS)} topics")
    seed = args.semilla if args.semilla is not None else random.randrange(1000)
    cost, options = best_assignments(groups)
    choice = random.Random(seed).choice(options)

    width = max(len(name) for name, _ in groups)
    for (name, prefs), t in zip(groups, choice):
        rank = f"{prefs.index(t) + 1}ª preferencia" if t in prefs else "SIN PREFERENCIA"
        print(f"{name:<{width}}  {t}. {TOPICS[t]}  ({rank})")
    free = sorted(set(TOPICS) - set(choice))
    print(f"\nTemas libres: {', '.join(map(str, free)) or 'ninguno'}")
    print(f"Costo {cost}; {len(options)} asignación(es) empatada(s); semilla {seed}")


if __name__ == "__main__":
    main()
