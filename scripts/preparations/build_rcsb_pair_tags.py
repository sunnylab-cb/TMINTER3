"""Observed-residue protein-pair tags from mmCIF biological assemblies.

No raw writes, no UniProt completion, no asymmetric-unit fallback.
Outputs are atomic per-entry gzip JSON; failures remain explicit in shard logs.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import gemmi
import numpy as np
from scipy.spatial import cKDTree

CONTRACT = "observed-polymer-heavy-atom-5p5-assembly-v2"


def prepare_sources(structure, block):
    asym_entity = dict(zip(block.find_values("_struct_asym.id"), block.find_values("_struct_asym.entity_id")))
    protein_entities = {e for e, t in zip(block.find_values("_entity_poly.entity_id"), block.find_values("_entity_poly.type"))
                        if "polypeptide" in t.lower()}
    groups = {}
    for chain in structure[0]:
        for residue in chain:
            if asym_entity.get(residue.subchain) not in protein_entities:
                continue
            if residue.label_seq is None:
                raise ValueError("protein residue lacks label_seq: {} {}".format(residue.subchain, residue.seqid))
            groups.setdefault(residue.subchain, []).append((chain.name, residue))
    sources = {}
    for asym, residues in sorted(groups.items()):
        # Do not silently pick between microheterogeneous residue identities.
        numbers = [r.label_seq for _, r in residues]
        if len(numbers) != len(set(numbers)):
            raise ValueError("microheterogeneous duplicate label_seq in " + asym)
        residues.sort(key=lambda x: x[1].label_seq)
        sequence, mapping, coords, atom_rows = [], [], [], []
        for auth, residue in residues:
            occupancy = {}
            for atom in residue:
                if atom.element.name in ("H", "D", "X") or atom.occ <= 0:
                    continue
                if atom.altloc not in ("\x00", " "):
                    occupancy[atom.altloc] = occupancy.get(atom.altloc, 0.) + atom.occ
            conformer = min(occupancy, key=lambda k: (-occupancy[k], k)) if occupancy else None
            selected = {}
            for atom in residue:
                if atom.element.name in ("H", "D", "X") or atom.occ <= 0:
                    continue
                if atom.altloc not in ("\x00", " ", conformer):
                    continue
                xyz = [atom.pos.x, atom.pos.y, atom.pos.z]
                if not np.isfinite(xyz).all():
                    continue
                if atom.name not in selected or atom.occ > selected[atom.name][0]:
                    selected[atom.name] = (atom.occ, xyz)
            if not selected:
                continue
            code = gemmi.find_tabulated_residue(residue.name).one_letter_code.upper()
            if code not in "ACDEFGHIKLMNPQRSTVWYBXZ":
                code = "X"
            index = len(sequence)
            sequence.append(code)
            mapping.append(dict(index=index, label_seq_id=residue.label_seq,
                                auth_seq_id=residue.seqid.num, insertion_code=residue.seqid.icode.strip(),
                                resname=residue.name, auth_asym_id=auth, altloc=conformer))
            coords.extend(v[1] for v in selected.values())
            atom_rows.extend([index] * len(selected))
        if not sequence:
            continue
        sequence = "".join(sequence)
        sources[asym] = dict(sequence=sequence, seq_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
                             residue_map=mapping, entity_id=asym_entity[asym],
                             coords=np.asarray(coords), atom_rows=np.asarray(atom_rows),
                             label_seq_gaps=[[a["label_seq_id"], b["label_seq_id"]] for a, b in zip(mapping, mapping[1:])
                                             if b["label_seq_id"] != a["label_seq_id"] + 1])
    return sources


def process(task):
    raw, relative, root, max_atoms = task
    pdb = Path(relative).name.removesuffix(".gz").removesuffix(".cif")
    output = Path(root) / "entries" / pdb[1:3] / (pdb + ".json.gz")
    try:
        raw = Path(raw).resolve()
        path = (raw / relative).resolve()
        if raw not in path.parents:
            raise ValueError("Inventory path escapes raw-root")
        resolved = output.resolve()
        if raw == resolved or raw in resolved.parents or Path(root).resolve() not in resolved.parents:
            raise ValueError("Output path escapes destination or points into raw")
        if output.exists():
            with gzip.open(output, "rt") as handle:
                old = json.load(handle)
            if (old.get("contract"), old.get("source_path"), old.get("source_bytes"), old.get("source_mtime_ns")) != (CONTRACT, str(path), path.stat().st_size, path.stat().st_mtime_ns):
                raise ValueError("Incompatible/stale existing output " + str(output))
            return dict(pdb_id=pdb, status="reused", pairs=len(old["pairs"]))
        block = gemmi.cif.read(str(path)).sole_block()
        structure = gemmi.make_structure_from_block(block)
        if not len(structure):
            raise ValueError("no coordinate model")
        sources = prepare_sources(structure, block)
        chains, pairs = [], []
        for assembly in structure.assemblies:
            instances, seen = [], set()
            for gi, generator in enumerate(assembly.generators):
                for oi, operator in enumerate(generator.operators):
                    matrix = np.array(operator.transform.mat.tolist())
                    vector = np.array(operator.transform.vec.tolist())
                    for asym, source in sources.items():
                        auth = source["residue_map"][0]["auth_asym_id"]
                        if asym not in generator.subchains and auth not in generator.chains:
                            continue
                        key = (asym, tuple(np.round(matrix.ravel(), 10)), tuple(np.round(vector, 10)))
                        if key in seen:
                            continue
                        seen.add(key)
                        identifier = "{}:asm{}:{}:g{}o{}".format(pdb, assembly.name, asym, gi, oi)
                        xyz = source["coords"] @ matrix.T + vector
                        instances.append((identifier, asym, xyz, source["atom_rows"]))
                        chains.append(dict(chain_instance_id=identifier, assembly_id=assembly.name,
                                           source_label_asym_id=asym, operator_name=operator.name,
                                           operator_matrix=matrix.tolist(), operator_vector=vector.tolist()))
            total_atoms = sum(len(x[2]) for x in instances)
            if max_atoms and total_atoms > max_atoms:
                raise ValueError("assembly atom resource guard: {} > {}; not silently skipped".format(total_atoms, max_atoms))
            for left, right in combinations(instances, 2):
                a, b = left[2], right[2]
                delta = np.maximum(0, np.maximum(a.min(0)-b.max(0), b.min(0)-a.max(0)))
                if np.linalg.norm(delta) > 5.5:
                    continue
                neighbors = cKDTree(a).query_ball_tree(cKDTree(b), 5.5)
                contacts = set()
                for ia, near in enumerate(neighbors):
                    contacts.update((int(left[3][ia]), int(right[3][ib])) for ib in near)
                if contacts:
                    pairs.append(dict(left=left[0], right=right[0],
                                      left_positive=sorted({i for i, _ in contacts}),
                                      right_positive=sorted({j for _, j in contacts}),
                                      residue_contacts=sorted(contacts)))
        payload = dict(contract=CONTRACT, pdb_id=pdb, source_path=str(path),
                       source_bytes=path.stat().st_size, source_mtime_ns=path.stat().st_mtime_ns,
                       coordinate_models=len(structure), model_policy="first_model",
                       experimental_methods=list(block.find_values("_exptl.method")),
                       assembly_count=len(structure.assemblies), chains=chains, pairs=pairs,
                       sources={k: {a: b for a, b in v.items() if a not in ("coords", "atom_rows")} for k, v in sources.items()})
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_name(output.name + ".{}.tmp".format(os.getpid()))
        with gzip.open(temp, "xt") as handle:
            json.dump(payload, handle)
        os.replace(temp, output)
        return dict(pdb_id=pdb, status="ok", sources=len(sources), assemblies=len(structure.assemblies), pairs=len(pairs))
    except Exception as error:
        return dict(pdb_id=pdb, status="error", error=repr(error))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", required=True, type=Path)
    p.add_argument("--raw-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--limit", type=int)
    p.add_argument("--max-assembly-atoms", type=int, default=300000)
    args = p.parse_args()
    assert 0 <= args.shard < args.shards
    raw, output = args.raw_root.resolve(), args.output.resolve()
    if raw == output or raw in output.parents:
        raise ValueError("Output inside raw is prohibited")
    with args.inventory.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    rows = rows[args.shard::args.shards]
    if args.limit:
        rows = rows[:args.limit]
    output.mkdir(parents=True, exist_ok=True)
    tasks = [(str(raw), r["relative_path"], str(output), args.max_assembly_atoms) for r in rows]
    status = output / "status_{}.jsonl".format(args.shard)
    if status.is_symlink() or (status.exists() and status.stat().st_nlink != 1):
        raise ValueError("Status must not be a symlink or hardlink")
    with status.open("a") as handle, ProcessPoolExecutor(args.workers) as pool:
        for result in pool.map(process, tasks, chunksize=1):
            line = json.dumps(result)
            handle.write(line + "\n")
            handle.flush()
            print(line, flush=True)


if __name__ == "__main__":
    main()
