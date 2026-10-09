"""Small contract tests without production data or GPUs."""
import importlib.util
import hashlib
import sys
from pathlib import Path
import gemmi
import numpy as np

root = Path(__file__).resolve().parents[1] / "scripts/preparations"
sys.path.insert(0, str(root))
from build_rcsb_pair_tags import prepare_sources
import rcsb_pairwise_si as si

structure = gemmi.Structure()
model = gemmi.Model("1")
chain = gemmi.Chain("A")
for label, name, x in [(1, "ALA", 0.), (3, "GLY", 10.)]:
    residue = gemmi.Residue()
    residue.name, residue.subchain, residue.label_seq = name, "X1", label
    residue.seqid = gemmi.SeqId(label + 100, " ")
    for atom_name, element, dx, altloc, occupancy in [("CA", "C", 0., "\x00", 1.), ("CB", "C", 1., "A", .8), ("CB", "C", 99., "B", .2), ("H", "H", 2., "\x00", 1.)]:
        atom = gemmi.Atom()
        atom.name, atom.element, atom.pos = atom_name, gemmi.Element(element), gemmi.Position(x + dx, 0, 0)
        atom.altloc, atom.occ = altloc, occupancy
        residue.add_atom(atom)
    chain.add_residue(residue)
model.add_chain(chain)
structure.add_model(model)
block = gemmi.cif.read_string("data_t\nloop_\n_struct_asym.id\n_struct_asym.entity_id\nX1 1\nloop_\n_entity_poly.entity_id\n_entity_poly.type\n1 'polypeptide(L)'\n").sole_block()
source = prepare_sources(structure, block)["X1"]
assert source["sequence"] == "AG"
assert source["label_seq_gaps"] == [[1, 3]]
assert [r["auth_seq_id"] for r in source["residue_map"]] == [101, 103]
np.testing.assert_array_equal(source["coords"][:, 0], [0, 1, 10, 11])
assert source["seq_sha256"] == hashlib.sha256(b"AG").hexdigest()

rows = [dict(unique_id=str(i), sequence=s) for i, s in enumerate(["ACDEFGHIK", "ACDEFGHIL", "WWWWWWWWW", "ACDEFGHIK" * 10])]
si.init(rows)
_, edges, skipped, aligned = si.compare_row(0)
assert len(edges) == 1 and edges[0][:2] == ("0", "1")
assert edges[0][2:4] == (8, 9)
assert skipped == 2 and aligned == 1
print("PASS observed sequence/gaps/altloc/hydrogen filtering; SI identity and safe pruning")
