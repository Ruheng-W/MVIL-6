from rdkit import Chem
from rdkit.Chem import AllChem
smi='CC(C)OC(=O)C(C)NP(=O)(OCC1C(C(C(O1)N2C=CC(=O)NC2=O)(C)F)O)OC3=CC=CC=C3'
mol = Chem.MolFromSmiles(smi)
t = Chem.MolToSequence(mol)
print(t)