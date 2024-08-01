import re
import torch
from openbabel import openbabel
from typing import List, Literal, Optional, Tuple, Union
from dataclasses import dataclass
from tempfile import NamedTemporaryFile
from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.data.parsers import parse_mol
from rf2aa.data.small_molecule import compute_features_from_obmol
from rf2aa.util import get_bond_feats


@dataclass
class MoleculeToMoleculeBond:
    chain_index_first: int
    absolute_atom_index_first: int
    chain_index_second: int
    absolute_atom_index_second: int
    new_chirality_atom_first: Optional[Literal['null','CW','CCW']]
    new_chirality_atom_second: Optional[Literal['null','CW','CCW']]

@dataclass
class AtomizedResidue:
    chain: str
    chain_index_in_combined_chain: int
    absolute_N_index_in_chain: int
    absolute_C_index_in_chain: int
    original_chain: str
    index_in_original_chain: int

@dataclass
class CovalentBond:
    """
    A class representing a covalent bond between a protein and a small molecule.

    Attributes:
    - protein_chain: Identifier for the protein chain, of string type.
    - protein_residue_number: Residue number in the protein, can be an integer or a string representation of a number.
    - protein_atom_name: Name of the atom in the protein participating in the covalent bond, string type.
    - small_molecule_chain: Identifier for the small molecule chain, string type.
    - sm_atom_index: Index of the atom in the small molecule participating in the covalent bond, can be an integer or a string representation of a number.
    - new_chirality_atom_1: Orientation of the new chiral center atom 1, can be 'null', 'CW' (clockwise), or 'CCW' (counterclockwise).
    - new_chirality_atom_2: Orientation of the new chiral center atom 2, can be 'null', 'CW', or 'CCW'.

    Returns:
    - None. This initializer directly constructs the object.
    """

    protein_chain: str
    protein_residue_number: int
    protein_atom_name: str
    small_molecule_chain: str
    sm_atom_index: int

    new_chirality_atom_1: Literal['null','CW','CCW']
    new_chirality_atom_2: Literal['null','CW','CCW']

    def __post_init__(self):
        """
        Special method called immediately after the object is initialized, used to validate and convert residue_number and atom_index values.
        """
        # Validate and convert residue_number to ensure it's an integer
        if isinstance(self.protein_residue_number, str):
            if not self.protein_residue_number.isdigit():
                raise ValueError("Residue number must be a number")
            self.protein_residue_number = int(self.protein_residue_number)
        
        # Validate and convert atom_index to ensure it's an integer
        if isinstance(self.sm_atom_index, str):
            if not self.sm_atom_index.isdigit():
                raise ValueError("atom_index must be a number")
            self.sm_atom_index = int(self.sm_atom_index)

@dataclass
class CovalentBondsParser:
    """
    A class for parsing covalent bond information.

    Attributes:
        input (str): A string containing a series of covalent bond entries, separated by semicolons. Each entry follows the pattern: 'Chain,Residue,Atom:Chain,AtomIndex:NewChiralityAtom1,NewChiralityAtom2'.
        sub_pattern (Optional[str]): A regular expression pattern for matching the input format. Defaults to `r'([A-Z]{1}),([\d]{1,4}),(\w+):([A-Z]{1}),([\d]{1,2}):(\w+),(\w+)'`.

    Methods:
        __post_init__: Validates the input format after initialization.
        parse_one_covalent_bond(sub_input: str) -> CovalentBond: Parses a single covalent bond entry and returns a CovalentBond object.
        formatted -> Tuple[CovalentBond]: Returns a tuple of parsed CovalentBond objects from the input string.
    """

    input: str # A case: 'A,74,ND2:B,1:CW,null;...'
    sub_pattern: Optional[str]=r'([A-Z]{1}),([\d]{1,4}),(\w+):([A-Z]{1}),([\d]{1,2}):(\w+),(\w+)'

    def __post_init__(self):
        """
        Validates the input format after initialization.

        Raises:
            ValueError: If the input format is invalid.
        """
        for sub_input in self.input.split(';'):
            if not (':' in sub_input and ',' in sub_input):
                raise ValueError(f"Invalid input format: {sub_input=}.\nPlease follow comma-separated pattern like `r'{self.sub_pattern}'`")
    def parse_one_covalent_bond(self, sub_input: str) -> CovalentBond:
        """
        Parses a single covalent bond entry and returns a CovalentBond object.

        Args:
            sub_input (str): A covalent bond entry to be parsed.

        Returns:
            CovalentBond: The parsed covalent bond object.

        Raises:
            ValueError: If the input format is invalid.
        """
        if not (is_match:= re.match(self.sub_pattern, sub_input)):
            raise ValueError(f"Invalid input format: {sub_input=}")
        return CovalentBond(
            protein_chain=is_match.group(1),
            protein_residue_number=is_match.group(2),
            protein_atom_name=is_match.group(3),
            small_molecule_chain=is_match.group(4),
            sm_atom_index=is_match.group(5),
            new_chirality_atom_1=is_match.group(6),
            new_chirality_atom_2=is_match.group(7))

    @property
    def formatted(self) -> Tuple[CovalentBond]:
        """
        Returns a tuple of parsed CovalentBond objects from the input string.

        Returns:
            Tuple[CovalentBond]: A tuple containing CovalentBond objects.
        """
        return tuple(self.parse_one_covalent_bond(sub_input) for sub_input in self.input.split(';'))

def load_covalent_molecules(protein_inputs, config, model_runner):
    if config.covale_inputs is None:
        return None
    
    if config.sm_inputs is None:
        raise ValueError("If you provide covale_inputs, you must also provide small molecule inputs")
     
    covalent_bonds = CovalentBondsParser(config.covale_inputs).formatted
    sm_inputs = delete_leaving_atoms(config.sm_inputs)
    residues_to_atomize, combined_molecules, extra_bonds = find_residues_to_atomize(protein_inputs, sm_inputs, covalent_bonds, model_runner)
    chainid_to_input = {}
    for chain, combined_molecule in combined_molecules.items():
        extra_bonds_for_chain = extra_bonds[chain]
        msa, bond_feats, xyz, Ls = get_combined_atoms_bonds(combined_molecule)
        residues_to_atomize = update_absolute_indices_after_combination(residues_to_atomize, chain, Ls)
        mol = make_obmol_from_atoms_bonds(msa, bond_feats, xyz, Ls, extra_bonds_for_chain)
        xyz = recompute_xyz_after_chirality(mol)
        input = compute_features_from_obmol(mol, msa, xyz, model_runner)
        chainid_to_input[chain] = input
    
    return chainid_to_input, residues_to_atomize

def find_residues_to_atomize(protein_inputs, sm_inputs, covalent_bonds: Tuple[CovalentBond], model_runner):
    residues_to_atomize = [] # hold on to delete wayward inputs
    combined_molecules = {} # combined multiple molecules that are bonded
    extra_bonds = {}
    for bond in covalent_bonds:
        prot_chid = bond.protein_chain
        prot_res_idx=bond.protein_residue_number
        atom_to_bond=bond.protein_atom_name
        sm_chid=bond.small_molecule_chain
        sm_atom_num=bond.sm_atom_index
        chirality_first_atom=bond.new_chirality_atom_1
        chirality_second_atom = bond.new_chirality_atom_2
        if chirality_first_atom.strip() == "null":
            chirality_first_atom = None
        if chirality_second_atom.strip() == "null":
            chirality_second_atom = None

        sm_atom_num = int(sm_atom_num) - 1 # 0 index 
        try:
            assert sm_chid in sm_inputs, f"must provide a small molecule chain {sm_chid} for covalent bond: {bond}"
        except:
            print(f"Skipping bond: {bond} since no sm chain {sm_chid} was provided")
            continue
        assert sm_inputs[sm_chid].input_type == "sdf", "only sdf inputs can be covalently linked to proteins"
        try:
            protein_input = protein_inputs[prot_chid]
        except Exception as e:
            raise ValueError(f"first atom in covale_input must be present in\
                             a protein chain. Given chain: {prot_chid} was not in \
                            given protein chains: {list(protein_inputs.keys())}")
        
        residue = (prot_chid, prot_res_idx, atom_to_bond)
        file, atom_index = convert_residue_to_molecule(protein_inputs, residue, model_runner) 
        if sm_chid not in combined_molecules:
            combined_molecules[sm_chid] = [sm_inputs[sm_chid].input]
        combined_molecules[sm_chid].insert(0, file) # this is a bug, revert
        absolute_chain_index_first = combined_molecules[sm_chid].index(sm_inputs[sm_chid].input)
        absolute_chain_index_second = combined_molecules[sm_chid].index(file)
        
        if sm_chid not in extra_bonds:
            extra_bonds[sm_chid] = []
        extra_bonds[sm_chid].append(MoleculeToMoleculeBond(
            absolute_chain_index_first,
            sm_atom_num,
            absolute_chain_index_second,
            atom_index,
            new_chirality_atom_first=chirality_first_atom,
            new_chirality_atom_second=chirality_second_atom
        ))
        residues_to_atomize.append(AtomizedResidue(
            sm_chid,
            absolute_chain_index_second,
            0, 
            2,
            prot_chid,
            int(prot_res_idx) -1
        ))

    return residues_to_atomize, combined_molecules, extra_bonds

def convert_residue_to_molecule(protein_inputs, residue, model_runner):
    """convert residue into sdf and record index for covalent  bond"""
    prot_chid, prot_res_idx, atom_to_bond = residue
    protein_input = protein_inputs[prot_chid]
    prot_res_abs_idx = int(prot_res_idx) -1
    residue_identity_num = protein_input.query_sequence()[prot_res_abs_idx]
    residue_identity = ChemData().num2aa[residue_identity_num]
    molecule_info = model_runner.molecule_db[residue_identity]
    sdf = molecule_info["sdf"]
    temp_file = create_and_populate_temp_file(sdf)
    is_heavy = [i for i, a in enumerate(molecule_info["atom_id"]) if a[0] != "H"]
    is_leaving = [a for i,a  in enumerate(molecule_info["leaving"]) if i in is_heavy]

    sdf_string_no_leaving_atoms = delete_leaving_atoms_single_chain(temp_file, is_leaving )
    temp_file = create_and_populate_temp_file(sdf_string_no_leaving_atoms)
    atom_names = molecule_info["atom_id"]
    atom_index = atom_names.index(atom_to_bond.strip())
    return temp_file, atom_index

def get_combined_atoms_bonds(combined_molecule):
    atom_list  = []
    bond_feats_list = []
    xyzs = []
    Ls = []
    for molecule in combined_molecule:
        obmol, msa, ins, xyz, mask = parse_mol(
            molecule, 
            filetype="sdf", 
            string=False,
            generate_conformer=True,
            find_automorphs=False    
        )
        bond_feats = get_bond_feats(obmol)

        atom_list.append(msa)
        bond_feats_list.append(bond_feats)
        xyzs.append(xyz)
        Ls.append(msa.shape[0])
    
    atoms = torch.cat(atom_list)
    L_total = sum(Ls)
    bond_feats = torch.zeros((L_total, L_total)).long()
    offset = 0
    for bf in bond_feats_list:
        L = bf.shape[0]
        bond_feats[offset:offset+L, offset:offset+L] = bf
        offset += L
    xyz = torch.cat(xyzs, dim=1)[0]
    return atoms, bond_feats, xyz, Ls
        
def make_obmol_from_atoms_bonds(msa, bond_feats, xyz, Ls, extra_bonds):
    mol = openbabel.OBMol()    
    for i,k in enumerate(msa):
        element = ChemData().num2aa[k]
        atomnum = ChemData().atomtype2atomnum[element]
        a = mol.NewAtom()
        a.SetAtomicNum(atomnum)
        a.SetVector(float(xyz[i,0]), float(xyz[i,1]), float(xyz[i,2]))

    first_index, second_index = bond_feats.nonzero(as_tuple=True)
    for i, j in zip(first_index, second_index):
        order = bond_feats[i,j]
        bond = make_openbabel_bond(mol, i.item(), j.item(), order.item())
        mol.AddBond(bond)

    for bond in extra_bonds:
        absolute_index_first = get_absolute_index_from_relative_indices(
            bond.chain_index_first,
            bond.absolute_atom_index_first, 
            Ls
        ) 
        absolute_index_second = get_absolute_index_from_relative_indices(
            bond.chain_index_second,
            bond.absolute_atom_index_second,
            Ls
        )
        order = 1 #all covale bonds are single bonds
        openbabel_bond = make_openbabel_bond(mol, absolute_index_first, absolute_index_second, order)
        mol.AddBond(openbabel_bond)
        set_chirality(mol, absolute_index_first, bond.new_chirality_atom_first)
        set_chirality(mol, absolute_index_second, bond.new_chirality_atom_second)
    return mol

def make_openbabel_bond(mol, i, j, order):
    obb = openbabel.OBBond()
    obb.SetBegin(mol.GetAtom(i+1))
    obb.SetEnd(mol.GetAtom(j+1))
    if order == 4:
        obb.SetBondOrder(2)
        obb.SetAromatic()
    else:
        obb.SetBondOrder(order)
    return obb

def set_chirality(mol, absolute_atom_index, new_chirality):
    stereo = openbabel.OBStereoFacade(mol)
    if stereo.HasTetrahedralStereo(absolute_atom_index+1):
        tetstereo = stereo.GetTetrahedralStereo(mol.GetAtom(absolute_atom_index+1).GetId())
        if tetstereo is None:
            return

        assert new_chirality is not None, "you have introduced a new stereocenter, \
            so you must specify its chirality either as CW, or CCW"
        
        config = tetstereo.GetConfig()
        config.winding = chirality_options[new_chirality]
        tetstereo.SetConfig(config)
        print("Updating chirality...")
    else:
        assert new_chirality is None, "you have specified a chirality without creating a new chiral center"
    
chirality_options = {
    "CW": openbabel.OBStereo.Clockwise,
    "CCW": openbabel.OBStereo.AntiClockwise,
}

def recompute_xyz_after_chirality(obmol):
    builder = openbabel.OBBuilder()
    builder.Build(obmol)
    ff = openbabel.OBForceField.FindForceField("mmff94")
    did_setup = ff.Setup(obmol)
    if did_setup:
        ff.FastRotorSearch()
        ff.GetCoordinates(obmol)
    else:
        raise ValueError(f"Failed to generate 3D coordinates for molecule {obmol}.")
    atom_coords = torch.tensor([[obmol.GetAtom(i).x(),obmol.GetAtom(i).y(), obmol.GetAtom(i).z()] 
                                for i in range(1, obmol.NumAtoms()+1)]).unsqueeze(0) # (1, natoms, 3)
    return atom_coords

def delete_leaving_atoms(sm_inputs):
    updated_sm_inputs = {}
    for chain in sm_inputs:
        if "is_leaving" not in sm_inputs[chain]:
            continue
        is_leaving = eval(sm_inputs[chain]["is_leaving"])
        sdf_string = delete_leaving_atoms_single_chain(sm_inputs[chain]["input"], is_leaving)
        updated_sm_inputs[chain] = {
            "input": create_and_populate_temp_file(sdf_string),
            "input_type": "sdf"
            }
    
    sm_inputs.update(updated_sm_inputs)
    return sm_inputs

def delete_leaving_atoms_single_chain(filename, is_leaving):
    obmol, msa, ins, xyz, mask = parse_mol(
        filename,
        filetype="sdf", 
        string=False,
        generate_conformer=True    
    )
    assert len(is_leaving) == obmol.NumAtoms()
    leaving_indices = torch.tensor(is_leaving).nonzero()
    for idx in leaving_indices:
        obmol.DeleteAtom(obmol.GetAtom(idx.item()+1))
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("sdf", "sdf")
    sdf_string = obConversion.WriteString(obmol)
    return sdf_string

def get_absolute_index_from_relative_indices(chain_index, absolute_index_in_chain, Ls):
    offset = sum(Ls[:chain_index])
    return offset + absolute_index_in_chain

def update_absolute_indices_after_combination(residues_to_atomize, chain, Ls):
    updated_residues_to_atomize  = []
    for residue in residues_to_atomize:
        if residue.chain == chain:
            absolute_index_N = get_absolute_index_from_relative_indices(
                residue.chain_index_in_combined_chain,
                residue.absolute_N_index_in_chain,
                Ls)
            absolute_index_C = get_absolute_index_from_relative_indices(
                residue.chain_index_in_combined_chain,
                residue.absolute_C_index_in_chain,
                Ls)
            updated_residue = AtomizedResidue(
                residue.chain,
                None,
                absolute_index_N,
                absolute_index_C,
                residue.original_chain,
                residue.index_in_original_chain
            )
            updated_residues_to_atomize.append(updated_residue)
        else:
            updated_residues_to_atomize.append(residue)
    return updated_residues_to_atomize

def create_and_populate_temp_file(data):
    # Create a temporary file
    with NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        # Write the string to the temporary file
        temp_file.write(data)

        # Get the filename
        temp_file_name = temp_file.name

    return temp_file_name
