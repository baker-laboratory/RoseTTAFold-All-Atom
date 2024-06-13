"""
Input UI for RoseTTAfold All Atom 

using two custom gradio components: gradio_molecule3d and gradio_cofoldinginput
"""



import json
import yaml
import os
import zipfile


import torch 

import gradio as gr
import plotly.express as px

from openbabel import openbabel
from gradio_cofoldinginput import CofoldingInput
from gradio_molecule3d import Molecule3D

baseconfig = """job_name: "structure_prediction"
output_path: ""
checkpoint_path: RFAA_paper_weights.pt
database_params:
  sequencedb: ""
  hhdb: "pdb100_2021Mar03/pdb100_2021Mar03"
  command: make_msa.sh
  num_cpus: 4
  mem: 64
protein_inputs: null
na_inputs: null
sm_inputs: null
covale_inputs:  null
residue_replacement: null

chem_params:
  use_phospate_frames_for_NA: True
  use_cif_ordering_for_trp: True

loader_params:
  n_templ: 4
  MAXLAT: 128
  MAXSEQ: 1024
  MAXCYCLE: 4
  BLACK_HOLE_INIT: False
  seqid: 150.0


legacy_model_param:
  n_extra_block: 4
  n_main_block: 32
  n_ref_block: 4
  n_finetune_block: 0
  d_msa: 256
  d_msa_full: 64
  d_pair: 192
  d_templ: 64
  n_head_msa: 8
  n_head_pair: 6
  n_head_templ: 4
  d_hidden_templ: 64
  p_drop: 0.0
  use_chiral_l1: True
  use_lj_l1: True
  use_atom_frames: True
  recycling_type: "all"
  use_same_chain: True
  lj_lin: 0.75
  SE3_param: 
    num_layers: 1
    num_channels: 32
    num_degrees: 2
    l0_in_features: 64
    l0_out_features: 64
    l1_in_features: 3
    l1_out_features: 2
    num_edge_features: 64
    n_heads: 4
    div: 4
  SE3_ref_param:
    num_layers: 2
    num_channels: 32
    num_degrees: 2
    l0_in_features: 64
    l0_out_features: 64
    l1_in_features: 3
    l1_out_features: 2
    num_edge_features: 64
    n_heads: 4
    div: 4
"""

def convert_format(input_file, jobname, chain, deleteIndexes, attachmentIndex):

    conv = openbabel.OBConversion()
    conv.SetInAndOutFormats('cdjson', 'sdf')

    # Add options
    conv.AddOption("c", openbabel.OBConversion.OUTOPTIONS, "1")
    with open(f"{jobname}_sm_{chain}.json", "w+") as fp:
        fp.write(input_file)
    mol = openbabel.OBMol()
    conv.ReadFile(mol, f"{jobname}_sm_{chain}.json") 
    
    deleted_count = 0
    # delete atoms in delete indexes
    for index in sorted(deleteIndexes, reverse=True):
        if index < attachmentIndex:
            deleted_count += 1
        atom = mol.GetAtom(index)
        mol.DeleteAtom(atom)
    
    attachmentIndex -= deleted_count

    conv.WriteFile(mol, f"{jobname}_sm_{chain}.sdf")
    return attachmentIndex


def prepare_input(input, jobname, baseconfig, hard_case):
    input_categories = {"protein":"protein_inputs", "DNA":"na_inputs","RNA":"na_inputs", "ligand":"sm_inputs"}

    # convert input to yaml format
    yaml_dict = {"defaults":["base"], "job_name":jobname, "output_path": jobname}
    list_of_input_files = []

    if len(input["chains"]) == 0:
        raise gr.Error("At least one chain must be provided")
    for chain in input["chains"]:
        if input_categories[chain["class"]] not in yaml_dict.keys():
            yaml_dict[input_categories[chain["class"]]] = {}

        if input_categories[chain["class"]] in ["protein_inputs", "na_inputs"]:
            #write fasta 
            with open(f"{jobname}_{chain['chain']}.fasta", "w+") as fp:
                fp.write(f">chain A\n{chain['sequence']}")
            if input_categories[chain["class"]] == "na_inputs":
                entry = {"input_type":chain["class"].lower(), "fasta":f"{jobname}/{jobname}_{chain['chain']}.fasta"}
            else:
                entry = {"fasta_file": f"{jobname}/{jobname}_{chain['chain']}.fasta"}
            list_of_input_files.append(f"{jobname}_{chain['chain']}.fasta")
            yaml_dict[input_categories[chain["class"]]][chain['chain']] =  entry
        
        if input_categories[chain['class']] == "sm_inputs":
            if "smiles" in chain.keys():
                entry = {"input_type": "smiles", "input": chain["smiles"]}
            elif "sdf" in chain.keys():
                # write to file 
                with open(f"{jobname}_sm_{chain['chain']}.sdf", "w+") as fp:
                    fp.write(chain["sdf"])
                list_of_input_files.append(f"{jobname}_sm_{chain['chain']}.sdf")
                entry = {"input_type": "sdf", "input": f"{jobname}/{jobname}_sm_{chain['chain']}.sdf"}
            elif "name" in chain.keys():
                list_of_input_files.append(f"metal_sdf/{chain['name']}_ideal.sdf")
                entry = {"input_type": "sdf", "input": f"{jobname}/{chain['name']}_ideal.sdf"}
            yaml_dict["sm_inputs"][chain['chain']] =  entry

    covale_inputs = []
    if len(input["covMods"])>0:
        yaml_dict["covale_inputs"]=""

    for covMod in input["covMods"]:
        if len(covMod["deleteIndexes"])>0:
            new_attachment_index = convert_format(covMod["mol"],jobname, covMod["ligand"], covMod["deleteIndexes"], covMod["attachmentIndex"])
        chirality_ligand = "null"
        chirality_protein = "null"
        if covMod["protein_symmetry"] in ["CW", "CCW"]:
            chirality_protein = covMod["protein_symmetry"]
        if covMod["ligand_symmetry"] in ["CW", "CCW"]:
            chirality_ligand = covMod["ligand_symmetry"]
        covale_inputs.append(((covMod[ "protein"], covMod["residue"], covMod["atom"]), (covMod["ligand"], new_attachment_index), (chirality_protein, chirality_ligand)))
    if len(input["covMods"])>0:
        yaml_dict["covale_inputs"] = json.dumps(json.dumps(covale_inputs))[1:-1].replace("'", "\"")
    
    if hard_case:
        yaml_dict["loader_params"]= {}
        yaml_dict["loader_params"]["MAXCYCLE"] = 10
    # write yaml to tmp 
    with open(f"/tmp/{jobname}.yaml", "w+") as fp:
        # need to convert single quotes to double quotes
        fp.write(yaml.dump(yaml_dict).replace("'", "\""))
    
    # write baseconfig 
    with open(f"/tmp/base.yaml", "w+") as fp:
        fp.write(baseconfig)

    list_of_input_files.append(f"/tmp/{jobname}.yaml")
    list_of_input_files.append(f"/tmp/base.yaml")
    # convert dictionary to YAML
    with zipfile.ZipFile(os.path.join("/tmp/", f"{jobname}.zip"), 'w') as zip_archive:        
        for file in set(list_of_input_files):
            zip_archive.write(file, arcname= os.path.join(jobname,os.path.basename(file)),compress_type=zipfile.ZIP_DEFLATED)
    
    return yaml.dump(yaml_dict).replace("'", "\""),os.path.join("/tmp/", f"{jobname}.zip")


def convert_bfactors(pdb_path):
    with open(pdb_path, 'r') as f:
        lines = f.readlines()
    for i,line in enumerate(lines):
        # multiple each bfactor by 100
        if line[0:6] == 'ATOM  ' or line[0:6] == 'HETATM':
            bfactor = float(line[60:66])
            bfactor *= 100
            line = line[:60] + f'{bfactor:6.2f}' + line[66:]
            lines[i] = line
    with open(pdb_path.replace(".pdb", "_processed.pdb"), 'w') as f:
        f.write(''.join(lines))


def run_rf2aa(jobname, zip_archive):
    current_dir = os.getcwd()
    try:
        with zipfile.ZipFile(zip_archive, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(current_dir))
        os.system(f"python -m rf2aa.run_inference --config-name {jobname}.yaml --config-path {current_dir}/{jobname}")
        # scale pLDDT to 0-100 range in pdb output file
        convert_bfactors(f"{current_dir}/{jobname}/{jobname}.pdb")
        aux = torch.load(f"{current_dir}/{jobname}/{jobname}_aux.pt")
        fig_pde = px.imshow(aux["pde"][0],  template="simple_white", labels={"x": "Scored residue", "y": "Aligned residue",  "color":"PDE",})
        fig_pde.update_layout(coloraxis_colorbar=dict(title="PDE (Å)"))
        fig_pae = px.imshow(aux["pae"][0],  template="simple_white", labels={"x": "Scored residue", "y": "Aligned residue",  "color":"PAE",})
        fig_pae.update_layout(coloraxis_colorbar=dict(title="PAE (Å)"))
        fig_plddt = px.line(y=aux["plddts"].flatten().numpy()*100, template="simple_white", labels={"y": "pLDDT", "x":"residue"})

    except Exception as e:
        raise gr.Error(f"Error running RFAA: {e}")
    return f"{current_dir}/{jobname}/{jobname}_processed.pdb", fig_plddt, fig_pae, fig_pde
    


def predict(input, jobname, dry_run, baseconfig, hard_case):
    yaml_input, zip_archive = prepare_input(input, jobname, baseconfig, hard_case)

    reps = []

    for chain in input["chains"]:
        if chain["class"] in ["protein", "RNA", "DNA"]:
            reps.append({
                "model": 0,
                "chain": chain["chain"],
                "resname": "",
                "style": "cartoon",
                "color": "alphafold",
                "residue_range": "",
                "around": 0,
                "byres": False
            })
        elif chain["class"] == "ligand" and "name" not in chain.keys():
            reps.append({
                "model": 0,
                "chain": chain["chain"],
                "resname": "LG1",
                "style": "stick",
                "color": "whiteCarbon",
                "residue_range": "",
                "around": 0,
                "byres": False
            })
        else:
            reps.append({
                "model": 0,
                "chain": chain["chain"],
                "resname": "LG1",
                "style": "sphere",
                "color": "whiteCarbon",
                "residue_range": "",
                "around": 0,
                "byres": False
            })
    if dry_run:
        return gr.Code(yaml_input, visible=True), gr.File(zip_archive, visible=True), gr.Markdown(f"""You can run your RFAA job using the following command: <pre>python -m rf2aa.run_inference --config-name {jobname}.yaml --config-path absolute/path/to/unzipped/{jobname}</pre>""", visible=True), Molecule3D(visible=False), gr.Plot(visible=False), gr.Plot(visible=False), gr.Plot(visible=False)
    else:
        pdb_file, pldtt_plot, pae_plot, pde_plot = run_rf2aa(jobname, zip_archive)
        return gr.Code(yaml_input, visible=True), gr.File(zip_archive, visible=True),gr.Markdown(visible=False), Molecule3D(pdb_file,reps=reps,visible=True), gr.Plot(pldtt_plot, visible=True), gr.Plot(pae_plot, visible=True), gr.Plot(pde_plot, visible=True)

with gr.Blocks() as demo:
    gr.Markdown("# RoseTTAFold All Atom UI")
    gr.Markdown("""This UI allows you to generate input files for RoseTTAFold All Atom (RFAA) using the CofoldingInput widget. The input files can be used to run RFAA on your local machine. <br /> 
                If you launch the UI directly on your local machine you can also directly run the RFAA prediction. <br />
                More information in the official GitHub repository: [baker-laboratory/RoseTTAFold-All-Atom](https://github.com/baker-laboratory/RoseTTAFold-All-Atom)
                """)
    jobname = gr.Textbox("job1", label="Job Name")
    with gr.Tab("Input"):
        inp=CofoldingInput(label="Input")
        hard_case = gr.Checkbox(False, label="Hard case (increase MAXCYCLE to 10)")
        # only allow running the predictions if local 
        if os.environ.get("SPACE_HOST")!=None:
            dry_run = gr.Checkbox(True, label="Only generate input files (dry run)", interactive=False)
        else:
            dry_run = gr.Checkbox(True, label="Only generate input files (dry run)")
    with gr.Tab("Base config"):
        base_config = gr.Code(baseconfig, label="Base config")
    btn = gr.Button("Run")
    config_file = gr.Code(label="YAML Hydra config for RFAA", visible=True)
    runfiles = gr.File(label="files to run RFAA", visible=False)
    instructions = gr.Markdown(visible=False)

    
    out = Molecule3D(visible=False, label="Predicted Structure")
    with gr.Row():
        plddt = gr.Plot(visible=False, label="pLDDT")
        pae = gr.Plot(visible=False, label="Predicted aligned error")
        pde = gr.Plot(visible=False, label="Predicted distance error")
    btn.click(predict, inputs=[inp, jobname, dry_run, base_config, hard_case], outputs=[config_file, runfiles, instructions, out, plddt, pae, pde])

if __name__ == "__main__":
    demo.launch(share=True)
