import os
from hydra import initialize, compose
from pathlib import Path
import subprocess

#from rf2aa.run_inference import ModelRunner
script_dir=os.path.dirname(os.path.abspath(__file__)) #RoseTTAFold-All-Atom/rf2aa/data
msa_script_dir=os.path.abspath(os.path.join(script_dir, '..','input_prep'))   #RoseTTAFold-All-Atom/rf2aa/input_prep


def make_msa(
    fasta_file,
    chain,
    model_runner
): 
    out_dir_base = Path(model_runner.config.output_path)
    hash = model_runner.config.job_name
    out_dir = out_dir_base / hash / chain
    out_dir.mkdir(parents=True, exist_ok=True)

    command = model_runner.config.database_params.command

# sequence databases
    DB_UR30=model_runner.config.database_params.DB_UR30
    DB_BFD=model_runner.config.database_params.DB_BFD
    num_cpus = model_runner.config.database_params.num_cpus
    ram_gb = model_runner.config.database_params.mem
    template_database = model_runner.config.database_params.DB_PDB100

    out_a3m = out_dir / "t000_.msa0.a3m"
    out_atab = out_dir / "t000_.atab"
    out_hhr = out_dir / "t000_.hhr"
    if out_a3m.exists() and out_atab.exists() and out_hhr.exists():
        return out_a3m, out_hhr, out_atab

    search_command = f"{msa_script_dir}/{command} {os.path.abspath(fasta_file)} {os.path.abspath(out_dir)} {num_cpus} {ram_gb} {DB_UR30} {DB_BFD} {template_database}"
    print(search_command)
    _ = subprocess.run(search_command, shell=True)

    if _.returncode != 0:
        raise RuntimeError(f"Failed to execute command {search_command}")
    return out_a3m, out_hhr, out_atab

