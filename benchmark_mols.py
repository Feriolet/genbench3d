import json
import yaml
import logging
import argparse
import os
import numpy as np

from concurrent.futures import as_completed, ProcessPoolExecutor
from rdkit import Chem
from genbench3d import GenBench3D
from genbench3d.data.source import CSDDrug, CrossDocked, SDFSource, MolListSource
from genbench3d.data.structure import Protein, Pocket
from genbench3d.data import ComplexMinimizer
from genbench3d.utils import preprocess_mols, preprocess_mol
from genbench3d.geometry import ReferenceGeometry

from rdkit import RDLogger 
from tqdm import tqdm
from time import time
RDLogger.DisableLog('rdApp.*')

from warnings import simplefilter
simplefilter(action='ignore', category=DeprecationWarning)
import pandas as pd

def align_mol_name_with_results(mol_l, original_mol_name, results):
    results_with_individual_value_dict = {}
    cel_mol_name = [mol.GetProp('_Name') for mol in mol_l]
    total_valid_mol = max([len(val) for val in results.values() if type(val) == list])

    for key, val in results.items():
        if type(val) == list and len(val) == total_valid_mol:
            DATA_COLUMN = 'genbench_key'
            data = pd.DataFrame(val, index=cel_mol_name, columns=[DATA_COLUMN])
            results_with_individual_value_dict[key] = list(data.reindex(original_mol_name).to_dict()[DATA_COLUMN].values())
        else:
            results_with_individual_value_dict[key] = val
    
    return results_with_individual_value_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", 
                        default='config/default.yaml', 
                        type=str,
                        help="Path to config file.")
    parser.add_argument("-i", "--input_sdf", 
                        # default='examples/pocket2mol_generated_2z3h.sdf', 
                        type=str,
                        help="Path to sdf file containing molecules to benchmark.")
    parser.add_argument("-o", "--output_json", 
                        # default='examples/results_pocket2mol_generated_2z3h.json', 
                        type=str,
                        help="Path to json file to store benchmark results.")
    parser.add_argument("-s", "--source",
                        default='ligboundconf',
                        type=str,
                        help="Source of the reference geometry.",
                        choices=['csd_drug', 'crossdocked', 'ligboundconf'])
    parser.add_argument("-m", "--minimize",
                        action='store_true',
                        help="Whether to minimize the molecules before benchmarking.")
    parser.add_argument("-p", "--pdb_structure",
                        # default='test_set/BSD_ASPTE_1_130_0/2z3h_A_rec.pdb',
                        type=str,
                        help="PDB structure for the pocket used to generate the molecules")
    parser.add_argument("-n", "--native_ligand_sdf",
                        # default='test_set/BSD_ASPTE_1_130_0/2z3h_A_rec_1wn6_bst_lig_tt_docked_3.sdf',
                        help="Native ligand corresponding to the pocket used to generate the molecules")
    parser.add_argument('--log_output',
                        type=str, default='benchmark.log',
                        help="log Output directory")
    args = parser.parse_args()


    if args.log_output:
        logging.basicConfig(format='%(asctime)s [%(levelname)s] %(funcName)s: %(message)s',
                            datefmt='%d/%m/%Y %I:%M:%S %p',
                            filemode='w',
                            filename=args.log_output, 
                            encoding='utf-8', 
                            level=logging.INFO)
    else:
        logging.basicConfig(format='%(asctime)s [%(levelname)s] %(funcName)s: %(message)s',
                        datefmt='%d/%m/%Y %I:%M:%S %p',
                        filemode='w',
                        filename='sb_benchmark.log', 
                        encoding='utf-8', 
                        level=logging.INFO)
        

    config = yaml.safe_load(open(args.config_path, 'r'))

    if args.source == 'csd_drug':
        source = CSDDrug(subset_path=config['data']['csd_drug_subset_path'])
    elif args.source == 'crossdocked':
        source = CrossDocked(root=config['benchmark_dirpath'],
                            config=config['data'],
                            subset='train')
    elif args.source == 'ligboundconf':
        source = SDFSource(ligands_path=config['data']['ligboundconf_path'],
                        name='LigBoundConf')
        # mol_list = Chem.SDMolSupplier(config['data']['ligboundconf_path'], removeHs=False)
        # source = MolListSource(mol_list=mol_list,
        #                         name='LigBoundConf')
    else:
        raise ValueError(f"Unknown source: {args.source}")
        
    reference_geometry = ReferenceGeometry(source=source,
                                        root=config['benchmark_dirpath'],
                                        minimum_pattern_values=config['genbench3d']['minimum_pattern_values'],)

    benchmark = GenBench3D(reference_geometry=reference_geometry,
                            config=config['genbench3d'])

    if args.minimize:
        assert args.pdb_structure is not None, "PDB structure path is required for minimization."
        assert args.native_ligand_sdf is not None, "Native ligand path is required for minimization."
        # absolute paths are required for Glide and Gold
        original_structure_path = os.path.abspath(args.pdb_structure)
        native_ligand_path = os.path.abspath(args.native_ligand_sdf)
        native_ligand = [mol 
                            for mol in Chem.SDMolSupplier(native_ligand_path, 
                                                        removeHs=False)][0]
        native_ligand = Chem.AddHs(native_ligand, addCoords=True)

        protein = Protein(pdb_filepath=original_structure_path)
        pocket = Pocket(pdb_filepath=protein.protein_clean_filepath, 
                        native_ligand=native_ligand,
                        distance_from_ligand=config['pocket_distance_from_ligand'])
        complex_minimizer = ComplexMinimizer(pocket,
                                                config=config['minimization'])


    start = time()
    gen_mols : list[Chem.Mol] = []
    start_sdf = [mol for mol in Chem.MultithreadedSDMolSupplier(args.input_sdf, removeHs=False, numWriterThreads=10)]
    n_total_mols = len(start_sdf) # Used to compute the molecular graph Validity metric
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for x in  tqdm(start_sdf):
            futures.append(executor.submit(preprocess_mol, x))
            # keep memory under control
            if len(futures) >= 1000:  # process in chunks of 100
                for f in as_completed(futures):
                    result = f.result()
                    
                    if result:
                        gen_mols.append(result)
                futures = []

        # handle leftovers
        for f in as_completed(futures):
            result = f.result()
            if result:
                gen_mols.append(result)

    print(f'finish reading in {time() - start}')

    name_l = [mol.GetProp('_Name') if mol else f"unknown_{i}" for i, mol in enumerate(gen_mols) ]
    if args.minimize:
        
        gen_mols = [complex_minimizer.minimize_ligand(mol) 
                    for mol in gen_mols]
        gen_mols = preprocess_mols(gen_mols)

    results = benchmark.get_results_for_mol_list(gen_mols,
                                                n_total_mols=n_total_mols)

    results = align_mol_name_with_results(mol_l=gen_mols,
                                        original_mol_name=name_l,
                                        results=results)

    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=4)
        
    summary = {}
    for metric_name, values in results.items():
        if isinstance(values, dict): # e.g. Ring proportion
            for key, value in values.items():
                summary[metric_name + str(key)] = value
                print(f'{metric_name + str(key)}: {np.around(value, 4)}')
        elif isinstance(values, list):
            median = np.nanmedian(values)
            summary[metric_name] = median # values can have nan
            print(f'Median {metric_name}: {np.around(median, 4)}')
        else: # float or int
            summary[metric_name] = values
            print(f'{metric_name}: {np.around(values, 4)}')
            