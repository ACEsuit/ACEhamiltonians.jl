from ase.db import connect
import numpy as np
from ase.db import connect
import h5py
import math
import numpy as np
import os


basis_definition = dict(H=("2s","1p"), C=("3s", "2p", "1d"), O=("3s", "2p", "1d"))

def split_orbitals(symbols:list, basis_definition:dict)->list:
    orbitals = [j for i in symbols for j in basis_definition[i]]
    orbitals = [i[-1] for i in orbitals for _ in range(int(i[:-1]))]
    return orbitals

def reorder_idx(orbitals:list)->list:
    ct = -1
    idx_list = []
    # orb_orders = dict(s = [1], p = [2,3,1], d = [3,4,2,5,1], f = [4,5,3,6,2,7,1]) #ChatGPT
    # orb_orders = dict(s = [1], p = [2,1,3], d = [4,2,1,3,5], f = [6,4,2,1,3,5,7]) #(0,-1,1)
    # orb_orders = dict(s = [1], p = [1,2,3], d = [1,2,3,4,5], f = [1,2,3,4,5,6,7])
    orb_orders = dict(s = [1], p = [3,1,2], d = [5,3,1,2,4], f = [7,5,3,1,2,4,6]) #(0,+1,-1)
    # orb_orders = dict(s = [1], p = [1,3,2], d = [4,2,1,3,5], f = [6,4,2,1,3,5,7])
    # orb_orders = dict(s = [1], p = [3,2,1], d = [4,2,1,3,5], f = [6,4,2,1,3,5,7])

    for orb in orbitals:
        idx_list.extend([ct+i for i in orb_orders[orb]])
        ct += len(orb_orders[orb])
    return idx_list

def orca_to_fhiaims_ordering(symbols: list, basis_definition:dict, matrix:np.ndarray)->np.ndarray:
    orbitals = split_orbitals(symbols, basis_definition)
    index_map = reorder_idx(orbitals)
    reordered_matrix = matrix[np.ix_(index_map, index_map)]
    return reordered_matrix

def prefactor_corection(symbols: list, basis_definition:dict)->np.ndarray:
    orbitals = split_orbitals(symbols, basis_definition)
    vec = []
    for orbital in orbitals:
        if orbital == 's':
            vec.append(1)
        elif orbital == 'p':
            vec.extend([1, 1, -1])
        elif orbital == 'd':
            vec.extend([1, 1, 1, -1, 1])
        elif orbital == 'f':
            vec.extend([1, 1, 1, 1, -1, 1, -1])
    vec = np.array(vec)
    map_mat = np.matmul(vec.reshape(-1, 1), vec.reshape(1, -1))
    return map_mat


def write_dataset_h5(db_path:str="./Data/dbs/schnorb_hamiltonian_water.db", outp_direc:str='./Data/Datasets'):

    db = connect(db_path)
    rows = db.select()

    Atoms_list = []
    H_data_list = []
    S_data_list = []
    energy_data_list = []
    force_data_list = []
    for row in rows:
        Atoms_list.append(row.toatoms())
        atoms = row.toatoms()
        symbols = atoms.get_chemical_symbols()
        map_mat = prefactor_corection(symbols, basis_definition)
        H_data_list.append(orca_to_fhiaims_ordering(symbols, basis_definition, row.data['hamiltonian'])*map_mat)
        S_data_list.append(orca_to_fhiaims_ordering(symbols, basis_definition, row.data['overlap'])*map_mat)
        # H_data_list.append(orca_to_fhiaims_ordering(symbols, basis_definition, row.data['hamiltonian']))
        # S_data_list.append(orca_to_fhiaims_ordering(symbols, basis_definition, row.data['overlap']))
        energy_data_list.append(row.data['energy'])
        force_data_list.append(row.data['forces'])

    n_geometries = len(Atoms_list)
    length_number = int(math.log10(n_geometries))+1

    file_name = os.path.basename(db_path).rstrip('.db')
    output_path = os.path.join(outp_direc, file_name+'.h5')

    with h5py.File(output_path, "w") as f:
        for i, (atoms, H, S, energy, force) in enumerate(zip(Atoms_list, H_data_list, S_data_list, energy_data_list, force_data_list)):
            gs = f.create_group(f'{i:0{length_number}}')
            gsd = gs.create_group('Data')
            gsdd = gsd.create_group('DoS')
            gsdd.create_dataset('broadening', dtype=np.float64)
            gsdd.create_dataset('energies', dtype=np.float64)
            gsdd.create_dataset('values', dtype=np.float64)
            gsd.create_dataset('H', data=np.expand_dims(H, axis=-1).T, dtype=np.float64)
            gsd.create_dataset('H_gama', data=H.T)
            gsd.create_dataset('S', data=np.expand_dims(S, axis=-1).T, dtype=np.float64)
            gsd.create_dataset('S_gama', data=S.T)
            # gsd.create_dataset('dm', data=np.expand_dims(dm, axis=-1).T, dtype=np.float64)
            # gsd.create_dataset('dm_gama', data=dm.T)
            gsd.create_dataset('fermi_level', dtype=np.float64)
            gsd.create_dataset('forces', data=force.T, dtype=np.float64)
            gsd.create_dataset('total_energy', data=energy, dtype=np.float64)
            gsi = gs.create_group('Info')
            gsib = gsi.create_group('Basis')
            gsib.create_dataset('14', dtype=np.float64)
            gsi.create_dataset('Translations', data=np.array([[0, 0, 0]], dtype=np.int64))
            gsi.create_dataset('k-points', data=np.array([[0., 0., 0.]]), dtype=np.float64)
            gss = gs.create_group('Structure')
            gss.create_dataset('atomic_numbers', data=atoms.numbers, dtype=np.int64)
            # gss.create_dataset('lattice', data=atoms.cell, dtype=np.float64)
            margins = atoms.positions.max(axis=0) - atoms.positions.min(axis=0)
            cell = np.zeros((3, 3))
            np.fill_diagonal(cell, margins+100)
            gss.create_dataset('lattice', data=cell, dtype=np.float64)
            gss.create_dataset('pbc', data=atoms.pbc, dtype=np.bool_)
            gss.create_dataset('positions', data=atoms.positions, dtype=np.float64)


db_path_list = [f'./Data/dbs/schnorb_hamiltonian_{i}.db' for i in ["water", "ethanol_dft"]]

for db_path in db_path_list:
    write_dataset_h5(db_path)