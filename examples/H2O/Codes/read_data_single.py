from sparseutils import from_coo_tuple
from ase.db import connect
import h5py
import math
import numpy as np
import os

def write_dataset_h5(db_path:str='../Data/dbs/dyn-wd-300K_3.db', outp_direc:str='../Data/Datasets'):

    db = connect(db_path)
    rows = db.select()

    Atoms_list = []
    H_data_list = []
    S_data_list = []
    for row in rows:
        Atoms_list.append(row.toatoms())
        H_data_list.append(from_coo_tuple(row.data['hamiltonian']))
        S_data_list.append(from_coo_tuple(row.data['overlap']))

    n_geometries = len(Atoms_list)
    length_number = int(math.log10(n_geometries))+1

    file_name = os.path.basename(db_path).rstrip('.db')
    output_path = os.path.join(outp_direc, file_name+'.h5')

    with h5py.File(output_path, "w") as f:
        for i, (atoms, H, S) in enumerate(zip(Atoms_list, H_data_list, S_data_list)):
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
            gsd.create_dataset('fermi_level', dtype=np.float64)
            gsd.create_dataset('forces', dtype=np.float64)
            gsd.create_dataset('total_energy', dtype=np.float64)
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


db_path_list = ["/home/c/chenqian3/ACEhamiltonians/H2O_PASHA/H2O_Pasha/Data/dbs/H2O_H_aims.db"]

for db_path in db_path_list:
    write_dataset_h5(db_path)