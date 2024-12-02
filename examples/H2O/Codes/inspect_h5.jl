using HDF5

db_path_list = ["../Data/Datasets/smpl10_md_w$(i).h5" for i in [6, 12, 21]]

db_path_list = ["./Data/Datasets/smpl10_md_w$(i).h5" for i in [6, 12, 21]]

file = h5open(db_path_list[1])