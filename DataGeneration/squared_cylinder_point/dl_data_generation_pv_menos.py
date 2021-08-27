import numpy as np
import h5py
import os
from tqdm import tqdm
import pyvista as pv

def reduce(list1,factor):
      list2 = []
      list3 = []
      print(list1[0].shape)
      for i in range(int(list1[0].shape[0]/factor)):
         list2.append(list1[0][i*factor])
      print(len(list2))
      arr = np.array(list2)
      list3.append(arr)
      return list3

def reduce_points(array,factor):
   new_array = []
   for i in range(int(array.shape[0]/factor)):
      new_array.append(array[i*factor])

   new_array = np.array(new_array)
   return new_array

def padding(array):
	x=[]
	max = 50*80
	a =np.pad(array, [(0, max - array.shape[0] ) ], mode = 'constant', constant_values = -100.0)
	x.append(a)
	x= np.array(x).reshape((max,))
	return x

def extract_simulation_data(serial=None):
    data = {'Ux': [], 'Uy': [], 'p': [], 'Cx': [], 'Cy': [] }
    for entity in ['p', 'U', 'Cx', 'Cy']:
      for time in range(int(total_times)):
	          
        vtk_list = []
        path = 'simulation_data/' + serial + '/VTK/' + serial + '_' + str(int(time * 200 * deltat_write)) + '.vtk'
        mesh_cell_data = pv.read(path).cell_arrays 
        vtk_list.append(reduce_points(mesh_cell_data[entity],4))
        #vtk_list = reduce(vtk_list,10)        
	
        #boundaries
        for patch in ['inlet', 'obstacle', 'outlet', 'top']:
    
          path = 'simulation_data/' + serial + '/VTK/'+ patch + '/'+ patch + '_' + str(int(time * 200 * deltat_write)) + '.vtk'
          mesh_cell_data = pv.read(path).cell_arrays 
          vtk_list.append(reduce_points(mesh_cell_data[entity],2))

        if entity == 'U':
          data['Ux'].append(padding(np.concatenate(vtk_list)[:,0]))
          data['Uy'].append(padding(np.concatenate(vtk_list)[:,1]))

        else:
          extent = 0, 3, 0, 1
          data[entity].append(padding(np.concatenate(vtk_list)))

    return data



directory_list = next(os.walk('simulation_data/'))[1]
total_sim = len(directory_list)

directory_list_sim = next(os.walk('simulation_data/0/'))[1]
total_times = len(directory_list_sim) - 4 #constant, dynamicCode, system and VTK files
deltat_write = 1

train_shape = (total_sim, int(total_times), 50 * 80 , 5)

hdf5_path = 'dl_data/all_data_vanKarman_reduced.hdf5'
hdf5_file = h5py.File(hdf5_path, mode='w')
hdf5_file.create_dataset('sim_data', train_shape, np.float32)
hdf5_file.create_dataset('sim_no', (total_sim, 1), np.int16)

X, Y, XY = ([] for i in range(3))
count = 0

for sim_no in tqdm(directory_list):
    with open('simulation_data/' + sim_no + '/cellInformation') as file:
        for line in file:
            x, y = (int(i) for i in line.split())
            X.append(x), Y.append(y), XY.append(x * y)

    hdf5_file['sim_no'][count, 0] = int(sim_no)

    data = extract_simulation_data(sim_no)
    hdf5_file['sim_data'][count, ..., 0] = data['Ux']
    hdf5_file['sim_data'][count, ..., 1] = data['Uy']
    hdf5_file['sim_data'][count, ..., 2] = data['p']
    hdf5_file['sim_data'][count, ..., 3] = data['Cx']
    hdf5_file['sim_data'][count, ..., 4] = data['Cy']

    X.clear(), Y.clear(), XY.clear()
    count = count + 1

hdf5_file.close()
