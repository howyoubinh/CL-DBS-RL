import math
import numpy as np 

def normalize(i, min, max):
    return (i - min) / (max - min)

def hasher(n, arr, min=0., max=0.1001): # previously 2.0001 / 0.1001
    back_flag = False
    out = np.zeros(n)
    for item in arr: 
        norm = normalize(item, min, max)
        index = math.floor(norm * n)

        # Clamp initial index to be safe
        if index >= n: index = n - 1
        if index < 0: index = 0

        while out[index] == 1:
            if index == n - 1:
                back_flag = True 

            if back_flag: 
                index -= 1 
            else:
                index += 1
            
            # If we've gone out of bounds, we can't place this spike.
            if index < 0 or index >= n:
                break
        
        back_flag = False 
        
        # Only place spike if we found a valid, empty slot.
        if index >= 0 and index < n and out[index] == 0:
            out[index] = 1

    return out

def unpack_hash(data, num_steps):
    unpacked = [hasher(num_steps, item['times']) for item in data] 
    arr = np.array(unpacked)

    return np.fliplr(np.rot90(arr, -1))

def max_len(mat_list, elem):
    max_elem = 0 
    for mat in mat_list:
        data = mat[elem]
        temp = max([len(item[0]) for item in data[0]])
        
        if(max_elem < temp):
            max_elem = temp 

    return max_elem

def constructor_hash(mat, num_steps=150):
    TH_APs_data = unpack_hash(mat['TH_APs'], num_steps)
    STNAPs_data = unpack_hash(mat['STN_APs'], num_steps)
    GPe_APs_data = unpack_hash(mat['GPe_APs'], num_steps)
    GPi_APs_data = unpack_hash(mat['GPi_APs'], num_steps)
    Striat_APs_indr_data = unpack_hash(mat['Striat_APs_indr'], num_steps)
    Striat_APs_dr_data = unpack_hash(mat['Striat_APs_dr'], num_steps)
    Cor_APs_data = unpack_hash(mat['Cor_APs'], num_steps) # Cor_APs = Cor_E + Cor_I

    build = np.append(TH_APs_data, STNAPs_data, axis=1)
    build = np.append(build, GPe_APs_data, axis=1)
    build = np.append(build, GPi_APs_data, axis=1)
    build = np.append(build, Striat_APs_indr_data, axis=1)
    build = np.append(build, Striat_APs_dr_data, axis=1)
    build = np.append(build, Cor_APs_data, axis=1)
        
    return build

