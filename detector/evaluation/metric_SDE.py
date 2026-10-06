import argparse
import os
import scipy.io
import numpy as np
import gpytoolbox as gpy
import glob

def calplaneloss(plane, vertices, faces, points):
    #Add a column to points filled with 1s
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    lam = points.dot(plane.T)
    
    planepoints = points - 2*lam*plane
    
    d, ind, b = gpy.squared_distance(planepoints[:,0:3], vertices, faces, use_aabb=True, use_cpp=True)
    loss = np.mean(d)

    return loss

def distPlanes(plane, planes):
    minDist = 1000000
    for p in planes:
        val1 = np.linalg.norm(plane-p)
        val2 = np.linalg.norm(plane+p)
        val = min(val1, val2)
        if val < minDist:
            minDist = val
    
    return minDist

def sde_score_calc(file_names, mesh_path, sym_path, threshold_confidence):
    with open(file_names, 'r') as f:
        names = f.readlines()
    
    for i in range(len(names)):
        names[i] = names[i].strip().split('\t')[0]
    
    threshold_inlier = 0.2
    sde = []
    
    maxName = ''
    maxSDE = 0

    for i in range(len(names)): #Se recorre cada objeto del groundtruth
        fil = glob.glob(mesh_path + '/' + names[i] + '.obj', recursive=True)
        
        v,f = gpy.read_mesh(fil[0])
        sample = gpy.random_points_on_mesh(v, f, 1000)

        if os.path.exists(sym_path + '/' + names[i] + '_res.txt') == False:
            continue

        with open(sym_path + '/' + names[i] + '_res.txt', 'r') as fil:
            num_symmetries = int(fil.readline().strip())
            aux = []

            for j in range(num_symmetries):
                L = fil.readline().strip().split()
                L = L[1:]
                L = list(map(float, L))
                
                normal = np.array(L[:3])
                point = np.array(L[3:6])
                
                confidence = L[6]
                if confidence >= threshold_confidence:
                    
                    #convert normal and point to plane
                    d = -point.dot(normal)
                    plane = np.array([normal[0], normal[1], normal[2], d])
                
                    #convert plane to (1,4) array
                    plane = plane.reshape(1, 4)
                    if distPlanes(plane, aux) > threshold_inlier:
                        aux.append(plane)
        
        for plane in aux:
            loss = calplaneloss(plane, v, f, sample)
            sde.append(loss)
            if loss > maxSDE:
                maxSDE = loss
                maxName = names[i]

    sde = np.array(sde)
    print(f'SDE: {np.mean(sde)}')
    print(f'Min SDE: {np.min(sde)}')
    print(f'Max SDE: {np.max(sde)}')
    print(f'Max SDE alone: {maxSDE} {maxName}')
    return np.mean(sde)

if __name__=='__main__':
    np.seterr(divide='ignore', invalid='ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_names', type=str, default='')
    parser.add_argument('--mesh_path', type=str, default='')
    parser.add_argument('--sym_path', type=str, default='')
    parser.add_argument('--threshold_confidence', type=float, default=1.00)
    opt = parser.parse_args()
    if opt.file_names and opt.mesh_path and opt.sym_path:
        sde_score_calc(opt.file_names, opt.mesh_path, opt.sym_path, opt.threshold_confidence)
    else:
        print("Usage: python metric_SDE.py --file_names <path_to_file_names.txt> --mesh_path <path_to_meshes_folder> --sym_path <path_to_symmetries_folder> --threshold_confidence <confidence_threshold>")