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

def f1_score_calc(file_names, mesh_path, sym_path, threshold_confidence):
    #Se lee el archivo con los nombres de los objetos
    with open(file_names, 'r') as f:
        names = f.readlines()
    
    #Se eliminan los saltos de línea y se separan los nombres de los objetos
    for i in range(len(names)):
        names[i] = names[i].strip().split('\t')[0]
    
    set_threshold = [0.05, 0.1, 0.15, 0.2]
    f1_mean = 0

    for threshold_inlier in set_threshold:
        threshold_plane = 0.05
        tp = 0
        fp = 0
        fn = 0

        #Se recorre cada objeto del groundtruth
        for i in range(len(names)): 
            
            #Leemos el archivo de simetrías groundtruth
            with open(mesh_path + '/' + names[i] + '.txt', 'r') as fil:
                num_gt_symmetries = int(fil.readline().strip())

                planes_gt = [] #Contiene los planos de GT, en forma de plano

                for j in range(num_gt_symmetries):
                    L = fil.readline().strip().split()
                    L = L[1:]
                    L = list(map(float, L))
                    
                    normal = np.array(L[:3])
                    point = np.array(L[3:6])

                    #convert normal and point to plane
                    d = -point.dot(normal)
                    plane = np.array([normal[0], normal[1], normal[2], d])
                    #convert plane to (1,4) array
                    plane = plane.reshape(1, 4)
                    planes_gt.append(plane)
                
            if os.path.exists(sym_path + '/' + names[i] + '_res.txt') == False:
                num_symmetries = 0
                predicted = []
            else:            
                with open(sym_path + '/' + names[i] + '_res.txt', 'r') as fil:
                    num_symmetries = int(fil.readline().strip())
                    predicted = []

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
                            if distPlanes(plane, predicted) > threshold_plane:
                                predicted.append(plane)
                    

            mask = np.zeros(len(planes_gt), dtype=bool)
            #Hasta aquí tenemos dos conjuntos de simetrías, el groundtruth y el predicho con alta confidencia
            for pred in predicted:
                for ind, gt in enumerate(planes_gt):
                    if mask[ind]:
                        continue
                    val1 = np.linalg.norm(pred-gt)
                    val2 = np.linalg.norm(pred+gt)
                    val = min(val1, val2)
                    if val < threshold_inlier:
                        mask[ind] = True
                        tp += 1
                    else:
                        fp += 1
            fn += np.sum(~mask)

        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2*(precision*recall)/(precision+recall)
        f1_mean += f1

    #print(f'Precision: {precision}')
    #print(f'Recall: {recall}')
    print(f'F1: {f1_mean/len(set_threshold)}')
    return f1_mean/len(set_threshold)

if __name__=='__main__':
    np.seterr(divide='ignore', invalid='ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_names', type=str, default='')
    parser.add_argument('--mesh_path', type=str, default='')
    parser.add_argument('--sym_path', type=str, default='')
    parser.add_argument('--threshold_confidence', type=float, default=1.00)
    opt = parser.parse_args()
    if opt.file_names and opt.mesh_path and opt.sym_path:
        f1_score_calc(opt.file_names, opt.mesh_path, opt.sym_path, opt.threshold_confidence)
    else:
        print('Usage: python metric_F1.py --file_names <file_names> --mesh_path <mesh_path> --sym_path <sym_path> --threshold_confidence <threshold_confidence>')
