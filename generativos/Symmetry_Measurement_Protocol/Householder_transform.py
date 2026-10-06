import numpy as np
import argparse

def householder_transformation(pc):
    try:
        # Define the vector "v" for the YZ plane - Transformation in X
        v = np.array([1,0,0])
        
        # Create the Householder matrix
        H = np.eye(3) - 2 * np.outer(v,v) # Entity matrix

        # Apply transformation
        transformed_points = pc @ H.T # (N, 3) x (3, 3) - Matrix product

        # Save result
        #np.save("archivoreflejado.npy", transformed_points)

        return transformed_points # Return the transformed points

    except:
        raise Exception("Error while trying to do the householder transformation.")

'''
# Argument parser
parser = argparse.ArgumentParser(description="Compute chamfer distance")
parser.add_argument("file", type=str, help="Path to the first point cloud")

args = parser.parse_args()
file = args.file

# Load point cloud
points = np.load(file)

ht = householder_transformation(points)
np.save("archivoreflejado.npy", ht)
print("Done")'''