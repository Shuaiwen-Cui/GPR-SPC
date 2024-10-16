"""
This script is to build up the model for the event simulation.
Author: Shuaiwen Cui
Date: May 14, 2024

Log:
-  May 14, 2024: initial version

"""

import numpy as np
import sys

def model_build(nDOF=10, me=29.13, ke=1190e3, zeta=0.00097, damaged_floor=None, reduction_factor=0.0):
    '''
    Description: This function is to build up the model for the event simulation.
    
    Parameters:
    
    nDOF: int, the number of degrees of freedom (default is 10)
    
    me: float, the mass for each floor (default is 29.13)
    
    ke: float, the stiffness unit N/m (default is 1190e3)
    
    zeta: float, the damping ratio - actual damp / critical damp (default is 0.00097)
    
    damaged_floor: int or None, the index of the floor where stiffness reduction occurs (default is None, meaning no damage)
    
    reduction_factor: float, the factor by which the stiffness is reduced for the damaged floor (default is 0.0, meaning no reduction)
    
    Returns:
    
    M: numpy array, the mass matrix
    
    K: numpy array, the stiffness matrix
    
    C: numpy array, the damping matrix
    '''
    
    # M - mass matrix - diagonal matrix with mass me for each floor
    M = np.zeros((nDOF, nDOF))
    for i in range(nDOF):
        M[i, i] = me

    # K - stiffness matrix
    K = np.zeros((nDOF, nDOF))
    for i in range(nDOF):
        K[i, i] = 2 * ke
        if i > 0:
            K[i, i - 1] = -ke
        if i < nDOF - 1:
            K[i, i + 1] = -ke
    
    K[nDOF - 1, nDOF - 1] = ke

    # Apply stiffness reduction to the damaged floor
    if damaged_floor is not None and 0 <= damaged_floor < nDOF:
        reduction = 1 - reduction_factor
        if reduction < 0:
            raise ValueError("Reduction factor must result in positive stiffness.")
        
        # Apply reduction to the damaged floor itself (on diagonal)
        K[damaged_floor, damaged_floor] *= reduction
        
        # Apply reduction to the stiffness between damaged floor and adjacent floors
        if damaged_floor > 0:
            K[damaged_floor, damaged_floor - 1] *= reduction  # Connection to the floor below
            K[damaged_floor - 1, damaged_floor] *= reduction  # Symmetry for lower floor
        if damaged_floor < nDOF - 1:
            K[damaged_floor, damaged_floor + 1] *= reduction  # Connection to the floor above
            K[damaged_floor + 1, damaged_floor] *= reduction  # Symmetry for upper floor

    # C - damping matrix

    # Calculate the eigenvalues and eigenvectors of the system
    INV_M = np.linalg.inv(M)
    INV_M_K = np.dot(INV_M, K)
    eigenvalues, eigenvectors = np.linalg.eig(INV_M_K)

    # Characteristic frequency
    omega = np.sqrt(eigenvalues)
    omega = np.diag(omega)

    # Damping matrix
    C = 2 * zeta * me * np.linalg.inv(np.transpose(eigenvectors)) @ omega @ np.linalg.inv(eigenvectors)
    
    # return the matrices
    return M, K, C


# testing
if __name__ == '__main__':
    M, K, C = model_build()
    print('Model built successfully!')
    print('Mass matrix: ', M)
    print('Stiffness matrix: ', K)
    print('Damping matrix: ', C)

