"""Linear algebra utilities

Provide utilities related to linear algebra

Functions defined here:
- sparse
- vec
- schur_complement


A Large Language Model (GPT-5.5 from Open AI, free version) was used to help with the code and generate
the docstrings of the functions. The authors have written the initial code, carefully checked and post-edited
the content of this file, and take full responsability of its content.
This software is provided "as is" without warranty of any kind, and can be used, shared ad modified under the terms of GNU LGPL license.
"""

#%% Metadata

__author__ = "Théodore Cherrière"
__copyright__ = "Copyright 2026, CentraleSupélec, SAFRAN"
__credits__ = ["Théodore Cherrière", "Alexis Pons", "Guillaume Krebs",
                    "Adrien Mercier", "Loucif Benmamas", "Sulivan Küttler"]
__license__ = "GNU LGPL"
__version__ = "0.2"
__maintainer__ = "Théodore Cherrière"
__email__ = "theodore.cherriere@centralesupelec.fr"
__status__ = "Development"

#%% Import

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv

#%% Vector and matrices

def sparse(bf, freedofs_rows = None, freedofs_cols=None):
    r,c,vals  = bf.COO()
    K = csr_matrix((vals,(r,c)))
    if freedofs_rows is not None:
        K = K[freedofs_rows,:]
    if freedofs_cols is not None:
        K = K[:,freedofs_cols]
    return K

def vec(lf, freedofs = None):
    f = lf.Evaluate().FV().NumPy().reshape(-1,1)
    if freedofs is not None:
        f = f[freedofs]
    return f

#%% Condensation

def split_mat(K,
               freedofs,
               excluded_dofs = None,
               )-> tuple:
    
    if excluded_dofs is None:
        excluded_dofs = ~freedofs
        
    K = sparse(K)
    A = K[freedofs,:][:,freedofs]
    B = K[freedofs,:][:,excluded_dofs]
    C = K[excluded_dofs,:][:,freedofs]
    D = K[excluded_dofs,:][:,excluded_dofs]
    
    return A, B, C, D

def split_vec(F,
              freedofs,
              excluded_dofs = None,
              )-> tuple:
    
    if excluded_dofs is None:
        excluded_dofs = ~freedofs
        
    F = vec(F)
    F1 = F[freedofs]
    F2 = F[excluded_dofs]
    
    return F1, F2
