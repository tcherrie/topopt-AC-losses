"""Linear algebra utilities

Provide utilities related to linear algebra.

Functions defined here:
- L2norm
- H1norm
- sparse
- vec
- split_mat
- split_vec
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

from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
import ngsolve as ngs
from utils.physics_symmetric import integrate

#%% Vector and matrices

def L2norm(field,
           state,
           zone : str = ".*"):
    """
    Compute the L2 norm of a field over a specified region.

    Evaluates :math:`\\|f\\|_{L^2} = \\sqrt{\\int_\\Omega |f|^2 \\, dx}`.

    Parameters
    ----------
    field : ngs.CoefficientFunction or ngs.GridFunction
        Field whose L2 norm is to be computed.
    state : dict
        Simulation state dictionary (provides mesh and material information).
    zone : str, optional
        Regular expression matching material names for the integration zone.
        Default is ``".*"`` (entire domain).

    Returns
    -------
    float
        The L2 norm of the field over the specified zone.

    See Also
    --------
    H1norm : Computes the H1 norm (includes gradient contribution).
    """
    return ngs.sqrt(integrate(ngs.Norm(field)**2, state, zone))


def H1norm(field,
           state,
           gradfield : callable = None,
           zone : str = ".*"):
    """
    Compute the H1 norm of a field over a specified region.

    Evaluates :math:`\\|f\\|_{H^1} = \\sqrt{\\int_\\Omega (|f|^2 + |\\nabla f|^2) \\, dx}`.

    If a precomputed gradient is provided via ``gradfield``, it is used instead
    of ``ngs.grad(field)`` (useful when the gradient is available from a
    previous computation).

    Parameters
    ----------
    field : ngs.CoefficientFunction or ngs.GridFunction
        Field whose H1 norm is to be computed.
    state : dict
        Simulation state dictionary (provides mesh and material information).
    gradfield : callable, optional
        Precomputed gradient expression for ``field``. If ``None``,
        :math:`\\nabla f` is computed automatically via ``ngs.grad(field)``.
    zone : str, optional
        Regular expression matching material names for the integration zone.
        Default is ``".*"`` (entire domain).

    Returns
    -------
    float
        The H1 norm of the field over the specified zone.

    See Also
    --------
    L2norm : Computes the L2 norm (no gradient contribution).
    """
    if gradfield is None:
        expr = ngs.Norm(field)**2 + ngs.Norm(ngs.grad(field))**2
    else:
        expr = ngs.Norm(field)**2 + ngs.Norm(gradfield)**2
    return ngs.sqrt(ngs.integrate(expr, state, zone))
    
def sparse(bf, 
           freedofs_rows = None, 
           freedofs_cols =None,
           type : str = "coo"):
    """
    Convert a bilinear form into a sparse matrix, optionally restricting
    it to selected rows and columns.

    Parameters
    ----------
    bf : object
        Bilinear form providing a ``COO()`` method that returns row indices,
        column indices, and corresponding matrix values.
    freedofs_rows : array-like, optional
        Row indices to retain. If ``None``, all rows are retained.
    freedofs_cols : array-like, optional
        Column indices to retain. If ``None``, all columns are retained.
    type : str, optional
        Type of sparse matrix (``csc``, ``csr``, ``coo``). Default is ``coo``.

    Returns
    -------
    scipy.sparse.spmatrix
        Sparse matrix constructed from the COO representation of ``bf``,
        optionally restricted to the specified rows and columns. The exact
        type depends on the ``type`` parameter (``csc``, ``csr``, or ``coo``).
    """
    r,c,vals  = bf.COO()
    if type.lower() == "csc":
        K = csc_matrix((vals,(r,c)))
    elif type.lower() == "csr":
        K = csr_matrix((vals,(r,c)))
    elif type.lower() == "coo":
        K = coo_matrix((vals,(r,c)))
    if freedofs_rows is not None:
        K = K[freedofs_rows,:]
    if freedofs_cols is not None:
        K = K[:,freedofs_cols]
    return K

def vec(lf, freedofs = None):
    """
    Convert a linear form into a column vector, optionally restricting it
    to selected degrees of freedom.

    Parameters
    ----------
    lf : object
        Linear form providing an ``Evaluate()`` method whose result can be
        converted to a NumPy array via ``FV().NumPy()``.
    freedofs : array-like, optional
        Indices of the degrees of freedom to retain. If ``None``, all
        degrees of freedom are retained.

    Returns
    -------
    numpy.ndarray
        Column vector containing the values of the linear form, optionally
        restricted to the specified degrees of freedom.
    """
    f = lf.Evaluate().FV().NumPy().reshape(-1,1)
    if freedofs is not None:
        f = f[freedofs]
    return f

#%% Condensation

def split_mat(K,
               freedofs,
               excluded_dofs = None,
               type : str = "coo"
               )-> tuple:
    """
    Split a matrix into submatrices corresponding to free and excluded
    degrees of freedom.

    Parameters
    ----------
    K : object
        Matrix or bilinear form to split. If not already a sparse matrix,
        it is converted using ``sparse()``.
    freedofs : array-like
        Boolean mask or indices identifying the free degrees of freedom.
    excluded_dofs : array-like, optional
        Boolean mask or indices identifying the excluded degrees of freedom.
        If ``None``, the complement of ``freedofs`` is used.

    Returns
    -------
    tuple
        Four sparse matrix blocks ``(A, B, C, D)`` corresponding to the
        partition

        ``K = [[A, B], [C, D]]``

        where ``A`` contains free-to-free entries, ``B`` free-to-excluded
        entries, ``C`` excluded-to-free entries, and ``D`` excluded-to-
        excluded entries.
    """
    if excluded_dofs is None:
        excluded_dofs = ~freedofs
        
    K = sparse(K, type = type)
    A = K[freedofs,:][:,freedofs]
    B = K[freedofs,:][:,excluded_dofs]
    C = K[excluded_dofs,:][:,freedofs]
    D = K[excluded_dofs,:][:,excluded_dofs]
    
    return A, B, C, D

def split_vec(F,
              freedofs,
              excluded_dofs = None,
              )-> tuple:
    """
    Split a vector into components corresponding to free and excluded
    degrees of freedom.

    Parameters
    ----------
    F : object
        Linear form to split. It is converted to a column vector using
        ``vec()``.
    freedofs : array-like
        Boolean mask or indices identifying the free degrees of freedom.
    excluded_dofs : array-like, optional
        Boolean mask or indices identifying the excluded degrees of freedom.
        If ``None``, the complement of ``freedofs`` is used.

    Returns
    -------
    tuple
        Two vectors ``(F1, F2)`` where ``F1`` contains the entries
        corresponding to the free degrees of freedom and ``F2`` contains
        the entries corresponding to the excluded degrees of freedom.
    """
    if excluded_dofs is None:
        excluded_dofs = ~freedofs
        
    F = vec(F)
    F1 = F[freedofs]
    F2 = F[excluded_dofs]
    
    return F1, F2
