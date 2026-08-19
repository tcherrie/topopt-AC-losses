"""Physics utilities

Provide utilities related to the physical formulation and associated solvers.

Functions defined here:
- state2gfu                 (helper)
- gfu2state                 (helper)
- Curl                      (helper)
- surface                   (helper)
- integrate                 (helper)
- average_property          (helper)
- magnetization_halbach     (helper)
- solve_magnetoharmonic     (main physical solvers)
- solve_magnetoharmonic2     (main physical solvers), to check consistency of formulations
- newton                    (main physical solvers)
- dual_trace                (post-processing)
- electric_field            (post-processing)
- electric_field2           (post-processing), to check consistency of formulations
- current_density           (post-processing)
- joule_losses              (post-processing)
- matrix_arkkio             (post-processing)
- average_torque            (post-processing)


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

import ngsolve as ngs
import numpy as np
from time import time
import scipy.sparse as sp
from re import match


#%% Helpers


def state2gfu(state : dict) -> ngs.GridFunction:
    """
    Extract the solution state from a simulation result dictionary and return
    it as an NGSolve GridFunction.

    The function reconstructs a GridFunction associated with the finite element
    space stored in ``state["info"]["fes"]`` and copies the magnetic vector
    potential and, when present, the electric potentials of the conducting
    bundles from the solution dictionary.

    Parameters
    ----------
    state : dict
        Simulation state dictionary, typically returned by a magneto-harmonic
        solver. It is expected to contain:

        ``state["info"]["fes"]``
            The NGSolve finite element space associated with the solution.

        ``state["solution"]["a"]``
            The magnetic vector potential GridFunction.

        ``state["solution"]["e"]``
            A dictionary mapping conducting bundle names to their corresponding
            electric potential GridFunctions.

        ``state["bundles"]``
            Iterable containing the names of the conducting bundles. The
            corresponding electric potentials are stored in the components
            following the magnetic vector potential.

    Returns
    -------
    ngs.GridFunction
        A GridFunction defined on ``state["info"]["fes"]`` containing the
        complete solution state. The first component contains the magnetic
        vector potential, while subsequent components contain the electric
        potentials associated with the conducting bundles.

        If no bundle electric potentials are present, the magnetic vector
        potential is copied directly into the returned GridFunction.

    Notes
    -----
    The function supports both compound finite element spaces, where the
    solution consists of multiple components, and single-component spaces
    containing only the magnetic vector potential.
    """

    fes = state["info"]["fes"]
    sol = ngs.GridFunction(fes)
    try:
        sol.components[0].vec.data = state["solution"]["a"].vec
        for i, bundle in enumerate(state["bundles"]):
                    sol.components[i + 1].vec.data = state["solution"]["e"][bundle].vec
    except:
        sol.vec.data = state["solution"]["a"].vec
    return sol


def gfu2state(gfu : ngs.GridFunction,
              state : dict) -> ngs.GridFunction:
    """
    Update a simulation state dictionary from an NGSolve GridFunction.

    The function creates a shallow copy of the provided state dictionary and
    replaces the solution fields with the components of ``gfu``. For a
    compound finite element space, the first component is interpreted as the
    magnetic vector potential and the subsequent components as the electric
    potentials associated with the conducting bundles. For a single-component
    space, ``gfu`` is interpreted directly as the magnetic vector potential.

    Parameters
    ----------
    gfu : ngs.GridFunction
        NGSolve GridFunction containing the solution state. For a compound
        finite element space, its components are expected to be ordered as:

        - component 0: magnetic vector potential ``a``.
        - components 1..N: electric potentials ``e`` for the conducting bundles.

    state : dict
        Existing simulation state dictionary. It is expected to contain:

        ``state["solution"]``
            Dictionary containing the magnetic vector potential ``"a"`` and,
            when applicable, the bundle electric potentials ``"e"``.

        ``state["bundles"]``
            Iterable containing the names of the conducting bundles, in the
            same order as the corresponding components of ``gfu``.

    Returns
    -------
    dict
        A shallow copy of ``state`` with its solution fields updated from
        ``gfu``. The original top-level state dictionary is not modified.

    Notes
    -----
    If the GridFunction is not a compound space, the entire ``gfu`` is stored
    as the magnetic vector potential under ``state_copy["solution"]["a"]``.
    """
    state_copy = state.copy()
    try:
        state_copy["solution"]["a"] = gfu.components[0]
        for i, bundle in enumerate(state["bundles"]):
            state_copy["solution"]["e"][bundle] = gfu.components[i + 1]
    except:
        state_copy["solution"]["a"] = gfu
    return state_copy


def Curl(u: ngs.GridFunction | ngs.CoefficientFunction
         ) -> ngs.CoefficientFunction:
    """
    Compute the 2D curl operator.

    This function returns the 2D vector curl of a scalar field, corresponding
    to the curl of a z-directed vector quantity into the xy-plane.

    Parameters
    ----------
    u : ngs.GridFunction or ngs.CoefficientFunction
        Scalar field representing the z-component of a vector field.

    Returns
    -------
    ngs.CoefficientFunction
        2D vector field representing Curl(u), defined as:

        b_xy = Curl(a_z)

    Notes
    -----
    - Implements the 2D curl as a 90-degree rotation of the gradient.
    - Equivalent to applying the operator:
    
        [[ 0,  1],
         [-1,  0]] ∇u
    """
    return ngs.CF(((0, 1), (-1, 0)), dims=(2, 2)) * ngs.grad(u)


def surface(zone: str,
            mesh: ngs.comp.Mesh
            ) -> float:
    """
    Compute the surface measure of a specified mesh region.

    Parameters
    ----------

    zone : str
        Name of the material region whose surface measure is to be computed.
        The region is selected using ``mesh.Materials(zone)``.
    
    mesh : ngs.comp.Mesh
            NGSolve computational mesh on which the surface is evaluated.

    Returns
    -------
    float
        Surface measure of the specified region, computed by integrating the
        constant function ``1`` over the selected mesh region.

    Notes
    -----
    The integration order is set to the curve order of the mesh using
    ``mesh.GetCurveOrder()``.
    """
    return ngs.Integrate(1, mesh.Materials(zone), order = mesh.GetCurveOrder())

def integrate(property: ngs.GridFunction | ngs.CoefficientFunction,
              results: dict,
              zone: str = ".*",
              order_min = 5,
              ) -> float:
    """
    Compute the integral of a field over a given mesh region.

    Parameters
    ----------
    property : ngs.GridFunction or ngs.CoefficientFunction
        Field to be averaged over the domain.

    results : dict
        Simulation results dictionary containing at least the FESpace
        and mesh information under `results["info"]["fes"]`.

    zone : str, optional
        Material or region selector (regex-style). Default is ".*" (whole domain).

    Returns
    -------
    float
        Spatial average of the given property over the selected zone.

    Notes
    -----
    - The integration is performed using NGSolve integration utilities.
    """

    mesh = results["info"]["fes"].mesh

    # Compute integral of the field over the region and normalize
    try: order = max([mesh.GetCurveOrder(), 2*results["info"]["fes"].components[0].globalorder + 1 , order_min])
    except: order = max([mesh.GetCurveOrder(), 2*results["info"]["fes"].globalorder + 1 , order_min])
    
    return ngs.Integrate(property, mesh.Materials(zone), order = order)

def average_property(property: ngs.GridFunction | ngs.CoefficientFunction,
                     results: dict,
                     zone: str = ".*",
                     order_min = 5,
                     ) -> float:
    """
    Compute the spatial average of a field over a given mesh region.

    This function evaluates the mean value of a scalar (or scalar-valued)
    finite-element field over a selected geometric zone.

    Parameters
    ----------
    property : ngs.GridFunction or ngs.CoefficientFunction
        Field to be averaged over the domain.

    results : dict
        Simulation results dictionary containing at least the FESpace
        and mesh information under `results["info"]["fes"]`.

    zone : str, optional
        Material or region selector (regex-style). Default is ".*" (whole domain).

    Returns
    -------
    float
        Spatial average of the given property over the selected zone.

    Notes
    -----
    - The average is computed as:
        ⟨f⟩ = (∫_Ω f dx) / (∫_Ω 1 dx)
    - The integration is performed using NGSolve integration utilities.
    """

    mesh = results["info"]["fes"].mesh
    return integrate(property, results, zone, order_min) / surface(zone, mesh)

def magnetization_halbach(br: float = 1,
                          mu: float = 4e-7 * ngs.pi,
                          p: int = 4
                          ) -> ngs.CoefficientFunction:
    """
    Construct a complex-valued Halbach magnetization field.

    This function defines the magnetization of a Halbach array in polar
    coordinates, producing a rotating magnetization vector with harmonic
    order controlled by the pole pair number.

    Parameters
    ----------
    br : float, optional
        Remanent flux density magnitude.

    mu : float, optional
        Magnetic permeability (default is vacuum permeability).

    p : int, optional
        Number of pole pairs defining the spatial harmonic order.

    Returns
    -------
    ngs.CoefficientFunction
        Complex-valued 2D magnetization field in Cartesian components.

    Notes
    -----
    - The field is expressed in polar coordinates using the angle
      alpha = atan2(y, x).
    - The Halbach distribution is encoded using complex exponentials,
      representing a rotating magnetization pattern.
    - Commonly used to model ideal permanent magnet arrays in rotating
      electrical machines.
    """

    alpha = ngs.atan2(ngs.y, ngs.x)

    return br / mu * ngs.CF((ngs.exp(-1j * (p - 1) * alpha),
                             ngs.exp(-1j * ((p - 1) * alpha + ngs.pi / 2)) )
    )

#%% Main physical solver

def solve_magnetoharmonic(
    fes: ngs.FESpace,  # finite element space
    reluctivity: ngs.GridFunction | ngs.CoefficientFunction,  # magnetic reluctivity
    magnetization: ngs.GridFunction | ngs.CoefficientFunction,  # complex magnetization
    frequency: float,   # electrical frequency
    supply: dict, # supply of electrical conductors
    conductivity: ngs.GridFunction | ngs.CoefficientFunction | float = 6e7,    # conductivity
    Kinv=None,  # optional precomputed inverse system matrix
    solver: str = "pardiso",  # linear solver type
    bonus_intorder : int = 3,       # bonus order of integration in the assembly
    verbose: int = 0,  # for controlling print statements
    taskmanager: bool = True, # for paralelizing assembly process
    # Slot model - mixed boundary conditions
    # on selected boundary, apply : robin_coeff*a           + (1-robin_coeff)*nu*da/dn
    #                             = robin_coeff*a_dirichlet + (1-robin_coeff)*Trace(h_neumann)
    robin_bnd   : str = None, # Boundary name where apply mixed condition
    robin_coeff : ngs.GridFunction | ngs.CoefficientFunction | float = 0,     # Robin coefficient in [0 = neumann, 1 = dirichlet)
    a_dirichlet : ngs.GridFunction | ngs.CoefficientFunction | float = 0,     # non-zero Dirichlet, in fes
    h_tangential   : ngs.GridFunction | ngs.CoefficientFunction | float = 0,  # Neumann trace
    fix1dof : bool = False,   # if true, fix one dof to a_dirichlet value
    ) -> dict:
    """
    Solve a linear magneto-quasistatic problem in the frequency domain.

    This function assembles and solves the finite-element formulation of a
    time-harmonic magneto-quasistatic problem using the magnetic vector
    potential. Eddy currents in electrical conductors are modeled through a
    coupled formulation with Lagrange multipliers enforcing prescribed bundle
    currents. The solver supports permanent magnet excitation, spatially varying
    material properties, optional mixed (Robin) boundary conditions, and
    pre-factorized system matrices for efficient repeated simulations.

    Parameters
    ----------
    fes : ngs.FESpace
        Finite element space for the magnetic vector potential.

    reluctivity : ngs.GridFunction or ngs.CoefficientFunction
        Magnetic reluctivity (1/μ). May vary spatially. Magnetic materials are
        assumed linear.

    magnetization : ngs.GridFunction or ngs.CoefficientFunction
        Complex-valued magnetization source term, typically representing
        rotating permanent magnets.

    frequency : float
        Electrical excitation frequency [Hz].

    supply : dict
        Electrical supply of the conducting bundles. Keys correspond to 
        conductor bundle names and values to the imposed complex currents [A].

    conductivity : float or ngs.GridFunction or ngs.CoefficientFunction, optional
        Electrical conductivity distribution.

    Kinv : optional
        Precomputed inverse (or factorization) of the system matrix. Providing
        this argument skips assembly and factorization of the stiffness matrix,
        greatly accelerating repeated solves with unchanged material properties.

    solver : str, optional
        Direct solver backend used to factorize the system matrix (default:
        ``"pardiso"``).

    bonus_intorder : int, optional
        Additional integration order used for conductivity and magnetization
        terms to improve quadrature accuracy.

    verbose : int, optional
        Verbosity level. Values greater than zero print timing information for
        each stage of the solve.

    taskmanager : bool, optional
        If True, assembles the finite element matrices using NGSolve's
        ``TaskManager`` for parallel execution.

    robin_bnd : str, optional
        Name of the boundary where mixed (Robin) boundary conditions are applied.
        If ``None``, the default natural (Neumann) boundary condition is used.

    robin_coeff : float or ngs.GridFunction or ngs.CoefficientFunction, optional
        Robin coefficient α satisfying

            α a + (1-α) ν curl(a) x n
            = α a_dirichlet + (1-α) h_tangential.

        Typical α values are:
        - 0 : pure Neumann condition
        - values approaching 1 : approximate Dirichlet condition.

    a_dirichlet : float or ngs.GridFunction or ngs.CoefficientFunction, optional
        Prescribed magnetic vector potential used in the Robin boundary
        condition.

    h_tangential : float or ngs.GridFunction or ngs.CoefficientFunction, optional
        Prescribed tangential magnetic field trace used in the Robin boundary
        condition.

    fix1dof : bool, optional
        If True, fixes one degree of freedom of the magnetic vector potential to
        remove the null space associated with pure Neumann problems.

    Returns
    -------
    results : dict
        Dictionary containing

        ``"solution"``
            Solution fields:
            - ``"a"``: magnetic vector potential.
            - ``"E"``: bundle time integral of electric potentials.

        ``"test"``
            Test functions used in the weak formulation.

        ``"bundles"``
            Names of the conducting bundles.

        ``"info"``
            Simulation metadata including the finite element space, material
            properties, excitation, solver information, cached matrix inverse,
            and execution times.

    Notes
    -----
    - The formulation assumes linear magnetostatics in the frequency domain.
    - Eddy currents are modeled only inside conducting bundles.
    - Total bundle currents are imposed using Lagrange multipliers.
    - Reusing ``Kinv`` is highly recommended for repeated simulations with
    varying currents or magnetization, as it avoids reassembling and
    refactorizing the system matrix.
    - When ``fix1dof=True``, one degree of freedom is constrained to eliminate
    the singularity associated with pure Neumann boundary conditions.
    """
    
    if verbose >= 1:
        print(f"-- START MAGNETOHARMONIC SOLVER --")
        print(f"Solver : {solver.lower()}")
        
    t0 = time()
    jw = 1j * 2 * ngs.pi * frequency
    K = None
    txtref = f"Matrix decomposition with {solver}... "
    
    if verbose >= 1:
        txt = "Setup function space... "
        print(txt, *[""]*(len(txtref)-len(txt)), end="")
        
    # Identify conductor bundles
    bundles = supply.keys()

    mesh = fes.mesh
    if fix1dof:
        dummy = ngs.GridFunction(fes)
        dummy.Set(1)
        ind = np.nonzero(dummy.vec.FV().NumPy())[0][0]
        fes.FreeDofs()[ind] = False
        
    # Normalize supplied currents by bundle volume
    Jcplx = {
        bundle: supply[bundle] / surface(bundle, mesh)
        for bundle in bundles
    }

    # Extend FE space with Lagrange multipliers (bundle constraints)
    for _ in bundles:
        fes *= ngs.NumberSpace(mesh, complex=True)

    # Define trial and test functions
    trials = fes.TrialFunction()
    tests = fes.TestFunction()
    if type(trials) is list:
        a, a_ = trials[0], tests[0]
    else:
        a, a_ = trials, tests

    E = {bundle: trials[i + 1] for i, bundle in enumerate(bundles)}
    E_ = {bundle: tests[i + 1] for i, bundle in enumerate(bundles)}

    t_fes = time() - t0
    
    if verbose >= 1:
        print(f"done in {t_fes*1000:.0f} ms")

    # ------------------------------------------------------------
    # Assemble system matrix
    # ------------------------------------------------------------
    t_assembly = 0
    t_decomposition = 0
    if (Kinv is None) or fix1dof:
        tic = time()
        if verbose >= 1:
            txt = "Assemble matrix... "
            print(txt, *[""]*(len(txtref) - len(txt)), end="")

        bf = Curl(a_) * reluctivity * Curl(a) * ngs.dx
        
        # Optional Robin term
        if robin_bnd is not None:
            bf += a_ * robin_coeff / (1-robin_coeff) * a * ngs.ds(robin_bnd)

        # Eddy-current + constraint coupling in each bundle
        for bundle in bundles:
            bf += a_ * conductivity * jw * (a + E[bundle]) * ngs.dx(bundle, bonus_intorder = bonus_intorder)
            bf += E_[bundle] * conductivity * jw * (a + E[bundle]) * ngs.dx(bundle, bonus_intorder = bonus_intorder)

        # Assemble matrix
        if taskmanager:
            with ngs.TaskManager():
                K = ngs.BilinearForm(bf, symmetric = True).Assemble().mat
        else:
            K = ngs.BilinearForm(bf, symmetric = True).Assemble().mat

        t_assembly = time() - tic
        
        if verbose >= 1:
            print(f"done in {t_assembly*1000:.0f} ms")

        
        # Factorize system
        if verbose >= 1:
            txt = txtref
            print(txtref, end="")
    
        tic = time()
        if (Kinv is None):
            if solver.lower() != "superlu":
                Kinv = K.Inverse(fes.FreeDofs(), inverse=solver)
            else:
                rows,cols,vals = K.COO()
                Ksp =  sp.csc_matrix((vals,(rows,cols)))
                Ksp = Ksp[fes.FreeDofs(),:][:,fes.FreeDofs()]
                Kinv = sp.linalg.splu(Ksp)          
                            
        t_decomposition = time() - tic
        
        if verbose >= 1:
            print(f"done in {t_decomposition*1000:.0f} ms")

    # ------------------------------------------------------------
    # Assemble right-hand side
    # ------------------------------------------------------------
    if verbose >= 1:
        txt = "Assemble right hand side... "
        print(txt, *[""]*(len(txtref) - len(txt)), end="")
    tic = time()
    lf = Curl(a_) * magnetization * ngs.dx(bonus_intorder = bonus_intorder)
    
    # Optional Robin term
    if robin_bnd is not None:
        lf += a_ * robin_coeff / (1-robin_coeff) * a_dirichlet * ngs.ds(robin_bnd)
        lf += a_ * h_tangential * ngs.ds(robin_bnd)
     
    for bundle in bundles:
        lf += -E_[bundle] * Jcplx[bundle] * ngs.dx(bundle)
        
    F = ngs.LinearForm(lf).Assemble().vec
    t_rhs = time() - tic
    if verbose >= 1:
        print(f"done in {t_rhs*1000:.0f} ms")

    # ------------------------------------------------------------
    # Solve linear system
    # ------------------------------------------------------------
    if verbose >= 1:
        txt = "Solve the problem... "
        print(txt, *[""]*(len(txtref) - len(txt)), end="")
    tic = time()
    sol = ngs.GridFunction(fes)
    res = sol.vec.CreateVector()
    if fix1dof:
        sol.vec.data[ind] = a_dirichlet.vec[ind]
        res.data = K*sol.vec - F
    else:
        res.data =  -F

    if type(Kinv) != sp.linalg.SuperLU:
        sol.vec.data -= Kinv * res
    else:
        spsol = Kinv.solve(res.FV().NumPy()[fes.FreeDofs()])
        sol.vec.data.FV().NumPy()[fes.FreeDofs()] = - spsol
    
    t_solve = time() - tic

    if verbose >= 1:
        print(f"done in {t_solve*1000:.0f} ms")

    # ------------------------------------------------------------
    # Package results
    # ------------------------------------------------------------
    if verbose >= 1:
        txt = "Pack the results... "
        print(txt, *[""]*(len(txtref) - len(txt)), end="")

    time_total = time() -t0
    if type(trials) is list:
        solution = {
                "a": sol.components[0],
                "E": {
                    bundle: sol.components[i + 1]
                    for i, bundle in enumerate(bundles)
                }
                }
    else:
        solution = {
                "a": sol,
                "E": {}}
    results = {
        "solution": solution,
        "trial": {"a": a, "E": E},
        "test": {"a": a_, "E": E_},
        "bundles": bundles,
        "info": {
            "fes": fes,
            "reluctivity": reluctivity,
            "magnetization": magnetization,
            "frequency": frequency,
            "supply": supply,
            "conductivity": conductivity,
            "Kinv": Kinv,
            "K" : K,
            "F": -res,
            "solver": solver,
            "walltime" : {"fes": t_fes, 
                          "assembly": t_assembly, 
                          "decomposition": t_decomposition, 
                          "rhs": t_rhs, 
                          "solve": t_solve,
                          "total":time_total}
            },
        }  

    if verbose >= 1:
        print(f"total time: {time_total:.3f} s")
        print(f"-- END MAGNETOHARMONIC SOLVER --")
       
    return results


def solve_magnetoharmonic2(
    fes: ngs.FESpace,  # finite element space
    reluctivity: ngs.GridFunction | ngs.CoefficientFunction,  # magnetic reluctivity
    magnetization: ngs.GridFunction | ngs.CoefficientFunction,  # complex magnetization
    frequency: float,   # electrical frequency
    supply: dict, # supply of electrical conductors
    conductivity: ngs.GridFunction | ngs.CoefficientFunction | float = 6e7,    # conductivity
    Kinv=None,  # optional precomputed inverse system matrix
    solver: str = "pardiso",  # linear solver type
    bonus_intorder : int = 3,       # bonus order of integration in the assembly
    verbose: int = 0,  # for controlling print statements
    taskmanager: bool = True, # for paralelizing assembly process
    # Slot model - mixed boundary conditions
    # on selected boundary, apply : robin_coeff*a           + (1-robin_coeff)*nu*da/dn
    #                             = robin_coeff*a_dirichlet + (1-robin_coeff)*Trace(h_neumann)
    robin_bnd   : str = None, # Boundary name where apply mixed condition
    robin_coeff : ngs.GridFunction | ngs.CoefficientFunction | float = 0,     # Robin coefficient in [0 = neumann, 1 = dirichlet)
    a_dirichlet : ngs.GridFunction | ngs.CoefficientFunction | float = 0,     # non-zero Dirichlet, in fes
    h_tangential   : ngs.GridFunction | ngs.CoefficientFunction | float = 0,  # Neumann trace
    fix1dof : bool = False,   # if true, fix one dof to a_dirichlet value
    ) -> dict:
    """
    Solve a linear magneto-quasistatic problem in the frequency domain.

    This function assembles and solves the finite-element formulation of a
    time-harmonic magneto-quasistatic problem using the magnetic vector
    potential. Eddy currents in electrical conductors are modeled through a
    coupled formulation with Lagrange multipliers enforcing prescribed bundle
    currents. The solver supports permanent magnet excitation, spatially varying
    material properties, optional mixed (Robin) boundary conditions, and
    pre-factorized system matrices for efficient repeated simulations.

    Parameters
    ----------
    fes : ngs.FESpace
        Finite element space for the magnetic vector potential.

    reluctivity : ngs.GridFunction or ngs.CoefficientFunction
        Magnetic reluctivity (1/μ). May vary spatially. Magnetic materials are
        assumed linear.

    magnetization : ngs.GridFunction or ngs.CoefficientFunction
        Complex-valued magnetization source term, typically representing
        rotating permanent magnets.

    frequency : float
        Electrical excitation frequency [Hz].

    supply : dict
        Electrical supply of the conducting bundles. Keys correspond to 
        conductor bundle names and values to the imposed complex currents [A].

    conductivity : float or ngs.GridFunction or ngs.CoefficientFunction, optional
        Electrical conductivity distribution.

    Kinv : optional
        Precomputed inverse (or factorization) of the system matrix. Providing
        this argument skips assembly and factorization of the stiffness matrix,
        greatly accelerating repeated solves with unchanged material properties.

    solver : str, optional
        Direct solver backend used to factorize the system matrix (default:
        ``"pardiso"``).

    bonus_intorder : int, optional
        Additional integration order used for conductivity and magnetization
        terms to improve quadrature accuracy.

    verbose : int, optional
        Verbosity level. Values greater than zero print timing information for
        each stage of the solve.

    taskmanager : bool, optional
        If True, assembles the finite element matrices using NGSolve's
        ``TaskManager`` for parallel execution.

    robin_bnd : str, optional
        Name of the boundary where mixed (Robin) boundary conditions are applied.
        If ``None``, the default natural (Neumann) boundary condition is used.

    robin_coeff : float or ngs.GridFunction or ngs.CoefficientFunction, optional
        Robin coefficient α satisfying

            α a + (1-α) ν curl(a) x n
            = α a_dirichlet + (1-α) h_tangential.

        Typical α values are:
        - 0 : pure Neumann condition
        - values approaching 1 : approximate Dirichlet condition.

    a_dirichlet : float or ngs.GridFunction or ngs.CoefficientFunction, optional
        Prescribed magnetic vector potential used in the Robin boundary
        condition.

    h_tangential : float or ngs.GridFunction or ngs.CoefficientFunction, optional
        Prescribed tangential magnetic field trace used in the Robin boundary
        condition.

    fix1dof : bool, optional
        If True, fixes one degree of freedom of the magnetic vector potential to
        remove the null space associated with pure Neumann problems.

    Returns
    -------
    results : dict
        Dictionary containing

        ``"solution"``
            Solution fields:
            - ``"a"``: magnetic vector potential.
            - ``"E"``: bundle time integral of electric potentials.

        ``"test"``
            Test functions used in the weak formulation.

        ``"bundles"``
            Names of the conducting bundles.

        ``"info"``
            Simulation metadata including the finite element space, material
            properties, excitation, solver information, cached matrix inverse,
            and execution times.

    Notes
    -----
    - The formulation assumes linear magnetostatics in the frequency domain.
    - Eddy currents are modeled only inside conducting bundles.
    - Total bundle currents are imposed using Lagrange multipliers.
    - Reusing ``Kinv`` is highly recommended for repeated simulations with
    varying currents or magnetization, as it avoids reassembling and
    refactorizing the system matrix.
    - When ``fix1dof=True``, one degree of freedom is constrained to eliminate
    the singularity associated with pure Neumann boundary conditions.
    """
    
    if verbose >= 1:
        print(f"-- START MAGNETOHARMONIC SOLVER --")
        print(f"Solver : {solver.lower()}")
        
    t0 = time()
    jw = 1j * 2 * ngs.pi * frequency
    K = None
    txtref = f"Matrix decomposition with {solver}... "
    
    if verbose >= 1:
        txt = "Setup function space... "
        print(txt, *[""]*(len(txtref)-len(txt)), end="")
        
    # Identify conductor bundles
    bundles = supply.keys()

    mesh = fes.mesh
    if fix1dof:
        dummy = ngs.GridFunction(fes)
        dummy.Set(1)
        ind = np.nonzero(dummy.vec.FV().NumPy())[0][0]
        fes.FreeDofs()[ind] = False
        
    # Normalize supplied currents by bundle volume
    Jcplx = {
        bundle: supply[bundle] / surface(bundle, mesh)
        for bundle in bundles
    }

    # Extend FE space with Lagrange multipliers (bundle constraints)
    for _ in bundles:
        fes *= ngs.NumberSpace(mesh, complex=True)

    # Define trial and test functions
    trials = fes.TrialFunction()
    tests = fes.TestFunction()
    if type(trials) is list:
        a, a_ = trials[0], tests[0]
    else:
        a, a_ = trials, tests

    E = {bundle: trials[i + 1] for i, bundle in enumerate(bundles)}
    E_ = {bundle: tests[i + 1] for i, bundle in enumerate(bundles)}

    t_fes = time() - t0
    
    if verbose >= 1:
        print(f"done in {t_fes*1000:.0f} ms")

    # ------------------------------------------------------------
    # Assemble system matrix
    # ------------------------------------------------------------
    t_assembly = 0
    t_decomposition = 0
    if (Kinv is None) or fix1dof:
        tic = time()
        if verbose >= 1:
            txt = "Assemble matrix... "
            print(txt, *[""]*(len(txtref) - len(txt)), end="")

        bf = Curl(a_) * reluctivity * Curl(a) * ngs.dx
        
        # Optional Robin term
        if robin_bnd is not None:
            bf += a_ * robin_coeff / (1-robin_coeff) * a * ngs.ds(robin_bnd)

        # Eddy-current + constraint coupling in each bundle
        for bundle in bundles:
            bf += a_ * conductivity * jw * (a + E[bundle]) * ngs.dx(bundle, bonus_intorder = bonus_intorder)
            bf += E_[bundle] * conductivity * jw * (a + E[bundle]) * ngs.dx(bundle, bonus_intorder = bonus_intorder)

        # Assemble matrix
        if taskmanager:
            with ngs.TaskManager():
                K = ngs.BilinearForm(bf, symmetric = True).Assemble().mat
        else:
            K = ngs.BilinearForm(bf, symmetric = True).Assemble().mat

        t_assembly = time() - tic
        
        if verbose >= 1:
            print(f"done in {t_assembly*1000:.0f} ms")

        
        # Factorize system
        if verbose >= 1:
            txt = txtref
            print(txtref, end="")
    
        tic = time()
        if (Kinv is None):
            if solver.lower() != "superlu":
                Kinv = K.Inverse(fes.FreeDofs(), inverse=solver)
            else:
                rows,cols,vals = K.COO()
                Ksp =  sp.csc_matrix((vals,(rows,cols)))
                Ksp = Ksp[fes.FreeDofs(),:][:,fes.FreeDofs()]
                Kinv = sp.linalg.splu(Ksp)          
                            
        t_decomposition = time() - tic
        
        if verbose >= 1:
            print(f"done in {t_decomposition*1000:.0f} ms")

    # ------------------------------------------------------------
    # Assemble right-hand side
    # ------------------------------------------------------------
    if verbose >= 1:
        txt = "Assemble right hand side... "
        print(txt, *[""]*(len(txtref) - len(txt)), end="")
    tic = time()
    lf = Curl(a_) * magnetization * ngs.dx(bonus_intorder = bonus_intorder)
    
    # Optional Robin term
    if robin_bnd is not None:
        lf += robin_coeff / (1-robin_coeff) * a_dirichlet * a_ * ngs.ds(robin_bnd)
        lf += h_tangential * a_ * ngs.ds(robin_bnd)
     
    for bundle in bundles:
        #lf += Jcplx[bundle] * a_ * ngs.dx(bundle)
        Jdc = supply[bundle] * conductivity / ngs.Integrate(conductivity, mesh.Materials(bundle))
        lf += Jdc * a_ * ngs.dx(bundle)

    if taskmanager:
        with ngs.TaskManager():
            F = ngs.LinearForm(lf).Assemble().vec
    else:
        F = ngs.LinearForm(lf).Assemble().vec

    t_rhs = time() - tic
    if verbose >= 1:
        print(f"done in {t_rhs*1000:.0f} ms")

    # ------------------------------------------------------------
    # Solve linear system
    # ------------------------------------------------------------
    if verbose >= 1:
        txt = "Solve the problem... "
        print(txt, *[""]*(len(txtref) - len(txt)), end="")
    tic = time()
    sol = ngs.GridFunction(fes)
    res = sol.vec.CreateVector()

    if fix1dof:
        sol.vec.data[ind] = a_dirichlet.vec[ind]
        res.data = K*sol.vec - F
    else:
        res.data =  -F

    if type(Kinv) != sp.linalg.SuperLU:
        sol.vec.data -= Kinv * res
    else:
        spsol = Kinv.solve(res.FV().NumPy()[fes.FreeDofs()])
        sol.vec.data.FV().NumPy()[fes.FreeDofs()] = - spsol
    
    t_solve = time() - tic

    if verbose >= 1:
        print(f"done in {t_solve*1000:.0f} ms")

    # ------------------------------------------------------------
    # Package results
    # ------------------------------------------------------------
    if verbose >= 1:
        txt = "Pack the results... "
        print(txt, *[""]*(len(txtref) - len(txt)), end="")

    time_total = time() -t0
    if type(trials) is list:
        solution = {
                "a": sol.components[0],
                "E": {
                    bundle: sol.components[i + 1]
                    for i, bundle in enumerate(bundles)
                }
                }
    else:
        solution = {
                "a": sol,
                "E": {}}
    results = {
        "solution": solution,
        "trial": {"a": a, "E": E},
        "test": {"a": a_, "E": E_},
        "bundles": bundles,
        "info": {
            "fes": fes,
            "reluctivity": reluctivity,
            "magnetization": magnetization,
            "frequency": frequency,
            "supply": supply,
            "conductivity": conductivity,
            "Kinv": Kinv,
            "K" : K,
            "F": -res,
            "solver": solver,
            "walltime" : {"fes": t_fes, 
                          "assembly": t_assembly, 
                          "decomposition": t_decomposition, 
                          "rhs": t_rhs, 
                          "solve": t_solve,
                          "total":time_total}
            },
        }  

    if verbose >= 1:
        print(f"total time: {time_total:.3f} s")
        print(f"-- END MAGNETOHARMONIC SOLVER --")
       
    return results


def newton(residual : ngs.BilinearForm,                   # residual (written using a bilinearform, actually not bilinear)
           initial_state :  dict,                         # initial guess (state structure)
           # Inspection parameters
           verbose : int = 1,                             # verbosity level (0 - silent to 3 - detailed)
           # Newton parameters
           maxit_newton : int = 50,              # maximum number of Newton outer iterations
           atol_decrement : float = 1e-10,        # (absolute) tolerance on Newton decrement : sqrt( < residual(uOld), du > )
           atol_residual : float = 1e-10,         # (absolute) tolerance on residual 
           rtol_residual : float = 1e-10,        # relative tolerance on the residual between 2 iterations (to save 1 useless iteration in case of linear problem)
           # Line search parameters
           linesearch : bool = True,             # flag to enable line search (recommended)
           maxit_linesearch : int = 33,          # maximum iteration number within the line search
           minstep_linesearch : float = 1e-10,   # minimum step size allowed in the line search 
           armijo_factor_linesearch : float = 0.1,      # Armijo coefficient in [0, 1) such that |residual(u-step*du)|² < residual²(u) - armijo_linesearch*step*(|residual(u)|²)'(du)
           step_factor_linesearch : float = 0.5, # step size reduction factor in (0, 1) to reduce the step if too big
           taskmanager : bool = False,          # flag to enable parallelization with TaskManager during assembly
           solver : str = "pardiso"              # method to solve the linear systems
           ) -> dict:
    """
    Solve a nonlinear finite element problem using Newton's method with optional
    backtracking line search.

    Parameters
    ----------
    residual : ngs.BilinearForm
        NGSolve bilinear form representing the nonlinear residual operator. Although
        implemented using a ``BilinearForm``, the operator may be nonlinear in the
        solution. Its linearization is assembled through
        ``residual.AssembleLinearization(...)`` and is used to compute the Newton
        correction.

    initial_state : dict
        Initial simulation state containing the initial guess and associated
        metadata. The state must be compatible with :func:`state2gfu` and
        :func:`gfu2state`.

    verbose : int, optional
        Verbosity level controlling diagnostic output:

        - ``0``: silent.
        - ``1``: print failure messages.
        - ``2``: print Newton iteration and convergence information.
        - ``3``: additionally print detailed timing information.

        Default is ``1``.

    maxit_newton : int, optional
        Maximum number of Newton iterations. Default is ``50``.

    atol_decrement : float, optional
        Absolute tolerance on the Newton decrement. The iteration is stopped when
            ``sqrt(abs(<residual, descent>)) < tol_decrement``.
        Default is ``1e-10``.

    atol_residual : float, optional
        Absolute tolerance on the norm of the residual. The iteration is stopped
        when the residual norm falls below this value. 
        Default is ``1e-10``.

    rtol_residual : float, optional
        Relative residual tolerance. It is used both to detect an effectively
        linear problem from the ratio of two successive residuals and as a stopping
        criterion based on the residual relative to the initial residual.
        Default is ``1e-10``.

    linesearch : bool, optional
        If True, perform a backtracking line search on each Newton update.
        Disabling line search is not recommended but can sometimes save time.
        Default is ``True``.

    maxit_linesearch : int, optional
        Maximum number of backtracking iterations allowed during the line search.
        Default is ``33``.

    minstep_linesearch : float, optional
        Minimum allowed line-search step size. The line search fails if the step
        becomes smaller than this value. 
        Default is ``1e-10``.

    armijo_factor_linesearch : float, optional
        Armijo parameter controlling the sufficient decrease condition. It should
        satisfy ``0 <= armijo_factor_linesearch < 1``. 
        Default is ``0.1``.

    step_factor_linesearch : float, optional
        Factor by which the line-search step is reduced when the Armijo condition
        is not satisfied. It should satisfy ``0 < step_factor_linesearch < 1``.
        Default is ``0.5``.

    taskmanager : bool, optional
        If True, use NGSolve's ``TaskManager`` during residual evaluation and
        linearization to enable parallel assembly. 
        Default is ``False``.

    solver : str, optional
        Linear solver used to compute the Newton correction. The default,
        ``"pardiso"``, uses NGSolve's direct solver interface. ``"superlu"`` uses
        SciPy's sparse LU factorization instead. 
        Default is ``"pardiso"``.

    Returns
    -------
    dict
        Updated simulation state containing the converged or final solution and
        Newton solver information. In addition to the fields already present in
        ``initial_state``, the result contains:

        ``"residual"``
            List of residual norms recorded during the Newton iterations.

        ``"decrement"``
            List of Newton decrement values for each iteration.

        ``"info"]["status"]``
            Solver status code:

            - ``0``: successful convergence.
            - ``1``: maximum number of Newton iterations reached.
            - ``2``: minimum line-search step reached.
            - ``3``: maximum number of line-search iterations reached.
            - ``4``: NaN detected in the residual.

        ``"info"]["linear_detected"]``
            Indicates whether the problem was detected as effectively linear from
            the relative change in the residual.

        ``"info"]["iteration"]``
            Number of the last Newton iteration performed.

        ``"info"]["Kinv"]``
            Factorization or inverse of the final linearized system matrix.

        ``"info"]["K"]``
            Final assembled linearized system matrix.

        ``"info"]["wall_time"]``
            Dictionary containing timing information for the solver. Individual
            assembly and solve timings are currently not populated, while
            ``"total"`` contains the total wall-clock time.

    Notes
    -----
    At each Newton iteration, the nonlinear residual is evaluated and its
    linearization is assembled. The resulting linear system is solved for the
    Newton descent direction. If line search is enabled, the update is reduced
    until a sufficient decrease of the squared residual norm satisfies the
    Armijo condition.

    The returned state is reconstructed using :func:`gfu2state`, preserving the
    state structure used by the surrounding finite element formulation.
    """

    # 1) Initialization
    tStart = time()
    if verbose >= 2 : print(f"******************** START NEWTON ********************")
    if verbose >= 2 : print(f"Solver: {solver.lower()} | Multithreaded Assembly: {taskmanager}")
    if verbose >= 3 : print(f"Initializing  ..... ", end = "")
    state = state2gfu(initial_state)
    fes = state.space
    res = state.vec.CreateVector()
    res_linesearch = state.vec.CreateVector()
    state_linesearch = state.vec.CreateVector()
    descent = state.vec.CreateVector()
    fes = state.space
    status = 0
    res2 = lambda res : np.dot(res.FV().NumPy()[fes.FreeDofs()],res.FV().NumPy()[fes.FreeDofs()])

    decrement_list = []
    residual_list = []
    if verbose >= 3 : print(f"done ({(time()-tStart) * 1000 :.2f} ms).")
    if verbose >= 3 : print(f"     ---------------- Start loop  ----------------")

    if taskmanager:
        with ngs.TaskManager(): residual.Apply(state.vec, res)
    else: residual.Apply(state.vec, res)
    residual_list.append(np.sqrt(res2(res)))

    # 2) Newton loop
    for counter_newton in range(1,maxit_newton+1):
        if verbose >= 2 : print(f" It {counter_newton} -------------------------------------------------")

        # a) NaN check
        if np.isnan(residual_list[-1]):
            status = 4
            if verbose >= 1 : 
                print(f"❌ FAILURE: NaN detected !!")
            break
        # b) Compute residual and linearization
        tStartAssembly = time()
        if verbose >= 3 : print(f" - Assembly ....... ", end = "")
        if taskmanager:
            with ngs.TaskManager():
                residual.Apply(state.vec, res)
                residual.AssembleLinearization(state.vec)
        else:
            residual.Apply(state.vec, res)
            residual.AssembleLinearization(state.vec)
        if verbose >= 3 : print(f"done ({(time()-tStartAssembly) * 1000 :.2f} ms).")

        # c) Solve
        tStartSolve = time()
        if verbose >= 3 : print(f" - Solve .......... ", end = "")
        if solver.lower() != "superlu":
            Kinv  = residual.mat.Inverse(freedofs=fes.FreeDofs(), inverse = solver)  
            descent.data = Kinv * res
        else:
            rows,cols,vals = residual.mat.COO()
            Ksp =  sp.csc_matrix((vals,(rows,cols)))
            Ksp = Ksp[fes.FreeDofs(),:][:,fes.FreeDofs()]
            Kinv = sp.linalg.splu(Ksp)
            spsol = Kinv.solve(res.FV().NumPy()[fes.FreeDofs()])
            descent.data.FV().NumPy()[fes.FreeDofs()] = spsol
        if verbose >= 3 : print(f"done ({(time()-tStartSolve) * 1000 :.2f} ms).")


        # d) Calculation of Newton's decrement
        decrement = np.sqrt(abs(ngs.InnerProduct(res, descent)))
        decrement_list.append(decrement)

        if verbose >= 2 : print(f" - Conv : ||residual|| = {residual_list[-1]:.4e} | decr = {decrement_list[-1] :.4e}")

        # e) Line search
        if linesearch:
            tStartLineSearch = time()
            if verbose >= 2 : print(f" - Line search .... ")
            step_linesearch = 1.0
            counter_linesearch = 0
            state_linesearch.data = state.vec - step_linesearch * descent
            if taskmanager:
                with ngs.TaskManager():
                    residual.Apply(state_linesearch, res_linesearch)
            else:
                residual.Apply(state_linesearch, res_linesearch)
            res2_state = res2(res)
            res2_linesearch = res2(res_linesearch)
            if verbose >= 2 : print(f"   it {counter_linesearch} : ||residual|| = {np.sqrt(res2_linesearch) :.4e} | step = {step_linesearch :.2e}")

            while not (res2_linesearch < (1 - 2 * armijo_factor_linesearch * step_linesearch) * res2_state):
                counter_linesearch += 1
                step_linesearch *= step_factor_linesearch
                state_linesearch.data = state.vec - step_linesearch * descent
                if taskmanager:
                    with ngs.TaskManager():
                        residual.Apply(state_linesearch, res_linesearch)
                else:
                    residual.Apply(state_linesearch, res_linesearch)
                res2_linesearch = res2(res_linesearch)
                if verbose >= 2 : print(f"   it {counter_linesearch} : ||residual|| = {np.sqrt(res2_linesearch) :.4e} | step = {step_linesearch :.2e}")

                if counter_linesearch >= maxit_linesearch:
                    if verbose >= 1 : print(f"❌ FAILURE: maximal number of line search iterations reached !!")
                    status = 3
                    break

                if step_linesearch < minstep_linesearch:
                    if verbose >= 1 : print(f"❌ FAILURE: minimal line search step reached !!")
                    status = 2
                    break
            
            if verbose >= 3 : print(f" - Line search done ({(time()-tStartLineSearch) * 1000 :.2f} ms).")
        
        if verbose >= 3 : print(f"     ---------------- End loop ----------------")
        # f) Update and residual computation
        if not status: 
            state.vec.data = state_linesearch
            res.data = res_linesearch
            residual_list.append(np.sqrt(res2(res)))
        else:
            state.vec.data = state.vec - step_linesearch * descent
            if taskmanager:
                with ngs.TaskManager(): residual.Apply(state.vec, res)
            else: residual.Apply(state.vec, res)
            residual_list.append(np.sqrt(res2(res)))

        # g) Stopping criteria
        if residual_list[-1] / residual_list[-2] < rtol_residual:
            if verbose >= 2 : print(f"Stop because linear problem detected.")
            linear = True
            break
        
        if decrement_list[-1] < atol_decrement : 
            if verbose >= 2 : print(f"Stop because decrement is lower than tol_decrement.")
            break

        if residual_list[-1] < atol_residual : 
            if verbose >= 2 : print(f"Stop because residual is lower than tol_residual.")
            break

        if residual_list[-1] / residual_list[0] < rtol_residual:
            if verbose >= 2 : print(f"Stop because relative residual is lower than rtol_residual.")
            break

        if counter_newton >= maxit_newton: 
            if verbose >= 1 : print(f"❌ FAILURE: maximum number of Newton iterations reached !!")
            status = 1
            break

    # 3) Export results
    if verbose >=2 and not status : 
        print(f"-------------------------------------------------------")  
        print(f" ✅ SUCCESS: Newton has converged in {counter_newton} iteration", end = "")
        if  counter_newton > 1 : print("s.")
        else : print(".") 
    if verbose >=2 :  print(f" Total wall time: {(time() - tStart) :.2f} s.")

    result = initial_state.copy()
    result["info"]["status"] = status
    result["info"]["linear_detected"] = linear
    result["info"]["iteration"] = counter_newton
    result["info"]["Kinv"] = Kinv
    result["info"]["K"] = residual.mat
    result["info"]["status"] = status
    result["linear_detected"] = linear
    result["iteration"] = counter_newton
    result["residual"] = residual_list
    result["decrement"] = decrement_list
    result["info"]["wall_time"] = { "fes": None, 
                                    "assembly": None, 
                                    "decomposition": None, 
                                    "rhs": None, 
                                    "solve": None,
                                    "total":time() - tStart}

    if verbose >=2 : print(f" ********************* END NEWTON ********************* ")  
    return gfu2state(state, result)



def operator_magnetostatic(state : dict,
                           type : str,
                           t : float = 0,
                           bonus_intorder : int = 3,
                           # Slot model - mixed boundary conditions
                           # on selected boundary, apply : robin_coeff*a           + (1-robin_coeff)*nu*da/dn
                           #                             = robin_coeff*a_dirichlet + (1-robin_coeff)*Trace(h_neumann)
                           robin_bnd   : str = None, # Boundary name where apply mixed condition
                           robin_coeff :  callable = lambda t: 0,     # Robin coefficient in [0 = neumann, 1 = dirichlet)
                           )-> ngs.comp.SumOfIntegrals:

    reluctivity = state["info"]["reluctivity"]
    a_ = state["test"]["a"]
    a = state[type]["a"]

    # Magnetostatic part
    K = reluctivity(ngs.Norm(Curl(a))) * Curl(a) * Curl(a_) * ngs.dx(bonus_intorder = bonus_intorder)

        #Optional Robin term
    if robin_bnd is not None:
        alpha = robin_coeff(t)
        K += a_ * alpha / (1-alpha) * a * ngs.ds(robin_bnd, bonus_intorder = bonus_intorder)

    return K.Compile()

def rhs_magnetostatic(state : dict,
                      t : float = 0,
                      js_flag = False,              # flag to consider supply as constant 
                      bonus_intorder : int = 3,
                      # Slot model - mixed boundary conditions
                      # on selected boundary, apply : robin_coeff*a           + (1-robin_coeff)*nu*da/dn
                      #                             = robin_coeff*a_dirichlet + (1-robin_coeff)*Trace(h_neumann)
                      robin_bnd   : str = None, # Boundary name where apply mixed condition
                      robin_coeff :  callable = lambda t: 0,  # Robin coefficient in [0 = neumann, 1 = dirichlet)
                      a_dirichlet :  callable = lambda t: 0,   # non-zero Dirichlet, in fes
                      h_tangential :  callable = lambda t: 0,  # Neumann trace
                      )-> ngs.comp.SumOfIntegrals:

    """ Right-hand side of asymmetric magnetoquasistatic formulation """

    mesh = state["info"]["fes"].mesh
    
    magnetization = state["info"]["magnetization"]
    supply = state["info"]["supply"]
    bundles = supply.keys()

    J = {bundle: supply[bundle](t) / surface(bundle, mesh)
        for bundle in bundles}
    
    a_ = state["test"]["a"]

    F = 0

    # Eddy-current + constraint coupling in each bundle
    if js_flag:
        for bundle in bundles:
            F += J[bundle] * a_ * ngs.dx(bundle)

    else:
        for bundle in bundles:
            F += J[bundle] * a_ * ngs.dx(bundle)

    # Optional Robin term
    if robin_bnd is not None:
        alpha = robin_coeff(t)
        lf += alpha / (1-alpha) * a_dirichlet(t) * a_ * ngs.ds(robin_bnd, bonus_intorder = bonus_intorder)
        lf += h_tangential(t) * a_ * ngs.ds(robin_bnd, bonus_intorder = bonus_intorder)

    F.Compile()

    F += magnetization(t) * Curl(a_) * ngs.dx(bonus_intorder = bonus_intorder)

    return F # F.Compile() might not work here because of derivation of real part of complex magnetization


def operator_magnetoquasistatic_time_domain(state : dict,
                                            t : float,
                                            dt : float,
                                            type : str,
                                            theta : float = 0.5,
                                            bonus_intorder : int = 3,
                                            # Slot model - mixed boundary conditions
                                            # on selected boundary, apply : robin_coeff*a           + (1-robin_coeff)*nu*da/dn
                                            #                             = robin_coeff*a_dirichlet + (1-robin_coeff)*Trace(h_neumann)
                                            robin_bnd   : str = None, # Boundary name where apply mixed condition
                                            robin_coeff : ngs.GridFunction | ngs.CoefficientFunction | float = 0,     # Robin coefficient in [0 = neumann, 1 = dirichlet)
                                            )-> ngs.comp.SumOfIntegrals:
    """ Operator of asymmetric magnetoquasistatic formulation """

    conductivity = state["info"]["conductivity"]
    e_ = state["test"]["e"]
    a_ = state["test"]["a"]
    eNew = state[type]["e"]
    aNew = state[type]["a"]

    K = 0
    # Magnetostatic part
    if theta >0:
        K += theta * operator_magnetostatic(state = state, type = type,
                                            t = t + dt,
                                            bonus_intorder = bonus_intorder,
                                            robin_bnd = robin_bnd,
                                            robin_coeff = robin_coeff)

    # Eddy-current + constraint coupling in each bundle
    for bundle in state["info"]["supply"].keys():
        K += conductivity * ( aNew/dt  + theta * eNew[bundle]) * (a_ + e_[bundle]) * ngs.dx(bundle, 
                                                                                            bonus_intorder = bonus_intorder)

    return K.Compile() 


def rhs_magnetoquasistatic_time_domain(state : dict,
                                       t : float,
                                       dt : float,
                                       theta : float = 0.5,
                                       bonus_intorder : int = 3,
                                       # Slot model - mixed boundary conditions
                                       # on selected boundary, apply : robin_coeff*a           + (1-robin_coeff)*nu*da/dn
                                       #                             = robin_coeff*a_dirichlet + (1-robin_coeff)*Trace(h_neumann)
                                       robin_bnd   : str = None, # Boundary name where apply mixed condition
                                       robin_coeff : callable = lambda t: 0,     # robin_coeff(t) -> GridFunction or CoefficientFunction [0 = neumann, 1 = dirichlet); 
                                       a_dirichlet : callable = lambda t: 0,     # a_dirichlet(t) -> GridFunction or CoefficientFunction (non-zero Dirichlet, in fes)
                                       h_tangential : callable = lambda t: 0,    # h_tangential(t)->  GridFunction or CoefficientFunction (Neumann trace)
                                       )-> ngs.comp.SumOfIntegrals:

    """ Right-hand side of asymmetric magnetoquasistatic formulation """

    conductivity = state["info"]["conductivity"]
    supply = state["info"]["supply"]
    bundles = supply.keys()

    J = {bundle: lambda t: supply[bundle](t) / surface(bundle,  state["info"]["fes"].mesh)
        for bundle in bundles}
    
    e_ = state["test"]["e"]
    a_ = state["test"]["a"]
    eOld = state["solution"]["e"]
    aOld = state["solution"]["a"]

    # Eddy-current coupling in each bundle
    F = 0
    for bundle in bundles:
        F += conductivity * ( aOld/dt  - (1-theta) * eOld[bundle]) * (a_ + e_[bundle]) * ngs.dx(bundle, bonus_intorder = bonus_intorder)

    # Magnetostatic part
    F += -(1-theta) * operator_magnetostatic(state = state, 
                                            type = "solution",
                                            t = t,
                                            bonus_intorder = bonus_intorder,
                                            robin_bnd = robin_bnd,
                                            robin_coeff = robin_coeff)
        
    F.Compile()

    F += (1-theta) * rhs_magnetostatic(state = state, t = t,
                                      bonus_intorder = bonus_intorder,  
                                      robin_bnd  = robin_bnd, 
                                      robin_coeff  = robin_coeff,
                                      a_dirichlet = a_dirichlet,
                                      h_tangential = h_tangential)

    F += theta * rhs_magnetostatic(state = state, t = t + dt,
                                   bonus_intorder = bonus_intorder,
                                   robin_bnd  = robin_bnd, 
                                   robin_coeff  = robin_coeff,
                                   a_dirichlet = a_dirichlet,
                                   h_tangential = h_tangential)

    return F


#%% Post-processing


def dual_trace(
    fes: ngs.FESpace,  # finite element space; should have dirichlet = bnd
    bnd: str,          # boundary name where to compute the dual trace
    nu: ngs.GridFunction | ngs.CoefficientFunction, # reluctivity
    a_ref: ngs.GridFunction | ngs.CoefficientFunction # reference magnetic vector potential
    ) -> ngs.GridFunction:
    """
    Compute the tangential magnetic field on a boundary by dual projection.

    This function projects the tangential magnetic field
    ``h = nu * Curl(a_ref)`` onto the specified boundary using an L²
    projection. The resulting grid function can be used, for example, as
    Neumann boundary data in another magnetostatic or magneto-harmonic
    simulation.

    Parameters
    ----------
    fes : ngs.FESpace
        H¹ finite element space with Dirichlet boundary conditions defined
        on the boundary where the trace is computed.

    bnd : str
        Name of the boundary where the tangential trace is evaluated.

    nu : ngs.GridFunction or ngs.CoefficientFunction
        Magnetic reluctivity.

    a_ref : ngs.GridFunction or ngs.CoefficientFunction
        Reference magnetic vector potential.

    Returns
    -------
    ngs.GridFunction
        Boundary projection of the tangential magnetic field.
    """
    h, h_ = fes.TnT()

    # Boundary mass matrix
    K = ngs.BilinearForm(h_ * h * ngs.ds(bnd)).Assemble().mat

    # Reference magnetic field
    h_ref = nu * Curl(a_ref)

    # Right-hand side corresponding to the dual trace
    f = ngs.LinearForm(Curl(h_) * h_ref * ngs.dx).Assemble().vec

    # Solve the boundary projection problem
    ht = ngs.GridFunction(fes)
    ht.vec.data = K.Inverse(freedofs=~fes.FreeDofs()) * f

    return ht
    

def electric_field(results: dict,              # result of solve_magnetoharmonic
                   type: str = "solution"      # "solution" or "test" for directional derivative
                   ) -> ngs.CoefficientFunction:
    """
    Compute the electric field in conducting regions by post-processing.

    This function reconstructs the complex electric field from the magnetic
    vector potential and bundle-wise electric potentials obtained from a
    magneto-harmonic finite element solve.

    Parameters
    ----------
    results : dict
        Output dictionary from `solve_magnetoharmonic`, containing the
        solution fields and metadata.

    type : str, optional
        Specifies which fields to use:
        - "solution": use primal solution fields
        - "test": use adjoint/test fields for sensitivity analysis

    Returns
    -------
    ngs.CoefficientFunction
        Complex electric field distribution, restricted to conducting regions.

    Notes
    -----
    - The electric field is computed as:
        E = -jω (A - e_bundle)
    - A conductivity mask is applied to restrict the field to conductors
      and avoid division by zero.
    """

    jw = 1j * 2 * ngs.pi * results["info"]["frequency"]

    # Time-harmonic contribution from magnetic vector potential
    electric_field = -jw * results[type]["a"]

    mesh = results["info"]["fes"].mesh

    # Add bundle electric potential contributions
    for bundle in results[type]["E"].keys():
        E = results[type]["E"][bundle]
        electric_field += -jw * E * mesh.MaterialCF({bundle: 1})

    # Apply conductivity mask (avoid division by zero)
    sigma = results["info"]["conductivity"]

    return electric_field * sigma / (sigma + 1e-300)

def electric_field_eddy_current(results: dict,              # result of solve_magnetoharmonic
                                type: str = "solution"      # "solution" or "test" for directional derivative
                                ) -> ngs.CoefficientFunction:
    """
    Compute the electric field due to Eddy current in conducting regions 
    by post-processing.

    This function reconstructs the complex electric field from the magnetic
    vector potential and bundle-wise electric potentials obtained from a
    magneto-harmonic finite element solve.

    Parameters
    ----------
    results : dict
        Output dictionary from `solve_magnetoharmonic`, containing the
        solution fields and metadata.

    type : str, optional
        Specifies which fields to use:
        - "solution": use primal solution fields
        - "test": use adjoint/test fields for sensitivity analysis

    Returns
    -------
    ngs.CoefficientFunction
        Complex electric field distribution, restricted to conducting regions.

    Notes
    -----
    - The eddy current electric field is computed as:
        E = -jω (A - e_bundle)
    - A conductivity mask is applied to restrict the field to conductors
      and avoid division by zero.
    """

    jw = 1j * 2 * ngs.pi * results["info"]["frequency"]

    # Time-harmonic contribution from magnetic vector potential

    mesh = results["info"]["fes"].mesh
    E_eddy_current = -jw * results[type]["a"]

    # Add bundle electric potential contributions
    for bundle in results[type]["E"].keys():
        E = results[type]["E"][bundle]
        E_eddy_current += -jw * E* mesh.MaterialCF({bundle: 1})

    sigma = results["info"]["conductivity"]

    return E_eddy_current * sigma / (sigma + 1e-300)

def electric_field2(results: dict,              # result of solve_magnetoharmonic
                    type: str = "solution",      # "solution" or "test" for directional derivative
                    E_eddy : ngs.CoefficientFunction = None
                   ) -> ngs.CoefficientFunction:
    """
    Compute the electric field in conducting regions by post-processing.

    This function reconstructs the complex electric field from the magnetic
    vector potential and bundle-wise electric potentials obtained from a
    magneto-harmonic finite element solve.

    Parameters
    ----------
    results : dict
        Output dictionary from `solve_magnetoharmonic`, containing the
        solution fields and metadata.

    type : str, optional
        Specifies which fields to use:
        - "solution": use primal solution fields
        - "test": use adjoint/test fields for sensitivity analysis

    Returns
    -------
    ngs.CoefficientFunction
        Complex electric field distribution, restricted to conducting regions.

    Notes
    -----
    - The electric field is computed as:
        E = Js / sigma + electric_field_eddy_current
    """
    E = 0
    if E_eddy is None:
        E +=  electric_field_eddy_current(results = results,
                                          type = type)
    else:
        E += E_eddy

    mesh = results["info"]["fes"].mesh
    sigma = results["info"]["conductivity"]

    # Add DC contributions
    if  type.lower() == "solution":
        for bundle in results[type]["E"].keys():
            I = results["info"]["supply"][bundle]
            rho_avg = 1/integrate(sigma, results, bundle)
            E += rho_avg * I  * mesh.MaterialCF({bundle: 1}) 

    return E

def current_density(results: dict,          # result of solve_magnetoharmonic
                    type: str = "solution"  # "solution" or "test" for directional derivative
                    ) -> ngs.CoefficientFunction:
    """
    Compute the electric current density by post-processing.

    This function evaluates the conductive current density from the electric
    field obtained in a magneto-harmonic finite element simulation.

    Parameters
    ----------
    results : dict
        Output dictionary from `solve_magnetoharmonic`, containing solution
        fields and material properties.

    type : str, optional
        Specifies which field set to use:
        - "solution": use primal solution fields
        - "test": use test fields for sensitivity analysis

    Returns
    -------
    ngs.CoefficientFunction
        Complex current density field J = σ E (local Ohm's law)
    """

    sigma = results["info"]["conductivity"]

    return sigma * electric_field(results, type)

def current_density2(results: dict,          # result of solve_magnetoharmonic
                    type: str = "solution"  # "solution" or "test" for directional derivative
                    ) -> ngs.CoefficientFunction:
    """
    Compute the electric current density by post-processing.

    This function evaluates the conductive current density from the electric
    field obtained in a magneto-harmonic finite element simulation.

    Parameters
    ----------
    results : dict
        Output dictionary from `solve_magnetoharmonic`, containing solution
        fields and material properties.

    type : str, optional
        Specifies which field set to use:
        - "solution": use primal solution fields
        - "test": use test fields for sensitivity analysis

    Returns
    -------
    ngs.CoefficientFunction
        Complex current density field J = σ E (local Ohm's law)
    """

    sigma = results["info"]["conductivity"]

    return sigma * electric_field2(results, type)

def joule_losses(results : dict,
                 zone : str = ".*") -> float:
    """
    Compute the total Joule (ohmic) losses by post-processing.

    This function evaluates the resistive power dissipation in conducting
    regions based on the current density obtained from a magneto-harmonic
    finite element simulation.

    Parameters
    ----------
    results : dict
        Output dictionary from `solve_magnetoharmonic`, containing solution
        fields and material properties.

    Returns
    -------
    float
        Total Joule losses (time-averaged power dissipation).

    Notes
    -----
    - The Joule losses are computed as:
        P = 1/2 ∫_Ω (|J|² / σ) dx
      where J is the complex current density.
    - The factor 1/2 accounts for time-averaging in harmonic regime; 
    we assume phasors modules are amplitude and not RMS values.
    - A small regularization term is added to σ to avoid division by zero.
    """

    # Current density from post-processing
    j = current_density(results)

    # Conductivity with numerical safety offset
    sigma = results["info"]["conductivity"] + 1e-300

    # Time-averaged Joule losses: 1/2 ∫ |J|^2 / σ dx
    P = integrate(ngs.InnerProduct(j, j).real / sigma, results, zone) / 2

    return P

def joule_losses2(results : dict,
                  zone : str = ".*") -> float:
    """
    Compute the total Joule (ohmic) losses by post-processing.

    This function evaluates the resistive power dissipation in conducting
    regions based on the current density obtained from a magneto-harmonic
    finite element simulation.

    Parameters
    ----------
    results : dict
        Output dictionary from `solve_magnetoharmonic`, containing solution
        fields and material properties.

    Returns
    -------
    float
        Total Joule losses (time-averaged power dissipation).

    Notes
    -----
    - More efficient alternative implementation than
        P = 1/2 ∫_Ω (|J|² / σ) dx
      We integrate on each bundle separating DC and AC losses.
    """
    sigma = results["info"]["conductivity"] + 1e-300

    E =  electric_field_eddy_current(results = results)
    E.Compile()

    Pdc = 0
    Pac = 0
    I = results["info"]["supply"]
    for bundle in results["bundles"]:
        if match(zone, bundle):
            intSigma = integrate(sigma, results, bundle)
            Pac += integrate(ngs.InnerProduct(E,E).real * sigma, results, bundle) / 2
            Pdc += abs(I[bundle])**2 / intSigma / 2

    return Pac + Pdc

def matrix_arkkio() -> ngs.CoefficientFunction :
    """
    Construct the Arkkio torque weighting matrix.

    This operator defines the 2D geometric weighting matrix used in
    Arkkio's method for torque computation in electrical machines.
    It depends only on spatial coordinates (x, y).

    Returns
    -------
    ngs.CoefficientFunction
        2×2 matrix field used to weight the magnetic flux density
        in the torque integral.

    Notes
    -----
    - Implements the standard Arkkio geometric tensor:
        Q = (1/r) * [ -xy, -(y² - x²)/2 ; -(y² - x²)/2, xy ]
    - r = sqrt(x² + y²)
    - Used for torque evaluation in the airgap region.
    """

    xy, x2, y2 = ngs.x * ngs.y, ngs.x**2, ngs.y**2
    r = ngs.sqrt(x2 + y2)

    Q11 = -xy / r
    Q21 = -(y2 - x2) / (2 * r)

    return ngs.CF(((Q11, Q21), (Q21, -Q11)), dims=(2, 2))


def average_torque(results : dict, 
                   airgap : str = "airgap_rotor",
                   L : float = 1. # axial length
                   ) -> float:
    """
    Compute the average electromagnetic torque using Arkkio's method.

    This function evaluates the electromagnetic torque in the airgap
    region based on the magnetic flux density and a geometric weighting
    tensor.

    Parameters
    ----------
    results : dict
        Output dictionary from `solve_magnetoharmonic`, containing the
        magnetic vector potential solution.

    airgap : str, optional
        Material name corresponding to the airgap region where torque
        is evaluated.
    
    L :float, optional
        Axial length of the machine

    Returns
    -------
    float
        Average electromagnetic torque.

    Notes
    -----
    - Uses Arkkio's method:
        T = (L π / (μ₀ S)) ∫ (B · (Q B)) dΩ
        see this paper: https://arxiv.org/pdf/2511.07217
    - S is the airgap surface area.
    """

    # Quadrature order adapted to FE polynomial degree
    order = 2 * results["solution"]["a"].space.globalorder + 1

    mesh = results["info"]["fes"].mesh

    # Airgap normalization area
    S = surface(airgap, mesh)

    # Magnetic flux density
    b = Curl(results["solution"]["a"])

    mu0 = 4e-7 * ngs.pi

    # Arkkio torque formula
    integrand = ngs.InnerProduct(b, (matrix_arkkio() * b))
    factor = L * ngs.pi / (S * mu0) 
    return factor * ngs.Integrate(integrand , mesh.Materials(airgap), order=order).real