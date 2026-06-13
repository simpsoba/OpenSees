# -*- coding: utf-8 -*-
"""
Created on Sep 10, 2024

@author: garaujor

Description:
    Two-Story Moment Resisting Frame from Kolay and Ricles (2016, 2019).

Units:
    [Force] = kN
    [Length] = m
    [Time] = s

Reference:
    Kolay, C., & Ricles, J. M. (2019). Improved Explicit Integration Algorithms for Structural Dynamic Analysis with Unconditional Stability and Controllable Numerical Dissipation. Journal of Earthquake Engineering, 23(5), 771–792. https://doi.org/10.1080/13632469.2017.1326423

    Kolay, C., & Ricles, J. M. (2016). Assessment of explicit and semi-explicit classes of model-based algorithms for direct integration in structural dynamics. International Journal for Numerical Methods in Engineering, 107(1), 49–73. https://doi.org/10.1002/nme.5153

"""

# Modules
import os
import sys
from datetime import datetime

# Prefer locally built OpenSees Python module (build/Release/opensees.so).
# Import OpenSees before numpy/scipy/pandas: the local opensees.so embeds libstdc++
# (-static-libstdc++), and if numpy loads first, PathSeries -filePath reads fail.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ops_import import ops  # noqa: E402

import numpy as np

import ReadRecord

from plot_config import DT_ANALYSIS, FREE_VIBRATION_SECONDS

# Optional plotting/post-processing dependencies (not required for running the analysis)
try:
    import opsvis  # type: ignore
except Exception:  # pragma: no cover
    opsvis = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None

try:
    from scipy.sparse import lil_matrix  # type: ignore
except Exception:  # pragma: no cover
    lil_matrix = None

# Units and constants
kN = meter = sec = 1.0
kip = 4.44822 * kN
inch = 0.0254 * meter

foot = 12 * inch
lbf = kip / 1000
gravity = 9.80665 * meter / (sec * sec)
kg = (kN / 1000) * (sec * sec) / meter
kPa = kN / (meter * meter)
MPa = 1000 * kPa
GPa = 1000 * MPa

# Response monitoring: base 17, floor 1 node 10, roof node 2 (left roof-beam end).
MONITOR_NODES = (17, 10, 2)
MONITOR_DOFS = (1, 2, 3)


def create_model(apply_gravity=True, plot_model=True):
    
    # OpenSees model
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)
    
    # Basic Geometry
    L = 6.0 * meter 
    H = 3.0 * meter
    dLeaning = 0.2 * L
    Ljoint_beam = L / 12
    Hjoint_col = H / 6
    
    # Floor 2 nodes
    ops.node(1, 0.0, 2 * H)
    ops.node(2, Ljoint_beam, 2 * H)
    ops.node(3, 2 * Ljoint_beam, 2 * H)
    ops.node(4, L / 2, 2 * H)
    ops.node(5, L - 2 * Ljoint_beam, 2 * H)
    ops.node(6, L - Ljoint_beam, 2 * H)
    ops.node(7, L, 2 * H)
    ops.node(8, L + dLeaning, 2 * H) # Leaning Column Node
    
    # Floor 1 nodes
    ops.node(9, 0.0, H)
    ops.node(10, Ljoint_beam, H)
    ops.node(11, 2 * Ljoint_beam, H)
    ops.node(12, L / 2, H)
    ops.node(13, L - 2 * Ljoint_beam, H)
    ops.node(14, L - Ljoint_beam, H)
    ops.node(15, L, H)
    ops.node(16, L + dLeaning, H) # Leaning Column Node
    
    # Base nodes
    ops.node(17, 0.0, 0.0)
    ops.node(18, L, 0.0)
    ops.node(19, L + dLeaning, 0.0) # Leaning Column Node
    
    # Left Column Nodes
    ops.node(20, 0.0, (H - 2 * Hjoint_col) / 2)
    ops.node(21, 0.0, H - 2 * Hjoint_col)
    ops.node(22, 0.0, H - Hjoint_col)
    ops.node(23, 0.0, H + Hjoint_col)
    ops.node(24, 0.0, H + 2 * Hjoint_col)
    ops.node(25, 0.0, 2 * H - 2 * Hjoint_col)
    ops.node(26, 0.0, 2 * H - Hjoint_col)
    
    # Right Column Nodes
    ops.node(27, L, (H - 2 * Hjoint_col) / 2)
    ops.node(28, L, H - 2 * Hjoint_col)
    ops.node(29, L, H - Hjoint_col)
    ops.node(30, L, H + Hjoint_col)
    ops.node(31, L, H + 2 * Hjoint_col)
    ops.node(32, L, 2 * H - 2 * Hjoint_col)
    ops.node(33, L, 2 * H - Hjoint_col)
    
    # Supports
    ops.fix(17, 1, 1, 0)
    ops.fix(18, 1, 1, 0)
    ops.fix(19, 1, 1, 0)
    
    # Rigid diaphragms and other constraints
    ops.rigidDiaphragm(1, 8, 4) # Floor 2 constraint in DOF 1
    ops.rigidDiaphragm(1, 16, 12) # Floor 1 constraint in DOF 1
    
    # Nodal Masses
    ops.mass(8, 50.97e3 * kg, 1.0 * kg, 1.0 * kg * meter) # Floor 2
    ops.mass(16, 50.97e3 * kg, 1.0 * kg, 1.0 * kg * meter) # Floor 1
    
    # Materials
    Es = 200 * GPa # Elastic modulus
    Fy = 345 * MPa # Yield strength
    eta = 0.01 # Hardening ratio
    ops.uniaxialMaterial('Steel01', 1, Fy, Es, eta)
    
    # Sections
    # Beams, W24x55
    d = 23.6 * inch # section depth
    tw = 0.395 * inch # web thickness
    bf = 7.01 * inch # flange width
    tf = 0.505 * inch # flange thickness
    Nfw = 10 # number of fibers in the web
    Nff = 3 # number of fibers in each flange
    beam_linear_density = (55 * lbf / foot) / gravity
    beam_sec_tag = 1
    mat_tag = 1
    ops.section('WFSection2d', beam_sec_tag, mat_tag, d, tw, bf, tf, Nfw, Nff)

    # Columns, W14x120
    d = 14.5 * inch # section depth
    tw = 0.59 * inch # web thickness
    bf = 14.7 * inch # flange width
    tf = 0.94 * inch # flange thickness
    Nfw = 10 # number of fibers in the web
    Nff = 3 # number of fibers in each flange
    col_linear_density = (120 * lbf / foot) / gravity
    col_sec_tag = 2
    mat_tag = 1
    ops.section('WFSection2d', col_sec_tag, mat_tag, d, tw, bf, tf, Nfw, Nff)
        
    # Beams
    beam_geom_transf = 1
    ops.geomTransf('Linear', beam_geom_transf)
    beam_integ = 1
    NIPs = 2 # two Legendre integration points
    ops.beamIntegration('Legendre', beam_integ, beam_sec_tag, NIPs)
    # Floor 2 beams
    ops.element('dispBeamColumn', 1, 1, 2, beam_geom_transf, beam_integ, '-cMass', '-mass', beam_linear_density)
    ops.element('dispBeamColumn', 2, 2, 3, beam_geom_transf, beam_integ, '-cMass', '-mass', beam_linear_density)
    ops.element('dispBeamColumn', 3, 3, 4, beam_geom_transf, beam_integ, '-cMass', '-mass', beam_linear_density)
    ops.element('dispBeamColumn', 4, 4, 5, beam_geom_transf, beam_integ, '-cMass', '-mass', beam_linear_density)
    ops.element('dispBeamColumn', 5, 5, 6, beam_geom_transf, beam_integ, '-cMass', '-mass', beam_linear_density)
    ops.element('dispBeamColumn', 6, 6, 7, beam_geom_transf, beam_integ, '-cMass', '-mass', beam_linear_density)
    # Floor 1 beams
    ops.element('dispBeamColumn', 9, 9, 10, beam_geom_transf, beam_integ, '-cMass', '-mass', beam_linear_density)
    ops.element('dispBeamColumn', 10, 10, 11, beam_geom_transf, beam_integ, '-cMass', '-mass', beam_linear_density)
    ops.element('dispBeamColumn', 11, 11, 12, beam_geom_transf, beam_integ, '-cMass', '-mass', beam_linear_density)
    ops.element('dispBeamColumn', 12, 12, 13, beam_geom_transf, beam_integ, '-cMass', '-mass', beam_linear_density)
    ops.element('dispBeamColumn', 13, 13, 14, beam_geom_transf, beam_integ, '-cMass', '-mass', beam_linear_density)
    ops.element('dispBeamColumn', 14, 14, 15, beam_geom_transf, beam_integ, '-cMass', '-mass', beam_linear_density)

    # Columns
    col_geom_transf = 2
    ops.geomTransf('Linear', col_geom_transf)
    col_integ = 2
    NIPs = 2 # two Legendre integration points
    ops.beamIntegration('Legendre', col_integ, col_sec_tag, NIPs)
    # Left Column
    ops.element('dispBeamColumn', 15, 17, 20, col_geom_transf, col_integ, '-cMass', '-mass', col_linear_density)
    ops.element('dispBeamColumn', 16, 20, 21, col_geom_transf, col_integ, '-cMass', '-mass', col_linear_density)
    ops.element('dispBeamColumn', 17, 21, 22, col_geom_transf, col_integ, '-cMass', '-mass', col_linear_density)
    ops.element('dispBeamColumn', 18, 22, 9, col_geom_transf, col_integ, '-cMass', '-mass', col_linear_density)
    ops.element('dispBeamColumn', 19, 9, 23, col_geom_transf, col_integ, '-cMass', '-mass', col_linear_density)
    ops.element('dispBeamColumn', 20, 23, 24, col_geom_transf, col_integ, '-cMass', '-mass', col_linear_density)
    ops.element('dispBeamColumn', 21, 24, 25, col_geom_transf, col_integ, '-cMass', '-mass', col_linear_density)
    ops.element('dispBeamColumn', 22, 25, 26, col_geom_transf, col_integ, '-cMass', '-mass', col_linear_density)
    ops.element('dispBeamColumn', 23, 26, 1, col_geom_transf, col_integ, '-cMass', '-mass', col_linear_density)
    # Right Column
    ops.element('dispBeamColumn', 24, 18, 27, col_geom_transf, col_integ, '-cMass', '-mass', col_linear_density)
    ops.element('dispBeamColumn', 25, 27, 28, col_geom_transf, col_integ, '-cMass', '-mass', col_linear_density)
    ops.element('dispBeamColumn', 26, 28, 29, col_geom_transf, col_integ, '-cMass', '-mass', col_linear_density)
    ops.element('dispBeamColumn', 27, 29, 15, col_geom_transf, col_integ, '-cMass', '-mass', col_linear_density)
    ops.element('dispBeamColumn', 28, 15, 30, col_geom_transf, col_integ, '-cMass', '-mass', col_linear_density)
    ops.element('dispBeamColumn', 29, 30, 31, col_geom_transf, col_integ, '-cMass', '-mass', col_linear_density)
    ops.element('dispBeamColumn', 30, 31, 32, col_geom_transf, col_integ, '-cMass', '-mass', col_linear_density)
    ops.element('dispBeamColumn', 31, 32, 33, col_geom_transf, col_integ, '-cMass', '-mass', col_linear_density)
    ops.element('dispBeamColumn', 32, 33, 7, col_geom_transf, col_integ, '-cMass', '-mass', col_linear_density)
    
    # Leaning Column
    leaning_geom_transf = 3
    ops.geomTransf('PDelta', leaning_geom_transf)
    A_leaning = 9.76e-2 * meter**2
    I_leaning = 7.125e-4 * meter**4
    E_leaning = Es
    ops.element('elasticBeamColumn', 33, 19, 16, A_leaning, E_leaning, I_leaning, leaning_geom_transf, '-mass', 1e-3*kg/meter, '-cMass') # lumped mass at floor nodes
    ops.element('elasticBeamColumn', 34, 16, 8, A_leaning, E_leaning, I_leaning, leaning_geom_transf, '-mass', 1e-3*kg/meter, '-cMass') # lumped mass at floor nodes
    
    # Plot model
    if plot_model and opsvis is not None and plt is not None:
        opsvis.plot_model()
        plt.gca().set_title('Model')
    
    if apply_gravity:
        # Gravity loads
        Pfloor1, Pfloor2 = 500 * kN, 500 * kN 
        ops.timeSeries('Linear', 1)
        ops.pattern('Plain', 1, 1)
        ops.load(8, 0.0, -Pfloor2, 0.0)
        ops.load(16, 0.0, -Pfloor1, 0.0)

        # Gravity analysis setup
        n_steps = 10
        ops.wipeAnalysis()
        ops.constraints('Transformation')
        ops.numberer('RCM')
        ops.system('FullGeneral')
        ops.test('NormDispIncr', 1.0e-6, 10)
        ops.algorithm('KrylovNewton')
        ops.integrator('LoadControl', 1.0 / n_steps)
        ops.analysis('Static')
        ops.analyze(n_steps)

        # Set gravity loads to constant and time to 0.0
        ops.loadConst('-time', 0.0)
    
        if plot_model and opsvis is not None and plt is not None:
            # Plot applied gravity loads
            opsvis.plot_loads_2d(nep=17, sfac=False, fig_wi_he=False, fig_lbrt=False, 
                                 fmt_model_loads={'color': 'black', 'linestyle': 'solid', 
                                                  'linewidth': 1.2, 'marker': '', 
                                                  'markersize': 1},
                                 node_supports=True, truss_node_offset=0, ax=False)
            plt.gca().set_title('Applied gravity loads')

def get_eigenvalues(num_modes=2, plot_mode_shape=True):
    # Compute eigenvalues after applying gravity loads
    eigenvalues = ops.eigen(num_modes)
    omega_n = np.sqrt(eigenvalues)
    T_n = 2 * np.pi / omega_n
    
    # Create a DataFrame to display eigenvalues as a table
    # eigen_data = {
    #     'Eigenvalues': eigenvalues,
    #     'Omega': omega_n,
    #     'T': T_n
    # }
    # df = pd.DataFrame(eigen_data)
    # df.index = df.index + 1
    
    # Plot mode shapes
    if plot_mode_shape and opsvis is not None and plt is not None:
        for mode in range(1, num_modes + 1):
            opsvis.plot_mode_shape(mode)
            plt.gca().set_title(
                f"Mode {mode}, $T_{mode}$ = {T_n[mode-1]:.3f} s, $\\omega_{mode}$ = {omega_n[mode-1]:.2f} rad/s"
            )
    
    eigen_dict = ops.modalProperties('-return', '-file', 'ModalReport.txt', '-unorm')
    print("Eigen done! Check 'ModalReport.txt'")
    return eigen_dict

def setup_rayleigh_damping(omega_i, omega_j, zeta_i, zeta_j, use_NPD=True, plot_rayleigh=True):
    # Setup Rayleigh damping using mass and initial stiffness
    denominator = omega_i**2 - omega_j**2
    a0 = 2 * (zeta_j * omega_i - zeta_i * omega_j) * (omega_j * omega_i) / denominator
    a1 = 2 * (zeta_i * omega_i - zeta_j * omega_j) / denominator
    #             M    KT  KI  Kn
    ops.rayleigh(a0, 0.0, a1, 0.0)
    
    # Set stiffness-proportional damping coefficients to zero in elements subjected to highly nonlinear deformations 
    if use_NPD:
        for ele in [1, 2, 5, 6, 9, 10, 13, 14]:
            ops.setElementRayleighDampingFactors(ele, a0, 0.0, 0.0, 0.0)

    if plot_rayleigh and plt is not None:
        T_values = np.linspace(0.01, 5.01, 500)
        omega_values = 2 * np.pi / T_values
        zeta_values = 0.5 * (a0 / omega_values + a1 * omega_values)

        fig, ax1 = plt.subplots(figsize=(6, 4))

        # Plot damping versus frequency
        ax1.semilogx(omega_values, zeta_values * 100, c='gray', label=f'$a_0$={a0:.2e}, $a_1$={a1:.2e}')
        ax1.semilogx(
            omega_i,
            zeta_i * 100,
            "or",
            label=f"($\\omega_i$={omega_i:.2f} rad/s, $\\zeta_i$={zeta_i * 100:.2f}%)",
        )
        ax1.semilogx(
            omega_j,
            zeta_j * 100,
            "xr",
            label=f"($\\omega_j$={omega_j:.2f} rad/s, $\\zeta_j$={zeta_j * 100:.2f}%)",
        )
        ax1.set_xlabel("$\\omega$ (rad/s)")
        ax1.set_ylabel("$\\zeta$ (%)")
        ax1.set_xlim(1, 300)
        ax1.set_ylim(0, 10)
        ax1.set_title('Rayleigh Damping')
        ax1.grid(True)
        ax1.legend()
        
        # Plot damping  versus period
        ax2 = ax1.twiny()
        ax2.set_xscale('log')
        omega_ticks = ax1.get_xticks() # np.array([1, 10, 100, 300])
        T_secondary_ticks = 2 * np.pi / omega_ticks

        # Set the new ticks and labels for the secondary axis
        ax2.set_xticks(omega_ticks)
        ax2.set_xticklabels([f'{T:.2f}' for T in T_secondary_ticks])
        ax2.set_xlabel('$T$ (s)')
        
        plt.show()

def setup_modal_damping(zetas, use_quick=True, add_K_damping=True, use_NPD=True, plot_damping=True):
    Nmodes = len(zetas)
    eigenvalues = ops.eigen(Nmodes)
    omegas = np.sqrt(eigenvalues)
    a1 = 0.0
    if add_K_damping:
        a1 = 2*zetas[-1]/omegas[-1]
    ops.rayleigh(0.0, 0.0, a1, 0.0)
    damping = [zetas[i] - a1*omegas[i]/2 for i in range(Nmodes)]
    if use_quick:
        ops.modalDampingQ(*damping)
    else:
        ops.modalDamping(*damping)
    
    # Set stiffness-proportional damping coefficients to zero in elements subjected to highly nonlinear deformations 
    if add_K_damping and use_NPD:
        for ele in [1, 2, 5, 6, 9, 10, 13, 14]:
            ops.setElementRayleighDampingFactors(ele, 0.0, 0.0, 0.0, 0.0)
    
    if plot_damping and plt is not None:
        T_values = np.linspace(0.01, 5.01, 500)
        omega_values = np.sort(np.unique(np.concatenate([2 * np.pi / T_values, omegas])))
        zeta_values = 0.5 * (a1 * omega_values)
        match_idx = np.where(np.isclose(omega_values[:, None], omegas, rtol=1e-5, atol=1e-8))[0]

        fig, ax1 = plt.subplots(figsize=(6, 4))

        # Plot damping versus frequency
        ax1.semilogx(omega_values, zeta_values * 100, c='gray', label=f'K-proportional damping, $a_1$={a1:.2e}')
        ax1.semilogx(omega_values[match_idx], zeta_values[match_idx] * 100, 'o', markerfacecolor='white', markeredgecolor='gray')
        ax1.semilogx(omegas, np.array(zetas) * 100, 'or', label=f'Modal damping, $N_{{modes}}$={Nmodes}')


        ax1.set_xlabel("$\\omega$ (rad/s)")
        ax1.set_ylabel("$\\zeta$ (%)")
        ax1.set_xlim(1, 300)
        ax1.set_ylim(0, 10)
        ax1.set_title('Modal + K-Proportional Damping')
        ax1.grid(True)
        ax1.legend()
        
        # Plot damping  versus period
        ax2 = ax1.twiny()
        ax2.set_xscale('log')
        omega_ticks = ax1.get_xticks() # np.array([1, 10, 100, 300])
        T_secondary_ticks = 2 * np.pi / omega_ticks

        # Set the new ticks and labels for the secondary axis
        ax2.set_xticks(omega_ticks)
        ax2.set_xticklabels([f'{T:.2f}' for T in T_secondary_ticks])
        ax2.set_xlabel('$T$ (s)')
        
        plt.show()


def _is_newmark(integrator_method: str) -> bool:
    return integrator_method == "Newmark"


def _transient_algorithm(integrator_method: str) -> str:
    """Explicit KRAlpha/MKR family: Linear; Newmark: Newton."""
    return "Newton" if _is_newmark(integrator_method) else "Linear"


def _default_pflag(integrator: dict) -> int:
    if _is_newmark(integrator["method"]):
        return 0
    return 5 if integrator["maxIter"] == 1 else 2


def _integrator_ops_test_args(integrator, pFlag):
    """
    Build ops.test arguments for transient analysis.

    Default: NormUnbalance, tol=1e-8, iter=integrator['maxIter'], print flag=pFlag
    (same for Newmark, KRAlphaExplicit, MKR, *_TP variants, and *MultiSOE variants).

    Override with integrator['test'], e.g.
      {'type': 'NormDispIncr', 'tol': 1e-6, 'iter': 25, 'pFlag': 0}
    or ['NormDispIncr', 1e-6, 25, 0].
    """
    max_iter = integrator["maxIter"]
    if "test" in integrator:
        t = integrator["test"]
        if isinstance(t, dict):
            name = t["type"]
            tol = float(t["tol"])
            it = int(t.get("iter", max_iter))
            pf = int(t.get("pFlag", pFlag))
            return (name, tol, it, pf)
        if isinstance(t, (list, tuple)):
            name = t[0]
            tol = float(t[1])
            it = int(t[2]) if len(t) > 2 else max_iter
            pf = int(t[3]) if len(t) > 3 else pFlag
            return (name, tol, it, pf)
        raise TypeError("integrator['test'] must be a dict or a sequence")

    return ("NormUnbalance", 1.0e-8, max_iter, pFlag)


def _load_path_series_values(path: str) -> list[float]:
    values: list[float] = []
    with open(path, "r", encoding="utf-8") as gm_file:
        for line in gm_file:
            values.extend(float(x) for x in line.split())
    return values


def _default_system(integrator_method: str) -> str:
    if integrator_method.startswith("Cuda") or integrator_method == "Newmark":
        return "CuDSS"
    if "MultiSOE" in integrator_method:
        return "CuDSS"
    return "FullGeneral"


def _set_transient_linear_system(integrator_method: str, system: str | None = None) -> None:
    """CuDSS for Newmark, Cuda*, and *MultiSOE*; FullGeneral for dense KRAlpha."""
    if system is not None:
        ops.system(system)
        return
    ops.system(_default_system(integrator_method))


def _result_folder(integrator: dict) -> str:
    method = integrator["method"]
    params = integrator["params"]
    system = integrator.get("system")
    if system is not None and not (method == "Newmark" and system == "CuDSS"):
        return f"results/{method}_{system}_params-{params!s}"
    return f"results/{method}_params-{params!s}"


def _integrator_call_params(integrator: dict) -> list:
    return list(integrator["params"])


def run_dynamic_analysis(
    gm_file,
    dt_analysis=DT_ANALYSIS,
    scale_factor=1.0,
    integrator=None,
    printA=False,
    dt_gm=None,
):
    # Default integrator
    if integrator == None:
        integrator = {'method': 'Newmark', 
                      'params': [0.5, 0.25], 
                      'maxIter': 10}

    output_folder = _result_folder(integrator)
    os.makedirs(output_folder, exist_ok=True)

    # Ground motion input:
    # - If gm_file is numeric-only (no PEER header), do NOT call ReadRecord on it, otherwise
    #   it can overwrite/truncate the original file (inFilename == outFilename case).
    # - Each analysis writes its own gm.dat under output_folder (safe for parallel runs).
    if dt_gm is not None:
        gm_path_for_opensees = gm_file
        dt_file = float(dt_gm)
        try:
            with open(gm_file, "r") as f:
                nPts = sum(1 for _ in f)
        except Exception:
            nPts = 0
    else:
        gm_dat = os.path.join(output_folder, "gm.dat")
        dt_file, nPts = ReadRecord.ReadRecord(gm_file, gm_dat)
        gm_path_for_opensees = gm_dat

    # Set time series and uniform excitation pattern (KRAlphaSparse: -filePath)
    ops.timeSeries("Path", 2, "-filePath", gm_path_for_opensees, "-dt", dt_file, "-factor", gravity)
    ops.pattern('UniformExcitation', 2, 1, '-accel', 2, '-fact', scale_factor)
    
    # Setup dynamic analysis
    ops.wipeAnalysis()
    _set_transient_linear_system(integrator["method"], integrator.get("system"))
    ops.constraints('Transformation')
    ops.numberer('RCM')
    # OpenSees 'test' print flag: 0 for Newmark; verbose (5) for 1-iter explicit runs
    pFlag = integrator.get("pFlag", _default_pflag(integrator))
    test_args = _integrator_ops_test_args(integrator, pFlag)
    ops.test(test_args[0], test_args[1], test_args[2], test_args[3])
    algo = _transient_algorithm(integrator["method"])
    ops.algorithm(algo)
    ops.integrator(integrator['method'], *_integrator_call_params(integrator))
    ops.analysis('Transient')
    
    # Setup recorders of interest
    
    # General log file and OpenSees log file
    results = open(f'{output_folder}/results.txt','a+')
    ops.logFile(f'{output_folder}/OpenSees.log', '-noEcho')
    
    # Get the current date and time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results.write(f'{current_time} - Analysis STARTED.\n')
    gm_log_name = os.path.basename(gm_path_for_opensees)
    free_vibration = FREE_VIBRATION_SECONDS * sec
    motion_steps = int(round(nPts * dt_file / dt_analysis))
    free_steps = int(round(free_vibration / dt_analysis))
    n_steps = motion_steps + free_steps
    tFinal = n_steps * dt_analysis
    results.write(
        f'{current_time} - Running {gm_log_name} (dt = {dt_file} s, npts = {nPts}) '
        f'with dt_analysis = {dt_analysis} s, n_steps = {n_steps} (t_end = {tFinal} s), '
        f'scale factor = {scale_factor}.\n'
    )
    results.write(
        f'{current_time} - test {test_args[0]} {test_args[1]} {test_args[2]} {test_args[3]}; '
        f'algorithm {algo}; system {integrator.get("system") or _default_system(integrator["method"])}; '
        f'integrator {integrator["method"]} {integrator["params"]}.\n'
    )
    results.close()
    
    # Monitor stack: disp, vel, accel, Rayleigh forces (DOFs 1–3 at each MONITOR_NODES)
    # Record every converged step (fixed dt_analysis in analyze loop below).
    for resp in ('disp', 'vel', 'accel'):
        ops.recorder(
            'Node', '-file', f'{output_folder}/{resp}.out', '-time',
            '-node', *MONITOR_NODES, '-dof', *MONITOR_DOFS, resp,
        )
    ops.recorder(
        'Node', '-file', f'{output_folder}/rayleigh-forces.out', '-time',
        '-node', *MONITOR_NODES, '-dof', *MONITOR_DOFS, 'rayleighForces',
    )

    # Element 1 (roof beam 1–2) local forces
    ops.recorder(
        'Element', '-file', f'{output_folder}/element-1-forces.txt',
        '-time', '-ele', 1, 'localForce',
    )

    ok = 0
    step = 0
    
    if printA and lil_matrix is None:
        raise RuntimeError("printA=True requires scipy (lil_matrix) in this example script.")

    if printA:
        # Print LHS matrix in full dense form (temporary FullGeneral for diagnostics)
        ops.system("FullGeneral")
        # Using GimmeMCK
        ops.integrator('GimmeMCK', 1.0, 0.0, 0.0)
        ops.analyze(1, 0.0)
        ops.printB('-file', f'{output_folder}/GimmeMCK_vectorB.txt')
        # Using Integrator with Linear algorithm
        ops.integrator(integrator['method'], *_integrator_call_params(integrator))
        ops.algorithm("Linear")
        ops.analyze(1, dt_analysis)
        ops.printB('-file', f'{output_folder}/Linear_vectorB.txt')
        # Using Integrator with given algorithm
        ops.algorithm(algo)
        ops.analyze(1, dt_analysis)
        ops.printA('-file', f'{output_folder}/matrixA.txt')
        ops.printB('-file', f'{output_folder}/vectorB.txt')
        Neqn = ops.systemSize()
        _set_transient_linear_system(integrator["method"], integrator.get("system"))

        # Element Connectivity Matrix. 
        # Reference: Scott, M. H. (2022). OpenSees Spy. Portwood Digital. url=https://portwooddigital.com/2022/03/13/opensees-spy/
        # Step 1: Build connectivity matrix using a sparse format
        connectivity_matrix = lil_matrix((Neqn, Neqn))  # LIL format allows efficient construction

        # Step 2: Fill the spy matrix based on element connectivity
        for e in ops.getEleTags():
            dofs = []
            for nd in ops.eleNodes(e):
                dofs += ops.nodeDOFs(nd)
            # Mark connections in connectivity_matrix for all valid DOFs
            for idof in dofs:
                if idof < 0:  # Skip constrained DOFs
                    continue
                for jdof in dofs:
                    if jdof < 0:
                        continue
                    connectivity_matrix[idof, jdof] = 1.0

        # Convert connectivity_matrix to CSR for more efficient operations later
        connectivity_matrix = connectivity_matrix.tocsr()

        # Save the spy matrix to a text file
        np.savetxt(f'{output_folder}/connectivity_matrix.txt', connectivity_matrix.toarray(), fmt='%d')

        # Step 3: Determine bandwidth
        max_bandwidth = 0
        for i in range(Neqn):
            for j in range(i, Neqn):
                if connectivity_matrix[i, j] != 0.0:
                    bandwidth = j - i + 1  # Compute bandwidth for this row
                    if bandwidth > max_bandwidth:
                        max_bandwidth = bandwidth

        # Save the bandwidth
        with open(f'{output_folder}/connectivity_matrix_bandwidth.txt', 'w') as f:
            f.write(str(max_bandwidth))

    # Perform the transient analysis (record NR iterations / unbalance norm per step)
    time_per_step = []
    iters_per_step = []
    tol_per_step = []

    def _record_convergence_step() -> None:
        try:
            nit = ops.testIter()
            nit = int(nit[0] if isinstance(nit, (list, tuple)) else nit)
        except Exception:
            nit = 0
        try:
            norms = ops.testNorms()
            norms = list(norms) if isinstance(norms, (list, tuple)) else [float(norms)]
            fnorm = norms[nit - 1] if nit and nit <= len(norms) else (norms[-1] if norms else float('nan'))
        except Exception:
            fnorm = float('nan')
        time_per_step.append(float(ops.getTime()))
        iters_per_step.append(nit)
        tol_per_step.append(fnorm)

    while ok == 0 and step < n_steps:
        ok = ops.analyze(1, dt_analysis)
        # if the analysis fails try initial tangent iteration
        if ok != 0 and integrator['maxIter'] > 1:
            print(f"{algo} failed .. lets try an initial stiffness for this step")
            ops.test(test_args[0], test_args[1], test_args[2] * 100, test_args[3])
            ops.algorithm('ModifiedNewton', '-initial')
            ok = ops.analyze(1, dt_analysis)
            if ok == 0:
                print(f"that worked .. back to {algo}")
                ops.test(test_args[0], test_args[1], test_args[2], test_args[3])
                ops.algorithm(algo)
        _record_convergence_step()
        step += 1

    if time_per_step:
        conv = np.column_stack(
            [np.asarray(time_per_step), np.asarray(iters_per_step), np.asarray(tol_per_step)]
        )
        np.savetxt(
            f'{output_folder}/convergence.dat',
            conv,
            header='time iters final_norm',
            comments='# ',
        )

    # Write success or failure status to log file
    results = open(f'{output_folder}/results.txt', 'a+')
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if ok == 0:
        results.write(f'{current_time} - Analysis COMPLETED successfully.\n')
        print("Passed!")
    else:
        results.write(f'{current_time} - Analysis FAILED at time t = {ops.getTime()} s\n')
        print("Failed!")

    results.close()
    ops.remove('recorders')