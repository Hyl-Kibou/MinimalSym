import numpy as np

from .point_group import PointGroup
from .sym_ops import normalize, rotation_matrix, canonical, generate_cyclic_axes, issame_axis, float_isclose
from .pg_detect import PointGroupResult, _classify_general
from .special_geometry import _find_C4s_for_Oh
from .mol_ops import transform

def _print_to_vector(axes:np.array, shine:float=0.2, factor:float=1.0) -> None:
    """
    Internal debugging utility: print vectors in a visualization-friendly format.

    This helper outputs vectors as formatted lines (prefixed with "V" and "S")
    intended for use with external visualization tools or custom viewers.
    It is primarily used to inspect symmetry axes during development.

    Parameters
    ----------
    axes : np.ndarray
        Array of vectors to print (shape: (N, 3) or (3,)).
    shine : float, optional
        Visual intensity parameter included in the output format.
    factor : float, optional
        Scaling factor applied to vector length in the output.

    Returns
    -------
    None
    """
    np.set_printoptions(precision=6)

    axes = np.asarray(axes)
    # If single vector -> wrap it
    if axes.ndim == 1:
        axes = axes.reshape(1, 3)

    color = int(0)
    for axis in axes:
        axis = normalize(axis)
        print("V", end="   ")
        for a in axis:
            print(f"{-a*factor:.6f}", end="   ")
        for a in axis:
            print(f"{a*factor:.6f}", end="   ")
        color +=1
        print(f"S   0.02  0.05 0.{int(color/4)%2*9}    0.{int(color/2)%2*9}   0.{color%2*9}   {shine}")

    print("- - - - - - - -")


def _decompose_cyclic(family:str, n:int, subfamily:str, paxis:np.array, saxis:np.array, taxis:np.array) -> list[PointGroupResult]:
    """
    Decompose cyclic and dihedral point groups into symmetry-consistent subgroups.

    This function generates a set of candidate point groups derived from a
    cyclic (Cₙ/Sₙ) or dihedral (Dₙ) parent group, including all compatible
    subgroups and orientations that preserve the underlying symmetry axes.

    Parameters
    ----------
    family : str
        Point-group family ("C" or "D").
    n : int
        Order of the principal rotation axis. Special case n = 0 corresponds
        to linear groups (C∞v / D∞h).
    subfamily : str or None
        Subfamily label ("v", "h", "d", or None).
    paxis : ndarray, shape (3,)
        Principal symmetry axis.
    saxis : ndarray, shape (3,)
        Secondary axis defining the canonical orientation.
    taxis : ndarray, shape (3,)
        Tertiary axis orthogonal to paxis and saxis.

    Returns
    -------
    : list[PointGroupResult]
        List of candidate point groups with associated orientations.

    Notes
    -----
    - Includes:
        * Parent group (e.g., Cₙ, Dₙ)
        * Subgroups (Cₖ, Dₖ where k | n)
        * Mirror and improper groups (Cs, Ci, Sₙ)
        * All symmetry-equivalent orientations of axes
    - For dihedral groups, additional recursive decomposition is performed
      through cyclic subgroups (e.g., Cₙᵥ, Cₙₕ).
    - Linear groups (n = 0) are approximated using high-order finite groups.
    """
    pg_list = []
    zeros = np.zeros(3)

    # Cs(h) Sn Ci
    if subfamily == 'h':
        pg_list.append(PointGroupResult("Cs", paxis, saxis))
        if n != 0 and n % 2 == 0:
            pg_list.append(PointGroupResult(f"S{n}", paxis, saxis))
        if n % 2 == 0:
            pg_list.append(PointGroupResult("Ci", paxis, saxis))

    if n == 0:
        # Ckv
        pg_list.append(PointGroupResult("C16v", paxis, saxis))
        # C0
        pg_list.append(PointGroupResult("C16", paxis, zeros))
        # 0 C2
        pg_list.append(PointGroupResult("Cs", saxis, zeros))
        pg_list.append(PointGroupResult("Cs", taxis, zeros))
        if family == 'D':
            # Dkh
            pg_list.append(PointGroupResult("D16h", paxis, saxis))
            # D0
            pg_list.append(PointGroupResult("D16", paxis, saxis))
            # 0 D2
            pg_list.append(PointGroupResult("D2", saxis, paxis))
            pg_list.append(PointGroupResult("D2", taxis, paxis))
            # 0 C2
            pg_list.append(PointGroupResult("C2", saxis, zeros))
            pg_list.append(PointGroupResult("C2", taxis, zeros))
            # S0
            pg_list.append(PointGroupResult("S2", paxis, zeros))
            # 0 S2
            pg_list.append(PointGroupResult("S2", saxis, zeros))
            pg_list.append(PointGroupResult("S2", taxis, zeros))
            pass
        return pg_list

    # Family(n)
    if subfamily is not None:
        pg_list.append(PointGroupResult(f"{family}{n}", paxis, saxis))

    # Cs(v)
    if subfamily == 'v':
        cyclic_cs_axes = generate_cyclic_axes(paxis, taxis, n)

        for rotated_taxis in cyclic_cs_axes:
            pg_list.append(PointGroupResult("Cs", rotated_taxis, zeros))

    # S2n
    if subfamily == 'd':
        pg_list.append(PointGroupResult(f"S{2*n}", paxis, saxis))

    if family == 'D':
        # Cnv RECURSIVE
        if subfamily == 'h' or subfamily == 'd':
            cnv_pg = PointGroup.from_string(f"C{n}v")
            additional_c2_axis = generate_cyclic_axes(paxis, saxis, n, 2)[1, :]
            dihedral_axis = normalize((saxis + additional_c2_axis))
            if subfamily == 'd': # diherdral mirror planes, bisect angle between 2 nC2 axes
                if n % 2 == 0:
                    # For n == even; all dihedral angle perpendicular to dihedral angle
                    pg_list.extend(_decompose_point_group(cnv_pg, paxis, dihedral_axis))
                else:
                    # For Dnd where n == odd; taxis is a dihedral angle (collinear to dihedral mirror plane)
                    # So for Dnd where n == odd; saxis is norm of dihedral plane
                    # For Cnv where n == odd; taxis is aligned with norm mirror plane
                    # So for Cnv where n == odd; saxis is collinear to mirror plane
                    pg_list.extend(_decompose_point_group(cnv_pg, paxis, taxis))
            if subfamily == 'h': # vertical mirror planes, vertical to nC2 axes
                if n % 2 == 0:
                    pg_list.extend(_decompose_point_group(cnv_pg, paxis, taxis))
                    if n > 2:
                        pg_list.extend(_decompose_point_group(PointGroup.from_string(f"D{n//2}d"), paxis, saxis))
                        pg_list.extend(_decompose_point_group(PointGroup.from_string(f"D{n//2}d"), paxis, additional_c2_axis))
                else:
                    pg_list.extend(_decompose_point_group(cnv_pg, paxis, saxis))
                # Cnh RECURSIVE
                pg_list.extend(_decompose_point_group(PointGroup.from_string(f"C{n}h"), paxis, saxis))
        # Cn RECURSIVE
        else:
            cn_pg = PointGroup.from_string(f"C{n}")
            pg_list.extend(_decompose_point_group(cn_pg, paxis, saxis))

        # n C2 axes
        cyclic_c2_axes = generate_cyclic_axes(paxis, saxis, n)
        for rotated_saxis in cyclic_c2_axes:
            pg_list.append(PointGroupResult("C2", rotated_saxis, zeros))

    # Fam_k Fam_k_sub
    for k in range(2, n):
        if n % k != 0:
            continue

        pg_list.append(PointGroupResult(f"{family}{k}", paxis, saxis))
        if subfamily is not None:
            if subfamily == 'v':
                if n % 2 != k % 2:
                    pg_list.append(PointGroupResult(f"{family}{k}{subfamily}", paxis, taxis))
                else:
                    pg_list.append(PointGroupResult(f"{family}{k}{subfamily}", paxis, saxis))
            elif subfamily != 'd':
                pg_list.append(PointGroupResult(f"{family}{k}{subfamily}", paxis, saxis))

    return pg_list

def _decompose_T_family(subfamily:str, paxis:np.array, saxis:np.array, taxis:np.array) -> list[PointGroupResult]:
    """
    Decompose tetrahedral point groups into symmetry-consistent subgroups.

    Generates all subgroup candidates of the T family (T, T_d, T_h),
    including cyclic, dihedral, and improper subgroups derived from
    tetrahedral symmetry elements.

    Parameters
    ----------
    subfamily : str or None
        Subfamily label ("d", "h", or None).
    paxis, saxis, taxis : ndarray, shape (3,)
        Orthogonal axes defining the tetrahedral orientation.

    Returns
    -------
    list[PointGroupResult]
        List of candidate point groups with associated orientations.

    Notes
    -----
    - Includes:
        * C₂ axes (3)
        * C₃ axes (4, along tetrahedral corners)
        * Derived subgroups (D₂)
    - For T_d:
        * Adds S₄ axes (3) and σ_d planes (6)
        * Includes C₃ᵥ subgroups
    - For T_h:
        * Adds inversion (Ci), σ_h planes (3), and S₆ axes (4)
        * Includes D₂h subgroups
    - Recursive decomposition is applied to composite subgroups.
    """
    # Tetrahedral (n == 3): use two of the three C2 axes.

    pg_list = []
    zeros = np.zeros(3)

    # T
    if subfamily is not None:
        pg_list.append(PointGroupResult('T', paxis, saxis))

    # 3 C2
    pg_list.append(PointGroupResult("C2", paxis, saxis))
    pg_list.append(PointGroupResult("C2", saxis, paxis))
    pg_list.append(PointGroupResult("C2", taxis, saxis))

    # 4 C3
    # Tetrahedron corners
    tetra_corners = [
        normalize(paxis +  saxis +  taxis),
        normalize(paxis +  saxis + -taxis),
        normalize(paxis + -saxis +  taxis),
        normalize(paxis + -saxis + -taxis)]

    for corner in tetra_corners:
        pg_list.append(PointGroupResult("C3", corner, zeros))

    # Combinations
    # D2(=3C2)  RECURSIVE
    d2_pg = PointGroup.from_string("D2")
    pg_list.extend(_decompose_point_group(d2_pg, paxis, saxis))
    pg_list.extend(_decompose_point_group(d2_pg, saxis, paxis))
    pg_list.extend(_decompose_point_group(d2_pg, taxis, paxis))


    if subfamily == 'd':
        # 3 S4 axes (same as C2 axes)
        pg_list.append(PointGroupResult("S4", paxis, saxis))
        pg_list.append(PointGroupResult("S4", saxis, paxis))
        pg_list.append(PointGroupResult("S4", taxis, saxis))
        # 6Cs(d)
        mirror_axes = np.array([
            canonical(paxis + saxis),
            canonical(paxis - saxis),
            canonical(paxis + taxis),
            canonical(paxis - taxis),
            canonical(saxis + taxis),
            canonical(saxis - taxis),
        ])
        for mirror_axis in mirror_axes:
            pg_list.append(PointGroupResult("Cs", mirror_axis, zeros))

        # Combinations
        # D2d(3C2) RECURSIVE
        d2d_pg = PointGroup.from_string("D2d")
        pg_list.extend(_decompose_point_group(d2d_pg, paxis, saxis))
        pg_list.extend(_decompose_point_group(d2d_pg, saxis, paxis))
        pg_list.extend(_decompose_point_group(d2d_pg, taxis, paxis))
        # C3v(4C3, 6Cs)
        pg_list.append(PointGroupResult("C3v", tetra_corners[0], canonical(np.cross(mirror_axes[1],  tetra_corners[0]))))
        pg_list.append(PointGroupResult("C3v", tetra_corners[1], canonical(np.cross(mirror_axes[1],  tetra_corners[1]))))
        pg_list.append(PointGroupResult("C3v", tetra_corners[2], canonical(np.cross(mirror_axes[0],  tetra_corners[2]))))
        pg_list.append(PointGroupResult("C3v", tetra_corners[3], canonical(np.cross(mirror_axes[0],  tetra_corners[3]))))

    if subfamily == 'h':
        # Ci
        pg_list.append(PointGroupResult("Ci", paxis, saxis))
        # 3Cs(h)
        pg_list.append(PointGroupResult("Cs", paxis, saxis))
        pg_list.append(PointGroupResult("Cs", paxis, taxis))
        pg_list.append(PointGroupResult("Cs", saxis, taxis))
        # 4 S6 (same as C3 axes)
        for corner in tetra_corners:
            pg_list.append(PointGroupResult("S6", corner, zeros))
        # Combinations
        # D2h(3C2) RECURSIVE
        d2h_pg = PointGroup.from_string("D2h")
        pg_list.extend(_decompose_point_group(d2h_pg, paxis, saxis))
        pg_list.extend(_decompose_point_group(d2h_pg, saxis, paxis))
        pg_list.extend(_decompose_point_group(d2h_pg, taxis, paxis))

    return pg_list

def _decompose_O_family(subfamily:str, paxis:np.array, saxis:np.array, taxis:np.array) -> list[PointGroupResult]:
    """
    Decompose octahedral point groups into symmetry-consistent subgroups.

    Generates subgroup candidates of the O family (O, O_h), including
    tetrahedral, cyclic, and dihedral subgroups derived from cubic symmetry.

    Parameters
    ----------
    subfamily : str or None
        Subfamily label ("h" for O_h, or None for O).
    paxis, saxis, taxis : ndarray, shape (3,)
        Orthogonal C₄ axes defining the cubic orientation.

    Returns
    -------
    list[PointGroupResult]
        List of candidate point groups with associated orientations.

    Notes
    -----
    - Includes:
        * C₄ axes (3), C₃ axes (4, cube diagonals), C₂ axes (6)
        * Subgroups: D₄, D₃, D₂, T
    - For O_h:
        * Adds inversion, mirror planes, and improper rotations (Ci, S₄, S₆)
        * Subgroups: D₄h, D₃d, D₂h, T, Th, Td
        * Includes recursive decomposition into D₄h, D₃d, D₂h
    - Axis orientations are explicitly constructed from cube geometry.
    """
    # Octahedral: paxis and saxis are two orthogonal C4 axes.

    pg_list = []
    zeros = np.zeros(3)

    # O
    if subfamily is not None:
        pg_list.append(PointGroupResult('O', paxis, saxis))

    # T
    pg_list.append(PointGroupResult('T', paxis, saxis))

    # 3 C4 RECURSIVE
    c4_pg = PointGroup.from_string("C4")
    pg_list.extend(_decompose_point_group(c4_pg, paxis, saxis))
    pg_list.extend(_decompose_point_group(c4_pg, saxis, paxis))
    pg_list.extend(_decompose_point_group(c4_pg, taxis, saxis))

    # 4 C3
    # Cube diagonals
    cube_corners = [
        normalize(paxis +  saxis +  taxis),
        normalize(paxis +  saxis + -taxis),
        normalize(paxis + -saxis +  taxis),
        normalize(paxis + -saxis + -taxis)]

    for corner in cube_corners:
        pg_list.append(PointGroupResult("C3", corner, zeros))

    # 6 C2
    face_diagonals = [
        normalize(paxis + saxis),
        normalize(paxis - saxis),
        normalize(paxis + taxis),
        normalize(paxis - taxis),
        normalize(saxis + taxis),
        normalize(saxis - taxis),
    ]

    for diagonal in face_diagonals:
        pg_list.append(PointGroupResult("C2", diagonal, zeros))

    # Combinations
    # D4(=3C4)8
    d4_axis_pairs = [
        (paxis, saxis),
        (saxis, paxis),
        (taxis, paxis),
    ]
    for axis_pair in d4_axis_pairs:
        pg_list.append(PointGroupResult("D4", axis_pair[0], axis_pair[1]))

    # D3(=4C3,6C2)
    d3_axis_pairs = [
        (cube_corners[0], face_diagonals[1]),
        (cube_corners[1], face_diagonals[1]),
        (cube_corners[2], face_diagonals[0]),
        (cube_corners[3], face_diagonals[0]),
    ]
    for axis_pair in d3_axis_pairs:
        pg_list.append(PointGroupResult("D3", axis_pair[0], axis_pair[1]))

    # D2(=6C2)
    d2_axis_pairs = [
        (face_diagonals[0], taxis),
        (face_diagonals[1], taxis),
        (face_diagonals[2], saxis),
        (face_diagonals[3], saxis),
        (face_diagonals[4], paxis),
        (face_diagonals[5], paxis),
    ]
    for axis_pair in d2_axis_pairs:
        pg_list.append(PointGroupResult("D2", axis_pair[0], axis_pair[1]))

    if subfamily == 'h':
        # Th
        pg_list.append(PointGroupResult("Th", paxis, saxis))
        # Td
        pg_list.append(PointGroupResult("Td", paxis, saxis))

        # Ci
        pg_list.append(PointGroupResult("Ci", paxis, saxis))
        # 3 Cs(h)
        pg_list.append(PointGroupResult("Cs", paxis, saxis))
        pg_list.append(PointGroupResult("Cs", saxis, paxis))
        pg_list.append(PointGroupResult("Cs", taxis, paxis))

        # 6 Cs(d)
        for diagonal in face_diagonals:
            pg_list.append(PointGroupResult("Cs", diagonal, zeros))

        # 4 S6 (same as C3 axes)
        for corner in cube_corners:
            pg_list.append(PointGroupResult("S6", corner, zeros))

        # 3 S4 (same as C4 axes)
        pg_list.append(PointGroupResult("S4", paxis, saxis))
        pg_list.append(PointGroupResult("S4", saxis, paxis))
        pg_list.append(PointGroupResult("S4", taxis, paxis))

        # Combinations
        # D4h(=3C4) RECURSIVE
        d4h_pg = PointGroup.from_string("D4h")
        for axis_pair in d4_axis_pairs:
            pg_list.extend(_decompose_point_group(d4h_pg, axis_pair[0], axis_pair[1]))

        # D3d(=4C3,6C2) RECURSIVE
        d3d_pg = PointGroup.from_string("D3d")
        for axis_pair in d3_axis_pairs:
            pg_list.extend(_decompose_point_group(d3d_pg, axis_pair[0], axis_pair[1]))
        # D2h(=6C2) RECURSIVE
        d2h_pg = PointGroup.from_string("D2h")
        for axis_pair in d2_axis_pairs:
            pg_list.extend(_decompose_point_group(d2h_pg, axis_pair[0], axis_pair[1]))

    return pg_list

def _decompose_I_family(subfamily:str, paxis:np.array, saxis:np.array, taxis:np.array) -> list[PointGroupResult]:
    """
    Decompose icosahedral point groups into symmetry-consistent subgroups.

    Generates subgroup candidates of the I family (I, I_h), including all
    cyclic and dihedral subgroups derived from icosahedral symmetry.

    Parameters
    ----------
    subfamily : str or None
        Subfamily label ("h" for I_h, or None for I).
    paxis : ndarray, shape (3,)
        Principal C₅ axis.
    saxis : ndarray, shape (3,)
        Secondary C₂ axis.
    taxis : ndarray, shape (3,)
        Orthogonal axis completing the reference frame.

    Returns
    -------
    list[PointGroupResult]
        List of candidate point groups with associated orientations.

    Notes
    -----
    - Includes:
        * C₅ axes (6), C₃ axes (10), C₂ axes (15)
        * Subgroups: D₅, D₃, D₂, T
    - Axes are constructed from icosahedral geometry using the golden ratio.
    - Subgroup axes are selected by enforcing orthogonality constraints.
    - For I_h:
        * Adds inversion, mirror planes, and improper rotations (Ci, S₁₀ (6), S₆ (10))
        * Includes recursive decomposition into D₂h, D₃d, D₅d
    """
    pg_list = []
    zeros = np.zeros(3)

    # Icosahedral: paxis = C5 axis, saxis = C2 axis.

    # I
    if subfamily is not None:
        pg_list.append(PointGroupResult('I', paxis, saxis))

    phi = (1 + np.sqrt(5)) / 2
    rphi = 1/phi

    # 6 C5
    raw_dirs_faces = np.array([
        [0,  1,  phi],
        [0, -1,  phi],
        [1,  phi, 0],
        [-1, phi, 0],
        [phi, 0, 1],
        [-phi, 0, 1],
    ])

    B = np.column_stack((saxis, taxis, paxis))
    raw_dirs_faces = (B @ raw_dirs_faces.T).T

    raw_dirs_faces = np.array([normalize(raw_axis) for raw_axis in raw_dirs_faces])

    theta = np.arccos(abs(np.dot(paxis, raw_dirs_faces[1])))
    R_align = rotation_matrix(saxis, -theta)
    c5_axes = transform(raw_dirs_faces, R_align)

    for c5_axis in c5_axes:
        pg_list.append(PointGroupResult("C5", c5_axis, zeros))

    # 15 C2
    c2_axes = []
    base_c2_axes = generate_cyclic_axes(c5_axes[-1], saxis, 5, 3)

    for c2_axis in base_c2_axes:
        c2_axes.extend(generate_cyclic_axes(paxis, c2_axis, 5))

    c2_axes = np.array(c2_axes)

    for c2_axis in c2_axes:
        pg_list.append(PointGroupResult("C2", c2_axis, zeros))

    # 10 C3
    c3_axes = []

    base_c3_axes = [normalize(saxis + base_c2_axes[1] - c2_axes[3])]
    base_c3_axes = generate_cyclic_axes(c5_axes[-1], base_c3_axes[0], 5, 2)
    base_c3_axes = np.array(base_c3_axes)

    for c3_axis in base_c3_axes:
        c3_axes.extend(generate_cyclic_axes(paxis, c3_axis, 5))

    c3_axes = np.array(c3_axes)

    for c3_axis in c3_axes:
        pg_list.append(PointGroupResult("C3", c3_axis, zeros))

    # Combinations
    # D2(=15C2)
    ortho_c2_axes = []
    global_ortho_list=[]
    for ii in range(15):
        i_axis = c2_axes[ii]
        ortho_list=[]
        for jj in range(15):
            if ii ==  jj: continue
            j_axis = c2_axes[jj]
            if float_isclose(np.dot(i_axis, j_axis), 0.):
                ortho_list.append(jj)
        ortho_c2_axes.append([i_axis])
        for ortho_id in ortho_list:
            ortho_c2_axes[ii].append(c2_axes[ortho_id])
        ortho_list.append(ii)
        global_ortho_list.append(ortho_list)

    for axis_pair in ortho_c2_axes:
        pg_list.append(PointGroupResult("D2", axis_pair[0], axis_pair[1]))

    # T(=15C2)
    t_axis_pairs=[]
    present_c2_axes = set()
    for ii, axes in enumerate(ortho_c2_axes):
        if global_ortho_list[ii][0] not in present_c2_axes:
            t_axis_pairs.append(axes)
            for id in global_ortho_list[ii]:
                present_c2_axes.add(id)

    for axis_pair in t_axis_pairs:
        pg_list.append(PointGroupResult("T", axis_pair[0], axis_pair[1]))

    # D5(=6C5,15C2)
    d5_axis_pairs=[]
    for c5_axis in c5_axes:
        for c2_axis in c2_axes:
            if float_isclose(np.dot(c5_axis, c2_axis), 0.):
                d5_axis_pairs.append((c5_axis, c2_axis))
                break
    for axis_pair in d5_axis_pairs:
        pg_list.append(PointGroupResult("D5", axis_pair[0], axis_pair[1]))

    # D3(=10C3,15C2)
    d3_axis_pairs=[]
    for c3_axis in c3_axes:
        for c2_axis in c2_axes:
            if float_isclose(np.dot(c3_axis, c2_axis), 0.):
                d3_axis_pairs.append((c3_axis, c2_axis))
                break
    for axis_pair in d3_axis_pairs:
        pg_list.append(PointGroupResult("D3", axis_pair[0], axis_pair[1]))

    if subfamily == 'h':
        # Ci
        pg_list.append(PointGroupResult("Ci", paxis, saxis))
        # 15Cs(v for C5, C2, C3. h for C2)
        for c2_axis in c2_axes:
            pg_list.append(PointGroupResult("Cs", c2_axis, zeros))
        # 6S10 (same as C5 axes)
        for c5_axis in c5_axes:
            pg_list.append(PointGroupResult("S10", c5_axis, zeros))
        # 10S6 (same as C3 axes)
        for c3_axis in c3_axes:
            pg_list.append(PointGroupResult("S6", c3_axis, zeros))

        # Combinations
        # D2h(=15C2) RECURSIVE
        d2h_pg = PointGroup.from_string("D2h")
        for axis_pair in ortho_c2_axes:
            pg_list.extend(_decompose_point_group(d2h_pg, axis_pair[0], axis_pair[1]))
        # Th(=15C2)
        for axis_pair in t_axis_pairs:
            pg_list.append(PointGroupResult("Th", axis_pair[0], axis_pair[1]))
        # D5d(=6C5,15C2) RECURSIVE
        d5d_pg = PointGroup.from_string("D5d")
        for axis_pair in d5_axis_pairs:
            pg_list.extend(_decompose_point_group(d5d_pg, axis_pair[0], axis_pair[1]))
        # D3d(=10C3,15C2) RECURSIVE
        d3d_pg = PointGroup.from_string("D3d")
        for axis_pair in d3_axis_pairs:
            pg_list.extend(_decompose_point_group(d3d_pg, axis_pair[0], axis_pair[1]))

    return pg_list

def _decompose_point_group(pg: PointGroup, paxis:np.array = np.array([0., 0., 1.0]), saxis:np.array = np.array([1.0, 0., 0.])) -> list[PointGroupResult]:
    """
    Decompose a point group into its symmetry elements represented
    as PointGroupResult objects.

    Parameters
    ----------
    pg: PointGroup
        Point group to decompose
    paxis: np.array
        Principal axis of point group
    saxis: np.array
        Secondary axis of point group
    Returns
    -------
    list[PointGroupResult]
        Internal symmetry elements of given point group
    """
    family = pg.family
    n = pg.n
    subfamily = pg.subfamily

    taxis = normalize(np.cross(paxis, saxis))

    pg_list = [PointGroupResult(pg=pg.str, paxis=paxis, saxis=saxis)]

    if subfamily == 's' or subfamily == 'i':
        return pg_list

    if family == 'S':
        # Cn/2 RECURSIVE
        cn2_pg = PointGroup.from_string(f"C{n // 2}")
        pg_list.extend(_decompose_point_group(cn2_pg, paxis, saxis))

        if n == 2:
            pg_list.append(PointGroupResult("Ci", paxis, saxis))

    if family == 'C' or family == 'D':
        pg_list.extend(_decompose_cyclic(family, n, subfamily, paxis, saxis, taxis))

    elif family == 'T':
        pg_list.extend(_decompose_T_family(subfamily, paxis, saxis, taxis))

    elif family == 'O':
        pg_list.extend(_decompose_O_family(subfamily, paxis, saxis, taxis))

    elif family == 'I':
        pg_list.extend(_decompose_I_family(subfamily, paxis, saxis, taxis))

    return pg_list

def _check_O_point_group(mol, invertable:bool) -> list[PointGroupResult]:
    """
    Checks if a given mol has O family symmetry.
    If it does returns the decomposition of the found O point group.

    Parameters
    ----------
    mol: ase.Atoms
        Molecule to check for O family symmetry.
    invertable: bool
        True if the molecule has an inversion center

    Returns
    -------
    list[PointGroupResult]
        The internal decompositon of the found O point group.
        Empty if no O point group was found.
    """
    try:
        c4_axes = _find_C4s_for_Oh(mol)
        print(c4_axes)
        paxis = c4_axes[0]
        saxis = c4_axes[1]
        if invertable:
            pg_str = "Oh"
        else:
            pg_str = "O"
        return _decompose_point_group(PointGroup.from_string(pg_str), paxis, saxis)
    except RuntimeError:
        return []

def _check_general_point_group(mol) -> list[PointGroupResult]:
    """
    Finds a general symmetry PointGroupResult for a molecule and
    returns the internal decomposition of the found point group.

    Parameters
    ----------
    mol: ase.Atoms
        Molecule to get general point group from.

    Returns
    -------
    list[PointGroupResult]
        The internal decompositon of the found point group.
    """
    general_pgr = _classify_general(mol, mol.positions, mol.get_masses(), mol.info["geom_tol"])
    return _decompose_point_group(PointGroup.from_string(general_pgr.pg), general_pgr.paxis, general_pgr.saxis)

def _unique_point_group(pg_list: list[PointGroupResult]) -> list[PointGroupResult]:
    """
    Returns a list of PointGroupResult without repetitions.

    Parameters
    ----------
    pg_list : list[PointGroupResult]
        PointGroupResult list to be deduplicated.

    Returns
    -------
    list[PointGroupResult]
        Deduplicated PointGroupResult list.
    """
    unique_pg_list_per_type = {}
    unique_index = []
    for idx, pgr in enumerate(pg_list):
        pg_str = pgr.pg.lower()
        paxis = pgr.paxis
        saxis = pgr.saxis
        pgr_is_unique = True

        if pg_str not in unique_pg_list_per_type:
            unique_pg_list_per_type[pg_str] = []

        ignore_saxis = False
        if pg_str[0] == 'c' or pg_str[0] == 'd':
            ignore_saxis = ((pg_str[0] == 'c' and pg_str[-1] != 'v') or (pg_str[1] == '0'))

        for existent_pgr in unique_pg_list_per_type[pg_str]:
            if pg_str == "ci":
                pgr_is_unique = False

            if ignore_saxis:
                if issame_axis(paxis, existent_pgr.paxis):
                    pgr_is_unique = False
                    break
                continue

            if issame_axis(paxis, existent_pgr.paxis) and issame_axis(saxis, existent_pgr.saxis):
                pgr_is_unique = False
                break

        if pgr_is_unique:
            unique_pg_list_per_type[pg_str].append(pgr)
            unique_index.append(idx)

    unique_pg_list = []
    for idx in unique_index:
        unique_pg_list.append(pg_list[idx])
    return unique_pg_list


def _find_pg_score(pg_str: str) -> int:
    """
    Given a string of the Schoenflies symbol of a point group,
    returns a score from the number of symmetry operations it possess.

    Parameters
    ----------
    pg_str: str
        Schoenflies symbol of a point group.

    Returns
    -------
    : int
        Score.
    """
    pg = PointGroup.from_string(pg_str)
    family = pg.family
    n = pg.n
    sub = pg.subfamily

    # Linear groups
    if n == 0:
        if family == 'C':   # C∞v
            return float('inf')
        elif family == 'D': # D∞h
            return float('inf')

    # Simple groups
    if family == 'C':
        if sub is None:
            return n
        elif sub in ('v', 'h'):
            return 2 * n

    elif family == 'D':
        if sub is None:
            return 2 * n
        elif sub in ('h', 'd'):
            return 4 * n

    elif family == 'S':
        return n

    elif family == 'T':
        if sub is None:
            return 12
        elif sub in ('d', 'h'):
            return 24.1

    elif family == 'O':
        if sub is None:
            return 24.2
        elif sub == 'h':
            return 48

    elif family == 'I':
        if sub is None:
            return 60
        elif sub == 'h':
            return 120

    # Special cases
    if pg.str == 'C1':
        return 1
    if pg.str in ('Cs', 'Ci'):
        return 2

    return 0  # fallback