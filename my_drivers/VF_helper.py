import torch


def calculate_point(centroids, state):
    delta=centroids[:,:,state]-centroids[:,:,0]
    return delta


def construct_VF(V_NN,delta_xyz):
    Vf = (
        V_NN[0]
        + V_NN[1] * delta_xyz[:,0]
        + V_NN[2] * delta_xyz[:,1]
        + V_NN[3] * delta_xyz[:,2]
        + V_NN[4] * delta_xyz[:,0]**2
        + V_NN[5] * delta_xyz[:,1]**2
        + V_NN[6] * delta_xyz[:,2]**2
        + V_NN[7] * (delta_xyz[:,0] * delta_xyz[:,1])
        + V_NN[8] * (delta_xyz[:,0] * delta_xyz[:,2])
        + V_NN[9] * (delta_xyz[:,1] * delta_xyz[:,2])
    )
    return Vf






def construct_VF_gradients(V_NN: torch.Tensor, delta_xyz: torch.Tensor) -> torch.Tensor:
    """
    V_NN:  either shape (30,), (30,1), or (1,30)
    delta_xyz: shape (Nelements, 3)
    returns grad_V of shape (Nelements, 3, 3)
    """
    # flatten to (30,)
    coeffs = V_NN.squeeze()
    if coeffs.numel() != 30:
        raise ValueError(f"expected 30 coefficients, got {coeffs.shape}")

    dx = delta_xyz[:, 0]
    dy = delta_xyz[:, 1]
    dz = delta_xyz[:, 2]

    # split into three 10-coefficient chunks
    a = coeffs[ 0:10]  # Vx
    b = coeffs[10:20]  # Vy
    c = coeffs[20:30]  # Vz

    # alias the ones we need
    a1,a2,a3 = a[1],a[2],a[3]
    a4,a5,a6 = a[4],a[5],a[6]
    a7,a8,a9 = a[7],a[8],a[9]

    b1,b2,b3 = b[1],b[2],b[3]
    b4,b5,b6 = b[4],b[5],b[6]
    b7,b8,b9 = b[7],b[8],b[9]

    c1,c2,c3 = c[1],c[2],c[3]
    c4,c5,c6 = c[4],c[5],c[6]
    c7,c8,c9 = c[7],c[8],c[9]

    # compute the nine partials
    dVx_dx = -a1 - 2*a4*dx -    a7*dy -    a8*dz
    dVx_dy = -a2 -    a7*dx - 2*a5*dy -    a9*dz
    dVx_dz = -a3 -    a8*dx -    a9*dy - 2*a6*dz

    dVy_dx = -b1 - 2*b4*dx -    b7*dy -    b8*dz
    dVy_dy = -b2 -    b7*dx - 2*b5*dy -    b9*dz
    dVy_dz = -b3 -    b8*dx -    b9*dy - 2*b6*dz

    dVz_dx = -c1 - 2*c4*dx -    c7*dy -    c8*dz
    dVz_dy = -c2 -    c7*dx - 2*c5*dy -    c9*dz
    dVz_dz = -c3 -    c8*dx -    c9*dy - 2*c6*dz

    # pack into (Nelements, 3, 3)
    row1 = torch.stack((dVx_dx, dVx_dy, dVx_dz), dim=1)
    row2 = torch.stack((dVy_dx, dVy_dy, dVy_dz), dim=1)
    row3 = torch.stack((dVz_dx, dVz_dy, dVz_dz), dim=1)
    
    return torch.stack((row1, row2, row3), dim=1)


def Voigt_to_3d(stress_tensor):
    N = stress_tensor.shape[0]
    states=stress_tensor.shape[2]
    stress_tensor_3d = torch.zeros(N, 3, 3,states,
                    device=stress_tensor.device,
                    dtype=stress_tensor.dtype)

    # diagonals
    stress_tensor_3d[:,0,0] = stress_tensor[:,0]   # stress_tensor_3d_xx
    stress_tensor_3d[:,1,1] = stress_tensor[:,1]   # stress_tensor_3d_yy
    stress_tensor_3d[:,2,2] = stress_tensor[:,2]   # stress_tensor_3d_zz

    # off‐diagonals (symmetric)
    stress_tensor_3d[:,0,1] = stress_tensor[:,3]   # stress_tensor_3d_xy
    stress_tensor_3d[:,1,0] = stress_tensor[:,3]

    stress_tensor_3d[:,0,2] = stress_tensor[:,4]   # stress_tensor_3d_xz
    stress_tensor_3d[:,2,0] = stress_tensor[:,4]

    stress_tensor_3d[:,1,2] = stress_tensor[:,5]   # stress_tensor_3d_yz
    stress_tensor_3d[:,2,1] = stress_tensor[:,5]

    return stress_tensor_3d

def taylor3(x, y, z, coeffs):
    """
    Evaluate a 3-var quadratic Taylor expansion
      f(x,y,z) = c0 
               + cx*x + cy*y + cz*z
               + cxx*x**2 + cyy*y**2 + czz*z**2
               + cxy*x*y + cxz*x*z + cyz*y*z

    Parameters
    ----------
    x, y, z : array-like or scalars (broadcastable to same shape)
    coeffs  : length-10 sequence [c0, cx, cy, cz, cxx, cyy, czz, cxy, cxz, cyz]

    Returns
    -------
    f      : same shape as the broadcast of x,y,z
    """
    c0, cx, cy, cz, cxx, cyy, czz, cxy, cxz, cyz = coeffs.detach().cpu().numpy()
    return (
        c0
      + cx * x + cy * y + cz * z
      + cxx * x**2 + cyy * y**2 + czz * z**2
      + cxy * x * y + cxz * x * z + cyz * y * z
    )


def print_vf_equations(V_NN, fmt="{:+.4f}"):
    """
    Given V_NN of shape (30,) containing the 10 Taylor coefficients
    for each of Vx, Vy, Vz (in that order), print out the symbolic
    form of each component.
    
    Parameters
    ----------
    V_NN : torch.Tensor or np.ndarray, shape (30,) or (1,30)
        The concatenated Taylor coefficients:
          [c0..c9 for Vx,  c0..c9 for Vy,  c0..c9 for Vz].
    fmt : str
        A Python format string for each coefficient, e.g. "{:+.3f}".
    """
    # get it into a flat numpy array of length 30
    if isinstance(V_NN, torch.Tensor):
        coeffs = V_NN.detach().cpu().numpy().ravel()
    else:
        coeffs = np.array(V_NN).ravel()
    if coeffs.size != 30:
        raise ValueError("Expected 30 coefficients, got %d" % coeffs.size)
    
    # split into three 10‐length rows
    comps = coeffs.reshape(3, 10)
    names = ("Vx", "Vy", "Vz")
    
    # the monomial basis for a 2nd‐order Taylor in 3D:
    basis = [
        "1",
        "x",
        "y",
        "z",
        "x**2",
        "x*y",
        "x*z",
        "y**2",
        "y*z",
        "z**2",
    ]
    
    for row, name in zip(comps, names):
        terms = []
        for c, mono in zip(row, basis):
            # skip tiny terms for readability
            if abs(c) < 1e-8:
                continue
            terms.append(f"{fmt.format(c)}*{mono}")
        expr = " + ".join(terms) if terms else "0"
        print(f"{name}(x,y,z) = {expr}")
