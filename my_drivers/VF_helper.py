import torch


def calculate_point(centroids, state):
    delta=centroids[:,:,state] #-centroids[:,:,0]
    return delta

def calculate_point_all_states(centroids, state):
    delta=centroids #[:,:,state] -centroids[:,:,0]
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

def construct_VF_MD(V_NN,delta_xyz):
    Vf = (
        V_NN[0]
        + V_NN[1] * delta_xyz[:,0,:]
        + V_NN[2] * delta_xyz[:,1,:]
        + V_NN[3] * delta_xyz[:,2,:]
        + V_NN[4] * delta_xyz[:,0,:]**2
        + V_NN[5] * delta_xyz[:,1,:]**2
        + V_NN[6] * delta_xyz[:,2,:]**2
        + V_NN[7] * (delta_xyz[:,0,:] * delta_xyz[:,1,:])
        + V_NN[8] * (delta_xyz[:,0,:] * delta_xyz[:,2,:])
        + V_NN[9] * (delta_xyz[:,1,:] * delta_xyz[:,2,:])
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
    dVx_dx = +a1 + 2*a4*dx +    a7*dy +    a8*dz
    dVx_dy = +a2 +    a7*dx + 2*a5*dy +    a9*dz
    dVx_dz = +a3 +    a8*dx +    a9*dy + 2*a6*dz

    dVy_dx = +b1 + 2*b4*dx +    b7*dy +    b8*dz
    dVy_dy = +b2 +    b7*dx + 2*b5*dy +    b9*dz
    dVy_dz = +b3 +    b8*dx +    b9*dy + 2*b6*dz

    dVz_dx = +c1 + 2*c4*dx +    c7*dy +    c8*dz
    dVz_dy = +c2 +    c7*dx + 2*c5*dy +    c9*dz
    dVz_dz = +c3 +    c8*dx +    c9*dy + 2*c6*dz

    # pack into (Nelements, 3, 3)
    row1 = torch.stack((dVx_dx, dVx_dy, dVx_dz), dim=1)
    row2 = torch.stack((dVy_dx, dVy_dy, dVy_dz), dim=1)
    row3 = torch.stack((dVz_dx, dVz_dy, dVz_dz), dim=1)
    
    return torch.stack((row1, row2, row3), dim=1)



def construct_VF_gradients_MD(V_NN: torch.Tensor, delta_xyz: torch.Tensor) -> torch.Tensor:
    """
    V_NN:  either shape (30,), (30,1), or (1,30)
    delta_xyz: shape (Nelements, 3)
    returns grad_V of shape (Nelements, 3, 3)
    """
    # flatten to (30,)
    coeffs = V_NN.squeeze()
    if coeffs.numel() != 30:
        raise ValueError(f"expected 30 coefficients, got {coeffs.shape}")

    dx = delta_xyz[:, 0,:]
    dy = delta_xyz[:, 1,:]
    dz = delta_xyz[:, 2,:]

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
    dVx_dx = +a1 + 2*a4*dx +    a7*dy +    a8*dz
    dVx_dy = +a2 +    a7*dx + 2*a5*dy +    a9*dz
    dVx_dz = +a3 +    a8*dx +    a9*dy + 2*a6*dz

    dVy_dx = +b1 + 2*b4*dx +    b7*dy +    b8*dz
    dVy_dy = +b2 +    b7*dx + 2*b5*dy +    b9*dz
    dVy_dz = +b3 +    b8*dx +    b9*dy + 2*b6*dz

    dVz_dx = +c1 + 2*c4*dx +    c7*dy +    c8*dz
    dVz_dy = +c2 +    c7*dx + 2*c5*dy +    c9*dz
    dVz_dz = +c3 +    c8*dx +    c9*dy + 2*c6*dz

    # pack into (Nelements, 3, 3)
    row1 = torch.stack((dVx_dx, dVx_dy, dVx_dz), dim=1)
    row2 = torch.stack((dVy_dx, dVy_dy, dVy_dz), dim=1)
    row3 = torch.stack((dVz_dx, dVz_dy, dVz_dz), dim=1)
    
    return torch.stack((row1, row2, row3), dim=1)

def construct_VF_gradients_MD(V_NN: torch.Tensor, delta_xyz: torch.Tensor) -> torch.Tensor:
    """
    V_NN:  either shape (30,), (30,1), or (1,30)
    delta_xyz: shape (Nelements, 3)
    returns grad_V of shape (Nelements, 3, 3)
    """
    # flatten to (30,)
    coeffs = V_NN.squeeze()
    if coeffs.numel() != 30:
        raise ValueError(f"expected 30 coefficients, got {coeffs.shape}")

    dx = delta_xyz[:, 0,:]
    dy = delta_xyz[:, 1,:]
    dz = delta_xyz[:, 2,:]

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
    dVx_dx = +a1 + 2*a4*dx +    a7*dy +    a8*dz
    dVx_dy = +a2 +    a7*dx + 2*a5*dy +    a9*dz
    dVx_dz = +a3 +    a8*dx +    a9*dy + 2*a6*dz

    dVy_dx = +b1 + 2*b4*dx +    b7*dy +    b8*dz
    dVy_dy = +b2 +    b7*dx + 2*b5*dy +    b9*dz
    dVy_dz = +b3 +    b8*dx +    b9*dy + 2*b6*dz

    dVz_dx = +c1 + 2*c4*dx +    c7*dy +    c8*dz
    dVz_dy = +c2 +    c7*dx + 2*c5*dy +    c9*dz
    dVz_dz = +c3 +    c8*dx +    c9*dy + 2*c6*dz

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


def Voigt_to_3d_v2(stress_tensor):
    N = stress_tensor.shape[0]
    states=stress_tensor.shape[2]
    stress_tensor_3d = torch.zeros(N, 3, 3,states,
                    device=stress_tensor.device,
                    dtype=stress_tensor.dtype)
    print('V2')
    # diagonals
    stress_tensor_3d[:,0,0] = stress_tensor[:,0]   # stress_tensor_3d_xx
    stress_tensor_3d[:,1,1] = stress_tensor[:,1]   # stress_tensor_3d_yy
    stress_tensor_3d[:,2,2] = stress_tensor[:,2]   # stress_tensor_3d_zz

    # off‐diagonals (symmetric)
    stress_tensor_3d[:,0,1] = stress_tensor[:,3]   # stress_tensor_3d_xy
    stress_tensor_3d[:,1,0] = stress_tensor[:,3]

    stress_tensor_3d[:,0,2] = stress_tensor[:,5]   # stress_tensor_3d_xz
    stress_tensor_3d[:,2,0] = stress_tensor[:,5]

    stress_tensor_3d[:,1,2] = stress_tensor[:,4]   # stress_tensor_3d_yz
    stress_tensor_3d[:,2,1] = stress_tensor[:,4]

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


def organize_inputs(x):
    # For clarity, we slice out each component as a (batch_size×1) tensor:
    F11 = x[:, 0:1,:]  # shape = (B,1)
    F12 = x[:, 1:2,:]
    F13 = x[:, 2:3,:]

    F21 = x[:, 3:4,:]
    F22 = x[:, 4:5,:]
    F23 = x[:, 5:6,:]

    F31 = x[:, 6:7,:]
    F32 = x[:, 7:8,:]
    F33 = x[:, 8:9,:]
    
    #Centroid positions
    Xc  = x[:,  9:10,:]
    Yc  = x[:, 10:11,:]
    Zc  = x[:, 11:12,:]

    # 1) Build the Right Cauchy‐Green tensor C = F^T F, component‐wise:
    #    C11 = F11^2 + F21^2 + F31^2
    C11 = F11**2 + F21**2 + F31**2

    #    C12 = F11·F12 + F21·F22 + F31·F32
    C12 = F11*F12 + F21*F22 + F31*F32

    #    C13 = F11·F13 + F21·F23 + F31·F33
    C13 = F11*F13 + F21*F23 + F31*F33

    #    C22 = F12^2 + F22^2 + F32^2
    C22 = F12**2 + F22**2 + F32**2

    #    C23 = F12·F13 + F22·F23 + F32·F33
    C23 = F12*F13 + F22*F23 + F32*F33

    #    C33 = F13^2 + F23^2 + F33^2
    C33 = F13**2 + F23**2 + F33**2

    # (Note: C21 = C12, C31 = C13, C32 = C23, but we only need them for invariants.)

    # 2) Compute the three invariants of C:

    #   I1 = trace(C) = C11 + C22 + C33
    I1 = C11 + C22 + C33

    #   I2 = sum of principal 2×2 minors of C:
    #        I2 =  C11·C22 + C11·C33 + C22·C33  –  (C12^2 + C13^2 + C23^2)
    I2 = (C11 * C22) + (C11 * C33) + (C22 * C33) - (C12**2 + C13**2 + C23**2)

    #   I3 = det(C).  But det(C) = (det F)².  So first form det(F):
    detF = (
        F11 * (F22*F33 - F23*F32)
    - F12 * (F21*F33 - F23*F31)
    + F13 * (F21*F32 - F22*F31)
    )
    I3 = detF**2

    # 3) Now form the modified invariants K1, K2, K3:

    #    J  = sqrt(I3)
    J = torch.sqrt(I3)

    #    K1 = I1 * I3^(–1/3) – 3.0
    K1 = I1 * torch.pow(I3, -1.0/3.0) - 3.0

    #    K2 = I2 * I3^(–2/3) – 3.0
    K2 = I2 * torch.pow(I3, -2.0/3.0) - 3.0

    #    K3 = (J - 1.0)^2
    K3 = (J - 1.0)**2

    #K2 = (I1 + I3 - 1.) * torch.pow(I3,-2./3.) - 3.0
    # Concatenate feature

    x = torch.cat((K1,K2,K3,Xc,Yc,Zc),dim=1).float()

    return x


def evaluate_vf_values(V_NN: torch.Tensor, delta_xyz: torch.Tensor):
    """
    Evaluate p global Taylor‐expanded virtual fields at N points.

    Args:
        V_NN       (torch.Tensor): shape [p, 30], Taylor coefficients for p fields
        delta_xyz  (torch.Tensor): shape [N,  3], the (x,y,z) offsets

    Returns:
        torch.Tensor: shape [p, N, 3], where each slice [k] is the (Vx,Vy,Vz)
                      values of the k+th virtual field at all N points.
    """
    # Split out the 10‐wide blocks for Vx, Vy, Vz respectively
    Vx_coeff = V_NN[:,  0:10]   # [p,10]
    Vy_coeff = V_NN[:, 10:20]
    Vz_coeff = V_NN[:, 20:30]

    # Build the N×10 design matrix of monomials
    x, y, z = delta_xyz[:,0], delta_xyz[:,1], delta_xyz[:,2]
    B = torch.stack([
        torch.ones_like(x),
        x, y, z,
        x**2, y**2, z**2,
        x*y, x*z, y*z
    ], dim=1)  # [N,10]

    # Multiply coefficients by monomials: (p×10) @ (10×N) → (p×N)
    Vx_vals = Vx_coeff @ B.T
    Vy_vals = Vy_coeff @ B.T
    Vz_vals = Vz_coeff @ B.T

    # Stack into (p, N, 3)
    return torch.stack([Vx_vals, Vy_vals, Vz_vals], dim=2)



def construct_VF_gradients_batch(V_NN: torch.Tensor, delta_xyz: torch.Tensor) -> torch.Tensor:
    """
    Vectorized gradient of p global Taylor‐expanded VFs at N points.

    Args:
        V_NN       (torch.Tensor): shape [p, 30], Taylor coefficients for p VFs
        delta_xyz  (torch.Tensor): shape [N,  3], the (x,y,z) offsets

    Returns:
        torch.Tensor: shape [p, N, 3, 3], where
                      out[k,i] is the 3×3 gradient tensor of the k-th VF at point i.
    """
    p, coeff_len = V_NN.shape
    N, _         = delta_xyz.shape
    if coeff_len != 30:
        raise ValueError(f"Expected V_NN with 30 coefficients, got {coeff_len}")

    # Prepare delta for broadcasting: [1, N]
    dx = delta_xyz[:, 0].unsqueeze(0)
    dy = delta_xyz[:, 1].unsqueeze(0)
    dz = delta_xyz[:, 2].unsqueeze(0)

    # Split into Vx, Vy, Vz coefficient blocks [p,10]
    a = V_NN[:,  0:10]
    b = V_NN[:, 10:20]
    c = V_NN[:, 20:30]

    # Helper to extract and unsqueeze a given index
    def e(block, idx):
        return block[:, idx].unsqueeze(1)  # [p,1]

    # Extract and unsqueeze needed coeffs
    a1,a2,a3 = e(a,1), e(a,2), e(a,3)
    a4,a5,a6 = e(a,4), e(a,5), e(a,6)
    a7,a8,a9 = e(a,7), e(a,8), e(a,9)
    b1,b2,b3 = e(b,1), e(b,2), e(b,3)
    b4,b5,b6 = e(b,4), e(b,5), e(b,6)
    b7,b8,b9 = e(b,7), e(b,8), e(b,9)
    c1,c2,c3 = e(c,1), e(c,2), e(c,3)
    c4,c5,c6 = e(c,4), e(c,5), e(c,6)
    c7,c8,c9 = e(c,7), e(c,8), e(c,9)

    # Compute partials [p,N]
    dVx_dx = +a1 + 2*a4*dx +    a7*dy +    a8*dz
    dVx_dy = +a2 +    a7*dx + 2*a5*dy +    a9*dz
    dVx_dz = +a3 +    a8*dx +    a9*dy + 2*a6*dz

    dVy_dx = +b1 + 2*b4*dx +    b7*dy +    b8*dz
    dVy_dy = +b2 +    b7*dx + 2*b5*dy +    b9*dz
    dVy_dz = +b3 +    b8*dx +    b9*dy + 2*b6*dz

    dVz_dx = +c1 + 2*c4*dx +    c7*dy +    c8*dz
    dVz_dy = +c2 +    c7*dx + 2*c5*dy +    c9*dz
    dVz_dz = +c3 +    c8*dx +    c9*dy + 2*c6*dz

    # Stack into [p,N,3,3]
    grad = torch.stack([
        torch.stack([dVx_dx, dVx_dy, dVx_dz], dim=2),
        torch.stack([dVy_dx, dVy_dy, dVy_dz], dim=2),
        torch.stack([dVz_dx, dVz_dy, dVz_dz], dim=2)
    ], dim=2)

    return grad


def construct_vf_gradients_per_element(
    V_NN: torch.Tensor,       # [E, p, 30]
    delta_xyz: torch.Tensor   # [E, 3]
) -> torch.Tensor:           # returns [E, p, 3, 3]
    """
    Compute ∇u_v for each element’s Taylor‐expanded virtual fields.

    Args:
      V_NN      (E×p×30): TE coeffs for p fields on E elements
      delta_xyz (E×3)   : (x,y,z) offsets at each element

    Returns:
      grad      (E×p×3×3): gradient ∂(Vx,Vy,Vz)/∂(x,y,z) per element & field
    """
    E, p, K = V_NN.shape
    assert K == 30, f"Expect 30 coeffs, got {K}"
    # unpack offsets and make [E×1] for broadcasting
    dx = delta_xyz[:, 0].unsqueeze(-1)  # → [E,1]
    dy = delta_xyz[:, 1].unsqueeze(-1)
    dz = delta_xyz[:, 2].unsqueeze(-1)

    # split coeff blocks → each [E, p, 10]
    a = V_NN[:, :,  0:10]   # Vx coeffs
    b = V_NN[:, :, 10:20]   # Vy coeffs
    c = V_NN[:, :, 20:30]   # Vz coeffs

    # index‐shortcut
    def coeff(block, idx):
        # block: [E,p,10], idx in [0..9] → [E,p]
        return block[:, :, idx]

    # extract the needed coefficient slices [E,p]
    a1, a2, a3 = coeff(a,1),  coeff(a,2),  coeff(a,3)
    a4, a5, a6 = coeff(a,4),  coeff(a,5),  coeff(a,6)
    a7, a8, a9 = coeff(a,7),  coeff(a,8),  coeff(a,9)

    b1, b2, b3 = coeff(b,1),  coeff(b,2),  coeff(b,3)
    b4, b5, b6 = coeff(b,4),  coeff(b,5),  coeff(b,6)
    b7, b8, b9 = coeff(b,7),  coeff(b,8),  coeff(b,9)

    c1, c2, c3 = coeff(c,1),  coeff(c,2),  coeff(c,3)
    c4, c5, c6 = coeff(c,4),  coeff(c,5),  coeff(c,6)
    c7, c8, c9 = coeff(c,7),  coeff(c,8),  coeff(c,9)

    # now compute the analytic partials [E,p]
    dVx_dx = a1 + 2*a4*dx +    a7*dy +    a8*dz
    dVx_dy = a2 +    a7*dx + 2*a5*dy +    a9*dz
    dVx_dz = a3 +    a8*dx +    a9*dy + 2*a6*dz

    dVy_dx = b1 + 2*b4*dx +    b7*dy +    b8*dz
    dVy_dy = b2 +    b7*dx + 2*b5*dy +    b9*dz
    dVy_dz = b3 +    b8*dx +    b9*dy + 2*b6*dz

    dVz_dx = c1 + 2*c4*dx +    c7*dy +    c8*dz
    dVz_dy = c2 +    c7*dx + 2*c5*dy +    c9*dz
    dVz_dz = c3 +    c8*dx +    c9*dy + 2*c6*dz

    # stack to [E,p,3,3]
    #  first stack each row of the Jacobian:
    row0 = torch.stack([dVx_dx, dVx_dy, dVx_dz], dim=2)  # [E,p,3]
    row1 = torch.stack([dVy_dx, dVy_dy, dVy_dz], dim=2)
    row2 = torch.stack([dVz_dx, dVz_dy, dVz_dz], dim=2)
    grad = torch.stack([row0, row1, row2], dim=2)       # [E,p,3,3]

    return grad
def evaluate_vf_values_per_element(
    V_NN: torch.Tensor,      # [E, p, 30]
    delta_xyz: torch.Tensor  # [E, 3]
) -> torch.Tensor:          # returns [E, p, 3]
    E, p, K = V_NN.shape
    assert K == 30, "expected 30 coeffs per field"
    # Build per-element monomial basis B[e] = [1, x, y, z, x^2, y^2, z^2, x*y, x*z, y*z]
    x = delta_xyz[:, 0]
    y = delta_xyz[:, 1]
    z = delta_xyz[:, 2]
    B = torch.stack([
        torch.ones_like(x),
        x, y, z,
        x**2, y**2, z**2,
        x*y, x*z, y*z
    ], dim=1)                    # → [E, 10]

    # Split out the x/y/z coefficient blocks:
    Vx_coeff = V_NN[:, :,  0:10]  # [E, p, 10]
    Vy_coeff = V_NN[:, :, 10:20]  # [E, p, 10]
    Vz_coeff = V_NN[:, :, 20:30]  # [E, p, 10]

    # Broadcast B over the p-dimension, then dot-product along that last axis:
    #   [E, p, 10] * [E, 1, 10] → [E, p, 10], summing over dim=2 → [E, p]
    B_exp = B.unsqueeze(1)        # [E, 1, 10]

    Vx_vals = (Vx_coeff * B_exp).sum(dim=2)  # [E, p]
    Vy_vals = (Vy_coeff * B_exp).sum(dim=2)  # [E, p]
    Vz_vals = (Vz_coeff * B_exp).sum(dim=2)  # [E, p]

    # Stack into the final [E, p, 3] tensor:
    return torch.stack([Vx_vals, Vy_vals, Vz_vals], dim=2)
