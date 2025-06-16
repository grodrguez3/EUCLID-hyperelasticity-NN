import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# --- 3) Define the plotting function ---
def plot_cube_faces(
    M, X, Y, Z,
    cmap: str = 'viridis',
    shading: str = 'nearest',
    figsize=(15,10),
    dpi=100
):
    """
    Plot the six faces of a cube-volume heatmap for scalar field M on grid X,Y,Z.
    """
    xs = X[0,:,0]
    ys = Y[:,0,0]
    zs = Z[0,0,:]

    fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=dpi)
    face_defs = [
        ('XY','z',  0),
        ('XY','z', -1),
        ('XZ','y',  0),
        ('XZ','y', -1),
        ('YZ','x',  0),
        ('YZ','x', -1),
    ]

    for ax, (plane, axis, idx) in zip(axes.flat, face_defs):
        if axis == 'z':
            Fi = M[:,:,idx]
            Xi = X[:,:,idx]
            Yi = Y[:,:,idx]
            xl, yl = 'x','y'
            title = f"z = {zs[idx]:.2f}"
        elif axis == 'y':
            Fi = M[idx,:,:]
            Xi = X[idx,:,:]
            Yi = Z[idx,:,:]
            xl, yl = 'x','z'
            title = f"y = {ys[idx]:.2f}"
        else:  # axis == 'x'
            Fi = M[:,idx,:]
            Xi = Y[:,idx,:]
            Yi = Z[:,idx,:]
            xl, yl = 'y','z'
            title = f"x = {xs[idx]:.2f}"

        im = ax.pcolormesh(Xi, Yi, Fi,
                           cmap=cmap, shading=shading)
        fig.colorbar(im, ax=ax, label='|V|')
        ax.set_aspect('equal')
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"{plane} @ {title}")

    plt.tight_layout()
    return fig, axes


def plot_cube_face_quivers(
    X, Y, Z,
    Vx, Vy, Vz,
    scale=50,
    width=0.003,
    cmap='viridis'
):
    """
    Draw quiver plots on the 6 boundary faces of the cube, coloring
    each arrow by its in‐plane magnitude.
    """
    xs = X[0,:,0]
    ys = Y[:,0,0]
    zs = Z[0,0,:]

    fig, axes = plt.subplots(2, 3, figsize=(15,10), dpi=100)
    faces = [
        ('XY','z',  0),
        ('XY','z', -1),
        ('XZ','y',  0),
        ('XZ','y', -1),
        ('YZ','x',  0),
        ('YZ','x', -1),
    ]

    for ax, (plane, axis, idx) in zip(axes.flat, faces):
        if axis == 'z':
            # XY @ z-index
            Xi, Yi = X[:,:,idx], Y[:,:,idx]
            Ui, Vi = Vx[:,:,idx], Vy[:,:,idx]
            title   = f"z = {zs[idx]:.2f}"
            xl, yl  = 'x','y'
        elif axis == 'y':
            # XZ @ y-index
            Xi, Yi = X[idx,:,:], Z[idx,:,:]
            Ui, Vi = Vx[idx,:,:], Vz[idx,:,:]
            title   = f"y = {ys[idx]:.2f}"
            xl, yl  = 'x','z'
        else:  # axis == 'x'
            # YZ @ x-index
            Xi, Yi = Y[:,idx,:], Z[:,idx,:]
            Ui, Vi = Vy[:,idx,:], Vz[:,idx,:]
            title   = f"x = {xs[idx]:.2f}"
            xl, yl  = 'y','z'

        # compute in‐plane magnitude for coloring
        Ci = np.sqrt(Ui**2 + Vi**2)

        # build a common Normalize across all faces?
        # here we normalize each face separately, but you could
        # compute global vmin/vmax outside the loop if preferred
        norm = Normalize(vmin=Ci.min(), vmax=Ci.max())

        q = ax.quiver(
            Xi, Yi,
            Ui, Vi,
            Ci,                # color array
            cmap=cmap,
            norm=norm,
            angles='xy', scale_units='xy',
            scale=scale, width=width
        )
        cb = fig.colorbar(q, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label('|in-plane V|')
        ax.set_aspect('equal')
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"{plane} @ {title}")

    plt.tight_layout()
    return fig, axes