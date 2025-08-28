import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import magpylib as mag
import pyvista as pv


class Magnet:
    def __init__(self, position, radius, height, direction, mag_strength):
        self.position = position  # (x, y, z)
        self.radius = radius
        self.height = height
        self.direction = direction
        self.mag_strength = mag_strength

    def to_magpylib(self):
        return mag.magnet.Cylinder(
            magnetization=(0, 0, self.direction * self.mag_strength),
            dimension=(2 * self.radius, self.height),
            position=self.position
        )


def plot_field_xz(collection, magnets, xlim, zlim, resolution=100, density=10, plot_type='fieldline', show=True):
    """
    Plot the magnetic field lines in the XZ-plane (y=0).

    Parameters:
    - collection: Magpylib Collection
    - magnets: List of Magnet objects
    - xlim, zlim: Tuples defining plot limits
    - resolution: Grid resolution
    - density: Streamplot density
    """
    print('computing plot, please wait...')

    # Create grid in XZ-plane (Y=0)
    x = np.linspace(*xlim, resolution)
    z = np.linspace(*zlim, resolution)
    X, Z = np.meshgrid(x, z)
    Y = np.full_like(X, 2.5)
    positions = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    # Calculate magnetic field
    B = collection.getB(positions).reshape(X.shape + (3,))
    BX = B[:, :, 0]
    BZ = B[:, :, 2]
    magnitude = np.linalg.norm([BX, BZ], axis=0)

    # Plot field lines
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))

    if plot_type == 'fieldline':
        stream = ax.streamplot(
            X, Z, BX, BZ,
            color=np.log(magnitude + 1e-12),  # Avoid log(0)
            cmap='viridis',
            density=density,
            linewidth=1.0
        )
        plt.colorbar(stream.lines, label='Log( Field strength [T] )')
    else:
        qiver = ax.quiver(
            X, Z, BX/magnitude*0.04, BZ/magnitude*0.04,
            np.log(magnitude + 1e-12),  # Avoid log(0)
            cmap='viridis',
            scale=2,  # adjust to get good arrow length
            width=0.002,  # arrow width
            pivot='middle'  # arrow base in the middle of the vector
        )
        plt.colorbar(qiver, label='Log( Field strength [T] )')


    ax.set_facecolor((0.1, 0.1, 0.1))
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Z [mm]")
    ax.set_title("Magnetic Field Lines in XZ Plane")
    ax.grid(False)
    fig.set_facecolor((0.1, 0.1, 0.1))

    # Draw magnets as transparent rectangles
    for m in magnets:
        x0 = m.position[0] - m.radius
        z0 = m.position[2] - m.height / 2
        # Draw semi-transparent white filled rectangle
        face_rect = patches.Rectangle(
            (x0, z0),
            width=2 * m.radius,
            height=m.height,
            facecolor='white',
            edgecolor='none',
            alpha=0.3
        )
        ax.add_patch(face_rect)

        # Draw opaque black edge rectangle on top
        edge_rect = patches.Rectangle(
            (x0, z0),
            width=2 * m.radius,
            height=m.height,
            facecolor='none',
            edgecolor='white',
            linewidth=1.0,
            alpha=1.0  # fully opaque edge
        )
        ax.add_patch(edge_rect)

    plt.tight_layout()
    if show:
        plt.show()


def plot_field_rotation_xz_along_lines(collection, point_pairs, num_points=200, show=True):
    """
    Plots magnetic field rotation angles in the XZ-plane along multiple lines.

    Parameters:
    - collection: Magpylib Collection
    - point_pairs: List of (start_point, end_point) tuples; each point is (x, y, z)
    - num_points: Number of samples along each line
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 4))

    colors = ['#4ea6ff', '#ff9933', '#4dd26a', '#ff5c5c', '#b88cf0', '#a67c6c', '#f291d0', '#aaaaaa']

    for idx, (point_start, point_end, label) in enumerate(point_pairs):
        # Generate points along the line
        line_points = np.linspace(point_start, point_end, num_points)

        # Compute magnetic field
        B = collection.getB(line_points)

        # Project B onto XZ plane
        B_xz = B[:, [0, 2]]  # BX and BZ

        # Compute angle in XZ plane
        angles = -np.arctan2(B_xz[:, 1], B_xz[:, 0])  # angle in radians

        # Distance along the line
        distances = np.linalg.norm(line_points - point_start, axis=1)

        # Plot the curve
        color = colors[idx % len(colors)]
        ax.plot(distances, np.degrees(angles), color=color, linewidth=1.5, label=label)

    # Axis styling
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Field angle [degrees]")
    ax.set_title("Magnetic Field Rotation Along Lines in XZ Plane")
    ax.grid(True, linestyle='--', color='gray', linewidth=0.6, alpha=0.8)
    ax.set_facecolor((0.1, 0.1, 0.1))
    fig.set_facecolor((0.1, 0.1, 0.1))
    ax.legend()
    plt.tight_layout()
    if show:
        plt.show()


# ----------------------------------------------------------------------------------------------------------------------

def main():
    # Parameters
    num_magnets = 13
    spacing = 4             # mm between magnet centers
    radius = 1.9            # mm
    height = 4              # mm
    mag_strength = 800e3    # A/m
    show_3d_viewer = False  # show 3d viewer with simulation scene

    plot_range = [spacing * num_magnets + 20, 30]

    # Create magnet objects
    magnets = []
    for i in range(num_magnets):
        direction = 1 if i % 2 == 0 else -1
        x_pos = (i - num_magnets / 2) * spacing
        position = [x_pos, 0, 0]
        magnets.append(Magnet(position, radius, height, direction, mag_strength))

    # Create Magpylib collection
    mag_sources = [m.to_magpylib() for m in magnets]
    collection = mag.Collection(mag_sources)

    # show 3d viewer with simulation scene
    if show_3d_viewer:
        pl = pv.Plotter()
        collection.show(backend='pyvista', canvas=pl)
        pl.set_background((0.1, 0.1, 0.1))  # solid black

        plane = pv.Plane(center=(0, 2.5, 0), direction=(0, 1, 0), i_size=plot_range[1], j_size=plot_range[0])
        pl.add_mesh(plane, color='white', opacity=0.7)

        pl.show()

    # Plot the field
    plot_field_xz(
        collection=collection,
        magnets=magnets,
        xlim=(-plot_range[0]/2, plot_range[0]/2),
        zlim=(-plot_range[1]/2, plot_range[1]/2),
        resolution=100,
        density=10,
        show=False
    )

    plot_field_rotation_xz_along_lines(collection, [((-8.9, 2.5, 4), (8, 2.5, 4), 'Z=4mm'),
                                                    ((-8.9, 2.5, 6), (8, 2.5, 6), 'Z=6mm'),
                                                    ((-8.9, 2.5, 8), (8, 2.5, 8), 'Z=8mm')],
                                       show=False)

    # show all plots
    plt.show()


if __name__ == "__main__":
    main()