import numpy as np

def uniform_sampling_on_simplex(dim, n_pts=1):
    # Draw (dim-1) uniforms for each sample
    u = np.sort(np.random.rand(n_pts, dim-1), axis=1)
    # Add 0 at start and 1 at end
    u = np.hstack([np.zeros((n_pts,1)), u, np.ones((n_pts,1))])
    # Take differences along each row
    X = np.diff(u, axis=1)
    return X  # shape: (n_pts, dim)


if __name__ == "__main__":
    print(uniform_sampling_on_simplex(4, n_pts=5).shape)