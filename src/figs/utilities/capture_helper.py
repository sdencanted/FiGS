"""
Helper functions for capture generation.
"""

import numpy as np
import torch
from tqdm.auto import tqdm

def distribute_values(values,num_picks):
    values = np.array(values)
    selected_values = [values[0]]  # Start with the first element
    
    for _ in range(num_picks - 1):
        remaining = [x for x in values if x not in selected_values]
        max_dist = -1
        best_candidate = None
        
        for candidate in remaining:
            # Compute the minimum distance of this candidate to the already selected
            min_dist = min(abs(candidate - s) for s in selected_values)
            if min_dist > max_dist:
                max_dist = min_dist
                best_candidate = candidate
        
        selected_values.append(best_candidate)

    selected_values.sort()

    return selected_values


def compute_ransac_transform(W1:np.ndarray, W2:np.ndarray,
                             n_batch:int=3,threshold:float=5e-2, max_iterations:int=1000) -> np.ndarray:
    """
    Compute the RANSAC transformation between two sets of 3D points.

    Args:
        W1:             First set of 3D points (3 x N)
        W2:             Second set of 3D points (3 x N)
        threshold:      Inlier threshold
        max_iterations: Maximum number of iterations

    Returns:
        co:             Scaling factor
        Ro:             Rotation matrix
        to:             Translation
    """

    assert W1.shape == W2.shape, "Input arrays must have the same shape"
    assert W1.shape[0] == 3, "Input arrays must have 3 rows (3D points)"

    N = W1.shape[1]
    best_inliers = 0
    co,Ro,to = None,None,None

    for _ in tqdm(range(max_iterations)):
        # Randomly sample 3 points
        idx = np.random.choice(N, n_batch, replace=False)
        W1_sample = W1[:, idx]
        W2_sample = W2[:, idx]

        # Compute centroids
        centroid_W1 = np.mean(W1_sample, axis=1, keepdims=True)
        centroid_W2 = np.mean(W2_sample, axis=1, keepdims=True)

        # Center the points
        W1_centered = W1_sample - centroid_W1
        W2_centered = W2_sample - centroid_W2

        # Compute scaling factor
        ce = np.linalg.norm(W2_centered) / np.linalg.norm(W1_centered)

        # Apply scaling to W1
        W1_centered_scaled = W1_centered*ce

        # Compute cross-covariance matrix
        H = W1_centered_scaled @ W2_centered.T

        # Singular Value Decomposition (SVD)
        U, _, Vt = np.linalg.svd(H)
        Re = Vt.T @ U.T

        # Ensure a proper rotation matrix (det(R) = 1)
        if np.linalg.det(Re) < 0:
            Vt[2, :] *= -1
            Re = Vt.T @ U.T

        te = centroid_W2.flatten() - ce*Re@centroid_W1.flatten()

        # Count inliers
        W1_transformed = ce*Re@W1 + te[:, np.newaxis]
        distances = np.linalg.norm(W2 - W1_transformed, axis=0)
        inliers = np.sum(distances < threshold)

        # Update best transformation if current one is better
        if inliers > best_inliers:
            best_inliers = inliers
            co,Ro,to = ce,Re,te

    # Compute some statistics
    total_cost = 0.0
    for i in range(N):
        W1_transformed = co*Ro@W1[:, i] + to
        cost = np.linalg.norm(W2[:, i] - W1_transformed)
        total_cost += cost

    print(f"RANSAC: {best_inliers} inliers out of {N} points with threshold {threshold}m")
    
    return co.copy(),Ro.copy(),to.copy()

def compute_default_transform(X:np.ndarray, Y:np.ndarray) -> np.ndarray:
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    N, m = X.shape

    mux = torch.mean(X, 0, True)
    muy = torch.mean(Y, 0, True)

    Yd = (Y - muy).unsqueeze(-1)
    Xd = (X - mux).unsqueeze(1)
    sx = torch.sum(torch.norm(Xd.squeeze(), dim=1) ** 2) / N
    Sxy = (1 / N) * torch.sum(torch.matmul(Yd, Xd), dim=0)

    if torch.linalg.matrix_rank(Sxy) < m:
        raise NameError("Absolute orientation transformation does not exist!")

    U, D, Vt = torch.linalg.svd(Sxy, full_matrices=True)
    S = torch.eye(m).to(dtype=Vt.dtype)
    if torch.linalg.det(Sxy) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt
    c = torch.trace(torch.diag(D) @ S) / sx
    t = muy.T - c * (R @ mux.T)


    return c, R, t
