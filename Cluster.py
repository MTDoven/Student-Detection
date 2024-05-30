import warnings
import torch
from torch.nn.functional import pairwise_distance


def k_means(init_centers, data_points, num_iterations=20):
    """
    K-means clustering algorithm
    :parameter init_centers: shape = (num_clusters, features_dim)
    :parameter data_points: shape = (num_data_points, features_dim)
    :parameter num_iterations: max number of iterations
    :return cluster_assignments: shape = (num_clusters, features_dim)
    :return final_centers: shape = (num_clusters, features_dim)
    """
    for _ in range(num_iterations):
        # calculate distance to each center
        distances = pairwise_distance(data_points.unsqueeze(1), init_centers.unsqueeze(0), p=2)
        # assign each point to the center
        cluster_assignments = torch.argmin(distances, dim=1)
        # update center
        new_centers = []
        for i in range(init_centers.size(0)):
            cluster_points = data_points[cluster_assignments == i]
            if cluster_points.numel() > 0:  # avoid divide 0
                new_center = cluster_points.mean(dim=0)
            else:  # avoid an empty center
                new_center = init_centers[i]
            new_centers.append(new_center)
        new_centers = torch.stack(new_centers)
        # center not change
        if torch.allclose(new_centers, init_centers, atol=1e-4):
            break
        init_centers = new_centers
    else:
        warnings.warn("k-means algorithm may have not converged, but reached max number of iterations.")
    return cluster_assignments, init_centers


if __name__ == "__main__":
    # prepare data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_clusters = 50
    num_features = 2
    num_data_points = 1000
    init_centers = torch.randn(num_clusters, num_features).to(device)
    data_points = torch.randn(num_data_points, num_features).to(device)
    # do cluster
    cluster_assignments, final_centers = k_means(init_centers, data_points)
    print("cluster_assignments:", cluster_assignments)
    print("final_centers:", final_centers)