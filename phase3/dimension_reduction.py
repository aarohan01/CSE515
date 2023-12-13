# Add all the dimension reduction algotihms here
import numpy as np
import torch
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation as LDA

import numpy as np
from matplotlib import pyplot as plt
import distances

from normalisation import *
from utils import convert_higher_dims_to_2d
import tensorly as tl


def svd(data_matrix : np.ndarray, k=None, center=False ) -> np.ndarray :
    
    '''
    Single Value Decomposition : Dimensionality reduction using eigen decomposition.
    
    Takes data matrix a 2D numpy ndarray as input. Ex : imageID x feature descriptor. 
    Performs the SVD dimensionality reduction using eigen decomposition.
    Returns the left factor matrix U , core matrix S and right factor matrix V^T
    Where D = USV^T and k is the desired latent semantics.
    D [mxn] = U [mxk] S [kxk] V^T [kxn]
    
    By default gives all the latent features unless k is provided.
    '''
    data_matrix = convert_higher_dims_to_2d(data_matrix)

    #Check if data_matrix is a 2D np array
    if not isinstance(data_matrix, np.ndarray) or data_matrix.ndim != 2:
        raise ValueError("Input data matrix should be a 2D numpy array")

    #Datamatrix : images * feature_descriptor matrix
    D = data_matrix
    
    #Center data matrix by subracting column means
    if center :
        D = D - np.mean(D, axis=0)
   
    #Datamatrix transpose
    DT = np.transpose(D)

    #Create a symmetric matrix for decomposition
    DTD = DT @ D
    
    #Eigen decomposition for V 
    eigenvalues, eigenvectors = np.linalg.eig(DTD)
    
    #Eigen decomposition, np.linalg.eig does not return sorted values 
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:,sorted_indices]

    #Calculate the right factor matrix VT
    V = sorted_eigenvectors
    VT = V.T
    
    #Calculate the core matrix S
    singular_values = np.sqrt(np.abs(sorted_eigenvalues)) 

    S = np.diag(singular_values)  

    SI = np.linalg.inv(S)
    
    
    '''
    Derive U from S and V https://cs.fit.edu/~dmitra/SciComp/Resources/singular-value-decomposition-fast-track-tutorial.pdf
    Since V is orthonormal V^T is V Inverse and D = USV^T 
    To get U we can multiply both sides composition of V and S^I
    U = DVS^i
    Gives SVD where If D is mxn then U is mxk S is kxk and VT is kxn
    '''

    #Calculate the left factor matrix U
    U = D @ V @ SI
    

    #Return k latent sematics along with the truncated matrices 
    if k != None : 
        if k <= len(singular_values) :
            return U[:,:k], S[:k,:k], VT[:k,:]
        else  :
            raise ValueError("k is higher than the discovered latent features")

    return U,S,VT


def nmf_als(V, K, iteration=200, tol=1, alpha=0.01):
    """
    Non-negative Matrix Factorization (NMF) using Alternating Least Squares (ALS) method.

    Parameters:
        V (numpy.ndarray): The input data matrix of shape (m, ...).
        K (int): Desired number of latent semantics
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance to stop the iterations.
        alpha (float): Regularization parameter.

    Returns:
        W (numpy.ndarray): The factorization matrix of shape (m, rank).
        H (numpy.ndarray): The factorization matrix of shape (rank, n).
    """
    # convert higher dims to 2d
    V = convert_higher_dims_to_2d(V)

    # Shape of the original vector
    m, n = V.shape

    # normalise here to [0-1]
    normalisation = Normalisation()
    V = normalisation.train_normalize_min_max(V)

    # Initialize W and H with random non-negative values
    np.random.seed(0)
    W = np.random.rand(m, K)
    H = np.random.rand(K, n)

    for iter in tqdm(range(iteration)):
        # Update H 
        H = H * (np.dot(W.T, V)) / (np.dot(np.dot(W.T, W), H))
        # Update W 
        W = W * (np.dot(V, H.T) ) / (np.dot(W, np.dot(H, H.T)) )

        # Error or distance from the original matrix , NMF for very large matrix have distance > 300
        residual = np.square(np.linalg.norm(V - np.dot(W, H)))
        

        # Check for convergence
        if residual < tol:
            break
    print(residual)
    return W, H

def cp_decompose(data, rank):
    """CP deocompose using Tensorly with Parafac, This decomposes the 3rd Order tensor to weights and Factors
    Parameters:
        k (int): Desired input for dimensions required
        Data : 3rd Order tensor with shape of 4339(images) x fearure_shape x 101(labels)
    returns :
        Factors containing Image Weights , Label wieghts and Feature weight matrix
        Weights of the core tensor
        """
    factors = tl.decomposition.parafac(data, rank, n_iter_max=5, init='random', verbose=1, normalize_factors=True)

    return factors


def lda(data_collection: np.ndarray, k: int) -> np.ndarray:
    """
    returns reduced matrix using sklearn's LDA inbuilt function
    Parameters: 
        k (int): Desired input for dimensions required
        data_collections (numpy.ndarray): The input data matrix of shape (m, ...)
    Returns:
        reduced_data (numpy.ndarray): Data matrix of shape (m, k)
    source code reference: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html#sklearn.decomposition.LatentDirichletAllocation
    inbuilt-method: https://github.com/scikit-learn/scikit-learn/blob/d99b728b3/sklearn/base.py#L888
    explanation: https://scikit-learn.org/stable/modules/decomposition.html#latentdirichletallocation
    """
    # converting data_collection from multi dimensions to 2 dimensions
    data_collection = convert_higher_dims_to_2d(data_collection)
    
    # normalise value to [0,1]
    normalisation = Normalisation()
    data_collection = normalisation.train_normalize_min_max(data_collection)

    lda_model = LDA(n_components=k, max_iter=10, random_state=42, learning_method='batch', verbose=1)
    # print(train_data.shape[0] == len(train_label))
    reduced_data = lda_model.fit_transform(data_collection)
    print(f'Reducing {data_collection.shape} => {reduced_data.shape}')
    if k == 1: 
        # every value comes out as [1.]
        print(reduced_data)
        raise ValueError
    return reduced_data


def extractDistanceFeatures(X,C):
    X_r = []
    for i in range(C.shape[0]):
        ci = C[i,:]
        di = np.sum((X-ci)**2,axis=1)
        X_r.append(di)
    return np.array(X_r).T

def K_means(k, data_collection):
    print("Original Shape ", data_collection.shape)
    
    if data_collection.ndim >= 2:
        data_collection = data_collection.reshape(data_collection.shape[0], -1)
        print("Reshaped Data Shape ", data_collection.shape)
    
    # ------ TRIED With library
    # json_data = mongo_query.query_all("fc_layer")
    # all_labels = [json_data[key]["label"] for key in range(0,8677, 1)]
    # unique_labels = np.unique(all_labels)
    # print(unique_labels.shape)
    # feature_descriptors = [json_data[key]["feature_descriptor"] for key in range(0,8677, 2)]
    
    # kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    # cluster_assignments = kmeans.fit_predict(data_collection)

    def initialize_centroids(data, k, seed_):
        # random number generator
        np.random.seed(seed_) # IMP: to produce same sequence everytime
        indices = np.random.choice(len(data), k, replace=False)
        # selected 'k' random data points to be centroids
        centroids = data[indices]
        return centroids

    # Assign data point to the nearest centroid
    def assign_to_centroids(data, centroids):
        # Initialize an array to store the cluster assignments
        cluster_assignment = np.zeros(len(data), dtype=float)

        for i in range(len(data)): # for each data point
            min_distance = float('inf')
            for j in range(len(centroids)): # check with each centroid of cluster
                distance = distances.cosine_distance(data[i], centroids[j])
                if distance < min_distance:
                    min_distance = distance
                    cluster_assignment[i] = j  # Assign the data point to the nearest centroid
        return cluster_assignment

    # Update centroids as the mean of the assigned data points
    def update_centroids(data, cluster_assignment, k):
        new_centroids = np.zeros((k, data.shape[1]))
        for i in range(k):
            new_centroids[i] = np.mean(data[cluster_assignment == i], axis=0)
        return new_centroids
    
    # K-means clustering
    def k_means(data, k, max_iterations=300, n_init=10):
        best_centroids, best_inertia = None, float('inf')
        for x in range(n_init):
            # This is the basic approach
            centroids = initialize_centroids(data, k, x)
            for _ in range(max_iterations):
                cluster_assignments = assign_to_centroids(data, centroids)
                new_centroids = update_centroids(data, cluster_assignments, k)
                if np.all(centroids == new_centroids):
                    # same centroid is getting created
                    print(f"same centroid formed, breaking loop")
                    break
                centroids = new_centroids
            # We will check for intertia here
            current_inertia = 0
            for i in range(k):
                # Calculate the squared Euclidean distance for data points in cluster i
                cluster_points = data[cluster_assignments == i]
                centroid = centroids[i]
                squared_distances = distances.euclidean_distance(cluster_points, centroid)
                current_inertia += np.sum(squared_distances)
                # print("current intertia = ", current_inertia)
            
            if current_inertia < best_inertia:
                best_centroids, best_inertia = centroids, current_inertia
        return best_centroids
    
    Centroids = k_means(data_collection, k)


    # C = kmeans.cluster_centers_
    # print("Cluster Shape = ", C.shape)
    # Using the Clusters find the latent semantic features 
    # X_r = extractDistanceFeatures(data_collection,C)
    X_r = extractDistanceFeatures(data_collection, Centroids)
    print("Latent Semantics Shape = ", X_r.shape)
    return X_r
    

    # clf = SVC().fit(X_r, feature_descriptors_label)
    # X_test_r = extractDistanceFeatures(feature_descriptors_test,C)

    # y_pred = clf.predict(X_test_r)

    # mat = confusion_matrix(feature_descriptors_test_label,y_pred)
    # sns.set()
    # sns.heatmap(mat.T,square=True,annot=True,fmt='d',cbar=False,
    #             xticklabels=np.unique(all_labels),yticklabels=np.unique(all_labels))
    # plt.xlabel("True label")
    # plt.ylabel("predicted label")
    # plt.show()
    return
