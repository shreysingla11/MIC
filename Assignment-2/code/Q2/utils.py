import numpy as np
from copy import deepcopy
from sklearn.cluster import KMeans

neighbour_mask = np.array([
    [0,1,0],
    [1,0,1],
    [0,1,0]
]).astype(bool)

def computeLogPosterior(X,image,means,variances,imageMask,beta):
    log_prior = 0
    for i,j in np.argwhere(imageMask == True):
        label = X[i,j]
        log_prior += computeLogPrior(label, X[i-1:i+2,j-1:j+2][np.logical_and(neighbour_mask,imageMask[i-1:i+2,j-1:j+2])],beta) 
    return np.sum(computeLogLikelihood(image,means[X],variances[X])[imageMask]) + log_prior/2

def computeLogPrior(label, neighbours,beta):
    return -beta*np.sum(label != neighbours)

def computeLogLikelihood(y,mean,variance):

    return -(y - mean)**2/(2*variance)

def getBestLabel(X, i, j, means, variances, image, imageMask, beta):
    neighbours = X[i-1:i+2,j-1:j+2][np.logical_and(neighbour_mask,imageMask[i-1:i+2,j-1:j+2])]
    probs = -beta*np.sum(np.expand_dims(np.arange(means.size),-1) != neighbours,axis=-1) + computeLogLikelihood(image[i,j],means,variances)
    return np.argmax(probs)

def computeLikelihood(y,mean,variance):
    return 1/np.sqrt(2*np.pi*variance)*np.exp(-(y - mean)**2/(2*variance))

def computePrior(label, neighbours,beta):
    return np.exp(-beta*np.sum(label != neighbours))

def updateMapMem(X_map, memberships, means, variances, image, imageMask,beta):
    valid_ind = np.argwhere(imageMask == True) 
    for i,j in np.random.permutation(valid_ind):
        X_map[i,j] = getBestLabel(X_map,i,j,means,variances,image,imageMask,beta)

    for i,j in valid_ind:
        neighbours = X_map[i-1:i+2,j-1:j+2][np.logical_and(neighbour_mask,imageMask[i-1:i+2,j-1:j+2])]
        memberships[i,j,:] = computeLikelihood(image[i,j],means,variances)*np.exp(-beta*np.sum(np.expand_dims(np.arange(means.size),-1) != neighbours,axis=-1))
    
    memberships = memberships/(np.sum(memberships,axis=-1,keepdims=True)+1e-200)
    return X_map, memberships

def updateMeanVariance(means, variances, memberships, image, imageMask):
    image = np.expand_dims(image,-1)
    imageMask = np.expand_dims(imageMask,-1)
    means = np.sum(memberships*image*imageMask,axis=(0,1))/(np.sum(memberships*imageMask,axis=(0,1)) + 1e-200)
    variances = np.sum(memberships*imageMask*(image - means)**2,axis=(0,1))/(np.sum(memberships*imageMask,axis=(0,1)) + 1e-200)
    return means , variances


def getOptimalParams(image, imageMask ,X_map_init, means_init, variances_init,beta, max_iters=200,verbose=True,compute_posterior=True):
    X_map = deepcopy(X_map_init)
    means = deepcopy(means_init)
    variances = deepcopy(variances_init)
    memberships = np.zeros(X_map.shape + (3,))
    posterior_values = []
    
    for iter in range(max_iters):
        if compute_posterior:
            prev_posterior = computeLogPosterior(X_map,image,means,variances,imageMask,beta)
        
        X_map, memberships = updateMapMem(X_map,memberships,means,variances,image,imageMask,beta)
        
        if compute_posterior:
            next_posterior = computeLogPosterior(X_map,image,means,variances,imageMask,beta)
            posterior_values.append([prev_posterior,next_posterior])
        
        means,variances = updateMeanVariance(means,variances,memberships,image,imageMask)

        if verbose:
            print(iter)
            print('---------------------------------------')
            print("Means",means)
            print("Variances",variances)
            if compute_posterior:
                print("Log posterior",prev_posterior, next_posterior)
            print('---------------------------------------')
        if compute_posterior:
            assert(next_posterior >= prev_posterior)
    
    return X_map,means,variances,memberships,np.array(posterior_values)

def initializeParams(image, imageMask, k = 3):
    X = image[imageMask]
    kmeans = KMeans(n_clusters=k,random_state=0).fit(X.reshape(-1,1))
    
    means_init = kmeans.cluster_centers_.reshape(-1)
    
    X_map_init = np.zeros(image.shape).astype(int)
    X_map_init[imageMask] = kmeans.labels_
    
    variances_init = np.array([np.var(X[kmeans.labels_ == i]) for i in range(k)])

    return X_map_init,means_init,variances_init
def reconstructionError(image, reconstructed_image):
    return np.sum((image-reconstructed_image)**2)/np.sum(image**2)