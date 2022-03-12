import numpy as np
from copy import deepcopy
from sklearn.cluster import KMeans

def getNeighbours(X,i,j,imageMask = None):
    h,w = X.shape
    neighours = []
    if i - 1 >= 0 and imageMask[i-1,j]:
        neighours.append(X[i-1,j])
    if i + 1 < h and imageMask[i+1,j]:
        neighours.append(X[i+1,j])
    if j - 1 >= 0 and imageMask[i,j-1]:
        neighours.append(X[i,j-1])
    if j + 1 < w and imageMask[i,j+1]:
        neighours.append(X[i,j+1])
    return np.array(neighours)

def computeLogPosterior(X,image,means,variances,imageMask,beta):
    log_prior = 0
    log_likelihood = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if imageMask[i,j]:
                label = X[i,j]
                log_prior += computeLogPrior(label, getNeighbours(X,i,j,imageMask),beta) 
                log_likelihood += computeLogLikelihood(image[i][j],means[label],variances[label])
    return log_likelihood + log_prior/2

def computeLogPrior(label, neighbours,beta):
    return -beta*np.sum(label != neighbours)

def computeLogLikelihood(y,mean,variance):

    return -(y - mean)**2/(2*variance)

def getBestLabel(X, i, j, means, variances, image, imageMask, beta):
    k = means.size
    probs = np.zeros(k)
    for label in range(k):
        # print(means,variances,computeLogLikelihood(image[i][j],means[label],variances[label]))
        # print(computeLogPrior(label, getNeighbours(X,i,j,imageMask),beta))
        probs[label] = computeLogPrior(label, getNeighbours(X,i,j,imageMask),beta) + computeLogLikelihood(image[i][j],means[label],variances[label])
    return np.argmax(probs)
 

def computeLikelihood(y,mean,variance):
    return 1/np.sqrt(2*np.pi*variance)*np.exp(-(y - mean)**2/(2*variance))

def computePrior(label, neighbours,beta):
    return np.exp(-beta*np.sum(label != neighbours))

def updateMapMem(X_map, memberships, means, variances, image, imageMask,beta):
    shape = X_map.shape
    for i,j in np.random.permutation([(x//shape[0],x%shape[1]) for x in  range(shape[0]*shape[1])]):
            if imageMask[i,j]:
                X_map[i][j] = getBestLabel(X_map,i,j,means,variances,image,imageMask,beta)

    for i in range(shape[0]):
        for j in range(shape[1]):
            if imageMask[i,j]:
                neighbours = getNeighbours(X_map,i,j,imageMask)
                for k in range(means.size):
                    memberships[i,j,k] = computeLikelihood(image[i,j],means[k],variances[k])*computePrior(k,neighbours,beta)
    
    memberships = memberships/(np.sum(memberships,axis=-1,keepdims=True)+1e-200)
    return X_map, memberships

def updateMeanVariance(means, variances, memberships, image, imageMask):
    image = np.expand_dims(image,-1)
    imageMask = np.expand_dims(imageMask,-1)
    means = np.sum(memberships*image*imageMask,axis=(0,1))/(np.sum(memberships*imageMask,axis=(0,1)) + 1e-200)
    variances = np.sum(memberships*imageMask*(image - means)**2,axis=(0,1))/(np.sum(memberships*imageMask,axis=(0,1)) + 1e-200)
    return means , variances


def getOptimalParams(image, imageMask ,X_map_init, means_init, variances_init,beta, max_iters=200,verbose=True):
    X_map = deepcopy(X_map_init)
    means = deepcopy(means_init)
    variances = deepcopy(variances_init)
    memberships = np.zeros(X_map.shape + (3,))
    for iter in range(max_iters):
        prev_posterior = computeLogPosterior(X_map,image,means,variances,imageMask,beta)
        X_map, memberships = updateMapMem(X_map,memberships,means,variances,image,imageMask,beta)
        next_posterior = computeLogPosterior(X_map,image,means,variances,imageMask,beta)
        means,variances = updateMeanVariance(means,variances,memberships,image,imageMask)
        if verbose:
            print(iter)
            print('---------------------------------------')
            print("Means",means)
            print("Variances",variances)
            print("Log posterior",prev_posterior, next_posterior)
            print('---------------------------------------')
        assert(next_posterior >= prev_posterior)
    
    return X_map,means,variances,memberships

def initializeParams(image, imageMask, k = 3):
    X = image[imageMask]
    kmeans = KMeans(n_clusters=k,random_state=0).fit(X.reshape(-1,1))
    
    means_init = kmeans.cluster_centers_
    
    X_map_init = np.zeros(image.shape).astype(int)
    X_map_init[imageMask] = kmeans.labels_
    
    variances_init = np.array([np.var(X[kmeans.labels_ == i]) for i in range(k)])

    return X_map_init,means_init,variances_init
def reconstructionError(image, reconstructed_image):
    return np.sum((image-reconstructed_image)**2)/np.sum(image**2)