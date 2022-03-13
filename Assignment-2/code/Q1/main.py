import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse
import os

np.random.seed(42)

def kmeans(image, mask):
    u = np.random.randint(0,3,size=(256,256))
    means = np.zeros(3)
    for k in range(10):
        for i in range(3):
            means[i] = np.sum(image[(u == i) & (mask == 1)])/np.sum((u==i) & (mask == 1))
        for x in range(256):
            for y in range(256):
                if mask[x,y] == 0:
                    continue
                u[x,y] = np.argmin(np.abs(image[x,y] - means))
    return means

def w(p1, p2, sigma):
    d2 = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
    if np.sqrt(d2) > radius:
        return 0
    return np.exp((-d2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

def d(k, p, W, c, bias):
    return np.sum(W*(image[p] - c[k]*bias[max(p[0]-radius,0):min(p[0]+radius,255), max(p[1]-radius,0):min(p[1]+radius,255)])**2)

def D(d, W, c, bias):
    D = np.zeros((image.shape[0], image.shape[1], 3))
    for x in range(256):
        for y in range(256):
            for k in range(3):
                if mask[x,y] == 0:
                    continue
                D[x,y,k] = d(k,(x,y), W, c, bias)
    return D

def update_u(u, W, c, bias):
    for x in range(256):
        for y in range(256):
            if mask[x,y] == 0:
                continue
            for k in range(3):
                u[x,y,k] = 1/((d(k,(x,y),W,c,bias)+1e-4)**(1/(q-1)))
    u = u/np.sum(u, axis=-1, keepdims=True)
    return u

def update_b(u, c, W, bias):
    for x in range(256):
        for y in range(256):
            if mask[x,y] == 0:
                continue
            p = (x,y)
            a = np.zeros((2*radius, 2*radius))
            b = np.zeros((2*radius, 2*radius))
            for i in range(3):
                a += (u[p[0]-radius:p[0]+radius, p[1]-radius:p[1]+radius][:,:,i]**q)*c[i]
                b += (u[p[0]-radius:p[0]+radius, p[1]-radius:p[1]+radius][:,:,i]**q)*(c[i]**2)
            bias[x,y] = np.sum(W*image[p[0]-radius:p[0]+radius,p[1]-radius:p[1]+radius]*a)/(np.sum(W*b))

def update_c(W, bias, u, c):
    for k in range(3):
        num = 0
        den = 0
        for x in range(256):
            for y in range(256):
                if mask[x,y] == 0:
                    continue
                temp = W*bias[x-radius:x+radius, y-radius:y+radius]
                num += (u[x,y,k]**q)*image[x,y]*np.sum(temp)
                den += (u[x,y,k]**q)*np.sum(temp*bias[x-radius:x+radius, y-radius:y+radius])
        c[k] = num/den

def construct_bias_removed(u, c):
    a = np.zeros((256,256))
    for i in range(3):
        a += u[:,:,i]*c[i]
    return a

def train(q, radius, image, mask, res_dir):

    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.savefig(os.path.join(res_dir, "corrupted_image.png"))

    c = kmeans(image, mask)
    sigma = radius/3
    W = np.zeros((radius*2, radius*2))
    for i in range(radius*2):
        for j in range(radius*2):
            W[i,j] = w((i,j),(radius,radius), sigma)

    plt.figure()
    plt.imshow(W)
    plt.colorbar()
    plt.savefig(os.path.join(res_dir, "W.png"))
    
    u = np.ones((image.shape[0], image.shape[1], 3))/3
    bias = np.ones_like(image)
    objectives = []
    for i in range(100):
        u = update_u(u, W, c, bias)
        update_b(u, c, W, bias)
        update_c(W, bias, u, c)
        obj = np.sum(D(d, W, c, bias)*u)
        print("objective: ", obj)
        objectives.append(obj)
        if i > 0 and abs(objectives[-1] - objectives[-2]) < 0.1:
            break
    bias_removed = construct_bias_removed(u, c)
    residual = image - bias_removed*bias
    residual[mask == 0] = 0

    plt.figure()
    plt.plot(np.array(objectives))
    plt.title("Objective function")
    plt.xlabel("iterations")
    plt.ylabel("Value")
    plt.savefig(os.path.join(res_dir, "Objective.png"))


    for i in range(3):
        plt.figure()
        plt.imshow(u[:,:,i], cmap='gray')
        plt.savefig(os.path.join(res_dir, f"Membership_{i}.png"))
    
    plt.figure()
    plt.imshow(bias, cmap='gray')
    plt.savefig(os.path.join(res_dir, "Bias.png"))

    plt.figure()
    plt.imshow(bias_removed, cmap='gray')
    plt.savefig(os.path.join(res_dir, "Bias_removed.png"))

    plt.figure()
    plt.imshow(residual, cmap='gray')
    plt.savefig(os.path.join(res_dir, "Residual.png"))






if __name__ == '__main__':
    radius = 10
    q = 1.7

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default='../../data/assignmentSegmentBrain.mat')
    parser.add_argument("--res_dir", type=str, default='../../results/Q1')

    args = parser.parse_args()

    file_path = args.file_path
    res_dir = args.res_dir

    f = h5py.File(file_path, 'r')
    image = np.array(f.get('imageData'))
    mask = np.array(f.get('imageMask'))

    train(q, radius, image, mask, res_dir)