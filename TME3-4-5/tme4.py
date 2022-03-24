## Esther CHOI 3800370
## ML TME4

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mltools import plot_data, plot_frontiere, make_grid, gen_arti

def perceptron_loss(w,x,y):
    """
    array**3 -> float
    x : n lignes, d colonnes
    w : d lignes, 1 colonne
    y : n lignes, 1 colonne
    Retourne le cout perceptron (hinge-loss)
    """
    y = y.reshape(-1,1)
    w = w.reshape(-1,1)
    x = x.reshape(y.shape[0],w.shape[0])
    return np.maximum(0,-y*(x@w))

def perceptron_grad(w,x,y):
    """
    array**3 -> float
    x : n lignes, d colonnes
    w : d lignes, 1 colonne
    y : n lignes, 1 colonne
    Retourne le cout du gradient du perceptron (gradient du hinge-loss)
    """
    y = y.reshape(-1,1)
    w = w.reshape(-1,1)
    x = x.reshape(y.shape[0],w.shape[0])
    return np.where((-y*x)*perceptron_loss(w,x,y) < 0, 0, -y*x)
    

class Lineaire(object):
    def __init__(self,loss=perceptron_loss,loss_g=perceptron_grad,max_iter=100,eps=0.01):
        self.max_iter, self.eps = max_iter,eps
        self.w = None
        self.loss,self.loss_g = loss,loss_g
        
    def fit(self,datax,datay):
        costs = []
        self.w = np.random.randn(datax.shape[1],1)
        datay = datay.reshape(-1,1)
        
        for i in range(self.max_iter):
            self.w = self.w - self.eps * self.loss_g(self.w,datax,datay).mean(axis=0).reshape(-1,1)
            costs.append(self.loss(self.w,datax,datay).sum())

        return costs

    def predict(self,datax):
        return np.sign(datax@self.w)

    def score(self,datax,datay):
        yhat = self.predict(datax).reshape(-1)
        return (yhat==datay).mean()


def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def get_usps(l,datax,datay):
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy
    tmp =   list(zip(*[get_usps(i,datax,datay) for i in l]))
    tmpx,tmpy = np.vstack(tmp[0]),np.hstack(tmp[1])
    return tmpx,tmpy

def show_usps(data):
    plt.figure()
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="hot")
    plt.colorbar()



if __name__ =="__main__":
    jouet = False
    usps = True

    if jouet:
        ## Tirage d'un jeu de données aléatoire avec un petit bruit
        datax, datay = gen_arti(epsilon=0.05)
        ## Fabrication d'une grille de discrétisation pour la visualisation de la fonction de coût
        grid, x_grid, y_grid = make_grid(xmin=-2, xmax=2, ymin=-2, ymax=2, step=100)
        
        L = Lineaire(max_iter=200)
        costs = L.fit(datax,datay)
        print(costs[-1])

        plt.figure()
        ## Visualisation des données et de la frontière de décision pour un vecteur de poids w
        w = L.w
        plot_frontiere(datax,lambda x : np.sign(x.dot(w)),step=100)
        plot_data(datax,datay)

        ## Visualisation de la fonction de coût en 2D
        plt.figure()
        plt.plot(np.arange(len(costs)),costs)

        plt.show()

    if usps:
        uspsdatatrain = "data/USPS_train.txt"
        uspsdatatest = "data/USPS_test.txt"
        alltrainx,alltrainy = load_usps(uspsdatatrain)
        alltestx,alltesty = load_usps(uspsdatatest)

        # plt.figure()
        # plt.imshow(alltrainx[0].reshape(16,16))
        # plt.show()

        oneVSone = True
        oneVSall = True

        if oneVSone:
            # 6 vs 9
            pos,neg = 6,9
            datax,datay = get_usps([neg,pos],alltrainx,alltrainy)
            testx,testy = get_usps([neg,pos],alltestx,alltesty)
            datay_sign = np.where(datay==pos,1,-1)
            testy_sign = np.where(testy==pos,1,-1)

            L = Lineaire()

            costs = L.fit(datax,datay)
            show_usps(L.w)
            plt.title(f"{pos} vs {neg} : w")

            print(f"{pos} vs {neg} :")
            print("score train =", L.score(datax,datay_sign))
            print("score test =", L.score(testx,testy_sign))

            plt.figure()
            plt.plot(costs)
            plt.title(f"{pos} vs {neg} : costs")

            plt.show()

        if oneVSall:
            # 6 vs others
            neg = 6
            pos = [i for i in range(10) if i != neg]
            datax,datay = get_usps([neg,pos],alltrainx,alltrainy)
            testx,testy = get_usps([neg,pos],alltestx,alltesty)
            datay_sign = np.where(datay==neg,-1,1)
            testy_sign = np.where(testy==neg,-1,1)

            L = Lineaire()

            costs = L.fit(datax,datay_sign)
            show_usps(L.w)
            plt.title(f"{neg} vs others : w")

            print(f"{neg} vs others :")
            print("score train =", L.score(datax,datay_sign))
            print("score test =", L.score(testx,testy_sign))

            plt.figure()
            plt.plot(costs)
            plt.title(f"{neg} vs others : costs")

            plt.show()



