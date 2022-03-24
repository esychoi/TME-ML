## Esther CHOI 3800370
## ML TME4

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mltools import plot_data, plot_frontiere, make_grid, gen_arti
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
import sklearn.svm

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
        
        for i in range(self.max_iter):
            self.w = self.w - self.eps * self.loss_g(self.w,datax,datay).mean()
            costs.append(self.loss(self.w,datax,datay).flatten().sum())

        return costs

    def predict(self,datax):
        return np.sign(np.dot(datax,self.w))

    def score(self,datax,datay):
        yhat = Lineaire.predict(self,datax)
        return np.where(yhat==datay)[0].size / datay.size


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
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")

def  plot_frontiere_proba(data,f,step=20):
    grid,x,y = make_grid(data=data,step=step)
    plt.contourf(x,y,f(grid).reshape(x.shape),255)
    plt.colorbar()


if __name__ =="__main__":
    ## Tirage d'un jeu de données aléatoire avec un bruit de 0.1
    ## mélange de 4 gaussiennes ou echiquier (car 2 gaussiennes c'est trop facile)
    datax, datay = gen_arti(epsilon=0.2,data_type=1)
    datay = datay.reshape(-1)
    X_train, X_test, y_train, y_test = train_test_split(datax,datay)

    ## Tests des différents kernels avec les parametres par defaut
    dothis = False
    aff = False
    kernels = ["linear","poly","rbf","sigmoid"]
    if dothis:
        for k in kernels:
            clf = sklearn.svm.SVC(kernel=k,probability=True)
            clf.fit(X_train,y_train)

            # score
            y_hat = clf.predict(X_test)
            print(k,clf.score(X_test,y_test))

            # affichage
            if aff:
                plt.figure()
                ## Visualisation des données et de la frontière de décision
                plot_frontiere_proba(X_train,lambda x : clf.predict_proba(x)[:,0],step=50)
                plot_data(datax,datay)
                plt.title(k)
        # On peut voir que le kernel rbf fonctionne le mieux pour les deux types de données (4 gaussiennes ou echiquier)


    ## Parametres pour le kernel lineaire
    dothis = True
    aff = True
    if dothis:
        C = np.linspace(0.1,2,30)
        nb_supp_vect = []
        scores = []

        ## Tirage d'un jeu de données aléatoire très bruité
        datax, datay = gen_arti(epsilon=0.8,data_type=0)
        datay = datay.reshape(-1)

        for i in range(len(C)):
            clf = sklearn.svm.SVC(C=C[i],kernel="linear",probability=True)
            scores.append(cross_validate(clf,datax,datay,cv=5,return_train_score=True))
        
        scores_train = []
        scores_test = []
        for i in range(len(scores)):
            scores_train.append(scores[i]["train_score"].mean())
            scores_test.append(scores[i]["test_score"].mean())
        scores_train = np.array(scores_train)
        scores_test = np.array(scores_test)
        print("best c :",C[scores_test.argmax()])

        if aff:
            plt.figure()
            plt.plot(C,scores_train,label='train')
            plt.plot(C,scores_test,label='test')
            plt.title("scores")
            plt.legend()

    ## Vecteurs support pour le kernel lineaire
    dothis = True
    aff = True
    if dothis:
        C = np.linspace(0.1,2,30)
        nb_supp_vect = []

        ## Tirage d'un jeu de données aléatoire très bruité
        datax, datay = gen_arti(epsilon=0.8,data_type=0)
        datay = datay.reshape(-1)
        X_train, X_test, y_train, y_test = train_test_split(datax,datay)
        
        for i in range(len(C)):
            clf = sklearn.svm.SVC(C=C[i],kernel="linear",probability=True)
            clf.fit(X_train,y_train)
            
            nb_supp_vect.append(len(clf.support_vectors_))

            # affichage
            if aff and (i%10==0):
                plt.figure()
                ## Visualisation des données et de la frontière de décision
                plot_frontiere_proba(X_train,lambda x : clf.predict_proba(x)[:,0],step=50)
                plot_data(datax,datay)
                plot_data(clf.support_vectors_)
                plt.title("linear")

        if aff:
            plt.figure()
            plt.plot(C,nb_supp_vect)
            plt.title("nombre vecteurs support") #diminue
        

    ## Parametres pour le kernel polynomial
    dothis = False
    aff = False
    #C = np.linspace(0.1,1,10)
    C = [0.2,1]
    lC = len(C)
    #D = np.arange(2,6)
    D = [3,6]
    lD = len(D)
    nb_supp_vect = dict()
    scores_train = dict()
    scores_test = dict()
    abscisses = dict()
    m = 0        
    for c in range(lC):
        for d in range(lD):
            abscisses[m] = (C[c],D[d])
            m += 1

    if dothis:
        for i in range(m):
            clf = sklearn.svm.SVC(C=abscisses[i][0],kernel="poly",degree = abscisses[i][1],probability=True)
            clf.fit(X_train,y_train)
            #sc = cross_val_score(clf,datax,datay,cv=3)
            y_hat_train = clf.predict(X_train)
            y_hat_test = clf.predict(X_test)
            nb_supp_vect[i] = len(clf.support_vectors_)
            scores_train[i] = clf.score(X_train,y_train)
            scores_test[i] = clf.score(X_test,y_test)

            # affichage
            if aff:
                plt.figure()
                ## Visualisation des données et de la frontière de décision
                plot_frontiere_proba(X_train,lambda x : clf.predict_proba(x)[:,0],step=50)
                plot_data(datax,datay)
                plot_data(clf.support_vectors_)
                plt.title("poly C="+str(C[c]) + " deg="+str(D[d]))    

        print(abscisses)
        plt.figure()
        plt.plot(list(nb_supp_vect.values()))
        plt.xticks(list(abscisses.keys()))
        plt.title("nb vecteurs support")

        plt.figure()
        plt.plot(list(scores_train.values()),label='train')
        plt.plot(list(scores_test.values()),label='test')
        plt.xticks(list(abscisses.keys()))
        plt.title("scores")
        plt.legend()
    
    plt.show()

    ups = False

    if ups:
        uspsdatatrain = "data/USPS_train.txt"
        uspsdatatest = "data/USPS_test.txt"
        alltrainx,alltrainy = load_usps(uspsdatatrain)
        alltestx,alltesty = load_usps(uspsdatatest)

        # plt.figure()
        # plt.imshow(alltrainx[0].reshape(16,16))
        # plt.show()

        #6 vs 9
        pos,neg = 6,9
        X_train,y_train = get_usps([neg,pos],alltrainx,alltrainy)
        X_test,y_test = get_usps([neg,pos],alltestx,alltesty)
        #y_train_sign = np.where(y_train==pos,1,-1)
        #y_test_sign = np.where(y_test==pos,1,-1)

        clf = sklearn.svm.SVC(kernel="linear",probability=True)
        clf.fit(X_train,y_train)

        y_hat_train = clf.predict(X_train)
        y_hat_test = clf.predict(X_test)
        nb_supp_vect = len(clf.support_vectors_)
        scores_train = clf.score(X_train,y_train)
        scores_test = clf.score(X_test,y_test)

        print(nb_supp_vect)
        print(scores_train)
        print(scores_test)
