## Esther CHOI 3800370
## ML TME2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

POI_FILENAME = "data/poi-paris.pkl"
parismap = mpimg.imread('data/paris-48.806-2.23--48.916-2.48.jpg')
## coordonnees GPS de la carte
xmin, xmax = 2.23, 2.48  # coord_x min et max
ymin, ymax = 48.806, 48.916  # coord_y min et max
coords = [xmin, xmax, ymin, ymax]


class Density(object):
    def fit(self,data):
        """
        data : matrice n*2 des coordonnées
        Differe entre histogramme et noyaux
        """
        pass

    def predict(self,data):
        """ 
        data : matrice n*2 des coordonnées
        Retourne un vecteur de taille n où chaque element est la densite
        du point de data correspondant
        """
        pass

    def score(self,data):
        """
        data : matrice n*2 des coordonnées
        Retourne la log-vraisemblance de data
        """
        eps = 10e-10
        dens = self.predict(data) + eps
        return np.sum(np.log(dens))/data.shape[0]

class Histogramme(Density):
    def __init__(self,steps=10):
        Density.__init__(self)
        self.steps = steps

    def fit(self,x):
        """
        x : x_train
        A compléter : apprend l'histogramme de la densité sur x
        """
        H, _ = np.histogramdd(x,bins=self.steps)
        self.histo = H

    def to_bin(self,data):
        n = data.shape[0]
        xmin, xmax = data[:,0].min(), data[:,0].max()
        ymin, ymax = data[:,1].min(), data[:,1].max()
        deltax = (xmax-xmin)/self.steps
        deltay = (ymax-ymin)/self.steps

        res = []
        for i in range(n):
            xi = int((data[i,0]-xmin)/deltax)
            yi = int((data[i,1]-ymin)/deltay)
            if xi == self.steps:
                xi -= 1
            if yi == self.steps:
                yi -= 1
            res.append((xi,yi))
        return res,deltax,deltay

    def predict(self,x):
        """
        x : x_test
        A compléter : retourne la densité associée à chaque point de x
        """
        n = x.shape[0] #nombre total de points
        ind,deltax,deltay = self.to_bin(x) #indices dans l'histogramme de chaque point
        V = deltax * deltay

        res = np.zeros(n) #vecteur des densites
        for i in range(n):
            cx,cy = ind[i]
            k = self.histo[cx,cy]
            res[i] = k
        return res/(n*V)


class KernelDensity(Density):
    def __init__(self,kernel=None,sigma=0.1):
        Density.__init__(self)
        self.kernel = kernel
        self.sigma = sigma

    def fit(self,x):
        self.x = x

    def predict(self,data):
        """
        data : matrice des données
        A compléter : retourne la densité associée à chaque point de data
        """
        n,d = data.shape
        res = np.zeros((n,d))
        for i in range(n):
            res[i] = self.kernel((data[i]-self.x)/self.sigma).sum()
        return (1/(n*(self.sigma**d))) * res

class Nadaraya(Density):
    def __init__(self,kernel=None,sigma=0.1):
        Density.__init__(self)
        self.kernel = kernel
        self.sigma = sigma

    def fit(self,x):
        self.x = x

    def predict(self, data):
        """
        Retourne un vecteur correspondant aux notes prédites pour data
        """
        #TODO
        pass 

    def error(self,data):
        """
        Retourne l'erreur au sens des moindres carres faite par le modele sur data
        """
        #TODO
        pass

    

def kernel_uniform(x):
    """
    x : matrice
    retourne un vecteur tq pour tout i, v_i = 1 si |x_i| <= 1/2 et 0 sinon
    """
    return np.where(np.abs(x)<=0.5,1,0)

def kernel_gaussian(x):
    d = x.shape[1]
    return ((2*np.pi)**(-1*d/2)) * np.exp(-0.5*(np.abs(x)**2))

def get_density2D(f,data,steps=100):
    """ Calcule la densité en chaque case d'une grille steps x steps dont les bornes sont calculées à partir du min/max de data. Renvoie la grille estimée et la discrétisation sur chaque axe.
    """
    xmin, xmax = data[:,0].min(), data[:,0].max()
    ymin, ymax = data[:,1].min(), data[:,1].max()
    xlin,ylin = np.linspace(xmin,xmax,steps),np.linspace(ymin,ymax,steps)
    xx, yy = np.meshgrid(xlin,ylin)
    grid = np.c_[xx.ravel(), yy.ravel()]
    res = f.predict(grid).reshape(steps, steps)
    return res, xlin, ylin

def show_density(f, data, steps=100, log=False):
    """ Dessine la densité f et ses courbes de niveau sur une grille 2D calculée à partir de data, avec un pas de discrétisation de steps. Le paramètre log permet d'afficher la log densité plutôt que la densité brute
    """
    res, xlin, ylin = get_density2D(f, data, steps)
    xx, yy = np.meshgrid(xlin, ylin)
    plt.figure()
    show_img()
    if log:
        res = np.log(res+1e-10)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.8, s=3)
    show_img(res)
    plt.colorbar()
    plt.contour(xx, yy, res, 20)


def show_img(img=parismap):
    """ Affiche une matrice ou une image selon les coordonnées de la carte de Paris.
    """
    origin = "lower" if len(img.shape) == 2 else "upper"
    alpha = 0.3 if len(img.shape) == 2 else 1.
    plt.imshow(img, extent=coords, aspect=1.5, origin=origin, alpha=alpha)
    ## extent pour controler l'echelle du plan


def load_poi(typepoi,fn=POI_FILENAME):
    """ Dictionaire POI, clé : type de POI, valeur : dictionnaire des POIs de ce type : (id_POI, [coordonnées, note, nom, type, prix])
    
    Liste des POIs : furniture_store, laundry, bakery, cafe, home_goods_store, 
    clothing_store, atm, lodging, night_club, convenience_store, restaurant, bar
    """
    poidata = pickle.load(open(fn, "rb"))
    data = np.array([[v[1][0][1],v[1][0][0]] for v in sorted(poidata[typepoi].items())])
    note = np.array([v[1][1] for v in sorted(poidata[typepoi].items())])
    return data,note

if __name__=="__main__":

    # Liste des POIs : furniture_store, laundry, bakery, cafe, home_goods_store, clothing_store, atm, lodging, night_club, convenience_store, restaurant, bar
    # La fonction charge la localisation des POIs dans geo_mat et leur note.
    geo_mat, notes = load_poi("night_club")

    # Split en train et test
    geo_train, geo_test, notes_train, notes_test = train_test_split(geo_mat, notes, test_size=0.3, random_state=0)


    ## Méthode par histogramme

    # Visualisation pour quelques pas
    dothis = False
    save = False
    if dothis:
        steps = [10,50,70]
        for s in steps:
            f = Histogramme(steps=s)
            f.fit(geo_train)
            show_density(f,geo_test,steps=f.steps)
            if save:
                plt.savefig("fig/histo-dens_steps="+str(s))
        plt.show()
        

    # Calcul du meilleur steps
    dothis = True
    save = True
    if dothis:
        steps = np.arange(1,100)
        best_steps = 0
        best_score = -np.inf
        scores_train = []
        scores_test = []
        for s in steps:
            f = Histogramme(steps=s)
            f.fit(geo_train)
            f_sc_train = f.score(geo_train)
            f_sc_test = f.score(geo_test)
            scores_train.append(f_sc_train)
            scores_test.append(f_sc_test)
            if best_score < f_sc_test:
                best_score = f_sc_test
                best_steps = s
        print("histogramme",best_steps,best_score)

        # Affichage
        show_carte = False
        show_dens = True
        courbes = True

        if show_carte or show_dens or courbes:
            if show_carte:
                #plt.ion()
                plt.figure()
                # Affiche la carte de Paris
                show_img()
                # Affiche les POIs
                plt.scatter(geo_mat[:,0],geo_mat[:,1],alpha=0.8,s=3)
                if save:
                    plt.savefig("fig/histo_carte")
            if show_dens:
                f = Histogramme(steps=best_steps)
                f.fit(geo_train)
                show_density(f,geo_test,steps=f.steps)
                if save:
                    plt.savefig("fig/histo_dens")
            if courbes:
                plt.figure()
                plt.plot(steps,scores_train,label="train")
                plt.plot(steps,scores_test,label="test")
                plt.legend()
                if save:
                    plt.savefig("fig/histo_plot")
            plt.show()

    ## Méthode à noyaux

    # Test
    #f = KernelDensity(kernel=kernel_uniform)
    #f.fit(geo_train)
    #print(f.score(geo_test))

    # Calcul du meilleur sigma pour kernel_uniform
    dothis = False
    save = False
    if dothis:
        sigmas_uni = np.linspace(0.1,0.5,10)
        best_score = -np.inf
        best_sigma_uni = 0
        scores_train_uni = []
        scores_test_uni = []
        for s in sigmas_uni:
            f = KernelDensity(kernel=kernel_uniform,sigma=s)
            f.fit(geo_train)
            f_sc_train = f.score(geo_train)
            f_sc_test = f.score(geo_test)
            scores_train_uni.append(f_sc_train)
            scores_test_uni.append(f_sc_test)
            if best_score < f_sc_test:
                best_score = f_sc_test
                best_sigma_uni = s
        print("uniform",best_sigma_uni,best_score)

        courbes = True
        if courbes:
            plt.figure()
            plt.plot(sigmas_uni,scores_train_uni,label="train")
            plt.plot(sigmas_uni,scores_test_uni,label="test")
            plt.legend()
            if save:
                plt.savefig("fig/noyau_plot_uni_best")
        plt.show()
    
    # Calcul du meilleur sigma pour kernel_gaussian
    dothis = False
    if dothis:
        sigmas_gauss = np.linspace(0.01,0.1,10)
        best_score = -np.inf
        best_sigma_gauss = 0
        scores_train = []
        scores_test = []
        for s in sigmas_gauss:
            f = KernelDensity(kernel=kernel_gaussian,sigma=s)
            f.fit(geo_train)
            f_sc_train = f.score(geo_train)
            f_sc_test = f.score(geo_test)
            scores_train.append(f_sc_train)
            scores_test.append(f_sc_test)
            if best_score < f_sc_test:
                best_score = f_sc_test
                best_sigma_gauss = s
        print("gaussian",best_sigma_gauss,best_score)

        # Affichage
        show_carte = False
        show_dens_uni = False
        show_dens_gauss = False
        courbes_uni = False
        courbes_gauss = False

        if show_carte or show_dens or courbes_uni or courbes_gauss:
            if show_carte:
                #plt.ion()
                plt.figure()
                # Affiche la carte de Paris
                show_img()
                # Affiche les POIs
                plt.scatter(geo_mat[:,0],geo_mat[:,1],alpha=0.8,s=3)
            if show_dens_uni:
                f = KernelDensity(kernel=kernel_uniform,sigma=best_sigma_uni)
                f.fit(geo_train)
                plt.figure()
                plt.plot(f.predict(geo_test))
                plt.title("Uniform")
            if show_dens_gauss:
                f = KernelDensity(kernel=kernel_gaussian,sigma=best_sigma_gauss)
                f.fit(geo_train)
                plt.figure()
                plt.plot(f.predict(geo_test))
                plt.title("Gaussian")
            if courbes_uni:
                plt.figure()
                plt.plot(sigmas_uni,scores_train_uni,label="train")
                plt.plot(sigmas_uni,scores_test_uni,label="test")
                plt.title("Uniform")
                plt.legend()
            if courbes_gauss:
                plt.figure()
                plt.plot(sigmas_gauss,scores_train,label="train")
                plt.plot(sigmas_gauss,scores_test,label="test")
                plt.title("Gaussian")
                plt.legend()
            plt.show()