## Esther CHOI 3800370
## ML TME3

import numpy as np
import matplotlib.pyplot as plt

from mltools import plot_data, plot_frontiere, make_grid, gen_arti

def mse(w,x,y):
    """vecteur(d,1) * matrice(n,d) * vecteur(n,1) -> vecteur(n,1)"""
    y = y.reshape(-1,1)
    w = w.reshape(-1,1)
    x = x.reshape(y.shape[0],w.shape[0])
    return ((x@w) - y)**2
    

def mse_grad(w,x,y):
    """vecteur(d,1) * matrice(n,d) * vecteur(n,1) -> matrice(n,d)"""
    y = y.reshape(-1,1)
    w = w.reshape(-1,1)
    x = x.reshape(y.shape[0],w.shape[0])
    return 2*x*((x@w) - y)

def reglog(w,x,y):
    """vecteur(d,1) * matrice(n,d) * vecteur(n,1) -> float"""
    y = y.reshape(-1,1)
    w = w.reshape(-1,1)
    x = x.reshape(y.shape[0],w.shape[0])
    return np.log(1 + np.exp(-y*(x@w)))

def reglog_grad(w,x,y):
    """vecteur(d,1) * matrice(n,d) * vecteur(n,1)"""
    y = y.reshape(-1,1)
    w = w.reshape(-1,1)
    x = x.reshape(y.shape[0],w.shape[0])
    return -x*y / (1 + np.exp(y * (x@w)))

def check_fonctions():
    ## On fixe la seed de l'aléatoire pour vérifier les fonctions
    np.random.seed(0)
    datax, datay = gen_arti(epsilon=0.1)
    wrandom = np.random.randn(datax.shape[1],1)
    assert(np.isclose(mse(wrandom,datax,datay).mean(),0.54731,rtol=1e-4))
    assert(np.isclose(reglog(wrandom,datax,datay).mean(), 0.57053,rtol=1e-4))
    assert(np.isclose(mse_grad(wrandom,datax,datay).mean(),-1.43120,rtol=1e-4))
    assert(np.isclose(reglog_grad(wrandom,datax,datay).mean(),-0.42714,rtol=1e-4))
    np.random.seed()

def descente_gradient(datax,datay,f_loss,f_grad,eps,iter):
    """
    matrice * vecteur * fonction * fonction * float * int -> vecteur
    """
    n = datax.shape[0]
    w = np.random.randn(datax.shape[1],1)
    w_list = []
    err_list = []

    for i in range(iter):
        w = w - eps * f_grad(w,datax,datay).mean(axis=0).reshape(-1,1)
        w_list.append(w)
        err_list.append(f_loss(w,datax,datay).mean())
        

    return w,np.array(w_list),np.array(err_list)

if __name__=="__main__":
    # check_fonctions()

    lineaire = True
    logistique = False

    if lineaire:
        separable = True
        
        if separable:
            save = False

            ## Tirage d'un jeu de données aléatoire (mélange de 2 gaussiennes) avec un petit bruit
            datax, datay = gen_arti(epsilon=0.05)
            
            ## Apprentissage
            eps=1e-3
            iter = 1000
            wopt,w_list,err_list = descente_gradient(datax,datay,mse,mse_grad,eps,iter)
            print(err_list[-1])

            ## Visualisation des données et de la frontière de décision pour un vecteur de poids w
            w = wopt
            plt.figure()
            plot_frontiere(datax,lambda x : np.sign(x.dot(w)),step=100)
            plot_data(datax,datay)

            if save:
                plt.savefig("fig3/lin_sep_data")

            ## Visualisation de la fonction de coût en 2D
            ## Fabrication d'une grille de discrétisation pour la visualisation de la fonction de coût
            lim = np.ceil(np.max(np.abs(w_list))+1)
            grid, x_grid, y_grid = make_grid(xmin=-lim, xmax=lim, ymin=-lim, ymax=lim, step=100)
            plt.figure()
            plt.contourf(x_grid,y_grid,np.array([mse(w,datax,datay).mean() for w in grid]).reshape(x_grid.shape),levels=20)
            plt.scatter(w_list[:,0],w_list[:,1])
            plt.colorbar()

            if save:
                plt.savefig("fig3/lin_sep_loss")

            ## Evolution de l'erreur
            plt.figure()
            plt.plot(err_list)
            plt.xlabel("iteration")
            plt.ylabel("erreur")

            if save:
                plt.savefig("fig3/lin_sep_err")

            plt.show()
        
        else:
            save = False
            ## Tirage d'un jeu de données aléatoire (mélange de 2 gaussiennes) avec un grand bruit
            datax, datay = gen_arti(epsilon=0.8)
            
            ## Apprentissage
            eps=1e-3
            iter = 1000
            wopt,w_list,err_list = descente_gradient(datax,datay,mse,mse_grad,eps,iter)
            print(err_list[-1])

            ## Visualisation des données et de la frontière de décision pour un vecteur de poids w
            w = wopt
            plt.figure()
            plot_frontiere(datax,lambda x : np.sign(x.dot(w)),step=100)
            plot_data(datax,datay)

            if save:
                plt.savefig("fig3/lin_nsep_data")

            ## Visualisation de la fonction de coût en 2D
            ## Fabrication d'une grille de discrétisation pour la visualisation de la fonction de coût
            lim = np.ceil(np.max(np.abs(w_list))+1)
            grid, x_grid, y_grid = make_grid(xmin=-lim, xmax=lim, ymin=-lim, ymax=lim, step=100)
            plt.figure()
            plt.contourf(x_grid,y_grid,np.array([mse(w,datax,datay).mean() for w in grid]).reshape(x_grid.shape),levels=20)
            plt.scatter(w_list[:,0],w_list[:,1])
            plt.colorbar()

            if save:
                plt.savefig("fig3/lin_nsep_loss")

            ## Evolution de l'erreur
            plt.figure()
            plt.plot(err_list)
            plt.xlabel("iteration")
            plt.ylabel("erreur")

            if save:
                plt.savefig("fig3/lin_nsep_err")

            plt.show()

    

    if logistique:
        separable = True
        
        if separable:
            save = False

            ## Tirage d'un jeu de données aléatoire (mélange de 2 gaussiennes) avec un petit bruit
            datax, datay = gen_arti(epsilon=0.05)
            
            ## Apprentissage
            eps=1e-3
            iter = 10000
            wopt,w_list,err_list = descente_gradient(datax,datay,reglog,reglog_grad,eps,iter)
            print(err_list[-1])

            ## Visualisation des données et de la frontière de décision pour un vecteur de poids w
            w = wopt
            plt.figure()
            plot_frontiere(datax,lambda x : np.sign(x.dot(w)),step=100)
            plot_data(datax,datay)

            if save:
                plt.savefig("fig3/log_sep_data")

            ## Visualisation de la fonction de coût en 2D
            ## Fabrication d'une grille de discrétisation pour la visualisation de la fonction de coût
            lim = np.ceil(np.max(np.abs(w_list))+1)
            grid, x_grid, y_grid = make_grid(xmin=-lim, xmax=lim, ymin=-lim, ymax=lim, step=100)
            plt.figure()
            plt.contourf(x_grid,y_grid,np.array([mse(w,datax,datay).mean() for w in grid]).reshape(x_grid.shape),levels=20)
            plt.scatter(w_list[:,0],w_list[:,1])
            plt.colorbar()

            if save:
                plt.savefig("fig3/log_sep_loss")

            ## Evolution de l'erreur
            plt.figure()
            plt.plot(err_list)
            plt.xlabel("iteration")
            plt.ylabel("erreur")

            if save:
                plt.savefig("fig3/log_sep_err")

            plt.show()
        
        else:
            save = False
            ## Tirage d'un jeu de données aléatoire (mélange de 2 gaussiennes) avec un grand bruit
            datax, datay = gen_arti(epsilon=0.8)
            
            ## Apprentissage
            eps=1e-3
            iter = 10000
            wopt,w_list,err_list = descente_gradient(datax,datay,reglog,reglog_grad,eps,iter)
            print(err_list[-1])

            ## Visualisation des données et de la frontière de décision pour un vecteur de poids w
            w = wopt
            plt.figure()
            plot_frontiere(datax,lambda x : np.sign(x.dot(w)),step=100)
            plot_data(datax,datay)

            if save:
                plt.savefig("fig3/log_nsep_data")

            ## Visualisation de la fonction de coût en 2D
            ## Fabrication d'une grille de discrétisation pour la visualisation de la fonction de coût
            lim = np.ceil(np.max(np.abs(w_list))+1)
            grid, x_grid, y_grid = make_grid(xmin=-lim, xmax=lim, ymin=-lim, ymax=lim, step=100)
            plt.figure()
            plt.contourf(x_grid,y_grid,np.array([mse(w,datax,datay).mean() for w in grid]).reshape(x_grid.shape),levels=20)
            plt.scatter(w_list[:,0],w_list[:,1])
            plt.colorbar()

            if save:
                plt.savefig("fig3/log_nsep_loss")

            ## Evolution de l'erreur
            plt.figure()
            plt.plot(err_list)
            plt.xlabel("iteration")
            plt.ylabel("erreur")

            if save:
                plt.savefig("fig3/log_nsep_err")

            plt.show()
        
