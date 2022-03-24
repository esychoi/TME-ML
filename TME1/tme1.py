## Esther CHOI 3800370
## ML TME1

import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn import svm
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier as DTree
import pydotplus

# Question 1

def entropie(vect):
    """
    array -> float
    retourne l'entropie de vect
    """
    n = len(vect)
    c = Counter(vect)
    res = 0.0
    for y in c.keys():
        py = c[y]/n
        res -= py * np.log(py)
    return res

# Question 2

def entropie_cond(list_vect):
    """
    list(array) -> float
    retourne l'entropie conditionnelle de list_vect
    """
    n = len(list_vect) #nombre de partitions
    m = 0   #nombre d'éléments total dans list_vect
    res = 0.0
    for i in range(n):
        Pi = list_vect[i]
        res += entropie(Pi) * len(Pi)
        m += len(Pi)
    return res/m

# Question 3

# data : tableau (films ,features), id2titles : dictionnaire  id -> titre
# fields : id  feature  -> nom
[data , id2titles , fields ]= pickle.load(open("imdb_extrait.pkl","rb"))
# la  derniere  colonne  est le vote
datax = data[: ,:28].astype(int)
datay = np.array([1 if x[33] > 6.5 else 0 for x in data])

# entropies des colonnes 0 à 28 (genres (variables binaires)) de datax
ent = np.zeros(28)
ent_cond = np.zeros(28)
diff_ent = np.zeros(28)
for i in range(28):
    vecti = datax[:,i]  #i-ème feature

    #calculs
    ent[i] = entropie(vecti)
    list_vecti = [datay[(vecti==1)],datay[vecti!=1]]
    ent_cond[i] = entropie_cond(list_vecti)
    diff_ent[i] = ent[i] - ent_cond[i]
    
    #affichage
    print("----------------------")
    print(fields[i])
    print("entropie :", ent[i])
    print("entropie conditionnelle :",ent_cond[i])
    print("différence :",diff_ent[i])

# print(fields[np.argmax(ent)])
# print(fields[np.argmin(ent_cond)])


# Question 4

id2genre = [x[1] for x in  sorted(fields.items ())[: -2]]
dt = DTree()
dt.max_depth = 5 #on fixe la  taille  max de l’arbre a 5
dt.min_samples_split = 2 #nombre  minimum d’exemples  pour  spliter  un noeud
dt.fit(datax ,datay)
dt.predict(datax [:5 ,:])
#print(dt.score(datax ,datay))
#export_graphviz(dt, out_file="/tmp/tree.dot",feature_names=id2genre)
# ou avec  pydotplus
#tdot = export_graphviz(dt,feature_names=id2genre)
#pydotplus.graph_from_dot_data(tdot).write_pdf('tree.pdf')

scores = []
save = False
for i in range(1,10,2):
    dt = DTree()
    dt.max_depth = i #on fixe la  taille  max de l’arbre a 5
    dt.min_samples_split = 2 #nombre  minimum d’exemples  pour  spliter  un noeud
    dt.fit(datax ,datay)
    dt.predict(datax [:5 ,:])
    scores.append(dt.score(datax ,datay))
    if save:
        # export_graphviz(dt, out_file="/tmp/tree.dot",feature_names=id2genre)
        # ou avec  pydotplus
        tdot = export_graphviz(dt,feature_names=id2genre)
        pydotplus.graph_from_dot_data(tdot).write_pdf('fig/tree_'+str(i)+'.pdf')


def generateDT(x,y,profondeur,min_split=2,save_file=False):
    dt = DTree()
    dt.max_depth = profondeur #on fixe la taille max de l’arbre
    dt.min_samples_split = min_split #nombre minimum d’exemples pour spliter un noeud
    dt.fit(x ,y)

    if save_file:
        #export_graphviz(dt, out_file="/tmp/tree.dot",feature_names=id2genre)
        # ou avec  pydotplus
        tdot = export_graphviz(dt,feature_names=id2genre)
        pydotplus.graph_from_dot_data(tdot).write_pdf('tree_'+str(profondeur)+'.pdf')
    
    return dt

# Question 7

proportion_test = [0.8,0.5,0.2]
save=False
show = False
for prop in proportion_test:
    dataxtrain, dataxtest, dataytrain, dataytest = train_test_split(datax, datay, test_size=prop, random_state=0)

    strain = []
    stest = []
    for p in range(1,16):
        dt = generateDT(dataxtrain,dataytrain,p)
        strain.append(1-dt.score(dataxtrain ,dataytrain))
        stest.append(1-dt.score(dataxtest ,dataytest))

    if show:
        plt.figure()
        plt.title("prop_test = "+str(prop))
        plt.plot(strain,label="train")
        plt.plot(stest,label="test")
        plt.xlabel("profondeur")
        plt.ylabel("erreur")
        plt.legend()
    
    if save:
        plt.savefig("fig/plot_errors-prop_test="+str(int(prop*10)))

plt.show()

# Question 8

# Question validation croisée
def cross_val_dt(X,y,n_folds=5,min_split=2,showplot=False,save=False):
    """
    Input :
        X (array) : data
        y (array) : labels
        min_split (int) : nombre minimum d’exemples pour spliter un noeud
        n_folds (int) : number of folds
        save (bool) : plots saved iff True
    Output : scores
    """
    strain = []
    stest = []

    for p in range(1,16):
        dt = DTree()
        dt.max_depth = p #on fixe la taille max de l’arbre
        dt.min_samples_split = min_split
        scores = cross_validate(dt,X,y,cv=n_folds,return_train_score=True)

        # plots    
        strain.append(1-scores['train_score'].mean())
        stest.append(1-scores['test_score'].mean())

    if showplot:
        plt.figure()
        plt.plot(strain,label="train")
        plt.plot(stest,label="test")
        plt.xlabel("profondeur")
        plt.ylabel("erreur")
        plt.legend()
        
    if save:
        plt.savefig("fig/plot_errors-crossval")

    if showplot:
        plt.show()

    return strain,stest

cross_val_dt(datax,datay,showplot=True,save=True)