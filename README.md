# ProjetSD
Projet de Science des Données - M2 MAS  
Slides / Présentation : ![Lien Overleaf - LateX](https://www.overleaf.com/project/5f9199e2a19ef20001a08ce4)

## Structure du Projet :

```
├── data `Dossier des fichiers`
│   ├── categories_string.csv
│   ├── template_submissions.csv
│   ├── test.json
│   ├── train.json
│   └── train_label.csv
├── exploration `Scripts & Notebook d'exploration`
│   ├── Dask_MacroDI.ipynb (OK)
│   ├── Dask_ML_TextPreprocessing.ipynb (OK)
│   ├── Gender_words.ipynb (OK)
│   ├── MarcHersant.py (OK)
│   ├── multiclass-text-classification-with-spacy-dask.ipynb (HS)
│   ├── PierreFinal.ipynb (HS)
│   ├── PierreLep.py (HS)
│   ├── Prof_reduction.ipynb (OK)
│   └── PYColson.py (OK)
├── global.py `fichier maître`
├── Historique_resultats.csv
├── __init__.py
├── ManipData `Scripts de Manipulation des données`
│   ├── __init__.py
│   ├── IO_Kaggle.py `Input/Output & Soumission Kaggle`
│   └── txt_prep.py `Preparation des chaînes de charactères`
├── ProjetSdEnv.yml
└── README.md
```

Tout les fichiers `(OK)` sont utilisables et stable (sans bugs) tout les fichiers `(HS)` sont non utilisables.

_Pour générer l'arboresence des fichiers:_ ![Aller ici](https://stackoverflow.com/questions/36321815/how-to-automatically-create-readme-md-markdown-of-directory-tree-listing?answertab=votes#tab-top)

---
## S'approprier le projet : 
### Nouvelle Utilisation :
1. `git clone https://github.com/Data-Brothers/ProjetSD.git`
2. `conda create -y -n ProjetSD python=3.8`
3. `conda activate ProjetSD`
4. `conda env update -f ProjetSdEnv.yml -n ProjetSD`
5. `conda run -n ProjetSD python -m ipykernel install --user --name ProjetSD`

### Utilisation standard
1. `conda activate ProjetSD`
2. `git pull`
3. Amuse-toi ! 

### Installation de Nouveau Packages
1. `conda env export > ProjetSdEnv.yml`
2. `git add .`
3. `git commit -m "Adding pkg"`
4. `git push`

---
## Etapes :

* Visualisation des données brutes
* Préparation de données
* Création du modèle


---
## Idées : 
* Interprétation du texte :
    * Découpage par :
        * `mots`
        * `phrase`
	(Aspect multi-echelle?)
    * Word-Embeeding :
        * Word2Vec
        * [Fairness in Word-Embeding](https://www.kdnuggets.com/2020/08/word-embedding-fairness-evaluation.html)
        * [Debiasing Word Embeding](http://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf)
