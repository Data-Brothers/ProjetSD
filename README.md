# ProjetSD
Projet de Science des Données - M2 MAS  
Slides / Présentation : ![Lien Overleaf - LateX](https://www.overleaf.com/project/5f9199e2a19ef20001a08ce4)


Pour générer l'arboresence des fichiers:_ ![Aller ici](https://stackoverflow.com/questions/36321815/how-to-automatically-create-readme-md-markdown-of-directory-tree-listing?answertab=votes#tab-top)

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
