"""Input/Ouput manipulation functions"""
import csv
import os
import re
import string
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass
class Inputfiles:
    """Class to store path to useful inputfiles"""

    default: Path = Path("data")
    train_data: str
    categories: str


@dataclass
class Outputfiles:
    """Class to store path to useful outputfiles"""

    default: str
    train_data: str
    submission_file: Path=


class IOHandler:
    """Class to handle submission to kaggle"""

    def __init__(self):
        self.input = Inputfiles()
        self.output = Inputfiles()

    def submit(self,custom_msg:str="New Submission from DataBrothers"):
        """
        Make a submission on kaggle

        Args:
            custom_msg (str, optional): custom message for submission.
                Defaults to "New Submission from DataBrothers".
        """
        submisson_filename =self.output.submission_file
        msg="new submission"
        command=f"kaggle competitions submit -c defi-ia-insa-toulouse -f {submisson_filename} -m '{msg}'"
        subprocess.run(command)

    def import(self)->pd.DataFrame:      
        """
        Import data

        Returns:
            pd.DataFrame: _description_
        """
        fileName_Ext = f"{fileName}.{fileExtension}"
        cmdread = f"pd.read_{fileExtension}('{fileName_Ext}')"
        dataset = eval(cmdread)
        return dataset


    def Export(
        pdDF=pd.DataFrame(),
        fileName="./data/unknown",
        fileExtension="csv",
        KaggleSubmission=False,
        MsgSubmit=,
    ):
        fileName_Ext = f"{fileName}.{fileExtension}"

        cmdto = f"pdDF.to_{fileExtension}('{fileName_Ext}', index = False, Header=True)"
        eval(cmdto)
        if KaggleSubmission:
            KaggleSubmit(fileName_Ext, MsgSubmit)
        return 0


def HistoriqueCsv(nom, preparation, modele, validation, f1score, fairness):
    """
    Parameters
    ----------
    nom : str
        Ton nom.
    preparation : str
        Méthode de préparation de text utilisé.
    modele : str
        Modèle utilisé (obliez pas les parametres).
    validation : str
        Validation effectuée (hold-out, VC...).
    f1score : float
        macro f1-score.
    fairness : float
        macro_disparate_impact.
    -------
    Ecrit le résultat d'un modèle dans le fichier Historique_resultats,
    pour pouvoir garder une trace des modèles testés.

    """
    jour = datetime.today()
    with open("Historique_resultats.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow([jour, nom, preparation, modele, validation, f1score, fairness])
