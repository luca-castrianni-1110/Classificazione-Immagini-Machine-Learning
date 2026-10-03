import os

CARTELLA_PROGETTO = os.path.dirname(os.path.abspath(__file__))

PERCORSO_DATASET_AID = './AID'
PERCORSO_DATASET_CLASSI = './UCMerced_LandUse/Images'

CARTELLA_RISULTATI = './File_per_i_calcoli/Risultati'

PERCORSO_SALVATAGGIO_DESCRITTORI = './File_per_i_calcoli/Risultati/tutti_descrittori_estratti.pkl'

ESTRATTORE_CARATTERISTICHE = "SIFT"
MAX_DESCRITTORI_PER_IMMAGINE = 2000

VALORI_K_DA_TESTARE = [50, 100, 500]

K_VINCENTE = 500

def ottieni_percorso_vocabolario(k):
    return os.path.join(CARTELLA_RISULTATI, f"vocabolario_K{k}.pkl")

def ottieni_percorso_dati_addestramento(k):
    return os.path.join(CARTELLA_RISULTATI, f"dati_addestramento_K{k}.pkl")