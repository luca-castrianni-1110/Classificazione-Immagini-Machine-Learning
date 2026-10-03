# Importa il modulo os per interagire con il sistema operativo e gestire i percorsi dei file.
import os

# Determina il percorso assoluto della cartella in cui si trova questo file di configurazione.
# os.path.abspath(__file__) prende il percorso completo del file corrente.
# os.path.dirname estrae solo la cartella che lo contiene.
# Si usa per assicurarsi che i percorsi funzionino su qualsiasi computer (Windows, Mac, Linux) senza rompersi.
CARTELLA_PROGETTO = os.path.dirname(os.path.abspath(__file__))

# Definisce la cartella in cui si trova il dataset AID (usato per l'estrazione dei descrittori del vocabolario).
PERCORSO_DATASET_AID = './AID'
# Definisce la cartella del dataset UCMerced (usato per l'estrazione delle etichette e delle caratteristiche finali).
PERCORSO_DATASET_CLASSI = './UCMerced_LandUse/Images'

# Definisce la cartella di output in cui verranno salvati tutti i file generati durante l'elaborazione.
CARTELLA_RISULTATI = './File_per_i_calcoli/Risultati'

# Specifica il percorso esatto del file binario (.pkl) in cui memorizzare la cache di tutti i descrittori estratti da AID.
# Si usa per evitare di dover ricalcolare i descrittori SIFT a ogni singolo avvio del programma.
PERCORSO_SALVATAGGIO_DESCRITTORI = './File_per_i_calcoli/Risultati/tutti_descrittori_estratti.pkl'

# Imposta l'algoritmo per l'estrazione delle caratteristiche locali. In questo caso si usa "SIFT".
ESTRATTORE_CARATTERISTICHE = "SIFT"
# Limita il numero massimo di punti chiave (Keypoints) estraibili da ogni singola immagine a 2000.
# Si usa come barriera di sicurezza per evitare che immagini enormi saturino la memoria RAM del computer.
MAX_DESCRITTORI_PER_IMMAGINE = 2000

# Definisce la lista dei numeri di cluster (K) da testare per il K-Means (ovvero la dimensione del vocabolario visivo).
# Più il valore è alto, più il vocabolario è ricco, ma aumentano drasticamente i tempi di calcolo.
VALORI_K_DA_TESTARE = [50, 100, 500]

# Indica l'iperparametro K ottimale selezionato dopo la fase di test per i calcoli successivi.
K_VINCENTE = 500

# Definisce una funzione per generare dinamicamente il percorso di salvataggio del vocabolario visivo associato a un preciso valore di K.
def ottieni_percorso_vocabolario(k):
    # Combina la cartella dei risultati con una stringa univoca formattata con il valore di K corrente.
    return os.path.join(CARTELLA_RISULTATI, f"vocabolario_K{k}.pkl")

# Definisce una funzione per generare dinamicamente il percorso di salvataggio della matrice degli istogrammi finale per un preciso valore di K.
def ottieni_percorso_dati_addestramento(k):
    # Combina la cartella dei risultati con una stringa contenente il nome del file finale pronto per i modelli di classificazione.
    return os.path.join(CARTELLA_RISULTATI, f"dati_addestramento_K{k}.pkl")