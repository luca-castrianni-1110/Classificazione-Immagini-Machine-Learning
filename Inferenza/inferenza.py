'''Cosa fa: Carica il pacchetto completo, intercetta le immagini inserite nella cartella Foto_da_aggiungere, ne estrae le feature locali SIFT, le mappa sui centroidi del dizionario visivo tramite la distanza euclidea minima (cdist), compila l'istogramma delle frequenze, lo normalizza in norma L2 e lo dà in pasto alla SVM per ottenere la classe testuale e la probabilità di confidenza.
A cosa serve: È lo stadio finale dell'applicazione (Deployment). Serve a dimostrare che l'intelligenza artificiale creata è in grado di generalizzare, ovvero di applicare la conoscenza appresa durante l'addestramento su fotografie scattate ex-novo, simulando il funzionamento reale di un software commerciale o di un sistema di monitoraggio territoriale satellitare.
Perché lo facciamo (Domande tipiche da esame):
Perché usi cdist e non un ciclo for? Un ciclo for in Python puro per confrontare 1000 descrittori SIFT contro 500 centroidi richiederebbe 1000×500=500.000 iterazioni ad ogni singola immagine, rendendo il software inutilizzabile in tempo reale. cdist (della libreria scipy) esegue il calcolo delle distanze euclidee interamente in linguaggio C sotto il cofano, sfruttando la vettorializzazione della memoria e abbattendo il tempo di esecuzione a frazioni di millisecondo.
Perché è obbligatoria la Normalizzazione L2 dell'istogramma? Perché il Bag of Visual Words conta le occorrenze pure. Se una foto ha una risoluzione doppia o contiene più luce, estrarrà molti più punti SIFT rispetto a un'immagine piccola o scura. La normalizzazione L2 trasforma il vettore in coordinate geometriche unitarie (lunghezza = 1), isolando la "proporzione" delle parole visive indipendentemente dal numero totale assoluto di punti chiave estratti.
Perché fai il reshape(1, -1) dell'istogramma? Scikit-Learn è progettato per elaborare dataset (matrici bidimensionali ad N righe e M colonne). Quando inseriamo una singola immagine, Python la vede come un vettore monodimensionale di lunghezza 500. Passarlo direttamente causerebbe un ValueError di incompatibilità di forma. Il reshape(1, -1) costringe l'array a comportarsi come una matrice strutturata di dimensioni 1×500, permettendo alla SVM di calcolare il prodotto 
scalare con l'iperpiano decisionale senza sollevare eccezioni di runtime.'''


# ==============================================================================
# SCRIPT: INFERENZA / DEPLOYMENT IN PRODUZIONE (`inferenza.py`)
# TRAGUARDO: Classificazione in tempo reale di immagini mai viste dal sistema
# ==============================================================================

# Importa glob per scansionare la cartella delle nuove immagini in input
import glob
# Importa OpenCV per caricare le immagini, convertirle in grigi ed estrarre le SIFT
import cv2
# Importa Numpy per la creazione dell'istogramma e la normalizzazione vettoriale
import numpy as np
# Importa pickle per caricare il pacchetto monolitico pre-addestrato
import pickle
# Importa matplotlib per mostrare a schermo il risultato visivo finale con il titolo della classe
import matplotlib.pyplot as plt
# Importa cdist (scipy) per effettuare il calcolo parallelo e velocissimo delle distanze euclidee
from scipy.spatial.distance import cdist
# Importa os per gestire i percorsi in modo nativo su qualunque sistema operativo
import os
# Importa la configurazione globale per leggere i limiti hardware dei descrittori
import configurazione

# Localizzazione del file binario del modello precedentemente congelato
cartella_corrente = os.path.dirname(os.path.abspath(__file__))
percorso_modello = os.path.join(cartella_corrente, "modello_finale_inferenza.pkl")

# [COSA FA]: Blocco di caricamento e deserializzazione iniziale del sistema congelato
try:
    with open(percorso_modello, 'rb') as f:
        pacchetto = pickle.load(f)
    print("[LOG]: Modello e Vocabolario caricati in memoria correttamente!")
except FileNotFoundError:
    print(f"[ERRORE CRITICO]: File non trovato in: {percorso_modello}. Esegui prima salva_modello_finale.py")
    exit()

# [A COSA SERVE]: Estrae dal pacchetto le singole componenti precedentemente salvate
modello = pacchetto['modello']          # La SVM pre-addestrata
vocabolario = pacchetto['vocabolario']  # La matrice dei 500 centroidi SIFT
encoder = pacchetto['encoder']          # Il decodificatore per tornare al testo
K = pacchetto['K']                      # Il numero di bin dell'istogramma (500)

def predici_immagine(percorso_immagine):
    """
    [A COSA SERVE]: Pipeline completa di inferenza su singola immagine:
    Lettura -> SIFT -> Quantizzazione (cdist) -> Istogramma BoW -> Normalizzazione L2 -> Predizione SVM.
    """
    # 1. CARICAMENTO E PRE-PROCESSING
    img_bgr = cv2.imread(percorso_immagine)
    if img_bgr is None:
        print(f"Immagine non valida o corrotta: {percorso_immagine}")
        return
    # [PERCHÉ LO FACCIAMO]: SIFT analizza le variazioni di luminanza (gradienti), il colore non serve
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. ESTRAZIONE DELLE CARATTERISTICHE LOCALI (SIFT)
    sift = cv2.SIFT.create(nfeatures=configurazione.MAX_DESCRITTORI_PER_IMMAGINE)
    kp, descrittori = sift.detectAndCompute(img_gray, None)

    # Controlla se l'immagine possiede tessitura e dettagli sufficienti
    if descrittori is not None:
        # 3. QUANTIZZAZIONE VETTORIALE (MAPPING SUL VOCABOLARIO VISIVO)
        # [PERCHÉ LO FACCIAMO]: cdist calcola la distanza euclidea tra OGNI descrittore SIFT trovato nell'immagine (N)
        # e TUTTI i 500 centroidi del vocabolario (K). Produce una matrice di distanze di forma (N x 500).
        distanze = cdist(descrittori, vocabolario, metric='euclidean')
        
        # [COSA FA]: np.argmin(..., axis=1) scorre la matrice riga per riga e trova l'indice (da 0 a 499)
        # del centroide più vicino in assoluto. Associa ogni punto geometrico alla sua "parola visiva".
        parole_visive = np.argmin(distanze, axis=1)

        # 4. COSTRUZIONE DELL'ISTOGRAMMA BAG OF VISUAL WORDS
        # Inizializza un vettore vuoto di 500 zeri (corrispondente al nostro vocabolario di taglia K)
        istogramma = np.zeros(K)
        # Cicla sulle parole visive assegnate e accumula le frequenze (voto di maggioranza)
        for p in parole_visive:
            istogramma[p] += 1

        # 5. NORMALIZZAZIONE VETTORIALE L2 (CRITICA PER L'ORALE!)
        # [PERCHÉ LO FACCIAMO]: Le immagini hanno risoluzioni e quantità di dettagli differenti. 
        # Un'immagine ricca di texture potrebbe generare 2000 punti SIFT, una liscia solo 200.
        # Senza la normalizzazione L2, i valori assoluti dell'istogramma ingannerebbero la SVM. 
        # Dividendo l'istogramma per la sua norma euclidea (lunghezza del vettore pari a 1), le frequenze 
        # diventano percentuali relative stabili, rendendo confrontabili immagini di qualsiasi tipo.
        norma = np.linalg.norm(istogramma)
        if norma > 0:
            istogramma = istogramma / norma

        # 6. RESHAPE MATRICIALE PER L'API DI SCIKIT-LEARN (CRITICA PER L'ORALE!)
        # [PERCHÉ LO FACCIAMO]: Il metodo .predict() di Scikit-Learn esige RIGOROSAMENTE una matrice a due dimensioni
        # nella forma (Numero_Campioni, Numero_Feature). Anche se stiamo predicendo UNA SOLA immagine, non possiamo passargli
        # un vettore piatto da 500 elementi. reshape(1, -1) trasforma il vettore in una matrice 1 x 500 (1 riga, 500 colonne),
        # assecondando l'architettura interna della libreria ed evitando crash di violazione del formato.
        istogramma = istogramma.reshape(1, -1) 
        
        # 7. INFERENZA E PREDIZIONE PROBABILISTICA
        # Sottopone l'istogramma normalizzato agli iperpiani della SVM per estrarre l'indice numerico predetto
        predizione_num = modello.predict(istogramma)[0]
        # [COSA FA]: Sfrutta il Platt Scaling per calcolare la confidenza associata a quella specifica classe
        probabilita = modello.predict_proba(istogramma)[0][predizione_num] * 100
        
        # [COSA FA]: Traduce l'indice numerico nella stringa testuale originaria (es. 3 -> "forest")
        nome_classe = encoder.inverse_transform([predizione_num])[0]

        # 8. RENDERING GRAFICO DEI RISULTATI
        # Converte l'immagine da BGR (OpenCV) a RGB (Matplotlib) per visualizzare i colori reali senza alterazioni
        plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        # Stampa a schermo la decisione del sistema e la percentuale di certezza matematica calcolata
        plt.title(f"Predizione: {nome_classe.upper()} ({probabilita:.2f}%)", fontsize=14, fontweight='bold', pad=10)
        plt.axis('off') # Nasconde i pixel degli assi cartesiani per un output pulito ed elegante
        plt.show()
        
        print(f"[PREDIZIONE]: L'immagine analizzata è: {nome_classe.upper()} | Confidenza: {probabilita:.2f}%")
    else:
        print("[AVVISO]: Impossibile classificare. Zero punti chiave SIFT estratti.")


if __name__ == "__main__":
    # CABINA DI REGIA: Scansione ricorsiva della cartella di test dell'utente
    cartella_dello_script = os.path.dirname(os.path.abspath(__file__))
    nome_cartella_foto = "Foto_da_aggiungere"
    percorso_cartella_foto = os.path.join(cartella_dello_script, nome_cartella_foto)

    # Verifica la presenza reale della cartella sul computer
    if not os.path.exists(percorso_cartella_foto):
        print(f"[ERRORE]: La cartella operativa '{nome_cartella_foto}' non esiste. Creala e inserisci le foto.")
    else:
        # [PERCHÉ LO FACCIAMO]: Usiamo le tuple di estensioni per accettare file da fotocamere diverse (.jpg, .png)
        estensioni = ('*.jpg', '*.JPG', '*.jpeg', '*.png')
        lista_immagini = []
        for est in estensioni:
            # glob funge da motore di ricerca integrato: analizza la cartella e colleziona i percorsi completi
            lista_immagini.extend(glob.glob(os.path.join(percorso_cartella_foto, est)))
            
        if not lista_immagini:
            print(f"[AVVISO]: Cartella '{nome_cartella_foto}' individuata ma priva di immagini supportate.")
            print("Contenuto attuale della cartella:", os.listdir(percorso_cartella_foto))
        else:
            print(f"[START]: Rilevate {len(lista_immagini)} immagini esterne. Avvio della pipeline di inferenza...")
            # Cicla sulle foto ed esegue la classificazione istantanea una per una
            for percorso_foto in lista_immagini:
                predici_immagine(percorso_foto)