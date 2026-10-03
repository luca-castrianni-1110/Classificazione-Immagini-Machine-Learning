'''Questo script incapsula la logica di estrazione dei descrittori locali a basso livello dalle immagini utilizzando la libreria OpenCV.
inizializza_estrattore
A COSA SERVE: Configura e crea l'istanza dell'oggetto estrattore di OpenCV scelto (SIFT o ORB).
PERCHÉ SI USA: Astrae la creazione dell'algoritmo. Se un domani volessi usare ORB invece di SIFT, basterebbe cambiare una stringa 
senza riscrivere il codice di estrazione. Passare nfeatures limita l'estrazione nativa per evitare sprechi computazionali.
estrai_descrittori_locali
A COSA SERVE: Carica un'immagine dal disco in modalità monocromatica, calcola i suoi punti di interesse e restituisce i relativi vettori 
descrittori SIFT (ogni punto viene descritto da un vettore di 128 numeri). Se i punti trovati superano la soglia massima, ne seleziona un sottoinsieme casuale.
PERCHÉ SI USA: Le immagini satellitari possono presentare un numero enorme di texture, generando decine di migliaia di punti SIFT. Il 
sottocampionamento casuale (np.random.choice) assicura che nessuna immagine sovrasti numericamente le altre durante la creazione del vocabolario, 
garantendo l'omogeneità statistica dei dati.'''


# Importa OpenCV, essenziale per la computer vision e la gestione degli algoritmi di estrazione caratteristiche.
import cv2
# Importa il modulo configurazione precedentemente analizzato per accedere alle costanti globali.
import configurazione
# Importa Numpy per manipolare vettori e matrici numeriche ad alta efficienza.
import numpy as np

def inizializza_estrattore(tipo_estrattore="SIFT"):
    # Verifica se la stringa passata corrisponde all'algoritmo SIFT.
    if tipo_estrattore == "SIFT":
        # Crea e restituisce l'oggetto estrattore SIFT nativo di OpenCV.
        # nfeatures configura il limite massimo di caratteristiche che l'algoritmo cercherit di individuare all'inizio.
        return cv2.SIFT.create(nfeatures=configurazione.MAX_DESCRITTORI_PER_IMMAGINE)
    
    # Verifica alternativa per l'algoritmo ORB (algoritmo più veloce ma meno robusto alle variazioni di scala).
    elif tipo_estrattore == "ORB":
        # Crea e restituisce l'oggetto estrattore ORB di OpenCV.
        return cv2.ORB_create()
    # Clausola di salvaguardia: se viene inserito un nome di algoritmo non censito o supportato.
    else:
        # Solleva un'eccezione bloccante per indicare lo sviluppatore l'errore di battitura o configurazione.
        raise ValueError("Errore estrattore")

def estrai_descrittori_locali(percorso_immagine, estrattore, max_descrittori=None):
    # Carica l'immagine dal percorso specificato sul disco rigido.
    # cv2.IMREAD_GRAYSCALE forza la conversione immediata in scala di grigi (un solo canale di intensità).
    # È necessario perché gli operatori di calcolo dei gradienti SIFT agiscono sulla luminanza, non sul colore.
    immagine = cv2.imread(percorso_immagine, cv2.IMREAD_GRAYSCALE)
    # Controllo di integrità: se l'immagine non è stata caricata correttamente (es. file corrotto o percorso errato).
    if immagine is None:
        # Ritorna None per evitare che il programma fallisca sulle righe successive nel tentativo di elaborare il nulla.
        return None
    
    # Funzione centrale di OpenCV: rileva contemporaneamente i Keypoints (punti geometrici stabili)
    # e calcola i Descrittori (array numerici associati all'intorno matematico di ogni punto chiave).
    # Il secondo parametro è impostato a None poiché non utilizziamo alcuna maschera di segmentazione.
    punti_chiave, descrittori = estrattore.detectAndCompute(immagine, None)
    
    # Controlla se sono stati effettivamente estratti dei descrittori e se è stato impostato un tetto massimo di contenimento.
    if descrittori is not None and max_descrittori is not None:
        # Se il numero di descrittori calcolati è numericamente superiore alla soglia tollerata.
        if len(descrittori) > max_descrittori:
            # np.random.choice genera un vettore di indici estratti casualmente.
            # len(descrittori) definisce il range degli indici disponibili (da 0 a N-1).
            # max_descrittori indica quanti indici vogliamo estrarre in totale.
            # replace=False garantisce che non venga mai estratto lo stesso indice più volte (campionamento senza ripetizione).
            indici = np.random.choice(len(descrittori), max_descrittori, replace=False)
            # Filtra la matrice originaria dei descrittori mantenendo esclusivamente le righe corrispondenti agli indici casuali generati.
            descrittori = descrittori[indici]
            
    # Restituisce la matrice finale dei descrittori (forma: Numero_Punti x 128).
    return descrittori