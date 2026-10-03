'''Questo file contiene il motore logico di quantizzazione vettoriale. Trasforma le descrizioni locali di SIFT in un unico vettore globale a dimensione fissa per l'immagine.
calcola_singolo_istogramma
A COSA SERVE: Prende i descrittori SIFT di una singola immagine e, per ciascuno di essi, trova qual è la parola più vicina all'interno del vocabolario visivo. Conta poi la frequenza di queste associazioni per generare un istogramma di dimensione K.
PERCHÉ SI USA: È la quantizzazione vera e propria. Invece di descrivere l'immagine tramite punti sparsi, la descriviamo dicendo quante volte compaiono determinate strutture visive standardizzate. L'uso di pairwise_distances_argmin è una scelta legata alle prestazioni: questa funzione è implementata nativamente in C e calcola la distanza tra migliaia di vettori quasi istantaneamente, sostituendo cicli Python che richiederebbero molto più tempo.
genera_dataset_addestramento
A COSA SERVE: Applica la funzione precedente a tutte le immagini del dataset di classificazione, raccoglie i rispettivi istogrammi e applica una normalizzazione L2 globale alla matrice risultante.
PERCHÉ SI USA: Prepara la matrice delle caratteristiche finale (X) e il vettore dei target (y) pronti per essere passati ai modelli di classificazione (es. SVM o Random Forest). La normalizzazione L2 finale è cruciale: garantisce che l'istogramma descriva la proporzione delle caratteristiche visive e non il loro numero assoluto. Senza questo passaggio, un'immagine molto grande o ricca di dettagli avrebbe valori intrinsecamente più alti rispetto a un'immagine piccola, alterando la classificazione a prescindere dal reale contenuto visivo.
salva_dati_addestramento
A COSA SERVE: Salva in formato binario il dizionario strutturato contenente le caratteristiche X normalizzate e le etichette target y.'''

# Importa Numpy per operare su array e calcolare gli istogrammi a intervalli discreti.
import numpy as np
# Importa pairwise_distances_argmin, funzione ultra-ottimizzata per trovare velocemente l'indice dell'elemento più vicino tra due matrici.
from sklearn.metrics import pairwise_distances_argmin
# Importa la funzione di normalizzazione dei vettori.
from sklearn.preprocessing import normalize
# Importa la barra di avanzamento visiva per monitorare l'andamento del loop a terminale.
from tqdm import tqdm
# Importa pickle per archiviare i dati pronti per l'addestramento.
import pickle

# Carica in memoria la funzione di estrazione delle caratteristiche dal file limitrofo.
from estrazione_caratteristiche import estrai_descrittori_locali

def calcola_singolo_istogramma(descrittori_immagine, vocabolario):
    # Legge il numero totale di parole visive disponibili estraendo la dimensione della prima dimensione del vocabolario (pari a K).
    numero_parole_visive = vocabolario.shape[0]
    # Controllo di sicurezza per immagini con texture piatte o assenza di dettagli utili (es. porzione omogenea di mare o deserto).
    if descrittori_immagine is None or len(descrittori_immagine) == 0:
        # Restituisce un vettore interamente composto da zeri di lunghezza pari a K. Previene il blocco dei calcoli successivi.
        return np.zeros(numero_parole_visive)
    
    # Esegue la quantizzazione vettoriale vettorializzata.
    # Per ogni descrittore SIFT (riga) della matrice 'descrittori_immagine', calcola la distanza euclidea con tutte le K righe di 'vocabolario'.
    # Restituisce un array 1D di lunghezza pari al numero di descrittori dell'immagine, dove ogni cella contiene l'indice (0 a K-1) del centroide più vicino.
    indici_parole = pairwise_distances_argmin(descrittori_immagine, vocabolario)
    
    # Costruisce l'istogramma di frequenza partendo dalle assegnazioni numeriche raccolte.
    # np.histogram conta quante volte un determinato indice compare dentro il vettore indici_parole.
    # bins=np.arange(numero_parole_visive + 1) definisce con precisione i confini dei contenitori (da 0 fino a K inclusi).
    istogramma, _ = np.histogram(indici_parole, bins=np.arange(numero_parole_visive + 1))
    
    # Restituisce l'istogramma di frequenze grezzo calcolato per l'immagine analizzata.
    return istogramma

def genera_dataset_addestramento(percorsi, etichette, estrattore, vocabolario):
    # Inizializza una lista vuota per raccogliere gli istogrammi di tutte le immagini elaborate.
    lista_istogrammi = []
    # Inizializza una lista parallela per memorizzare le etichette validi, escludendo eventuali file corrotti.
    etichette_valide = []
        
    # Avvia il ciclo principale scorrendo contemporaneamente i percorsi e le rispettive classi.
    # zip() unisce le liste, tqdm() avvolge il tutto generando una barra grafica interattiva di caricamento a schermo.
    for percorso, etichetta in tqdm(zip(percorsi, etichette), total=len(percorsi)):
        # Estrae i descrittori SIFT locali per l'immagine corrente del loop senza imporre limiti inferiori di campionamento.
        descrittori = estrai_descrittori_locali(percorso, estrattore, max_descrittori=None)
        
        # Verifica che l'estrazione non abbia riscontrato anomalie o restituito matrici nulle.
        if descrittori is not None:
            # Calcola l'istogramma BoW mappando i descrittori estratti sopra sul vocabolario comune.
            isto = calcola_singolo_istogramma(descrittori, vocabolario)
            # Aggiunge l'istogramma appena calcolato alla collezione generale.
            lista_istogrammi.append(isto)
            # Memorizza l'etichetta associata all'immagine allineandola all'indice della lista degli istogrammi.
            etichette_valide.append(etichetta)
            
    # Converte la lista complessiva degli istogrammi in una matrice bi-dimensionale standard di Numpy (Righe: Immagini, Colonne: K caratteristiche).
    X_matrice = np.array(lista_istogrammi)
    
    # Normalizza geometricamente la matrice finale degli istogrammi applicando la norma L2 lungo l'orizzontale.
    # Ogni riga della matrice (l'istogramma BoW di un'immagine) viene divisa per la sua lunghezza geometrica complessiva.
    # Questo trasforma le frequenze grezze in frequenze relative (probabilità).
    X_normalizzata = normalize(X_matrice, norm='l2')
    # Trasforma la lista delle etichette stringa in un array vettoriale monodimensionale compatto di Numpy.
    y_vettore = np.array(etichette_valide)
    
    # Restituisce la matrice normalizzata delle caratteristiche X e il corrispondente vettore delle classi Y.
    return X_normalizzata, y_vettore

def salva_dati_addestramento(X, y, percorso_salvataggio):
    # Mostra un riscontro a terminale che conferma l'avvio della scrittura fisica dei dati di addestramento finali.
    print(f"Salvataggio dataset di addestramento in {percorso_salvataggio}")
    # Apre il file di destinazione in modalità scrittura binaria.
    with open(percorso_salvataggio, 'wb') as file:
        # Impacchetta i dati inserendoli all'interno di un dizionario strutturato con chiavi 'X' e 'y'.
        pickle.dump({'X': X, 'y': y}, file)