'''Questo script gestisce l'addestramento e la serializzazione del vocabolario visivo (le "parole visive" del sistema BoW).
salva_descrittori / carica_descrittori / carica_vocabolario
A COSA SERVE: Scrivono e leggono oggetti complessi di Python (come liste di array numpy) direttamente su file binari del PC 
utilizzando la libreria pickle.
PERCHÉ SI USA: L'estrazione delle SIFT da migliaia di immagini può richiedere molto tempo. Salvando i risultati intermedi su disco 
tramite queste funzioni di utilità, è possibile riavviare il programma istantaneamente caricando i file pronti dalla cache.
esegui_clustering_e_salva
A COSA SERVE: Prende i descrittori SIFT estratti da tutto il dataset di supporto, li unisce in un'unica enorme matrice, applica la
normalizzazione L2 e lancia l'algoritmo MiniBatchKMeans per raggrupparli in K cluster distinti. I centroidi trovati vengono estratti e salvati come "vocabolario visivo".
PERCHÉ SI USA: Nel concetto BoW, un'immagine non può essere data a un classificatore sotto forma di punti SIFT sparsi perché ogni immagine ha un 
numero variabile di punti. Dobbiamo raggruppare lo spazio di tutti i possibili descrittori SIFT in K gruppi standard (le nostre parole visive). 
MiniBatchKMeans si usa al posto del KMeans classico perché elabora i dati a piccoli blocchi (batch), impedendo al computer di finire la memoria RAM a 
causa delle dimensioni della matrice complessiva. La normalizzazione L2 assicura che le distanze Euclidee calcolate tra i descrittori SIFT rimangano stabili e coerenti.'''


# Importa Numpy per la concatenazione verticale e la gestione strutturale delle matrici numeriche.
import numpy as np
# Importa pickle, la libreria standard per salvare oggetti Python complessi in file binari pronti al riutilizzo.
import pickle
# Importa os per verificare l'integrità dei file e manipolare l'interfaccia di scrittura.
import os
# Importa MiniBatchKMeans, una variante ottimizzata del K-Means adatta a dataset massivi.
from sklearn.cluster import MiniBatchKMeans
# Importa la funzione di normalizzazione geometrica delle matrici.
from sklearn.preprocessing import normalize

def salva_descrittori(lista_descrittori, percorso_salvataggio):
    # Apre un file nel percorso indicato in modalità scrittura binaria ('wb' = write binary).
    with open(percorso_salvataggio, 'wb') as file:
        # Scrive l'intero oggetto Python direttamente dentro il file binario serializzandolo.
        pickle.dump(lista_descrittori, file)

def carica_descrittori(percorso_caricamento):
    # Apre il file binario dei descrittori archiviato sul PC in modalità lettura binaria ('rb' = read binary).
    with open(percorso_caricamento, 'rb') as file:
        # Legge e ricostruisce fedelmente l'oggetto originale in memoria tramite deserializzazione.
        return pickle.load(file)
    
def carica_vocabolario(percorso_caricamento):
    # Apre il file binario del vocabolario (centroidi) in modalità lettura binaria.
    with open(percorso_caricamento, 'rb') as file:
        # Restituisce la matrice dei centroidi ricostruita in memoria.
        return pickle.load(file)

def esegui_clustering_e_salva(lista_descrittori, k_cluster, percorso_salvataggio_vocab):
    # np.vstack prende una lista contenente matrici SIFT separate e le "impila" in verticale.
    # Trasforma una lista di matrici in un'unica gigantesca matrice di descrittori globali.
    matrice_descrittori = np.vstack(lista_descrittori)
    # Stampa a schermo la dimensione totale della matrice per monitorare quanti descrittori complessivi verranno elaborati dal clustering.
    print(f"Forma della matrice: {matrice_descrittori.shape}")
    
    # Applica la normalizzazione L2 lungo le righe della matrice (norm='l2').
    # Significa che ogni riga (ogni singolo descrittore SIFT di 128 elementi) viene riscalata in modo che la somma dei quadrati dei suoi elementi sia pari a 1.
    # Questo passaggio standardizza l'impatto dei vettori impedendo a picchi di intensità isolati di falsare il calcolo delle distanze Euclidee.
    descrittori_normalizzati = normalize(matrice_descrittori, norm='l2')
    
    # Stampa un messaggio informativo che segnala l'avvio del clustering indicando il valore corrente di K.
    print(f"\nK-Means con K={k_cluster}\n")
    # Configura il motore di clustering MiniBatchKMeans.
    # n_clusters=k_cluster imposta il numero di parole visive che vogliamo estrarre.
    # random_state=42 fissa il seme di generazione casuale per garantire che rieseguendo il codice si ottengano sempre gli stessi identici centroidi.
    # batch_size=8192 dice all'algoritmo di elaborare 8192 descrittori alla volta per ottimizzare l'uso della memoria cache della CPU.
    # n_init="auto" lascia a scikit-learn la gestione ottimale dell'inizializzazione dei centroidi di partenza.
    algoritmo_kmeans = MiniBatchKMeans(n_clusters=k_cluster, random_state=42, batch_size=8192, n_init="auto")
    # Avvia l'addestramento del modello geometrico sui descrittori normalizzati (trova la posizione ottimale dei cluster).
    algoritmo_kmeans.fit(descrittori_normalizzati)
    
    # Estrae i centri geometrici finali dei cluster calcolati (centroidi).
    # Questa matrice rappresenta il nostro vero e proprio "Vocabolario Visivo" (Visual Vocabulary).
    # Ha una dimensione pari a (K_cluster x 128). Ogni riga è una "parola visiva".
    vocabolario_visivo = algoritmo_kmeans.cluster_centers_
    
    # Notifica l'avvenuta conclusione dei calcoli e la destinazione di salvataggio del file.
    print(f"\nSalvataggio vocabolario completato in: {percorso_salvataggio_vocab}\n")
    # Apre un canale di scrittura binaria verso il percorso stabilito.
    with open(percorso_salvataggio_vocab, 'wb') as file:
        # Archivia in modo permanente la matrice del vocabolario sul disco.
        pickle.dump(vocabolario_visivo, file)
        
    # Restituisce la matrice del vocabolario per poterla agganciare subito ai flussi successivi del Main.
    return vocabolario_visivo