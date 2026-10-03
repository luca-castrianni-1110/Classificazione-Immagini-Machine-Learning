'''COSA FA: Legge dal disco i file binari generati durante le fasi di estrazione e clustering. Estrae statistiche quantitative sui descrittori SIFT memorizzati in cache (numero di immagini elaborate, numero totale di vettori estratti, immagini sotto la soglia critica) e converte le matrici dei centroidi K×128 in mappe di calore colorate.
A COSA SERVE: Funge da strumento di Sanity Check (controllo di sanità dei dati). Permette di verificare che i file salvati non siano corrotti o vuoti, che le dimensioni delle strutture dati rispettino rigorosamente i vincoli dell'algoritmo SIFT (128 dimensioni per descrittore) e che il clustering non abbia subito un collasso dei centroidi.
PERCHÉ LO FACCIAMO: In un progetto di Machine Learning, l'analisi dei dati (Data Inspection) previene errori sistematici a valle.
Controllare se ci sono immagini con meno di 200 feature è fondamentale: immagini quasi uniformi (es. un quadrato perfetto di cielo o mare aperto nel telerilevamento) generano pochissimi punti chiave. Sapere quante sono ci aiuta a valutare se il dataset soffre di scarsità informativa.
Disegnare la matrice dei centroidi con la mappa di calore plasma mostra l'attivazione dei gradienti lungo le 128 dimensioni di SIFT. Se vedessimo righe completamente identiche o monocromatiche, significherebbe che il K-Means ha fallito il clustering unificando cluster distinti. La varietà visiva della heatmap prodotta conferma che il vocabolario è diversificato e cattura pattern geometrici differenti.'''

# Importa il modulo os per la gestione sicura dei percorsi e delle cartelle del sistema operativo
import os
# Importa pickle per caricare i file binari serializzati (le cache dei descrittori e dei vocabolari)
import pickle
# Importa l'interfaccia grafica di Matplotlib per generare mappe di calore dei centroidi
import matplotlib.pyplot as plt
# Importa il file di configurazione globale del progetto per leggerne i parametri e i percorsi
import configurazione

def ispeziona_descrittori():
    """
    Esegue un controllo di integrità sulla cache dei descrittori SIFT grezzi estratti dal dataset AID.
    """
    print("\n" + "="*40)
    print("ISPEZIONE CACHE DESCRITTORI")
    print("="*40)
    
    # Recupera dal file di configurazione il percorso in cui è salvato il file .pkl dei descrittori
    percorso_file = configurazione.PERCORSO_SALVATAGGIO_DESCRITTORI
    try:
        # Apre il file in modalità lettura binaria ('rb') usando il context manager 'with' per evitare leak di memoria
        with open(percorso_file, 'rb') as f:
            # Deserializza l'oggetto: ricostruisce in memoria la lista di matrici NumPy dei descrittori SIFT
            descrittori = pickle.load(f)
        
        # La lunghezza della lista corrisponde esattamente al numero di immagini correttamente elaborate
        print(f"- Immagini totali estate da AID: {len(descrittori)}")
        
        # Inizializza un contatore per individuare immagini critiche (con poche caratteristiche informative)
        immagini_poche_feature = 0
        for i in descrittori:
            # i è una matrice NumPy (N x 128). i.shape[0] restituisce N, ovvero il numero di punti chiave trovati in quell'immagine
            if i.shape[0] < 200:
                immagini_poche_feature += 1
        print(f"- Numero di immagini con meno di 200 descrittori: {immagini_poche_feature}")
        
        # Calcola la somma totale di tutti i vettori SIFT presenti nella cache accumulando le righe di ogni matrice
        totale_vettori = sum(len(d) for d in descrittori)
        print(f"- Numero Totale di descrittori SIFT salvati in cache: {totale_vettori}")
        
    except FileNotFoundError:
        # Gestisce il caso in cui lo script venga eseguito prima dell'effettiva estrazione dei dati da parte di main.py
        print(f"File cache '{percorso_file}' non trovato. Esegui prima main.py")

def ispeziona_vocabolari_multipli():
    """
    Carica i vocabolari visivi (centroidi di K-Means) per i diversi K ed effettua un'analisi matriciale e grafica.
    """
    print("\n" + "="*40)
    print("ISPEZIONE DEI VOCABOLARI VISIVI (K-MEANS)")
    print("="*40)
    
    # Cicla dinamicamente sulla lista dei valori di K impostati nel file di configurazione (es. [50, 100, 500])
    for K in configurazione.VALORI_K_DA_TESTARE:
        # Genera il percorso specifico per il file del vocabolario associato al K corrente
        percorso_file = configurazione.ottieni_percorso_vocabolario(K)
        print(f"\nAnalisi Vocabolario per K = {K}...")
        
        try:
            with open(percorso_file, 'rb') as f:
                # Carica la matrice dei centroidi calcolata dall'algoritmo MiniBatchKMeans
                vocabolario = pickle.load(f)
                
            # Mostra le dimensioni geometriche della matrice: deve essere una matrice di dimensioni (K x 128)
            print(f"Forma della matrice dei centroidi: {vocabolario.shape}")
            # Le righe (coordinate Y del grafico) rappresentano il numero di cluster (le parole visive del dizionario)
            print(f"Numero di parole visive (Cluster): {vocabolario.shape[0]}")
            # Le colonne (coordinate X) corrispondono alle 128 dimensioni dello spazio dei descrittori SIFT
            print(f"Dimensione del vettore SIFT: {vocabolario.shape[1]} elementi")
            
            # Rendering grafico del dizionario visivo tramite una Heatmap matriciale
            plt.figure(figsize=(10, 6))
            # imshow mappa l'intensità numerica di ogni cella della matrice in un colore dello spettro 'plasma'
            # aspect='auto' adatta le proporzioni per evitare che una matrice molto stretta e lunga si comprima visivamente
            plt.imshow(vocabolario, aspect='auto', cmap='plasma')
            # Aggiunge una barra laterale graduata per indicare il significato fisico dell'intensità del colore
            plt.colorbar(label='Valore del descrittore (Intensità del gradiente)')
            plt.title(f'Dizionario Visivo: {K} Parole (Centroidi K-Means)', fontsize=12, pad=10)
            plt.xlabel('Dimensioni del vettore SIFT (0 - 127)')
            plt.ylabel('ID Parola Visiva')
            plt.tight_layout()
            
            # Costruisce il nome dell'immagine finale includendo il valore di K ed esporta il grafico ad alta fedeltà
            nome_grafico = f"ispezione_vocabolario_K{K}.png"
            plt.savefig(os.path.join(configurazione.CARTELLA_RISULTATI, nome_grafico), dpi=300)
            plt.show()
            
        except FileNotFoundError:
            print(f"Vocabolario per K={K} non trovato al percorso: {percorso_file}")

if __name__ == "__main__":
    # Quando il file viene eseguito, lancia sequenzialmente i due test diagnostici di controllo dati
    ispeziona_descrittori()
    ispeziona_vocabolari_multipli()