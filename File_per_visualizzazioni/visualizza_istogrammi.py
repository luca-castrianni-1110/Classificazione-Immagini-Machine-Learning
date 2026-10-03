'''COSA FA: Prende la matrice globale delle caratteristiche BoW X, raggruppa le righe in base alla classe di appartenenza (usando il vettore delle etichette y) e calcola l'istogramma medio per ognuna delle 21 classi del dataset UCMerced. Compila infine una matrice finale 21×K rappresentata graficamente come una mappa di calore magma.
A COSA SERVE: Serve a dimostrare visivamente il potere discriminativo del vocabolario visivo e la separabilità delle classi. È l'esame visivo che giustifica la scelta del parametro K ottimale.
PERCHÉ LO FACCIAMO: Nella teoria del Bag of Visual Words, ogni riga della heatmap rappresenta la "firma visiva" media di un'intera categoria di territorio (es. aeroporto, foresta, quartiere residenziale).
Se il vocabolario visivo è efficace, classi diverse devono attivare parole visive differenti. Ad esempio, la riga della classe forest dovrà mostrare picchi luminosi (gialli) in corrispondenza degli ID di parole visive che descrivono texture frastagliate e dense (la vegetazione); la riga airplane mostrerà picchi su parole visive che modellano linee rette e angoli metallici delle ali.
Questo script permette di confrontare visivamente l'effetto dell'iperparametro K. Quando K=50, notiamo che molte classi presentano firme quasi identiche (colonne illuminate verticalmente su più classi contemporaneamente): questo si chiama underfitting del vocabolario, in cui le parole visive sono troppo generiche e non riescono a specializzarsi. Quando passiamo a K=500, la heatmap diventa estremamente frastagliata e "orizzontale": ogni classe possiede zone di attivazione uniche e ben distinte dalle altre categorie. Questa marcata diversificazione visiva della mappa per K=500 costituisce la prova matematica e visiva del perché i classificatori (SVM e Random Forest) ottengano accuratezze decisamente superiori con dizionari ampi.'''



# Importa il modulo os per strutturare i percorsi di salvataggio dei grafici finali
import os
# Importa pickle per caricare i file binari contenenti le matrici Bag of Visual Words (gli istogrammi generati)
import pickle
# Importa Numpy per calcolare operazioni statistiche vettorializzate (come la media lungo gli assi della matrice)
import numpy as np
# Importa l'ambiente grafico di Matplotlib per generare e personalizzare le heatmap delle firme visive
import matplotlib.pyplot as plt
# Importa la configurazione di progetto per recuperare la lista dei K da testare e le directory di output
import configurazione

def visualizza_firme_classi_multiple():
    """
    Aggrega gli istogrammi BoW calcolati per ogni classe del dataset UCMerced, creandone una firma visiva media.
    """
    print("\n" + "="*40)
    print("GENERAZIONE HEATMAP DELLE FIRME VISIVE (BoVW)")
    print("="*40)

    # Cicla in modo iterativo su ciascun valore di K configurato ([50, 100, 500]) per generare tre grafici comparativi distinti
    for K in configurazione.VALORI_K_DA_TESTARE:
        # Recupera il percorso del file .pkl contenente la matrice globale degli istogrammi BoW associata a questo specifico K
        percorso_file = configurazione.ottieni_percorso_dati_addestramento(K)
        print(f"\nGenerazione mappa delle frequenze per K = {K}...")
        
        try:
            with open(percorso_file, 'rb') as f:
                dati = pickle.load(f)
            
            # X è la matrice delle caratteristiche normalizzata ad alta dimensionalità: dimensioni (Numero_Immagini x K)
            X = dati['X']
            # y è il vettore unidimensionale delle etichette testuali (stringhe) corrispondenti a ogni riga di X
            y = dati['y']
            # Estrae l'elenco ordinato e privo di duplicati delle categorie presenti (le 21 classi di UCMerced)
            classi_uniche = np.unique(y)
            
            # Lista ausiliaria per collezionare i vettori medi calcolati categoria per categoria
            firme_medie = []
            for classe in classi_uniche:
                # X[y == classe] effettua un filtraggio booleano (masking) estraendo solo le righe appartenenti alla classe corrente
                # np.mean(..., axis=0) schiaccia la matrice filtrata calcolando la media aritmetica colonna per colonna.
                # Restituisce un singolo vettore lungo K che esprime la frequenza media di attivazione delle parole visive per quella classe.
                media_classe = np.mean(X[y == classe], axis=0)
                firme_medie.append(media_classe)
            
            # Converte la lista di vettori in una matrice compatta NumPy di dimensioni (21 x K) (ovvero Numero_Classi x K)
            firme_medie = np.array(firme_medie)

            # Inizializza la tela del grafico impostando dimensioni generose (16x10 pollici) per evitare compressioni del testo
            plt.figure(figsize=(16, 10))
            # Rappresenta la matrice 21xK sotto forma di immagine continua. Cmap='magma' adotta una scala cromatica 
            # che va dal nero (frequenza zero) al viola/arancione fino al giallo acceso (frequenza di attivazione massima)
            plt.imshow(firme_medie, aspect='auto', cmap='magma')
            
            # Configura i marker (ticks) dell'asse verticale Y associando a ciascuna delle 21 righe il rispettivo nome testuale della classe
            plt.yticks(range(len(classi_uniche)), classi_uniche, fontsize=10, fontweight='bold')
            
            # Calcola un passo di campionamento (passo_tick) intelligente per l'asse orizzontale X.
            # Se K=500, scrivere tutti i numeri da 0 a 499 creerebbe una macchia nera illeggibile; mettiamo una tacca ogni 50 parole.
            passo_tick = 5 if K == 50 else (10 if K == 100 else 50)
            plt.xticks(np.arange(0, X.shape[1], passo_tick)) 
            
            # Aggiunge la leggenda laterale del colore per quantificare numericamente le frequenze medie regolate dalla normalizzazione L2
            plt.colorbar(label='Frequenza media di attivazione della Parola Visiva')
            plt.title(f'Firme Visive Medie delle 21 Classi (Bag-of-Visual-Words, K={K})', fontsize=14, pad=15)
            plt.xlabel(f'ID Parola Visiva (0 - {X.shape[1] - 1})', fontsize=12)
            plt.ylabel('Classi del Dataset (UCMerced)', fontsize=12)
            
            plt.tight_layout()
            
            # Compone il nome del file immagine di output, lo unisce alla cartella dei risultati e salva il file a 300 DPI
            nome_file_output = f"firme_visive_heatmap_K{K}.png"
            percorso_salvataggio = os.path.join(configurazione.CARTELLA_RISULTATI, nome_file_output)
            plt.savefig(percorso_salvataggio, dpi=300)
            print(f"Mappa delle firme salvata con successo in: {percorso_salvataggio}")
            
            # Renderizza a schermo la figura prima di passare alla computazione del valore di K successivo
            plt.show()
            
        except FileNotFoundError:
            # Se l'utente salta la generazione degli istogrammi per un determinato K, evita il crash saltando l'iterazione
            print(f"File istogrammi '{percorso_file}' non trovato. Salto K={K}.")

if __name__ == "__main__":
    visualizza_firme_classi_multiple()