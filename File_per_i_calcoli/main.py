# Importa il modulo os per interfacciarsi con il file-system e verificare i percorsi.
import os
# Importa la barra di caricamento visiva tqdm per monitorare i cicli di estrazione.
from tqdm import tqdm
# Importa l'intero file di configurazione centrale per accedere a costanti, soglie e funzioni helper.
import configurazione
# Carica le funzioni deputate alla mappatura e scansione ricorsiva delle immagini sui dischi.
from caricatore_dati import ottieni_percorsi_immagini_aid, ottieni_percorsi_e_etichette
# Carica le funzioni di inizializzazione dell'algoritmo SIFT e di estrazione delle proprietà fisiche locali.
from estrazione_caratteristiche import inizializza_estrattore, estrai_descrittori_locali
# Carica i moduli pickle per il salvataggio dei file binari e la routine di esecuzione del MiniBatchKMeans.
from vocabolario import salva_descrittori, carica_descrittori, esegui_clustering_e_salva, carica_vocabolario
# Carica le funzioni necessarie a generare l'istogramma quantizzato combinato e l'esportazione finale del dataset.
from estrazione_istogrammi import genera_dataset_addestramento, salva_dati_addestramento

def principale():
    # Verifica se la cartella deputata a ospitare i risultati intermedi e finali esiste fisicamente nel PC.
    if not os.path.exists(configurazione.CARTELLA_RISULTATI):
        # Se non esiste, la crea in modo automatico chiamando le API del sistema operativo.
        os.makedirs(configurazione.CARTELLA_RISULTATI)

    # --- SOTTO-SISTEMA 1: ESTRAZIONE E CACHING DEI DESCRITTORI LOCALI ---
    # Inizializza una lista vuota destinata a contenere la totalità dei descrittori estratti dal dataset AID.
    tutti_i_descrittori = []
    # Controlla se sul PC esiste già il file binario cache contenente l'estrazione completa dei punti di interesse di AID.
    if os.path.exists(configurazione.PERCORSO_SALVATAGGIO_DESCRITTORI):
        # Messaggio a terminale per notificare che il sistema salterà il ricalcolo leggendo i dati pronti dalla cache.
        print(f"\nCaricamento descrittori SIFT dalla cache")
        # Popola la lista chiamando la routine di deserializzazione del modulo vocabolario.
        tutti_i_descrittori = carica_descrittori(configurazione.PERCORSO_SALVATAGGIO_DESCRITTORI)
    else:
        # Se la cache non esiste, è necessario procedere all'elaborazione ex-novo.
        print("\nEstrazione dei descrittori SIFT dal dataset AID")
        # Chiama la funzione di caricatore_dati per raccogliere tutti i percorsi assoluti delle immagini .jpg in AID.
        percorsi_immagini_aid = ottieni_percorsi_immagini_aid(configurazione.PERCORSO_DATASET_AID)
        # Controllo di sicurezza: se il dataset non è installato o il percorso è errato, interrompe il programma.
        if not percorsi_immagini_aid:
            print("Dataset AID non trovato")
            return
            
        # Attiva l'estrattore SIFT chiamando la configurazione globale tramite l'apposito wrapper di estrazione.
        estrattore = inizializza_estrattore(configurazione.ESTRATTORE_CARATTERISTICHE)
        # Avvia un ciclo iterativo per analizzare una ad una tutte le immagini collezionate in AID.
        # desc="Estrazione SIFT" personalizza la stringa di testo mostrata accanto alla barra di avanzamento dinamica.
        for percorso in tqdm(percorsi_immagini_aid, desc="Estrazione SIFT"):
            # Estrae i punti chiave SIFT locali forzando il campionamento massimo di sicurezza impostato a 2000.
            descrittori = estrai_descrittori_locali(percorso, estrattore, configurazione.MAX_DESCRITTORI_PER_IMMAGINE)
            # Se l'immagine ha prodotto dei descrittori validi ed è stata caricata correttamente.
            if descrittori is not None:
                # Appende la matrice dei descrittori dell'immagine singola all'elenco cumulativo generale.
                tutti_i_descrittori.append(descrittori)
        
        # Concluso il ciclo su tutte le immagini, scrive la lista cumulativa sul disco rigido in formato binario.
        # In questo modo, ai lanci successivi del programma l'intero blocco 'else' verrà ignorato, risparmiando tempo.
        salva_descrittori(tutti_i_descrittori, configurazione.PERCORSO_SALVATAGGIO_DESCRITTORI)

    # --- SOTTO-SISTEMA 2: LOOP GENERATIVO SUI VALORI DI K CONFIGURATI ---
    # Scansiona la cartella del dataset di classificazione UCMerced raccogliendo le liste parallele di percorsi ed etichette (.tif).
    percorsi_classi, etichette_classi = ottieni_percorsi_e_etichette(configurazione.PERCORSO_DATASET_CLASSI)
    # Controllo di coerenza: se non trova file blocca il processo per prevenire malfunzionamenti nei cicli successivi.
    if not percorsi_classi:
        print("Dataset UCMerced non trovato")
        return
        
    # Inizializza una seconda istanza dell'estrattore SIFT, dedicata alla successiva elaborazione del dataset UCMerced.
    estrattore_classi = inizializza_estrattore(configurazione.ESTRATTORE_CARATTERISTICHE)

    # Avvia il ciclo più esterno che itera su ciascun valore di K memorizzato nella lista globale di configurazione [50, 100, 500].
    # Questo approccio automatizza la creazione sequenziale dei vari scenari di test in un'unica esecuzione.
    for K in configurazione.VALORI_K_DA_TESTARE:
        # Stampa una linea estetica divisoria nel terminale per separare visivamente i log delle varie impostazioni di K.
        print(f"\n{'-'*50}\nElaborazione per K = {K}\n{'-'*50}")
        
        # Genera in modo automatico i percorsi di output specifici per il valore corrente di K usando le helper function.
        percorso_vocab_K = configurazione.ottieni_percorso_vocabolario(K)
        percorso_dati_K = configurazione.ottieni_percorso_dati_addestramento(K)

        # FASE 2.1: COSTRUZIONE O CARICAMENTO DEL VOCABOLARIO VISIVO (K-MEANS)
        # Verifica se per il valore di K corrente esiste già un vocabolario calcolato e memorizzato precedentemente su PC.
        if os.path.exists(percorso_vocab_K):
            print(f"Vocabolario K={K} già esistente.")
            # Carica direttamente la matrice dei centroidi saltando la fase pesante di clustering.
            vocabolario = carica_vocabolario(percorso_vocab_K)
        else:
            # Se il file manca, avvia l'elaborazione del clustering MiniBatchKMeans.
            print(f"Calcolo K-Means per K={K}")
            # Chiama la funzione passandole tutti i descrittori cumulativi raccolti in precedenza e il K desiderato.
            # Ritorna e memorizza la matrice dei centroidi finale generata dall'algoritmo.
            vocabolario = esegui_clustering_e_salva(tutti_i_descrittori, K, percorso_vocab_K)

        # FASE 2.2: QUANTIZZAZIONE ED ESTRAZIONE DEGLI ISTOGRAMMI BOW FINALI
        # Verifica se la matrice delle caratteristiche finale normalizzata per questo specifico valore di K è già presente sul disco.
        if os.path.exists(percorso_dati_K):
            # Notifica l'utente evitando inutili sovrascritture di file già completi e validi.
            print(f"Dati di addestramento K={K} già esistenti.")
        else:
            # Se la matrice non esiste, avvia la trasformazione delle immagini di UCMerced nel formato vettoriale BoW.
            print(f"Generazione istogrammi BoW per K={K}")
            # Esegue la quantizzazione e la normalizzazione L2 su tutte le immagini del dataset UCMerced.
            # X_dati conterrà la matrice finale normalizzata, y_etichette conterrà il vettore delle classi corrispondenti.
            X_dati, y_etichette = genera_dataset_addestramento(percorsi_classi, etichette_classi, estrattore_classi, vocabolario)
            # Salva in formato pickle il dizionario organizzato, pronto per essere importato dai modelli predittivi.
            salva_dati_addestramento(X_dati, y_etichette, percorso_dati_K)

# Costrutto standard Python: assicura che la funzione principale() venga eseguita solo se lo script
# viene lanciato direttamente dal terminale, impedendo esecuzioni involontarie in caso di importazione come modulo.
if __name__ == "__main__":
    principale()