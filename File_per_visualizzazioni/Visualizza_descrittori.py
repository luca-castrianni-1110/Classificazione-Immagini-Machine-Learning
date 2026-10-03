'''COSA FA: Esegue l'algoritmo di rilevamento SIFT su una sequenza di immagini del dataset AID e visualizza sullo schermo i punti chiave estratti usando dei cerchi grafici speciali.
A COSA SERVE: Consente una validazione qualitativa e visiva dell'estrazione delle caratteristiche. Permette all'operatore umano di osservare dove l'algoritmo SIFT decide di posizionare i punti di interesse nelle scene aeree e satellitari.
Per comprendere appieno l'importanza di questa visualizzazione, è utile ricordare come opera il rilevatore SIFT nello spazio di scala:
PERCHÉ LO FACCIAMO: La visualizzazione diretta è l'unico modo per accertarsi che il comportamento del descrittore locale sia coerente con l'applicazione di telerilevamento. Utilizzando il flag DRAW_RICH_KEYPOINTS, non vediamo semplici puntini, ma cerchi di raggio variabile provvisti di una linea radiale.
Il raggio del cerchio esprime la scala (nello spazio di scala della piramide gaussiana) a cui quella feature è risultata stabile. Un cerchio grande indica una struttura macroscopica (es. il tetto di un intero hangar o un incrocio stradale); un cerchio minuscolo rappresenta un micro-dettaglio tessiturale (es. la chioma di un singolo albero).
La linea radiale mostra l'orientamento dominante calcolato sull'istogramma dei gradienti locali. Questo vettore di rotazione viene sottratto durante il calcolo del descrittore finale, rendendo la rappresentazione intrinsecamente invariante per rotazione. Osservare i cerchi concentrici allinearsi lungo i bordi delle piste d'atterraggio o delle coste dimostra visivamente l'efficacia geometrica di SIFT rispetto a variazioni di scala e di orientamento dei voli satellitari.'''


# Importa la libreria OpenCV per l'elaborazione di immagini e la gestione dell'interfaccia utente (GUI)
import cv2
# Importa glob per eseguire ricerche di file basate su pattern testuali (wildcards) nel file system
import glob
# Importa os per assemblare i percorsi delle immagini in modo indipendente dalla piattaforma (Windows/Linux)
import os
# Importa la configurazione globale per conoscere il percorso del dataset e il tetto massimo di feature
import configurazione

def visualizza_sequenza_sift():
    """
    Crea una finestra interattiva OpenCV per mostrare ad hoc i punti chiave SIFT sovrimpressi sulle immagini reali.
    """
    percorso_cartella_immagini = configurazione.PERCORSO_DATASET_AID
    
    # Costruisce il pattern di ricerca: la stringa "**/*.jpg" indica una ricerca ricorsiva in qualsiasi sottocartella di AID
    percorso_ricerca = os.path.join(percorso_cartella_immagini, "**", "*.jpg")
    # glob.glob esamina il disco e restituisce una lista ordinata contenente i percorsi di tutte le immagini individuate
    percorsi = glob.glob(percorso_ricerca, recursive=True)
    
    # Controllo di sicurezza preventivo per scongiurare cicli vuoti ed errori di puntamento
    if not percorsi:
        print(f"ERRORE: Nessuna immagine trovata in: {percorso_cartella_immagini}")
        print("Verifica che il dataset AID sia posizionato correttamente.")
        return

    print(f"Trovate {len(percorsi)} immagini nel dataset AID.")
    print("COMANDI FINESTRA: Premi un tasto qualsiasi per la PROSSIMA foto. Premi 'q' per USCIRE.")
    
    # Istanzia l'oggetto rilevatore SIFT nativo di OpenCV
    # nfeatures fissa il tetto massimo di punti chiave riassuntivi da rilevare, ordinati per stabilità del contrasto
    estrattore_sift = cv2.SIFT.create(nfeatures=configurazione.MAX_DESCRITTORI_PER_IMMAGINE)
    
    # Inizializza una finestra grafica nativa di OpenCV. WINDOW_NORMAL permette all'utente di ridimensionare la finestra con il mouse
    cv2.namedWindow("Visualizzatore SIFT", cv2.WINDOW_NORMAL)

    # Scorre sequenzialmente i percorsi delle immagini trovate
    for percorso in percorsi:
        # Carica l'immagine a colori dal disco (BGR standard di OpenCV)
        immagine_colori = cv2.imread(percorso)
        # Salta l'iterazione se il file è corrotto o non è leggibile come immagine
        if immagine_colori is None:
            continue
            
        # Converte l'immagine da BGR a scala di grigi. SIFT analizza unicamente le variazioni di intensità del singolo canale (luminanza)
        immagine_grigia = cv2.cvtColor(immagine_colori, cv2.COLOR_BGR2GRAY)
        # Rileva esclusivamente le coordinate dei punti chiave geometrici (Keypoints) senza calcolare i vettori descrittori finali
        punti_chiave = estrattore_sift.detect(immagine_grigia, None)
        
        # Disegna sopra l'immagine originale a colori dei cerchietti in corrispondenza dei punti chiave individuati
        # DRAW_RICH_KEYPOINTS: per ogni punto viene disegnato un cerchio proporzionale alla scala (dimensione della feature) 
        # e una linea radiale che indica l'orientamento dominante del gradiente locale.
        immagine_con_punti = cv2.drawKeypoints(
            immagine_colori,
            punti_chiave,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        # Estrae il solo nome del file (es. "airport_1.jpg") dal percorso assoluto per usarlo come metadato visivo
        nome_file = os.path.basename(percorso)
        # Aggiorna dinamicamente la barra del titolo della finestra mostrando il nome del file e il numero esatto di punti estratti
        cv2.setWindowTitle("Visualizzatore SIFT", f"Immagine: {nome_file} - Punti Rilevati: {len(punti_chiave)}")
        # Renderizza l'immagine modificata all'interno della finestra GUI
        cv2.imshow("Visualizzatore SIFT", immagine_con_punti)
        
        # Blocca l'esecuzione del ciclo for a tempo indeterminato (0) in attesa che l'utente prema un tasto sulla tastiera
        # L'operatore bitwise '& 0xFF' pulisce il codice del tasto isolando i caratteri ASCII standard
        tasto_premuto = cv2.waitKey(0) & 0xFF
        # Se l'utente preme il carattere 'q' (quit), interrompe forzatamente il ciclo for chiudendo l'applicazione
        if tasto_premuto == ord('q'):
            break

    # Rilascia le risorse di sistema e distrugge le finestre grafiche create da OpenCV alla chiusura del programma
    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualizza_sequenza_sift()