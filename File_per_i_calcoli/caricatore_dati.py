
'''Questo file si occupa esclusivamente di scansionare il disco rigido alla ricerca delle immagini dei dataset e di organizzare 
i loro percorsi e le rispettive classi di appartenenza.
ottieni_percorsi_immagini_aid
A COSA SERVE: Cerca ricorsivamente tutti i file con estensione .jpg all'interno della cartella del dataset AID.
PERCHÉ SI USA: Il dataset AID contiene migliaia di immagini organizzate in sotto-cartelle. Questo approccio automatizza 
la raccolta senza dover specificare manualmente ogni singola sotto-directory.
ottieni_percorsi_e_etichette
A COSA SERVE: Legge la struttura delle cartelle del dataset UCMerced. Associa a ogni immagine il nome della cartella in cui 
si trova, che corrisponde alla sua etichetta di classe (es. "airplane", "forest").
PERCHÉ SI USA: Nel Machine Learning supervisionato, ogni dato deve essere strettamente accoppiato alla sua etichetta reale. 
Questa funzione mappa l'intero dataset strutturandolo in due liste parallele: una per i file e una per le classi.'''



# Importa os per il controllo dell'esistenza delle cartelle e l'unione dei percorsi.
import os
# Importa glob, una libreria specializzata nella ricerca di file tramite pattern (caratteri jolly come *).
import glob

def ottieni_percorsi_immagini_aid(percorso_dataset):
    # Costruisce il pattern di ricerca. "**" indica di cercare in qualsiasi sotto-cartella a qualsiasi livello,
    # mentre "*.jpg" indica che siamo interessati esclusivamente ai file in formato JPEG.
    percorso_ricerca = os.path.join(percorso_dataset, "**", "*.jpg")
    
    # Esegue la ricerca sul disco. recursive=True permette a glob di scendere in profondità in tutte le sotto-cartelle.
    # Restituisce una lista pulita contenente i percorsi completi di tutte le immagini trovate.
    percorsi_immagini = glob.glob(percorso_ricerca, recursive=True)
    
    # Controllo di sicurezza: se la lista è vuota significa che il percorso fornito è errato o il dataset manca.
    if not percorsi_immagini:
        # Stampa un avviso esplicito nel terminale per aiutare l'utente nel debug.
        print(f"Nessuna immagine trovato in {percorso_dataset}")
        
    # Restituisce la lista dei percorsi trovati.
    return percorsi_immagini

def ottieni_percorsi_e_etichette(percorso_dataset):
    # Inizializza una lista vuota per memorizzare i percorsi di ciascun file immagine.
    percorsi = []
    # Inizializza una lista vuota parallela per memorizzare il nome della classe corrispondente ad ogni file.
    etichette = []
    # Definisce l'estensione specifica dei file da cercare per il dataset UCMerced (.tif).
    estensioni = ('*.tif')
    
    # Verifica l'effettiva esistenza della cartella principale del dataset sul computer.
    if not os.path.exists(percorso_dataset):
        # Se non esiste, avvisa l'utente stampando un errore mirato.
        print(f"La cartella {percorso_dataset} non esiste!")
        # Restituisce le due liste vuote per interrompere l'esecuzione senza far crashare bruscamente l'intero programma.
        return percorsi, etichette

    # Ottiene la lista di tutti i nomi degli elementi contenuti dentro la cartella del dataset.
    # Nel dataset UCMerced, ogni elemento di primo livello è una cartella che dà il nome alla classe (es. "river").
    nomi_classi = os.listdir(percorso_dataset)
    
    # Avvia un ciclo per esaminare singolarmente ogni elemento trovato.
    for nome_classe in nomi_classi:
        # Costruisce il percorso completo verso la presunta cartella della classe corrente.
        percorso_classe = os.path.join(percorso_dataset, nome_classe)
        
        # Verifica che l'elemento analizzato sia effettivamente una cartella (directory) e non un file isolato.
        if os.path.isdir(percorso_classe):
            # Per ogni estensione configurata (in questo caso memorizzata nella stringa 'estensioni').
            for est in estensioni:
                # Costruisce il percorso di ricerca specifico per i file della cartella corrente (es. "./Images/river/*.tif").
                percorso_ricerca = os.path.join(percorso_classe, est)
                # Utilizza glob per raccogliere tutti i file che corrispondono alla sintassi stabilita.
                immagini_trovate = glob.glob(percorso_ricerca)
                
                # Itera su ciascun percorso immagine individuale trovato all'interno della cartella della classe.
                for img_path in immagini_trovate:
                    # Aggiunge il percorso specifico del file alla lista generale dei percorsi.
                    percorsi.append(img_path)
                    # Aggiunge il nome della cartella corrente alla lista delle etichette.
                    # Questo crea una perfetta corrispondenza ad indici condivisi tra le due liste.
                    etichette.append(nome_classe)
                    
    # Stampa un report finale a terminale indicando quante immagini sono state caricate e quante classi distinte esistono.
    # set(etichette) rimuove i duplicati dalla lista, permettendo a len() di contare solo le classi uniche.
    print(f"Trovate {len(percorsi)} immagini suddivise in {len(set(etichette))} classi.")
    # Restituisce le due liste parallele popolate e pronte per la fase di estrazione.
    return percorsi, etichette