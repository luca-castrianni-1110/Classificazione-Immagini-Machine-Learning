'''Questo script rappresenta la fase finale e analitica dell'intera pipeline: prende gli istogrammi BoW calcolati nei passaggi precedenti, li usa per addestrare due modelli di Machine Learning (una SVM e un Random Forest) e ne valuta rigorosamente le performance generando i grafici delle matrici di confusione.'''

# ==============================================================================
# SCRIPT DI VALUTAZIONE E GENERAZIONE DELLE MATRICI DI CONFUSIONE
# ==============================================================================

# Importa il modulo per interagire con il sistema operativo (creazione e gestione percorsi file)
import os
# Importa la libreria per caricare/scaricare oggetti Python serializzati in file binari (.pkl)
import pickle
# Importa Numpy per la manipolazione di vettori numerici e calcoli statistici (es. la media)
import numpy as np
# Importa l'interfaccia grafica di Matplotlib per la creazione di figure e sotto-grafici
import matplotlib.pyplot as plt
# Importa Seaborn, libreria di visualizzazione statistica usata qui per disegnare mappe di calore (heatmap)
import seaborn as sns
# Importa gli strumenti di Scikit-Learn per la validazione incrociata robusta e stratificata
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
# Importa il codificatore per convertire le etichette testuali (stringhe) in indici numerici discreti
from sklearn.preprocessing import LabelEncoder
# Importa il classificatore Support Vector Machine (SVC sta per Support Vector Classification)
from sklearn.svm import SVC
# Importa il classificatore Ensemble Random Forest (foresta di alberi decisionali indipendenti)
from sklearn.ensemble import RandomForestClassifier
# Importa la funzione matematica per calcolare i valori numerici della matrice di confusione
from sklearn.metrics import confusion_matrix
# Importa il modulo di configurazione locale per accedere ai parametri globali e ai percorsi dei dati
import configurazione

def carica_dati(percorso_file):
    """
    Carica la cache dei dati estratti (matrice BoW ed etichette) dal file binario specificato.
    """
    # Apre il file in modalità lettura binaria ('rb' = read binary) garantendo la chiusura automatica tramite 'with'
    with open(percorso_file, 'rb') as f:
        # Deserializza il file binario ricostruendo il dizionario originale salvato in precedenza
        dati = pickle.load(f)
    # Estrae e restituisce separatamente la matrice delle caratteristiche X e il vettore delle classi y
    return dati['X'], dati['y']

def valuta_e_mostra_modello(modello, nome_modello, X, y_numerico, nomi_classi, cv_stratificata, asse_grafico):
    """
    Esegue la cross-validation del modello, ne calcola l'accuratezza e ne disegna la matrice di confusione.
    """
    # Stampa a terminale lo stato di avanzamento per monitorare quale modello si sta calcolando
    print(f"Addestramento e valutazione: {nome_modello}")
    
    # Esegue la Cross-Validation per calcolare i punteggi di accuratezza sui diversi fold (blocchi)
    # modello: il classificatore da testare (SVM o Random Forest)
    # X, y_numerico: le caratteristiche BoW normalizzate e i target numerici corrispondenti
    # cv: l'oggetto che definisce lo schema di suddivisione dei blocchi di dati
    # scoring='accuracy': metrica di valutazione basata sulla percentuale di risposte corrette
    # n_jobs=-1: sfrutta tutti i core hardware disponibili nella CPU in parallelo per azzerare i tempi di attesa
    punteggi = cross_val_score(modello, X, y_numerico, cv=cv_stratificata, scoring='accuracy', n_jobs=-1)
    
    # Calcola la media aritmetica dei punteggi ottenuti nei singoli fold e la moltiplica per 100 per averla in percentuale
    acc_media = np.mean(punteggi) * 100
    
    # Genera le previsioni "fuori campione" (out-of-fold) per ogni singola immagine del dataset.
    # Ogni immagine viene predetta quando si trova nel blocco di test, garantendo una valutazione non falsata.
    y_previsto = cross_val_predict(modello, X, y_numerico, cv=cv_stratificata, n_jobs=-1)
    
    # Genera la matrice di confusione numerica (struttura quadrata Numero_Classi x Numero_Classi)
    # Confronta la realtà (y_numerico) con la risposta fornita dal modello (y_previsto)
    matrice_conf = confusion_matrix(y_numerico, y_previsto)
    
    # Disegna la mappa di calore (Heatmap) per rappresentare visivamente la matrice sul sotto-grafico corrente
    # annot=True: scrive esplicitamente il numero di campioni dentro ogni cella quadra
    # fmt='d': forza la formattazione dei numeri come interi decimali (evita la notazione scientifica)
    # cmap='Blues': adotta una sfumatura di colori basata sul blu (più scuro = più campioni stabili)
    # xticklabels/yticklabels: assegna i nomi reali delle categorie agli assi cartesiani del grafico
    # ax=asse_grafico: indica in quale specifico pannello della figura complessiva disegnare la mappa
    # annot_kws: imposta la dimensione del font del testo interno alle celle per renderlo leggibile
    # cbar_kws: riduce leggermente la barra laterale del colore per armonizzarla con le proporzioni del grafico
    sns.heatmap(matrice_conf, annot=True, fmt='d', cmap='Blues',
                xticklabels=nomi_classi, yticklabels=nomi_classi,
                ax=asse_grafico, annot_kws={"size": 9}, cbar_kws={'shrink': 0.8})
    
    # Configura il titolo del sotto-grafico stampando il nome del modello e l'accuratezza media con 2 cifre decimali
    asse_grafico.set_title(f'{nome_modello}\nAccuratezza Media: {acc_media:.2f}%', fontsize=16, pad=15)
    # Definisce l'etichetta dell'asse verticale (Y) che rappresenta la classe reale di appartenenza dell'immagine
    asse_grafico.set_ylabel('Classe Vera (Realtà)', fontsize=12, fontweight='bold')
    # Definisce l'etichetta dell'asse orizzontale (X) che rappresenta la classe ipotizzata/predetta dal modello
    asse_grafico.set_xlabel('Classe Predetta (Risposta del Modello)', fontsize=12, fontweight='bold')
    
    # Ruota le scritte dell'asse X di 45 gradi verso destra per evitare che i nomi lunghi si sovrappongano
    asse_grafico.set_xticklabels(asse_grafico.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    # Mantiene le scritte dell'asse Y perfettamente orizzontali (rotazione 0) per una lettura confortevole
    asse_grafico.set_yticklabels(asse_grafico.get_yticklabels(), rotation=0, fontsize=10)

def principale():
    """
    Funzione orchestratrice: configura i modelli, cicla sui valori di K e salva i grafici finali.
    """
    # Configura il protocollo di Cross-Validation Stratificata a 3 blocchi (K-Fold)
    # n_splits=3: divide i dati in 3 parti (due per addestrare, una per testare, a rotazione)
    # shuffle=True: mescola l'ordine delle immagini prima di dividerle per eliminare trend sistematici nel dataset
    # random_state=42: blocca il seme di generazione casuale per rendere l'esperimento riproducibile al 100%
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Avvia un ciclo iterativo per esaminare uno alla volta i valori di K configurati (es. [50, 100, 500])
    for K in configurazione.VALORI_K_DA_TESTARE:
        # Stampa linee divisorie estetiche a terminale per isolare visivamente i log di ciascun blocco K
        print(f"\n{'='*60}")
        print(f"Generazione matrice confusione per K = {K}")
        print(f"{'='*60}")
        
        # Genera in modo automatico il percorso del file contenente gli istogrammi BoW associati al K corrente
        percorso_dati = configurazione.ottieni_percorso_dati_addestramento(K)
        
        # Blocco di gestione delle eccezioni per prevenire crash improvvisi del software
        try:
            # Tenta di leggere e importare in memoria la matrice X e il vettore y dal file binario
            X, y_testo = carica_dati(percorso_dati)
        except FileNotFoundError:
            # Se il file non esiste (perché l'utente non ha lanciato prima main.py), cattura l'errore
            print(f"File '{percorso_dati}' non trovato. Salto K={K}.")
            print("Assicurati di aver eseguito prima 'main.py' per generare i dati BoW.")
            # Salta il resto del ciclo corrente e passa direttamente al valore di K successivo
            continue

        # Istanzia l'oggetto LabelEncoder per mappare le stringhe di testo in numeri interi indicizzati
        encoder = LabelEncoder()
        # Converte l'array delle etichette stringa (y_testo) in array di interi (y_numerico) [es. "forest" -> 3]
        y_numerico = encoder.fit_transform(y_testo)
        # Estrae la lista ordinata dei nomi testuali delle classi per usarla come etichetta nei grafici
        nomi_classi = encoder.classes_

        # Istanzia il modello Support Vector Machine (SVC) impostando i parametri ottimali di classificazione
        # kernel='rbf': adotta la funzione a base radiale per proiettare i dati in uno spazio a infinite dimensioni.
        # Questo permette alla SVM di tracciare confini decisionali curvi e non-lineari, riuscendo a
        # separare classi visivamente simili che una SVM lineare avrebbe confuso.
        # C=100.0: parametro di regolarizzazione alto che penalizza severamente gli errori di addestramento.
        # random_state=42: garantisce la stabilità e l'identità del comportamento del modello a ogni esecuzione.
        svm = SVC(kernel='rbf', C=100.0, random_state=42)
        
        # Istanzia il modello Random Forest con i parametri emersi dalla fase di tuning
        # n_estimators=500: costruisce una foresta robusta composta da ben 500 alberi decisionali indipendenti
        # max_depth=50: limita l'altezza massima di ciascun albero a 50 nodi per impedire un overfitting incontrollato
        # random_state=42: assicura che la scelta casuale di righe e colonne per gli alberi sia identica ad ogni avvio
        rf = RandomForestClassifier(n_estimators=500, max_depth=50, random_state=42)

        # Inizializza una nuova finestra grafica (Figure) ripartita in 1 riga e 2 colonne di sotto-grafici (assi)
        # figsize=(24, 11): imposta le dimensioni della tela in pollici (molto ampia per ospitare dettagli ad alta fedeltà)
        # constrained_layout=True: ottimizza automaticamente gli spazi interni eliminando le sovrapposizioni tra i testi
        figura, assi = plt.subplots(nrows=1, ncols=2, figsize=(24, 11), constrained_layout=True)
        
        # Chiama la funzione di calcolo e rendering per la SVM, associandola al primo pannello grafico (assi[0])
        valuta_e_mostra_modello(svm, f"SVM (Kernel RBF) [K={K}]", X, y_numerico, nomi_classi, cv, assi[0])
        # Chiama la funzione per il Random Forest, associandolo al secondo pannello grafico affiancato (assi[1])
        valuta_e_mostra_modello(rf, f"Random Forest [K={K}]", X, y_numerico, nomi_classi, cv, assi[1])
        
        # Costruisce la stringa del nome del file grafico incorporando dinamicamente il valore di K correntemente analizzato
        nome_file_grafico = f"matrici_confusione_K{K}.png"
        # Combina la cartella dei risultati globali con il nome del file per ottenere il percorso di destinazione assoluto
        percorso_salvataggio_grafico = os.path.join(configurazione.CARTELLA_RISULTATI, nome_file_grafico)
        # Salva fisicamente la figura sul disco ad alta risoluzione (dpi=300 garantisce nitidezza editoriale per la tesi)
        plt.savefig(percorso_salvataggio_grafico, dpi=300)
        # Conferma a terminale l'avvenuto salvataggio dell'immagine PNG sul computer
        print(f"Grafico salvato in: {percorso_salvataggio_grafico}")
        
        # Sblocca la memoria ed esibisce la finestra grafica interattiva direttamente sullo schermo dell'utente
        plt.show()

# Verifica se il file viene avviato come attore principale dal terminale ed esegue la funzione orchestratrice
if __name__ == "__main__":
    principale()