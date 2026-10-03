'''Cosa fa: Prende i dati ottimali (K=500), addestra una SVM con Kernel RBF e parametro probability=True, e salva in un unico file binario .pkl il "blocco" completo (Modello + Vocabolario + LabelEncoder).
A cosa serve: Serve a creare il modello di produzione pronto al deployment. In ambiente accademico e industriale si separa nettamente la fase di addestramento (lenta e pesante) dalla fase di inferenza (istantanea).
Perché lo facciamo: 1. Durante la validazione usiamo la Cross-Validation per non barare; ma per il modello finale usiamo il 100% dei dati. Più dati ha a disposizione la SVM, più robusti e precisi saranno i suoi iperpiani decisionali nel mondo reale.
2. Impostiamo probability=True perché una SVM standard restituisce solo la classe predetta in modo rigido. Abilitando il Platt Scaling (calcolo probabilistico tramite regressione logistica sulle distanze dall'iperpiano), il nostro sistema diventa in
grado di dirci quanto è sicuro di una determinata scelta, parametro fondamentale per l'ispezione visiva.'''


# ==============================================================================
# SCRIPT: SALVA MODELLO FINALE (`salva_modello_finale.py`)
# TRAGUARDO: Congelamento (Serialization) del sistema per l'ambiente di produzione
# ==============================================================================

# Importa pickle per la serializzazione binaria di oggetti complessi di Python
import pickle
# Importa il classificatore Support Vector Machine per l'addestramento definitivo
from sklearn.svm import SVC
# Importa lo strumento per convertire le etichette stringa in indici numerici discreti
from sklearn.preprocessing import LabelEncoder
# Importa os per l'elaborazione sicura dei percorsi dei file sul disco fisso
import os
# Importa la configurazione globale per leggere i parametri centralizzati (es. K_VINCENTE)
import configurazione

# [COSA FA]: Determina la posizione fisica assoluta sul computer di questo specifico script
cartella_attuale = os.path.dirname(os.path.abspath(__file__))
# [COSA FA]: Definisce il percorso finale in cui verrà salvato il file bundle .pkl completo
percorso_completo = os.path.join(cartella_attuale, "modello_finale_inferenza.pkl")

def esporta_modello_per_inferenza():
    """
    [A COSA SERVE]: Prende l'intera matrice dei dati, addestra la SVM finale sull'intero dataset
    senza Cross-Validation (sfruttando il 100% dell'informazione) e impacchetta tutto per l'inferenza.
    """
    print("\n" + "="*60)
    print("FASE DI PRODUZIONE: ADDESTRAMENTO FINALE E SERIALIZZAZIONE")
    print("="*60)

    # [PERCHÉ LO FACCIAMO]: Leggiamo K_VINCENTE (K=500) perché la fase di validazione ha dimostrato
    # che un vocabolario più ampio cattura meglio la variabilità geometrica del territorio.
    K = configurazione.K_VINCENTE
    percorso_dati = configurazione.ottieni_percorso_dati_addestramento(K)
    percorso_vocab = configurazione.ottieni_percorso_vocabolario(K)
    
    # [COSA FA]: Carica in memoria gli istogrammi BoW (matrice X) e le classi (vettore y)
    with open(percorso_dati, 'rb') as f:
        dati = pickle.load(f)
        X, y_testo = dati['X'], dati['y']
    
    # [COSA FA]: Carica il vocabolario visivo (i centroidi K-Means) associato al K vincente
    with open(percorso_vocab, 'rb') as f:
        vocabolario = pickle.load(f)

    # [COSA FA]: Converte le scritte testuali delle classi in indici numerici (es. "airport" -> 0)
    encoder = LabelEncoder()
    y_num = encoder.fit_transform(y_testo)
    
    # [CONFIGURAZIONE AVANZATA PER L'ORALE - PERCHÉ LO FACCIAMO]:
    # 1. kernel='rbf': Proietta gli istogrammi BoW in uno spazio a infinite dimensioni per tracciare confini curvi.
    # 2. C=100.0: Forte penalizzazione degli errori per massimizzare il margine di separazione tra le classi.
    # 3. probability=True: ABILITA IL CALCOLO DELLE PROBABILITÀ (Platt Scaling). 
    #    Di norma la SVM calcola solo le distanze dall'iperpiano decisionale; attivando questo flag, 
    #    la SVM esegue internamente una cross-validation a 5 fold per calcolare un modello logistico
    #    in grado di restituirci la confidenza in percentuale (es. "Prato al 94.2%"). È indispensabile per l'inferenza!
    modello_finale = SVC(kernel='rbf', C=100.0, probability=True, random_state=42)
    
    print(f"- Addestramento definitivo della SVM su tutte le immagini con K={K}...")
    modello_finale.fit(X, y_num)
    print("- Addestramento completato con successo.")

    # [ARCHITETTURA APPLICATIVA - A COSA SERVE]:
    # Creiamo un UNICO dizionario (monolitico) che racchiude tutte le componenti vitali del sistema.
    # In produzione non possiamo portarci dietro solo il modello; ci servono anche il vocabolario K-Means
    # per quantizzare le nuove immagini e il LabelEncoder per tradurre i numeri in testo.
    pacchetto_inferenza = {
        'modello': modello_finale,     # La mente matematica (iperpiani decisionali)
        'vocabolario': vocabolario,   # Il dizionario visivo (centroidi) per mappare i SIFT
        'encoder': encoder,           # Il traduttore Numeri -> Testo delle categorie
        'K': K                        # La dimensione fissa dell'istogramma (500)
    }
    
    # [COSA FA]: Salva fisicamente sul disco rigido il pacchetto binario
    with open(percorso_completo, 'wb') as f:
        pickle.dump(pacchetto_inferenza, f)
    print(f"\n[SUCCESSO]: Pacchetto di inferenza salvato in: {percorso_completo}")

if __name__ == "__main__":
    esporta_modello_per_inferenza()
    
    