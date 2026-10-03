import sys
import os
import pickle
import configurazione
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

# Gestione corretta dei percorsi
percorso_1 = os.path.abspath("./File_per_i_calcoli") 
percorso_2 = os.path.abspath("./File_per_visualizzazioni")

percorsi_da_controllare = [percorso_1, percorso_2]

for percorso in percorsi_da_controllare:
    if not os.path.exists(percorso):
        print(f"Cartella non trovata:\n{percorso}")
        sys.exit(1)
    else:
        sys.path.append(percorso)
        print(f"Trovato e aggiunto al path: {percorso}")

def principale():
    # Definisce la cartella dei risultati
    cartella_output = configurazione.CARTELLA_RISULTATI
    if not os.path.exists(cartella_output):
        os.makedirs(cartella_output)
        print(f"Creata cartella risultati in: {cartella_output}")

    percorso_report = os.path.join(cartella_output, "report_esperimento.txt")
    
    # Liste per memorizzare i risultati del grafico finale
    risultati_svm, risultati_rf = [], []

    # Inizializza il file del report (sovrascrive se già esistente)
    with open(percorso_report, "w", encoding="utf-8") as file_log:
        file_log.write(" Report esperimento: CLASSIFICAZIONE CON SIFT E BoVW\n")
        file_log.write("="*50 + "\n")

    # Inizializzazione della Cross-Validation a 3 Fold (Richiesta dall'assignment)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Cicla sulla lista di K definita centralmente in configurazione.py ([50, 100, 500])
    for K in configurazione.VALORI_K_DA_TESTARE:
        file_dati_K = configurazione.ottieni_percorso_dati_addestramento(K)

        print(f"\nCaricamento dati di addestramento per K = {K}")
        
        # Tenta di caricare gli istogrammi pre-calcolati dal main.py
        try:
            with open(file_dati_K, 'rb') as f:
                dati = pickle.load(f)
                X = dati['X']
                y_testo = dati['y']
        except FileNotFoundError:
            print(f"Il file '{file_dati_K}' non esiste")
            continue  # Salta questo K e passa al successivo

        # Codifica le etichette testuali delle 21 classi in numeri
        encoder = LabelEncoder()
        y_numerico = encoder.fit_transform(y_testo)
        nomi_classi = encoder.classes_

        #ADDESTRAMENTO E CROSS-VALIDATION SVM
        print(f"Esecuzione Cross-Validation su SVM (Kernel RBF, C=100) per K={K}")
        '''RBF Questo permette alla SVM di tracciare confini decisionali curvi e non-lineari, riuscendo a
        separare classi visivamente simili che una SVM lineare avrebbe confuso e C mi serve per determina quanto l'algoritmo
        deve essere "severo" nel punire gli errori di classificazione durante l'addestramento.
        con 100 ho visto ottimi miglioramenti ma oltre non sono andato se no si rischia overfitting
        ma metterlo troppo basso ero in underfitting e allora siccome ho usato 3-fold posso permetteri di usare 100'''
        svm = SVC(kernel='rbf', C=100.0, random_state=42)
        y_pred_svm = cross_val_predict(svm, X, y_numerico, cv=cv, n_jobs=-1) #facendo questa funzione non ho problemi sul instanziare sempre nuovi classificatori
                                                                            #fa tutto lei perche sotto ha una funzione di clone()
        f1_svm = f1_score(y_numerico, y_pred_svm, average='macro') #macro perchè fa la media tra le valutazioni f1_score delle single classi
        risultati_svm.append(f1_svm)

        #ADDESTRAMENTO E CROSS-VALIDATION RANDOM FOREST
        print(f"Esecuzione Cross-Validation su Random Forest per K={K}")
        rf = RandomForestClassifier(n_estimators=500, max_depth=50, random_state=42)
        y_pred_rf = cross_val_predict(rf, X, y_numerico, cv=cv, n_jobs=-1)
        f1_rf = f1_score(y_numerico, y_pred_rf, average='macro')
        risultati_rf.append(f1_rf)

        # Scrittura incrementale dei risultati dettagliati nel report di testo
        with open(percorso_report, "a", encoding="utf-8") as file_log:
            file_log.write(f"\n{'='*50}\n")
            file_log.write(f"RISULTATI DETTAGLIATI PER K = {K}\n")
            file_log.write(f"{'='*50}\n\n")
            
            file_log.write("SUPPORT VECTOR MACHINE (Kernel: RBF, C=100.0)\n")
            file_log.write(f"Macro F1-Score: {f1_svm:.4f}\n\n")
            file_log.write(classification_report(y_numerico, y_pred_svm, target_names=nomi_classi))
            
            file_log.write("\n" + "-"*40 + "\n")
            file_log.write("RANDOM FOREST (500 Estimators, Max Depth: 50)\n")
            file_log.write(f"Macro F1-Score: {f1_rf:.4f}\n\n")
            file_log.write(classification_report(y_numerico, y_pred_rf, target_names=nomi_classi))
            
        print(f"Risultati per K={K} salvati nel report.")

    print("\nGenerazione del grafico comparativo in corso")
    plt.figure(figsize=(10, 6))
    
    # Assicura la corrispondenza tra i valori dell'asse X e i F1-score raccolti
    valori_K_effettivi = configurazione.VALORI_K_DA_TESTARE[:len(risultati_svm)]
    
    plt.plot(valori_K_effettivi, risultati_svm, marker='o', linewidth=2, label='SVM (Kernel RBF)', color='blue')
    plt.plot(valori_K_effettivi, risultati_rf, marker='s', linewidth=2, label='Random Forest', color='green')
    
    plt.title('Confronto Classificatori al variare della dimensione del Vocabolario (K)', fontsize=14, pad=15)
    plt.xlabel('Dimensione Vocabolario (K)', fontsize=12)
    plt.ylabel('F1-Score (Macro Average)', fontsize=12)
    plt.xticks(valori_K_effettivi)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    percorso_grafico = os.path.join(cartella_output, "grafico_confronto_K.png")
    plt.savefig(percorso_grafico, dpi=300)
    print(f"Grafico salvato con successo in: {percorso_grafico}")
    
    plt.show()


if __name__ == "__main__":
    principale()