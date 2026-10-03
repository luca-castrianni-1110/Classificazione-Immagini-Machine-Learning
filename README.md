# Classificazione-Immagini-Machine-Learning
Modelli di classificazione di immagini aeree tramite SVM, Random Forest e preprocessing avanzato.


## Guida all'Utilizzo e Istruzioni d'Uso

> **IMPORTANTE (Prerequisito):** Prima di avviare qualsiasi script, scaricare i due dataset e posizionarli nella cartella principale del progetto, mantenendo la struttura originale. Questo passaggio è fondamentale affinché i percorsi (*path*) dei file vengano risolti correttamente.

---

### Passo 1: Addestramento
* Accedere alla cartella `File_per_i_calcoli` e avviare lo script principale (`main`).

### Passo 2: Visualizzazione e Analisi dei Dati
* Accedere alla cartella `File_per_visualizzazioni` e avviare i tre script dedicati per esplorare il vocabolario, ispezionare i dati o visualizzare gli istogrammi.
  * **Visualizza_Descrittori:** Premere `Invio` per scorrere le immagini a schermo o il tasto `q` per chiudere la finestra.
  * Gli altri grafici verranno mostrati in sequenza (alla chiusura del primo si aprirà il successivo). 
  * Tutti i risultati grafici sono comunque salvati e consultabili all'interno della cartella `Risultati` (situata dentro `File_per_i_calcoli`).

### Passo 3: Matrice di Confusione
* Accedere alla cartella `File_per_matrice_confusione` e avviare il file corrispondente per generare e visualizzare la matrice di confusione.

### Passo 4: Valutazione del Modello
* Accedere alla cartella `Valutazione_Modello` e avviare lo script. Al termine dei calcoli, verrà mostrato a schermo il grafico comparativo dei due classificatori. 
* Il file di testo (`.txt`) riepilogativo con tutti i valori calcolati è salvato nella cartella `Risultati` all'interno di `File_per_i_calcoli`.

### Passo 5: Inferenza (Test su nuove immagini)
1. Accedere alla cartella `Inferenza` e inserire l'immagine da classificare all'interno della sottocartella `Foto_da_aggiungere`.
2. Avviare prima lo script `Salva_modello_finale`.
3. Successivamente, avviare lo script di `inferenza`.
4. Tutti i risultati dell'inferenza verranno salvati automaticamente nella cartella `Risultati` (dentro `File_per_i_calcoli`).
