import pandas as pd
import os

# Variabili: nome del dataset e della colonna
dataset_name = "Fakeddit"  # Inserisci il nome del dataset
column_name = "2_way_label"   # Nome della colonna che contiene le label

# Funzione per contare le istanze di ciascuna label
def count_labels(file_path, column_name):
    # Leggi il file tsv
    df = pd.read_csv(file_path, sep='\t')
    
    # Conta le occorrenze di ciascuna label nella colonna specificata
    label_counts = df[column_name].value_counts()
    
    # Assicurati che entrambi i valori (0 e 1) siano presenti, altrimenti aggiungi 0
    label_counts = label_counts.reindex([0, 1], fill_value=0)
    
    return label_counts

# Percorsi dei file TSV
train_file = os.path.join(dataset_name, "train.tsv")
val_file = os.path.join(dataset_name, "val.tsv")
test_file = os.path.join(dataset_name, "test.tsv")

# Conta le istanze per ogni file
train_label_counts = count_labels(train_file, column_name)
val_label_counts = count_labels(val_file, column_name)
test_label_counts = count_labels(test_file, column_name)

# Stampa i risultati
print(f"Train set label counts:\n{train_label_counts}")
print(f"Validation set label counts:\n{val_label_counts}")
print(f"Test set label counts:\n{test_label_counts}")
