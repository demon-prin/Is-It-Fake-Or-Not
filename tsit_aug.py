import pandas as pd
import os
import random
from PIL import Image
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import albumentations as A
import numpy as np

# Definisci la variabile dataset
dataset = 'Fakeddit'

# Inizializza il lemmatizzatore
lemmatizer = WordNetLemmatizer()

# Funzione per ottenere il lemma di base di una parola
def get_base_lemma(word):
    return lemmatizer.lemmatize(word.lower())

# Funzione per augmentare il testo con più variazione
def augment_text(text):
    words = text.split()
    new_words = []
    for word in words:
        base_lemma = get_base_lemma(word)
        synonyms = wordnet.synsets(base_lemma)
        if synonyms and random.random() > 0.4:  # Sostituisce il 60% delle parole per maggiore variabilità
            synonym = random.choice(synonyms).lemmas()[0].name()
            new_words.append(lemmatizer.lemmatize(synonym.lower()))
        else:
            new_words.append(word.lower())
    
    # Aggiunge una variazione casuale in fondo al testo per aumentare l'unicità
    random_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5))
    return " ".join(new_words) + " " + random_suffix

# Funzione per augmentare le immagini con più variazione
def augment_image(image_path):
    image = Image.open(image_path)
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=25, p=0.7),  # Maggiore rotazione per aumentare la diversità
        A.RandomBrightnessContrast(p=0.6),
        A.GaussianBlur(p=0.3),
        A.Resize(height=320, width=320)
    ])
    augmented = transform(image=np.array(image))
    return Image.fromarray(augmented['image'])

# Funzione per gestire l'augmentation di un dataset
def augment_dataset(input_file, output_file, images_folder, output_images_folder):
    # Leggi il file tsv
    df = pd.read_csv(input_file, sep='\t')

    # Crea la directory per salvare le immagini augmentate se non esiste
    augmented_images_folder = os.path.join(output_images_folder, os.path.basename(input_file).split('.')[0])
    if not os.path.exists(augmented_images_folder):
        os.makedirs(augmented_images_folder)

    # Crea un set per memorizzare i record già visti (per evitare duplicati)
    seen_records = set()

    # Crea una lista per memorizzare i nuovi record augmentati
    augmented_data = []

    # Itera su ogni riga del DataFrame
    for index, row in df.iterrows():
        # Aggiungi il record originale e mantienilo unico
        record_tuple = tuple(map(lambda x: get_base_lemma(x.lower()) if isinstance(x, str) else x, row.values))
        if record_tuple not in seen_records:
            augmented_data.append(row.to_dict())
            seen_records.add(record_tuple)

        image_path = os.path.join(images_folder, f"{row['id']}.jpg")
        
        # Genera 5 versioni augmentate per ogni record
        for i in range(5):
            # Augmentazione del testo
            clean_title_aug = augment_text(row['clean_title'])
            title_aug = augment_text(row['title'])

            # Augmentazione dell'immagine
            if os.path.exists(image_path):
                augmented_image = augment_image(image_path)
                augmented_image_name = f"{row['id']}_aug_{i}.jpg"
                augmented_image.save(os.path.join(augmented_images_folder, augmented_image_name))
            else:
                augmented_image_name = ""

            # Crea un nuovo record augmentato basato sull'originale
            augmented_record = row.to_dict()

            # Mantieni tutte le colonne, modificando solo quelle necessarie
            augmented_record['clean_title'] = clean_title_aug
            augmented_record['title'] = title_aug
            augmented_record['image_url'] = augmented_image_name

            # Cambia l'id con un nome univoco basato sull'augmented image
            augmented_record['id'] = f"{row['id']}_aug_{i}"

            # Assicurati che il nuovo record augmentato sia unico
            augmented_tuple = tuple(map(lambda x: get_base_lemma(x.lower()) if isinstance(x, str) else x, augmented_record.values()))
            if augmented_tuple not in seen_records:
                augmented_data.append(augmented_record)
                seen_records.add(augmented_tuple)

    # Crea un nuovo DataFrame per i dati augmentati
    augmented_df = pd.DataFrame(augmented_data)

    # Salva il nuovo dataframe augmentato su un file
    augmented_df.to_csv(output_file, sep='\t', index=False)

# Crea le directory per le immagini augmentate
if not os.path.exists(f'{dataset}/augmented_images/train'):
    os.makedirs(f'{dataset}/augmented_images/train')

# Esegui l'augmentation su train.tsv
augment_dataset(f'{dataset}/train.tsv', f'{dataset}/train_augmented.tsv', f'./{dataset}/train', f'./{dataset}/augmented_images/train')

print("Data augmentation completata per train.tsv.")
