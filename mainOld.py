import streamlit as st

# import tensorflow as tf
# import keras
import librosa
import numpy as np
import pandas as pd
import soundfile
import audioread
import os


def save_uploaded_file(uploaded_file):
    # Specify the directory to save the uploaded files
    save_directory = "uploads"

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Create a file path based on the original file name
    file_path = os.path.join(save_directory, uploaded_file.name)

    # Write the file contents to the specified file path
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"File saved as {file_path}")


def audio_to_csv():
    def audio_pipeline(audio):
        features = []

        # Calcul du ZCR

        zcr = librosa.zero_crossings(audio)
        features.append(sum(zcr))

        # Calcul de la moyenne du Spectral centroid

        spectral_centroids = librosa.feature.spectral_centroid(y=audio)[0]
        features.append(np.mean(spectral_centroids))

        # Calcul du spectral rolloff point

        rolloff = librosa.feature.spectral_rolloff(y=audio)
        features.append(np.mean(rolloff))

        # Calcul des moyennes des MFCC

        mfcc = librosa.feature.mfcc(y=audio)

        for x in mfcc:
            features.append(np.mean(x))

        return features

    # audio = soundfile.read(audio)
    # audio = audioread.audio_open(audio)
    # my_audio = librosa.load("./uploads/blues.00002.wav")
    my_audio = librosa.load("./genres/blues/blues.00000.wav")

    column_names = [
        "zcr",
        "spectral_c",
        "rolloff",
        "mfcc1",
        "mfcc2",
        "mfcc3",
        "mfcc4",
        "mfcc5",
        "mfcc6",
        "mfcc7",
        "mfcc8",
        "mfcc9",
        "mfcc10",
        "mfcc11",
        "mfcc12",
        "mfcc13",
        "mfcc14",
        "mfcc15",
        "mfcc16",
        "mfcc17",
        "mfcc18",
        "mfcc19",
        "mfcc20",
        "label",
    ]

    df = pd.DataFrame(columns=column_names)

    df.loc[0] = audio_pipeline(my_audio)
    df.drop(labels="label")
    return df


st.title("Prédiction genre musical")

# Ajoutez des composants interactifs pour que l'utilisateur télécharge un fichier
uploaded_file = st.file_uploader("Télécharger un fichier audio", type=["wav"])

if uploaded_file is not None:
    # st.title(uploaded_file)
    st.audio(uploaded_file, format="audio/wav")

    save_uploaded_file(uploaded_file)
    # st.download_button(
    #     "Télécharger le fichier", data=uploaded_file, file_name="audio_file"
    # )
    # model = keras.models.load_model("./mymodelv1.keras")
    df = audio_to_csv()
    print(df)
    # model.predict(df)
    # print(image)
