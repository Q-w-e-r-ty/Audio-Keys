
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
import librosa.display
import librosa
import soundfile as sf
from keras.callbacks import EarlyStopping
from IPython.display import display, Audio
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score

class Mach:
    def __init__(self) -> None:
        #self.model=load_model("D:/Padhai/SEM/ML/Audio Keys/my_model.h5")
        pass
    def split_audio(self,input_file, output_folder, segment_length=500):
        # Load the audio file
        audio = AudioSegment.from_file(input_file)

        # Calculate the number of segments needed
        num_segments = len(audio) // segment_length

        # Iterate over each segment
        for i in range(num_segments):
            # Calculate the start and end time of the segment
            start_time = i * segment_length
            end_time = (i + 1) * segment_length

            # Extract the segment
            segment = audio[start_time:end_time]

            # Save the segment to a file
            output_file = output_folder + f"/segment_{i}.wav"
            segment.export(output_file, format="wav")

    def play_audio(self,audio_path):
    # Function to play audio file
        display(Audio(filename=audio_path))
        
    def generate_random_bytes(self):
        return os.urandom(16)

    def plot_audio_features(self,audio_path):
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)

        # Extract speaker name from the file path
        speaker_name = os.path.basename(audio_path).split('_')[0]

        # Plot the waveform
        plt.figure(figsize=(15, 10))
        plt.subplot(3, 1, 1)
        librosa.display.waveshow(y, sr=sr,color="blue")
        plt.title(f'Waveform - {speaker_name}')

        # Plot the spectrogram
        plt.subplot(3, 1, 2)
        D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram - {speaker_name}')

        # Plot the MFCCs
        plt.subplot(3, 1, 3)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        librosa.display.specshow(mfccs, x_axis='time')
        plt.colorbar()
        plt.title(f'MFCCs - {speaker_name}')

        plt.tight_layout()
        plt.show()

    def extract_features(self,parent_dir, speaker_folders):
        features = []
        labels = []

        for i, speaker_folder in enumerate(speaker_folders):
            speaker_folder_path = os.path.join(parent_dir, speaker_folder)

            for filename in os.listdir(speaker_folder_path):
                if filename.endswith(".wav"):
                    file_path = os.path.join(speaker_folder_path, filename)
                    audio, sr = librosa.load(file_path, sr=None, duration=500)
                    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                    
                    # Normalize MFCC features
                    mfccs = StandardScaler().fit_transform(mfccs)
                    
                    features.append(mfccs.T)
                    labels.append(i)

        return np.array(features), np.array(labels)


    def give_key_from_audio(self,filepath):
        self.input = filepath
        # Output directory to clear
        output_dir = ["E:/Padhai/SEM/ML/Audio Keys/combined_files","E:/Padhai/SEM/ML/Audio Keys/16000_pcm_speeches/Anjaneya","E:/Padhai/SEM/ML/Audio Keys/16000_pcm_speeches/Armaan","E:/Padhai/SEM/ML/Audio Keys/16000_pcm_speeches/Tanmay"]
        for single_path in output_dir:
            # Clear the contents of the output directory
            shutil.rmtree(single_path, ignore_errors=True)
            os.makedirs(single_path, exist_ok=True)
            print(f"Contents of {single_path} cleared.")

        
        input_file = "E:/Padhai/SEM/ML/Audio Keys/16000_pcm_speeches/Tanmay_Raw/Tanmay.wav"
        output_folder = "E:/Padhai/SEM/ML/Audio Keys/16000_pcm_speeches/Tanmay"
        self.split_audio(input_file, output_folder)

        input_file = "E:/Padhai/SEM/ML/Audio Keys/16000_pcm_speeches/Armaan_Raw/Armaan.wav"
        output_folder = "E:/Padhai/SEM/ML/Audio Keys/16000_pcm_speeches/Armaan"
        self.split_audio(input_file, output_folder)

        input_file = "E:/Padhai/SEM/ML/Audio Keys/16000_pcm_speeches/Anjaneya_Raw/Anjaneya.wav"
        output_folder = "E:/Padhai/SEM/ML/Audio Keys/16000_pcm_speeches/Anjaneya"

        self.split_audio(input_file, output_folder)







        # Path to the dataset
        dataset_path = "E:/Padhai/SEM/ML/Audio Keys/16000_pcm_speeches"

        # Output directory to save the combined files
        output_dir = "E:/Padhai/SEM/ML/Audio Keys/combined_files"

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # List of speaker folders
        speaker_folders = [
            "Tanmay",
            "Armaan",
            "Anjaneya"
        ]

        # Number of files to combine for each speaker
        num_files_to_combine = 240

        # Iterate over each speaker's folder
        for speaker_folder in speaker_folders:
            speaker_folder_path = os.path.join(dataset_path, speaker_folder)

            # List the first num_files_to_combine WAV files in the speaker's folder
            wav_files = [f"segment_{i}.wav" for i in range(num_files_to_combine)]

            # Combine all WAV files into a single long file
            combined_audio = []
            for wav_file in wav_files:
                wav_file_path = os.path.join(speaker_folder_path, wav_file)
                audio, sr = librosa.load(wav_file_path, sr=None)
                combined_audio.extend(audio)

            # Save the combined audio file
            output_file_path = os.path.join(output_dir, f"{speaker_folder}_combined.wav")
            sf.write(output_file_path, combined_audio, sr)

        #print("Combination complete. Combined files saved in:", output_dir)

    


        # Play a specific combined audio file
        speaker_folder = "Tanmay_combined"
        speaker_folder = "Armaan_combined"
        audio_path = os.path.join(output_dir, f"{speaker_folder}.wav")
        #print(f"Click the play button to listen: {audio_path}")
        #play_audio(audio_path)



        # Function to plot the waveform, spectrogram, and MFCCs


        # Paths to the combined audio files
        audio_paths = [
            'E:/Padhai/SEM/ML/Audio Keys/combined_files/Tanmay_combined.wav',
        ]

        # Plot features for each audio file
        #for audio_path in audio_paths:
            #plot_audio_features(audio_path)



        # Set the parent directory for speaker folders
        parent_dir = "E:/Padhai/SEM/ML/Audio Keys/16000_pcm_speeches"

        # List of speaker folders
        speaker_folders = [
            "Tanmay",
            "Armaan",
            "Anjaneya"
        ]



        # Extract features and labels
        X, y = self.extract_features(parent_dir, speaker_folders)


        #X.shape


        # Print the first few features
        #for feature in X[:1]:
            #print(feature)



        # Encode labels with explicit classes
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        self.label_encoder.classes_ = np.array(speaker_folders)

        # Split the data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Print the shapes of training and validation data
        #print("Training Data Shape:", X_train.shape)
        #print("Validation Data Shape:", X_val.shape)



        # Define the RNN model
        self.model = tf.keras.Sequential([
            #tf.keras.layers.LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])),
            tf.keras.layers.LSTM(128),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(speaker_folders), activation='softmax')
        ])


        # Compile the model
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


        # Define the EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
        

        # Train the model with EarlyStopping
        history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, callbacks=[early_stopping])
        

        # Check if EarlyStopping triggered
        #if early_stopping.stopped_epoch > 0:
        #    print("Early stopping triggered at epoch", early_stopping.stopped_epoch + 1)
        #else:
        #    print("Training completed without early stopping")
        # Plot training vs validation loss
        #plt.plot(history.history['loss'], label='Training Loss')
        #plt.plot(history.history['val_loss'], label='Validation Loss')
        #plt.xlabel('Epochs')
        #plt.ylabel('Loss')
        #plt.legend()
        #plt.show()
        



        key = os.urandom(16)
        # Evaluate the model on the test set
        #y_pred_probabilities = self.model.predict(X_test)
        #y_pred = np.argmax(y_pred_probabilities, axis=1)

        # Decode labels back to original format
        #y_test_decoded = self.label_encoder.inverse_transform(y_test)
        #y_pred_decoded = self.label_encoder.inverse_transform(y_pred)

        # Create a confusion matrix
        #conf_matrix = confusion_matrix(y_test_decoded, y_pred_decoded, labels=speaker_folders)

        # Calculate accuracy
        #accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
        #print(f"Test Evaluation Accuracy: {accuracy}")

        # Calculate F1 score
        #f1 = f1_score(y_test_decoded, y_pred_decoded, labels=speaker_folders, average='weighted')
        #print(f"Weighted F1 Score: {f1}")

        # Plot the confusion matrix
        #plt.figure(figsize=(8, 6))
        ##sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=speaker_folders, yticklabels=speaker_folders)

        # Rotate x-axis labels by 45 degrees
        #plt.xticks(rotation=45, ha="right")

        #plt.title("Confusion Matrix")
        
        #plt.xlabel("Predicted Label")
        #plt.ylabel("True Label")
        #plt.show()
        return key


    def get_speaker_index(self):
        
        if self.input.endswith("Tanmay.wav"):
            return 0
        elif self.input.endswith("Armaan.wav"):
            return 1
        elif self.input.endswith("Anjaneya.wav"):
            return 2
        else:
            print("F")
            

    def match_audio(self,filepath):
        #testing_features=[]
        #file_path_check="D:\Padhai\SEM-6\Crypto\Project\Testing\check\Anjaneya Check.wav"
        file_path_check=filepath
        #print("1")
        audio, sr = librosa.load(file_path_check, sr=None, duration=9)
        #print("11")
        # Normalize MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs = StandardScaler().fit_transform(mfccs)
        #print("111")
        #testing_features.append(mfccs.T)
        #x_check=np.array(testing_features)
        #y_pred_probabilities = model.predict(x_check)
        x_check = np.array([mfccs.T])  # Reshape the input to match the model's input shape
        #print("1111")
        
        #modell=load_model("D:/Padhai/SEM/ML/Audio Keys/my_model.h5")
        
        # Predict probabilities using the trained model
        #print(x_check.shape)
        #X_train_reshaped = x_check[:, :47, :]
        modell=load_model("E:/Padhai/SEM/ML/Audio Keys/my_model.h5")
        try:
            y_pred_probability = modell.predict(x_check)[0]
        except Exception as e:
            print("error is:",e)

        #print("11111")
        names=["Tanmay","Armaan","Anjaneya"]
        print(names)
        print(y_pred_probability)

        

        if y_pred_probability[self.get_speaker_index()]>0.7:
            return 1
        else:
            return 0


