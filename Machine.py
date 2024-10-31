
import numpy as np # linear algebra
import os
import shutil
import numpy as np
from pydub import AudioSegment
import librosa.display
import librosa
import soundfile as sf
from keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from keras.models import load_model

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
        output_dir = ["E:/Padhai/SEM/ML/Audio Keys/16000_pcm_speeches/processing/Anjaneya_Raw","E:/Padhai/SEM/ML/Audio Keys/16000_pcm_speeches/processing/Tanmay_Raw","E:/Padhai/SEM/ML/Audio Keys/16000_pcm_speeches/processing/Armaan_Raw"]
        for single_path in output_dir:
            # Clear the contents of the output directory
            shutil.rmtree(single_path, ignore_errors=True)
            os.makedirs(single_path, exist_ok=True)
            print(f"Contents of {single_path} cleared.")

        
        input_file = "E:/Padhai/SEM/ML/Audio Keys/16000_pcm_speeches/Raw/Tanmay_Raw/Tanmay_Raw.wav"
        output_folder = "E:/Padhai/SEM/ML/Audio Keys/16000_pcm_speeches/processing/Tanmay_Raw"
        self.split_audio(input_file, output_folder)

        input_file = "E:/Padhai/SEM/ML/Audio Keys/16000_pcm_speeches/Raw/Armaan_Raw/Armaan_Raw.wav"
        output_folder = "E:/Padhai/SEM/ML/Audio Keys/16000_pcm_speeches/processing/Armaan_Raw"
        self.split_audio(input_file, output_folder)

        input_file = "E:/Padhai/SEM/ML/Audio Keys/16000_pcm_speeches/Raw/Anjaneya_Raw/Anjaneya_Raw.wav"
        output_folder = "E:/Padhai/SEM/ML/Audio Keys/16000_pcm_speeches/processing/Anjaneya_Raw"
        self.split_audio(input_file, output_folder)

        # Path to the dataset
        
        # Output directory to save the combined files
        output_dir = "E:/Padhai/SEM/ML/Audio Keys/combined_files"

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        parent_dir = "E:/Padhai/SEM/ML/Audio Keys/16000_pcm_speeches/processing"

        # List of speaker folders
        speaker_folders = [
            "Tanmay_Raw",
            "Armaan_Raw",
            "Anjaneya_Raw"
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
        
        if self.input.endswith("Tanmay_Raw.wav"):
            return 0
        elif self.input.endswith("Armaan_Raw.wav"):
            return 1
        elif self.input.endswith("Anjaneya_Raw.wav"):
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


