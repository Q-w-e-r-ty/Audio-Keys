import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sounddevice as sd
import soundfile as sf
import ed
import Machine

class VoiceGuard:
    def __init__(self, master):
        self.master = master
        self.master.title("Voice Guard")

        self.mode_var = tk.StringVar()
        self.mode_var.set("")  # No initial selection

        self.file_path_var = tk.StringVar()
        self.audio_path_var = tk.StringVar()
        self.machine=Machine.Mach()
        self.create_widgets()
        


    def create_widgets(self):
        # Centering the window
        window_width = 800
        window_height = 350
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        x_coordinate = int((screen_width - window_width) / 2)
        y_coordinate = int((screen_height - window_height) / 2)
        self.master.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

        # Welcome text
        welcome_label = ttk.Label(self.master, text="Welcome to Voice Guard", font=('Helvetica', 35, 'bold'),background="#ADD8FF")
        welcome_label.pack(pady=20)

        # Mode selection
        mode_frame = ttk.Frame(self.master,style='Title.TFrame')
        mode_frame.pack(padx=10, pady=10)

        encrypt_btn = ttk.Button(mode_frame, text="Encrypt", command=self.show_encrypt_screen, style='Large.TButton')
        encrypt_btn.grid(row=0, column=0, padx=5, pady=5)

        decrypt_btn = ttk.Button(mode_frame, text="Decrypt", command=self.show_decrypt_screen, style='Large.TButton')
        decrypt_btn.grid(row=0, column=1, padx=5, pady=5)

        exit_btn = ttk.Button(self.master, text="Exit", command=self.master.destroy, style='Large.TButton')
        exit_btn.pack(padx=10)

    def show_encrypt_screen(self):
        self.clear_screen()
        self.mode_var.set("Encrypt")
        self.file_path_var.set("")
        self.audio_path_var.set("")

        self.create_file_selection_frame()
        self.create_audio_selection_frame()

        # Encrypt button
        encrypt_btn = ttk.Button(self.master, text="Encrypt", command=self.encrypt_files, style='Large.TButton')
        encrypt_btn.pack(padx=10, pady=10)

        # Exit and Back buttons frame
        button_frame = ttk.Frame(self.master,style='Title.TFrame')
        button_frame.pack(pady=10)

        back_btn = ttk.Button(button_frame, text="Back", command=self.show_mode_selection_screen, style='Large.TButton')
        back_btn.pack(side=tk.LEFT, padx=10)

        exit_btn = ttk.Button(button_frame, text="Exit", command=self.master.destroy, style='Large.TButton')
        exit_btn.pack(side=tk.RIGHT, padx=10)

    def create_file_selection_frame(self):
        # File selection frame
        file_frame = ttk.LabelFrame(self.master, text="Select the Text File to be encrypted",style='Title.TFrame')
        file_frame.pack(padx=10, pady=10, fill="x")

        file_btn = ttk.Button(file_frame, text="Select Text File", command=self.select_text_file, style='Large.TButton')
        file_btn.grid(row=0, column=0, padx=5, pady=5)
        self.ek_button=ttk.Label(file_frame, textvariable=self.file_path_var, wraplength=400,background="#ADD8FF", font=("Helvetica", 16)).grid(row=0, column=1, padx=5, pady=5)

    def create_audio_selection_frame(self):
        # Audio selection frame
        audio_frame = ttk.LabelFrame(self.master, text="Select Audio File (Minimum 2 mins)",style='Title.TFrame')
        audio_frame.pack(padx=10, pady=10, fill="x")

        audio_btn = ttk.Button(audio_frame, text="Select Audio File", command=self.select_audio_file, style='Large.TButton')
        audio_btn.grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(audio_frame, textvariable=self.audio_path_var, wraplength=400,background="#ADD8FF", font=("Helvetica", 16)).grid(row=0, column=1, padx=5, pady=5)
    
    
    def create_file_selection_frame1(self):
        # File selection frame
        file_frame = ttk.LabelFrame(self.master, text="Select the encrypted file",style='Title.TFrame')
        file_frame.pack(padx=10, pady=10, fill="x")

        file_btn = ttk.Button(file_frame, text="Select Text File", command=self.select_enc_file, style='Large.TButton')
        file_btn.grid(row=0, column=0, padx=5, pady=5)
        self.ek_button=ttk.Label(file_frame, textvariable=self.file_path_var, wraplength=400,background="#ADD8FF", font=("Helvetica", 16)).grid(row=0, column=1, padx=5, pady=5)

    def create_audio_selection_frame1(self):
        # Audio selection frame
        audio_frame = ttk.LabelFrame(self.master, text="Select Audio File (Minimum 10 secs)",style='Title.TFrame')
        audio_frame.pack(padx=10, pady=10, fill="x")

        audio_btn = ttk.Button(audio_frame, text="Select Audio File", command=self.select_audio_file, style='Large.TButton')
        audio_btn.grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(audio_frame, textvariable=self.audio_path_var, wraplength=400,background="#ADD8FF", font=("Helvetica", 16)).grid(row=0, column=1, padx=5, pady=5)

    def show_decrypt_screen(self):
        self.clear_screen()
        self.mode_var.set("Decrypt")
        self.file_path_var.set("")
        self.audio_path_var.set("")

        self.create_file_selection_frame1()
        self.create_audio_selection_frame1()

        # Decrypt button
        decrypt_btn = ttk.Button(self.master, text="Decrypt", command=self.decrypt_files, style='Large.TButton')
        decrypt_btn.pack(padx=10, pady=10)

        # Exit and Back buttons frame
        button_frame = ttk.Frame(self.master,style='Title.TFrame')
        button_frame.pack(pady=10)

        back_btn = ttk.Button(button_frame, text="Back", command=self.show_mode_selection_screen, style='Large.TButton')
        back_btn.pack(side=tk.LEFT, padx=10)

        exit_btn = ttk.Button(button_frame, text="Exit", command=self.master.destroy, style='Large.TButton')
        exit_btn.pack(side=tk.RIGHT, padx=10)

    def show_mode_selection_screen(self):
        self.clear_screen()
        self.create_widgets()

    def clear_screen(self):
        for widget in self.master.winfo_children():
            widget.destroy()
            
    def select_enc_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Encrypted files", "*.enc")])
        self.file_path_var.set(file_path)

    def select_text_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        self.file_path_var.set(file_path)

    def select_audio_file(self):
        audio_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav;*.mp3")])
        self.audio_path_var.set(audio_path)

    def encrypt_files(self):
        text_file = self.file_path_var.get()
        audio_file = self.audio_path_var.get()

        if text_file and audio_file:
            #key=Machine.give_key_from_audio("D:/Padhai/SEM/ML/Project/Testing/16000_pcm_speeches/Armaan/Raw/Armaan.wav")
            self.key=self.machine.give_key_from_audio(audio_file)
            ed.encrypt_file(text_file,self.key)
            # Perform encryption logic
            messagebox.showinfo("Success", "File has been encrypted and downloaded successfully.")
        else:
            messagebox.showerror("Error", "Please select both text and audio files.")

    def decrypt_files(self):
        text_file = self.file_path_var.get()
        audio_file = self.audio_path_var.get()

        if text_file and audio_file:
            if(self.machine.match_audio(audio_file)):
                ed.decrypt_file(text_file,self.key)

                messagebox.showinfo("Success", "Decryption successful.")
            else:
                messagebox.showinfo("Error","Voice did not match, Try Again")
        else:
            messagebox.showerror("Error", "Please select both text and audio files.")

def main():
    root = tk.Tk()
    root.configure(bg="#ADD8FF")
    style = ttk.Style(root)
    style.configure('Large.TButton', font=('Helvetica', 24, 'bold'))
    style.configure('Title.TFrame', background="#ADD8FF")
    app = VoiceGuard(root)
    root.mainloop()

if __name__ == "__main__":
    main()

