import tkinter as tk
from tkinter import filedialog, ttk
import os
import threading
from src.Connection_ai import *

def select_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        t = threading.Thread(target=process_folder, args=(folder_path,))
        t.start()
        #process_folder(folder_path)

def process_folder(folder_path):
    file_list = os.listdir(folder_path)
    num_files = len(file_list)

    # Parcourez tous les fichiers du dossier
    for index, file_name in enumerate(file_list):
        # Assurez-vous qu'il s'agit d'un fichier image
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, file_name)
            # Traitez l'image (par exemple, affichez le chemin du fichier)
            img = image(file_path)
            print("Image traitée :", file_path)
            save_path = os.path.join("processed", file_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            frame = analyze_image(img)
            save_image(frame, save_path)
        elif file_name.endswith(('.mp4')):
            file_path = os.path.join(folder_path, file_name)
            save_path = os.path.join("processed", file_name)
            analyze_video(file_path, save_path)
        # Mettez à jour la barre de progression
        progress = (index + 1) / num_files * 100
        progress_bar['value'] = progress
        root.update_idletasks()  # Mettre à jour l'interface utilisateur

# Créer la fenêtre principale de l'interface utilisateur
root = tk.Tk()
root.title("Sélection de dossier")

# Ajouter un bouton pour sélectionner un dossier
select_button = tk.Button(root, text="Sélectionner un dossier", command=select_folder)
select_button.pack(pady=20)


    
# Créez une barre de progression
progress_bar = ttk.Progressbar(root, orient="horizontal", length=100, mode="determinate")
progress_bar.pack(pady=20)

# Exécutez la boucle principale de l'interface utilisateur
root.mainloop()
