from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import os
import dlib
import numpy as np
from pinecone import Pinecone, ServerlessSpec
import tkinter as tk
from tkinter import font, messagebox, simpledialog
import winreg
import warnings
from PIL import Image, ImageTk
import sys
import time
import threading



# Pinecone initialisieren
pc = Pinecone(
    api_key="6ba67da3-f42a-4c9b-af51-a55ef5d139e4"  #API-Schlüssel
)

# Überprüfen, ob der Index existiert, und erstellen, falls nicht
if 'face-recognition-index' not in pc.list_indexes().names():
    pc.create_index(
        name='face-recognition-index',
        dimension=1536,  # Setze die Dimension entsprechend deinen Embeddings
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

warnings.filterwarnings("ignore", category=DeprecationWarning)

class GesichtserkennungApp:
    def __init__(self, master):
        self.master = master
        master.title("Gesichtserkennung")
        self.master.geometry("800x300")
        
        custom_font = font.Font(family="Arial", size=16)

        # Dlib-Modelle laden
        try:
            self.dlib_face_detector = dlib.get_frontal_face_detector()
            self.dlib_face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
            self.dlib_shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Laden der Dlib-Modelle: {e}")
            sys.exit(1)

        self.label = tk.Label(master, text="Willkommen zur Gesichtserkennung!", font=custom_font)
        self.label.pack(padx=30, pady=50)
        
        self.start_button = tk.Button(master, text="Nutzer überprüfen", command=self.zeige_webcam_fuer_upruefung)
        self.start_button.pack(padx=5, pady=5)
        
        self.new_customer_button = tk.Button(master, text="Neues Kundenbild speichern", command=self.zeige_webcam_fuer_neues_kundenbild)
        self.new_customer_button.pack(padx=5, pady=5)

        self.deleted_customer_button = tk.Button(master, text="Kundendaten löschen", command=self.loesche_kundendaten)
        self.deleted_customer_button.pack(padx=5, pady=5)
        
        self.quit_button = tk.Button(master, text="Beenden", command=self.beenden)
        self.quit_button.pack(padx=5, pady=5)
        
        self.video_capture = None
        self.deepface_ordner = self.lese_registry_wert(r"SOFTWARE\Tanoffice\facescan", "FotoPfad") or r"C:\TanOffice\DeepFace_Kunden"
        self.frame = None  # Zum Speichern des letzten Frames
        
        self.running_thread = None  # Thread für die Aktualisierung des Registry-Werts

        # Starten des Hintergrund-Threads für die Registry-Aktualisierung
        self.start_registry_thread()

    def set_registry_value(self, root, path, name, value):
        try:
            # Öffne den Registry-Schlüssel (oder erstelle ihn, falls er nicht existiert)
            key = winreg.CreateKey(root, path)
            # Setze den Wert
            winreg.SetValueEx(key, name, 0, winreg.REG_DWORD, value)
            # Schlüssel schließen
            winreg.CloseKey(key)
        except Exception as e:
            print(f"Fehler beim Schreiben in die Registry: {e}")


    def get_registry_value(self, key_path, value_name):
        try:
            # Öffne den Schlüssel im Lesezugriff
            registry_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_READ)
            
            # Lese den Wert des angegebenen Wertes
            value, regtype = winreg.QueryValueEx(registry_key, value_name)
            print(f"Registry value: {value}")
            
            # Schließe den Schlüssel
            winreg.CloseKey(registry_key)
            
            return value
        except FileNotFoundError:
            print(f"Registry key or value not found: {key_path}\\{value_name}")
            return None
        except Exception as e:
            print(f"Error reading registry: {e}")
            return None


    def start_registry_thread(self):
        """Startet den Hintergrund-Thread, der alle 5 Sekunden den Registry-Wert setzt."""
        self.running_thread = threading.Thread(target=self.registriere_lebensstatus)
        self.running_thread.daemon = True  # Wird beendet, wenn die Hauptanwendung beendet wird
        self.running_thread.start()



    def registriere_lebensstatus(self):
        registry_path = r"SOFTWARE\Tanoffice\facescan"
        registry_value_name = "IsRunning"
        registry_function_name = "Funktion"
        """Setzt den Registry-Wert 'IsRunning' alle 5 Sekunden auf 'True'."""
        while True:
            try:
                self.set_registry_value(winreg.HKEY_CURRENT_USER, registry_path, registry_value_name, 1)
                print("Registry-Wert gesetzt")
                current_value = self.get_registry_value(registry_path, registry_function_name)
                
                if current_value == 1:
                    self.set_registry_value(winreg.HKEY_CURRENT_USER, registry_path, registry_function_name, 0)
                    self.zeige_webcam_fuer_upruefung()
                    
                elif current_value == 2:
                    self.set_registry_value(winreg.HKEY_CURRENT_USER, registry_path, registry_function_name, 0)
                    self.zeige_webcam_fuer_neues_kundenbild()
                    
                elif current_value == 3:
                    self.set_registry_value(winreg.HKEY_CURRENT_USER, registry_path, registry_function_name, 0)
                    self.loesche_kundendaten()
                    
                elif current_value == 4:
                    self.set_registry_value(winreg.HKEY_CURRENT_USER, registry_path, registry_function_name, 0)
                    self.beenden()
                
                # Intervall für die Registry-Prüfung (z.B. alle 5 Sekunden)
                time.sleep(1)
                
            except Exception as e:
                print(f"Fehler beim Setzen des Registry-Werts: {e}")
                time.sleep(1)




    def beenden(self):
        """Setzt den Wert von 'IsRunning' auf False, wenn die Anwendung geschlossen wird."""
        try:
            self.set_registry_value(winreg.HKEY_CURRENT_USER, r"SOFTWARE\Tanoffice\facescan", "IsRunning", 0)
        except Exception as e:
            print(f"Fehler beim Setzen des Registry-Werts bei Beenden: {e}")
        self.master.quit()  # Beendet das Tkinter-Fenster

   

    def lese_registry_wert(self, registry_path, value_name):
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, registry_path) as key:
                wert, _ = winreg.QueryValueEx(key, value_name)
                return wert
        except FileNotFoundError:
            print(f"Der Registry-Pfad '{registry_path}' oder der Wert '{value_name}' wurde nicht gefunden.")
            return None

    def schreibe_registry(self, erkannter_kunde, ergebnis, ergebnis_text, zwischenstatus):
        try:
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"SOFTWARE\Tanoffice\facescan") as key:
                winreg.SetValueEx(key, "ErkannterKunde", 0, winreg.REG_SZ, erkannter_kunde)
                winreg.SetValueEx(key, "Ergebnis", 0, winreg.REG_SZ, ergebnis)
                winreg.SetValueEx(key, "ErgebnisText", 0, winreg.REG_SZ, ergebnis_text)
                winreg.SetValueEx(key, "Zwischenstatus", 0, winreg.REG_SZ, zwischenstatus)
            print(f"Erkannter Kunde in die Registry geschrieben: {erkannter_kunde}")
        except Exception as e:
            print(f"Fehler beim Schreiben in die Registry: {e}")
            

    # Funktion zum Löschen von Kundendaten
    def loesche_kundendaten(self):
        # Benutzer nach dem Kundennamen fragen
        name = simpledialog.askstring("Kundenname", "Geben Sie den Namen des Kunden ein, den Sie löschen möchten:")
        if not name:
            return

        # Bilddatei löschen
        bild_pfad = os.path.join(self.deepface_ordner, f"{name}.jpg")
        if os.path.exists(bild_pfad):
            os.remove(bild_pfad)
            print(f"Bild '{name}.jpg' wurde gelöscht.")
        else:
            messagebox.showwarning("Warnung", f"Bild für '{name}' wurde nicht gefunden.")

        # Vektor in Pinecone löschen
        try:
            index = pc.Index("face-recognition-index")
            index.delete(ids=[name])
            print(f"Vektor für {name} erfolgreich aus Pinecone entfernt.")
            self.schreibe_registry(name, "0", f"Kundendaten für '{name}' wurden erfolgreich gelöscht.", "Fertig")
            messagebox.showinfo("Erfolg", f"Kundendaten für '{name}' wurden erfolgreich gelöscht.")
        except Exception as e:
            fehler_text = f"Fehler beim Löschen des Vektors in Pinecone: {e}"
            self.schreibe_registry(name, "1", fehler_text, "Fehlgeschlagen")
            messagebox.showerror("Fehler", fehler_text)


    def zeige_webcam_fuer_upruefung(self):
        registry_path = r"SOFTWARE\Tanoffice\facescan"
        registry_value_name = "Funktion"

        self.set_registry_value(winreg.HKEY_CURRENT_USER, registry_path, registry_value_name, 1)

        """Funktion zum Starten der Webcam für die Nutzerüberprüfung."""
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            messagebox.showerror("Fehler", "Webcam konnte nicht geöffnet werden.")
            return

        while True:
            ret, frame = self.video_capture.read()
            # Anzeige der A nweisungen auf dem Kamerabild
            cv2.putText(frame, "Druecken Sie <Enter> zum Vergleichen", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Druecken Sie <Esc> zum Beenden", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Webcam', frame)
            if not ret:
                break
            cv2.imshow('Webcam', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:
                # Das aktuelle Frame wird für den Vergleich verwendet
                self.vergleiche_gesicht_mit_pinecone(frame)
                break  # Einmaliger Vergleich, dann Schleife beenden
            elif key == 27:
                break
        self.video_capture.release()
        cv2.destroyAllWindows()


    def lade_alle_kundenbilder(self):
        index = pc.Index("face-recognition-index")

        bilder = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for datei in os.listdir(self.deepface_ordner):
                if datei.lower().endswith(('.jpg', '.jpeg', '.png')):
                    pfad = os.path.join(self.deepface_ordner, datei)
                    futures.append(executor.submit(self.prozessiere_kundenbild, pfad, index))
            
            for future in as_completed(futures):
                pfad, embedding = future.result()
                if embedding is not None:
                    bilder.append(pfad)
        
        return bilder

    def prozessiere_kundenbild(self, pfad, index):
        try:
            bild = cv2.imread(pfad)
            if bild is not None:
                print(f"Lade Bild: {pfad}")
                # Berechne Embedding
                embedding = self.berechne_embedding(bild)
                if embedding is not None:
                    vector_id = os.path.splitext(os.path.basename(pfad))[0]
                    index.upsert([(vector_id, embedding)])
                    print(f"Vektor für {vector_id} erfolgreich in Pinecone gespeichert.")
                return pfad, embedding
        except Exception as e:
            print(f"Fehler beim Verarbeiten des Bildes {pfad}: {e}")
            return pfad, None

    def berechne_embedding(self, bild):
        det = self.dlib_face_detector(bild, 1)
        if len(det) > 0:
            shape = self.dlib_shape_predictor(bild, det[0])
            embedding = np.array(self.dlib_face_recognition_model.compute_face_descriptor(bild, shape))
            if embedding.size == 128:
                embedding = np.resize(embedding, (1536,))
            return embedding
        else:
            print(f"Kein Gesicht erkannt im Bild.")
            return None

    def vergleiche_gesicht_mit_pinecone(self, frame):
        index = pc.Index("face-recognition-index")

        embedding = self.berechne_embedding(frame)
        if embedding is not None:
            result = index.query(vector=embedding.tolist(), top_k=1, include_values=True, approximate=True)
            if result["matches"]:
                best_match = result["matches"][0]
                erkannter_kunde = best_match["id"]
                distance = best_match["score"]
                self.schreibe_registry(erkannter_kunde, "0", f"Kunde erkannt mit Distanz: {distance:.2f}", "Fertig")
                messagebox.showinfo("Kunde erkannt", f"Erkannter Kunde: {erkannter_kunde}\nDistanz: {distance:.2f}")
            else:
                self.schreibe_registry("Fehler", "1", "Kein Kunde erkannt", "Fertig")
                messagebox.showinfo("Kein Kunde erkannt", "Kein übereinstimmender Kunde wurde erkannt.")
        else:
            self.schreibe_registry("Fehler", "1", "Kein Gesicht erkannt", "Fertig")
            messagebox.showwarning("Kein Gesicht", "Es konnte kein Gesicht erkannt werden.")

    def vergleiche_gesicht_mit_alle_kundenbilder(self, frame):
        bilder = self.lade_alle_kundenbilder()
        if bilder:
            print(f"Vergleiche mit {len(bilder)} Kundenbildern")
            self.vergleiche_gesicht_mit_pinecone(frame)
        else:
            self.schreibe_registry("Fehler", "1", "Keine Kundenbilder gefunden.", "Fertig")
            messagebox.showerror("Fehler", "Keine Kundenbilder gefunden.")
    

    def zeige_webcam_fuer_neues_kundenbild(self):
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            self.schreibe_registry("Fehler", "1", "Kamera konnte nicht geöffnet werden.", "Fertig")
            messagebox.showerror("Fehler", "Kamera konnte nicht geöffnet werden.")
            return
        
        self.label.config(text="Drücken Sie 's', um das Kundenbild zu speichern oder 'q', um zu beenden.")
        
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                self.schreibe_registry("Fehler", "1", "Fehler beim Erfassen des Bildes.", "Fertig")
                break
            
            cv2.putText(frame, "Druecken Sie 's' zum Speichern", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Druecken Sie 'q' zum Beenden", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Webcam', frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                self.frame = frame  # Speichere das aktuelle Frame
                self.speichere_neues_kundenbild(frame)  # Speichere das Bild
                break
            elif key == ord('q'):
                break
        self.video_capture.release()
        cv2.destroyAllWindows()

    def speichere_neues_kundenbild(self, frame):
        name = simpledialog.askstring("Neuer Kunde", "Geben Sie den Namen des Kunden ein:")
        if not name:
            return
        
        dateiname = os.path.join(self.deepface_ordner, f"{name}.jpg")
        cv2.imwrite(dateiname, frame)
        print(f"Kundenbild für {name} wurde erfolgreich gespeichert.")
        
        self.index_kundenbild_in_pinecone(name, frame)

    def index_kundenbild_in_pinecone(self, name, frame):
        index = pc.Index("face-recognition-index")
        embedding = self.berechne_embedding(frame)
        if embedding is not None:
            index.upsert([(name, embedding.tolist())])
            print(f"Vektor für {name} erfolgreich in Pinecone gespeichert.")
            self.schreibe_registry(name, "0", f"Kundenbild für '{name}' erfolgreich gespeichert.", "Fertig")
            messagebox.showinfo("Erfolg", f"Kundenbild für '{name}' erfolgreich gespeichert.")
        else:
            self.schreibe_registry(name, "1", f"Fehler beim Berechnen des Embeddings für '{name}'", "Fehlgeschlagen")
            messagebox.showerror("Fehler", f"Fehler beim Berechnen des Embeddings für '{name}'")

# Tkinter Hauptprogramm starten
root = tk.Tk()
app = GesichtserkennungApp(root)
root.mainloop()