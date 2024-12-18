"""
Version: 1.0.0
Date: 2024-12-18
Description: Dieses Skript führt eine Gesichtserkennung durch.
"""



from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import os
import dlib
import numpy as np 
from pinecone import Pinecone, ServerlessSpec
import tkinter as tk
from tkinter import font, messagebox
import winreg
import warnings
import shutil

import sys
import time
import threading
import win32gui
import win32con


class RegistryHandler:
    def set_registry_value_global(self, key_path, value_name, Value):
        try:
            # Öffne den Registry-Schlüssel (oder erstelle ihn, falls er nicht existiert)
            registry_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_WRITE)
            # Setze den Wert
            winreg.SetValueEx(registry_key, value_name, 0, winreg.REG_SZ, Value)
            # Schlüssel schließen
            winreg.CloseKey(registry_key)
            return Value
        
        except Exception as e:
            print(f"Fehler beim Schreiben in die Registry-verfummelt: {e}")
            return None

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
            return None
        except Exception as e:
            print(f"Fehler beim Lesen der Registry: {e}")
            return None
        
    

#Klasse RegistryHandler() wird funktional gemacht
registry_handler = RegistryHandler()




# Registry-Wert abrufen
registry_path = r"SOFTWARE\Tanoffice\facescan"
registry_ErgebnisText = "ErgebnisText"
registry_Ergebnis = "Ergebnis"


registry_handler.set_registry_value_global(registry_path, registry_ErgebnisText, "Fehler: Problem mit API-Schlüssel")
#print(f"schreibe Registry 1 : {result}")
result = registry_handler.set_registry_value_global(registry_path, registry_Ergebnis, "1")
print(f"schreibe Registry 2 : {result}") 


key = registry_handler.get_registry_value(registry_path, "API_KEY")



        # Pinecone initialisieren
pc = Pinecone(
        api_key=key  #API-Schlüssel
        )

      

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

registry_handler.set_registry_value_global(registry_path, registry_ErgebnisText, "API-KEY Korrekt")
registry_handler.set_registry_value_global(registry_path, registry_Ergebnis, "0")






class GesichtserkennungApp:
    def __init__(self, master):
        registry_path = r"SOFTWARE\Tanoffice\facescan"
        registry_status = "Zwischenstatus"
        registry_function_name = "ErgebnisText"
        registry_function_value = "Funktion"

        self.master = master
        master.title("Gesichtserkennung")
        self.master.geometry("800x300")
        
        custom_font = font.Font(family="Arial", size=12)

       
        if not self.check_webcam():
            registry_path = r"SOFTWARE\Tanoffice\facescan"
            registry_function_name = "ErgebnisText"


            self.set_registry_value_sz(winreg.HKEY_CURRENT_USER, registry_path, registry_function_name, "Webcam nicht gefunden")
            print("Webcam nicht gefunden. Das Programm wird beendet.")
            sys.exit(1)

        

        # Dlib-Modelle laden
        try:
            self.dlib_face_detector = dlib.get_frontal_face_detector()
            self.dlib_face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
            self.dlib_shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        except Exception as e:
            self.set_registry_value(winreg.HKEY_CURRENT_USER, registry_path, registry_function_name, f"Fehler beim Laden der Dlib-Modelle: {e}")
            self.set_registry_value(winreg.HKEY_CURRENT_USER, registry_path, registry_status, "Fertig")
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
        self.deepface_ordner = self.lese_registry_wert(r"SOFTWARE\Tanoffice\facescan", "FotoPfad")
        self.frame = None  # Zum Speichern des letzten Frames
        
        self.running_thread = None  # Thread für die Aktualisierung des Registry-Werts

        self.master.iconify()

        # Starten des Hintergrund-Threads für die Registry-Aktualisierung
        self.start_registry_thread()

    


    def check_webcam(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():

            registry_path = r"SOFTWARE\Tanoffice\facescan"
            registry_function_name = "ErgebnisText"

            self.set_registry_value_sz(winreg.HKEY_CURRENT_USER, registry_path, registry_function_name, "Webcam nicht gefunden")
            print("Fehler", "Webcam nicht gefunden")
            return False

        else:
            print("Webcam gefunden")
            cap.release()
            return True


    def check_webcam_still_alive(self):

        registry_path = r"SOFTWARE\Tanoffice\facescan"
        registry_function_name = "ErgebnisText"


        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.set_registry_value_sz(winreg.HKEY_CURRENT_USER, registry_path, registry_function_name, "Webcam wurde getrennt")
            print("Webcam wurde getrennt.")
            return False
        cap.release()
        return True

        
    def monitor_webcam(self):
        while self.running_thread:  # Überprüfe ein Flag statt einer Endlosschleife
            if not self.check_webcam_still_alive():
                print("Webcam wurde getrennt. Programm wird beendet.")
                self.beenden()
                return
            time.sleep(2)

    def set_registry_value_sz(self, root, path, name, value):
        try:
            # Öffne den Registry-Schlüssel (oder erstelle ihn, falls er nicht existiert)
            key = winreg.CreateKey(root, path)
            # Setze den Wert
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)
            # Schlüssel schließen
            winreg.CloseKey(key)
        except Exception as e:
            print(f"Fehler beim Schreiben in die Registry (verfummelt): {e}")


    # Funktion zum Minimieren des Fensters
    def minimize_window():
        hwnd = win32gui.GetForegroundWindow()
        win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)

    # Fenster beim Start minimieren
    if hasattr(sys, 'frozen'):
        minimize_window()

    def set_registry_value(self, root, path, name, value):
        try:
            # Öffne den Registry-Schlüssel (oder erstelle ihn, falls er nicht existiert)
            key = winreg.CreateKey(root, path)
            # Setze den Wert
            winreg.SetValueEx(key, name, 0, winreg.REG_DWORD, value)
            # Schlüssel schließen
            winreg.CloseKey(key)
        except Exception as e:
            print(f"Fehler beim Schreiben in die Registry (verfummelt): {e}")

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
        """
        Scannt dauerhaft die Registry auf Änderungen und reagiert entsprechend.
        """
        registry_path = r"SOFTWARE\Tanoffice\facescan"
        registry_value_name = "IsRunning"
        registry_function_name = "Funktion"

        try:
            # Setzt den Registry-Wert 'IsRunning' auf 1, um anzuzeigen, dass das Programm läuft
            self.set_registry_value(winreg.HKEY_CURRENT_USER, registry_path, registry_value_name, 1)
            print("Registry-Wert 'IsRunning' gesetzt")

            while True:
                try:
                    # Liest den aktuellen Wert des 'Funktion'-Eintrags aus der Registry
                    current_value = self.get_registry_value(registry_path, registry_function_name)

                    if current_value == 1:
                        print("Funktion 1 erkannt: zeige_webcam_fuer_upruefung")
                        self.zeige_webcam_fuer_upruefung()
                        self.set_registry_value(winreg.HKEY_CURRENT_USER, registry_path, registry_function_name, 0)

                    elif current_value == 2:
                        print("Funktion 2 erkannt: zeige_webcam_fuer_neues_kundenbild")
                        self.zeige_webcam_fuer_neues_kundenbild()
                        self.set_registry_value(winreg.HKEY_CURRENT_USER, registry_path, registry_function_name, 0)

                    elif current_value == 3:
                        print("Funktion 3 erkannt: loesche_kundendaten")
                        self.loesche_kundendaten()
                        self.set_registry_value(winreg.HKEY_CURRENT_USER, registry_path, registry_function_name, 0)

                    elif current_value == 4:
                        print("Funktion 4 erkannt: Abbruch")
                        self.abbruch()
                        self.set_registry_value(winreg.HKEY_CURRENT_USER, registry_path, registry_function_name, 0)
                        break  # Schleife beenden, wenn Funktion 4 gesetzt ist

                    elif current_value == 5:
                        print("Funktion 5 erkannt: Beenden")
                        self.beenden()
                        self.set_registry_value(winreg.HKEY_CURRENT_USER, registry_path, registry_function_name, 0)
                        break  # Schleife beenden, wenn Funktion 4 gesetzt ist

                    # Eine kurze Pause, um die CPU-Auslastung zu minimieren
                    time.sleep(0.5)

                except Exception as inner_exception:
                    print(f"Fehler beim Auslesen der Registry: {inner_exception}")
                    time.sleep(1)

        except Exception as outer_exception:
            print(f"Fehler beim Setzen des Registry-Werts: {outer_exception}")
            time.sleep(1)




    def beenden(self):

        registry_path = r"SOFTWARE\Tanoffice\facescan"
        registry_value_name = "Funktion"
   

        """Setzt den Wert von 'IsRunning' auf False, wenn die Anwendung geschlossen wird."""
        try:
            self.set_registry_value(winreg.HKEY_CURRENT_USER, registry_path, registry_value_name, 5)
        except Exception as e:
            print(f"Fehler beim Setzen des Registry-Werts bei Beenden: {e}")
        self.master.quit()  # Beendet das Tkinter-Fenster



    def abbruch(self):
        registry_path = r"SOFTWARE\Tanoffice\facescan"
        registry_value_name = "Funktion"
        registry_ErgebnisText = "ErgebnisText"

        """Setzt den Wert von 'IsRunning' auf False und schließt nur das Fenster."""
        try:
            self.set_registry_value(winreg.HKEY_CURRENT_USER, registry_path, registry_value_name, 4)
            
        except Exception as e:
            print(f"Fehler beim Setzen des Registry-Werts bei Beenden: {e}")
        
        # Schließt nur das aktuelle Fenster, aber das Programm läuft weiter.
        self.master.destroy() 


   

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

        registry_path = r"SOFTWARE\Tanoffice\facescan"
        registry_function_name = "ErgebnisText"
        registry_kunde = "ErkannterKunde"
        registry_status = "Zwischenstatus"

        name = self.lese_registry_wert(registry_path, registry_kunde)

        # Benutzer nach dem Kundennamen fragen
        if not name:
            return

        # Bilddatei löschen
        bild_pfad = os.path.join(self.deepface_ordner, f"{name}.jpg")
        if os.path.exists(bild_pfad):
            os.remove(bild_pfad)
            print(f"Bild '{name}.jpg' wurde gelöscht.")
        else:
            self.set_registry_value(winreg.HKEY_CURRENT_USER, registry_path, registry_function_name, f"Bild für '{name}' wurde nicht gefunden.")
            self.set_registry_value(winreg.HKEY_CURRENT_USER, registry_path, registry_status, "Fertig")

        # Vektor in Pinecone löschen
        try:
            index = pc.Index("face-recognition-index")
            index.delete(ids=[name])
            print(f"Vektor für {name} erfolgreich aus Pinecone entfernt.")
            self.set_registry_value(winreg.HKEY_CURRENT_USER, registry_path, registry_function_name, f"Kundendaten für '{name}' wurden erfolgreich gelöscht.")
            self.set_registry_value(winreg.HKEY_CURRENT_USER, registry_path, registry_status, "Fertig")
        except Exception as e:
            self.set_registry_value(winreg.HKEY_CURRENT_USER, registry_path, registry_function_name, f"Fehler beim Löschen des Vektors in Pinecone: {e}")
            self.set_registry_value(winreg.HKEY_CURRENT_USER, registry_path, registry_status, "Fehlgeschlagen")
            
            

        self.master.iconify()


    def zeige_webcam_fuer_upruefung(self):

        

        registry_path = r"SOFTWARE\Tanoffice\facescan"
        registry_value_name = "Funktion"
        registry_function_name = "ErgebnisText"
        
        registry_status = "Zwischenstatus"


        self.set_registry_value(winreg.HKEY_CURRENT_USER, registry_path, registry_value_name, 1)

        """Funktion zum Starten der Webcam für die Nutzerüberprüfung."""
        self.video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.video_capture.isOpened():
            self.set_registry_value(winreg.HKEY_CURRENT_USER, registry_path, registry_function_name, "Webcam konnte nicht geöffnet werden")
            self.set_registry_value(winreg.HKEY_CURRENT_USER, registry_path, registry_status, "Fehlgeschlagen")
            return

        

        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break
            
            #Bildhöhe und Breite
            height, width, channels = frame.shape

            #Weise Fläche am unteren Rand hinzufügen
            white_stripe_heigtht = 50
            new_frame = np.vstack([frame, np.ones((white_stripe_heigtht, width, 3), dtype=np.uint8) * 255])

            #Anzeige der Anweisungen auf der weisen Fläche
            cv2.putText(new_frame, "<Esc> Beenden                             <Enter> Vergleichen", (10, height + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            
            #OpenCV-Bild anzeigen
            cv2.imshow('Webcam', new_frame)



        
            current_value = self.get_registry_value(registry_path, registry_value_name)
            if current_value == 4:
                break
           


            # Tastatureingaben verarbeiten
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter-Taste
                self.vergleiche_gesicht_mit_pinecone(frame)
                break
            elif key == 27:  # Escape-Taste
                break

        self.video_capture.release()
        cv2.destroyAllWindows()

        self.master.iconify()
    

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

        registry_path = r"SOFTWARE\Tanoffice\facescan"
        registry_function = "Score"

        str_Uebereinstimmung = self.get_registry_value(registry_path, registry_function)
        Uebereinstimmung= float(str_Uebereinstimmung)

        index = pc.Index("face-recognition-index")

        embedding = self.berechne_embedding(frame)
        if embedding is not None:
            # Pinecone-Abfrage
            result = index.query(vector=embedding.tolist(), top_k=1, include_values=True, approximate=True)
            if result["matches"]:
                best_match = result["matches"][0]
                erkannter_kunde = best_match["id"]
                score = best_match["score"]

                print(score)

                # Überprüfung auf 100 % Übereinstimmung
                if score >= Uebereinstimmung:  # Score für perfekte Übereinstimmung
                    self.schreibe_registry(erkannter_kunde, "0", f"Kunde erkannt mit perfekter Übereinstimmung", "Fertig")
                else:
                    self.schreibe_registry("Fehler", "1", f"Keine perfekte Übereinstimmung. Score: {score:.2f}", "Fertig")
            else:
                self.schreibe_registry("Fehler", "1", "Kein Kunde erkannt", "Fertig")
        else:
            self.schreibe_registry("Fehler", "1", "Kein Gesicht erkannt", "Fertig")
            

    def vergleiche_gesicht_mit_alle_kundenbilder(self, frame):
        bilder = self.lade_alle_kundenbilder()
        if bilder:
            print(f"Vergleiche mit {len(bilder)} Kundenbildern")
            self.vergleiche_gesicht_mit_pinecone(frame)
        else:
            self.schreibe_registry("Fehler", "1", "Keine Kundenbilder gefunden.", "Fertig")
            
    

    def zeige_webcam_fuer_neues_kundenbild(self):


        registry_path = r"SOFTWARE\Tanoffice\facescan"
        registry_value_name = "Funktion"

        
        self.video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.video_capture.isOpened():
            self.schreibe_registry("Fehler", "1", "Kamera konnte nicht geöffnet werden.", "Fertig")
            return
        
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            #Bildgröße und Höhe anpassen
            height, width, channels = frame.shape

            #Weise Fläche
            white_stripe_height = 50
            new_frame = np.vstack([frame, np.ones((white_stripe_height, width, 3), dtype=np.uint8) * 255])

             #Anzeige der Anweisungen auf der weisen Fläche
            cv2.putText(new_frame, "<Esc> Beenden                             <Enter> Speichern", (20, height + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            
            # OpenCV Fenster anzeigen
            cv2.imshow('Webcam', new_frame)
            
            current_value = self.get_registry_value(registry_path, registry_value_name)
            if current_value == 4:
                break


           # Tastatureingaben verarbeiten
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter-Taste
                self.speichere_neues_kundenbild(frame)
                break
            elif key == 27:  # Escape-Taste
                break

        self.video_capture.release()
        cv2.destroyAllWindows()
        self.master.iconify()


    def speichere_neues_kundenbild(self, frame):
            registry_path = r"SOFTWARE\Tanoffice\facescan"
            registry_function = "ErkannterKunde"

            # Holen des Kundennamens aus der Registry
            name = self.get_registry_value(registry_path, registry_function)
            if not name:
                print("Kein erkannter Kunde in der Registry.")
                return

            # Speichern des Kundenbildes
            dateiname = os.path.join(self.deepface_ordner, f"{name}.jpg")
            cv2.imwrite(dateiname, frame)
            print(f"Kundenbild für {name} wurde erfolgreich gespeichert.")
            
            # Optional: Indexieren des Kundenbildes in Pinecone, falls gewünscht
            self.index_kundenbild_in_pinecone(name, frame)


    def index_kundenbild_in_pinecone(self, name, frame):
        index = pc.Index("face-recognition-index")
        embedding = self.berechne_embedding(frame)
        if embedding is not None:
            index.upsert([(name, embedding.tolist())])
            print(f"Vektor für {name} erfolgreich in Pinecone gespeichert.")
            self.schreibe_registry(name, "0", f"Kundenbild für '{name}' erfolgreich gespeichert.", "Fertig")
        else:
            self.schreibe_registry(name, "1", f"Fehler beim Berechnen des Embeddings für '{name}'", "Fehlgeschlagen")
            

# Tkinter Hauptprogramm starten
root = tk.Tk()
app = GesichtserkennungApp(root)

 # Fenster immer im Vordergrund
root.wm_attributes("-topmost", True)

    # Fokus setzen
root.focus_force()



root.mainloop()


