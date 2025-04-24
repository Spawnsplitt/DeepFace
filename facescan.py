from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import os
import dlib
import numpy as np 
from pinecone import Pinecone, ServerlessSpec
import tkinter as tk
from tkinter import font
import winreg
import warnings
import ctypes
import sys
import time
import threading
import win32gui
import win32con
from datetime import datetime


REGISTRY_PATH = r"SOFTWARE\Tanoffice\facescan"
REGISTRY_SET_FUNCTION = "Funktion"
REGISTRY_FUNCTION_RESULT = "Ergebnis"
REGISTRY_FUNCTION_RESULT_TEXT = "ErgebnisText"
REGISTRY_STATUS = "Zwischenstatus"
REGISTRY_LIFE_STATUS = "IsRunning"
REGISTRY_KNOWN_CUSTOMER = "ErkannterKunde"
REGISTRY_SCORE = "Score"
REGISTRY_PICTURE = "FotoPfad"
REGISTRY_AKTUELLER_KUNDE = "AktuellerKunde"


class RegistryHandler:
    #Funktion zum Schreiben und Lesen von Registrierungswerten
    def registry_access(self, action, path, name=None, value=None, reg_type=None):
        try:
            if action == "set":
                # Schlüssel erstellen oder öffnen
                key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, path)
                
                if isinstance(value, dict):  # Mehrere Werte setzen
                    for k, v in value.items():
                        v_type = winreg.REG_DWORD if isinstance(v, int) else winreg.REG_SZ
                        winreg.SetValueEx(key, k, 0, v_type, v)
                else:  # Einzelnen Wert setzen
                    reg_type = reg_type or (winreg.REG_DWORD if isinstance(value, int) else winreg.REG_SZ)
                    winreg.SetValueEx(key, name, 0, reg_type, value)

                winreg.CloseKey(key)
                return True

            elif action == "get":
                # Schlüssel öffnen
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, path, 0, winreg.KEY_READ)
                reg_value, _ = winreg.QueryValueEx(key, name)
                winreg.CloseKey(key)
                return reg_value

        except FileNotFoundError:
            print(f"Registry-Pfad '{path}' oder Wert '{name}' nicht gefunden.")
            return None
        except Exception as e:
            print(f"Registry-Fehler: {e}")
            return None
        
    

#Klasse RegistryHandler() wird funktional gemacht
registry_handler = RegistryHandler()




# Registry-Wert abrufen
registry_handler.registry_access("set", REGISTRY_PATH, REGISTRY_FUNCTION_RESULT_TEXT, "Fehler: Problem mit API-Schlüssel", winreg.REG_SZ)
#print(f"schreibe Registry 1 : {result}")
result = registry_handler.registry_access("get", REGISTRY_PATH, REGISTRY_FUNCTION_RESULT, "1", winreg.REG_SZ)
print(f"schreibe Registry 2 : {result}") 


key = registry_handler.registry_access("get", REGISTRY_PATH, "API_KEY", winreg.REG_SZ)

database_name = registry_handler.registry_access("get", REGISTRY_PATH, "DatenbankName", winreg.REG_SZ)




        # Pinecone initialisieren
pc = Pinecone(
        api_key=key  #API-Schlüssel
        )


   
#Einbinden der Datenbank
if database_name not in pc.list_indexes().names():
            pc.create_index(
                name=database_name,
                dimension=1536,  # Setze die Dimension entsprechend den Embeddings
                metric='euclidean',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )

warnings.filterwarnings("ignore", category=DeprecationWarning)

registry_handler.registry_access("set", REGISTRY_PATH, REGISTRY_FUNCTION_RESULT_TEXT, "API-KEY Korrekt", winreg.REG_SZ)
registry_handler.registry_access("set", REGISTRY_PATH, REGISTRY_FUNCTION_RESULT, "0", winreg.REG_SZ)






class GesichtserkennungApp:
    def __init__(self, master):
        self.master = master
        master.title("Gesichtserkennung")
        self.master.geometry("800x300")
        
        custom_font = font.Font(family="Arial", size=12)

        #Überprüfung ob Webcam vorhanden ist
        if not self.check_webcam():
            self.registry_action("set", path=REGISTRY_PATH, name=REGISTRY_FUNCTION_RESULT, value="Webcam nicht gefunden", value_type=winreg.REG_SZ)
            print("Webcam nicht gefunden. Das Programm wird beendet.")
            self.log("[FEHLER] Webcam nicht gefunden")
            sys.exit(1)

        

        # Dlib-Modelle (KI-Modelle) laden
        try:
            self.dlib_face_detector = dlib.get_frontal_face_detector()
            self.dlib_face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
            self.dlib_shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        except Exception as e:
            self.registry_action("set", path=REGISTRY_PATH, name=REGISTRY_FUNCTION_RESULT, value=f"Fehler beim Laden der Dlib-Modelle: {e}", value_type=winreg.REG_SZ)
            self.registry_action("set", path=REGISTRY_PATH, name=REGISTRY_STATUS, value="Fertig", value_type=winreg.REG_SZ)
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
        self.deepface_ordner = self.registry_action("get", path=REGISTRY_PATH, name=REGISTRY_PICTURE)
        self.frame = None  # Zum Speichern des letzten Frames
        
        self.running_thread = None  # Thread für die Aktualisierung des Registry-Werts

        self.master.iconify()

        # Starten des Hintergrund-Threads für die Registry-Aktualisierung
        self.start_registry_thread()

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open("facescan.log", "a") as log_file:
            log_file.write(f"{timestamp} - {message}\n")
    

    #Funktion zum Überprüfen ob Webcam vorhanden ist
    def check_webcam(self):
        cap = cv2.VideoCapture(0)
        self.log("[INFO] Facescan gestartet")
    #Webcam nicht gefunden
        if not cap.isOpened():
            self.registry_action("set", path=REGISTRY_PATH, name=REGISTRY_FUNCTION_RESULT_TEXT, value="Webcam nicht gefunden", value_type=winreg.REG_SZ)
            print("Fehler", "Webcam nicht gefunden")
            self.log("[FEHLER] Webcam nicht gefunden")
            return False
    #Webcam gefunden
        else:
            print("Webcam gefunden")
            self.log("[INFO] Webcam gefunden")
            cap.release()
            return True


    # Funktion zum Minimieren des Fensters
    def minimize_window():
        hwnd = win32gui.GetForegroundWindow()
        win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)

    # Fenster beim Start minimieren
    if hasattr(sys, 'frozen'):
        minimize_window()

    #Funktion zum schreiben und lesen in der Windows Registry innerhalb des Main-Programm
    def registry_action(self, action, root=winreg.HKEY_CURRENT_USER, path="", name=REGISTRY_KNOWN_CUSTOMER, value=None, value_type=None):
        try:
            if action == "set":
                if value_type is None:
                    raise ValueError("value_type muss angegeben werden für 'set'")
                with winreg.CreateKey(root, path) as key:
                    winreg.SetValueEx(key, name, 0, value_type, value)

            elif action == "get":
                try:
                    with winreg.OpenKey(root, path) as key:
                        result, _ = winreg.QueryValueEx(key, name)
                        return result
                except FileNotFoundError:
                    print(f"Registry-Pfad oder Wert nicht gefunden: {path}\\{name} in Zeile {sys.exc_info()[2].tb_lineno} (Pfad: {path}, Name: {name})")
                    return None
                except Exception as e:
                    print(f"Fehler bei Registry-Operation 'get': {e} in Zeile {sys.exc_info()[2].tb_lineno} (Pfad: {path}, Name: {name})")
                    return None

            elif action == "set_multiple":
                if not isinstance(value, dict):
                    raise ValueError("Für 'set_multiple' muss 'value' ein Dictionary sein.")
                with winreg.CreateKey(root, path) as key:
                    for key_name, (val, v_type) in value.items():
                        winreg.SetValueEx(key, key_name, 0, v_type, val)

            else:
                raise ValueError("Ungültige Aktion. Verwende 'set', 'get' oder 'set_multiple'.")

        except Exception as e:
            print(f"Allgemeiner Fehler bei Registry-Operation '{action}': {e} in Zeile {sys.exc_info()[2].tb_lineno} (Pfad: {path}, Name: {name})")
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
        try:
            # Setzt den Registry-Wert 'IsRunning' auf 1, um anzuzeigen, dass das Programm läuft

            self.registry_action("set", path=REGISTRY_PATH, name=REGISTRY_LIFE_STATUS, value=1, value_type=winreg.REG_DWORD)
            print("Registry-Wert 'IsRunning' gesetzt")

            while True:
                try:
                    # Liest den aktuellen Wert des 'Funktion'-Eintrags aus der Registry
                    #current_value = self.get_registry_value(REGISTRY_PATH, REGISTRY_SET_FUNCTION)
                    current_value = self.registry_action("get", path=REGISTRY_PATH, name=REGISTRY_SET_FUNCTION)

                    if current_value == 1:
                        print("Funktion 1 erkannt: zeige_webcam_fuer_upruefung")
                        self.log("[INFO] Funktion 1 erkannt: zeige_webcam_fuer_upruefung")
                        self.zeige_webcam_fuer_upruefung()
                        self.registry_action("set", path=REGISTRY_PATH, name=REGISTRY_SET_FUNCTION, value=0, value_type=winreg.REG_DWORD)

                    elif current_value == 2:
                        print("Funktion 2 erkannt: zeige_webcam_fuer_neues_kundenbild")
                        self.log("[INFO] Funktion 2 erkannt: zeige_webcam_fuer_neues_kundenbild")
                        self.zeige_webcam_fuer_neues_kundenbild()
                        self.registry_action("set", path=REGISTRY_PATH, name=REGISTRY_SET_FUNCTION, value=0, value_type=winreg.REG_DWORD)

                    elif current_value == 3:
                        print("Funktion 3 erkannt: loesche_kundendaten")
                        self.log("[INFO] Funktion 3 erkannt: loesche_kundendaten")
                        self.loesche_kundendaten()
                        self.registry_action("set", path=REGISTRY_PATH, name=REGISTRY_SET_FUNCTION, value=0, value_type=winreg.REG_DWORD)

                    elif current_value == 4:
                        print("Funktion 4 erkannt: Abbruch")
                        self.log("[INFO] Funktion 4 erkannt: Abbruch")
                        self.abbruch()
                        self.registry_action("set", path=REGISTRY_PATH, name=REGISTRY_SET_FUNCTION, value=0, value_type=winreg.REG_DWORD)
                        break  # Schleife beenden, wenn Funktion 4 gesetzt ist

                    elif current_value == 5:
                        print("Funktion 5 erkannt: Beenden")
                        self.log("[INFO] Funktion 5 erkannt: Beenden")
                        self.registry_action("set", path=REGISTRY_PATH, name=REGISTRY_SET_FUNCTION, value=0, value_type=winreg.REG_DWORD)
                        self.beenden()
                        
                        break  # Schleife beenden, wenn Funktion 4 gesetzt ist

                    # Eine kurze Pause, um die CPU-Auslastung zu minimieren
                    time.sleep(0.5)

                except Exception as inner_exception:
                    print(f"Fehler beim Auslesen der Registry: {inner_exception}")
                    self.log(f"[ERROR] Fehler beim Auslesen der Registry: {inner_exception}")
                    time.sleep(1)

        except Exception as outer_exception:
            print(f"Fehler beim Setzen des Registry-Werts: {outer_exception}")
            self.log(f"[ERROR] Fehler beim Setzen des Registry-Werts: {outer_exception}")
            time.sleep(1)

    

    #Funktion zum beenden der gesamten Anwendung
    def beenden(self):
        """Setzt den Wert von 'IsRunning' auf False, wenn die Anwendung geschlossen wird."""
        try:
            self.registry_action("set", path=REGISTRY_PATH, name=REGISTRY_SET_FUNCTION, value=5, value_type=winreg.REG_DWORD)
            self.registry_action("set", path=REGISTRY_PATH, name=REGISTRY_LIFE_STATUS, value=0, value_type=winreg.REG_DWORD)
            self.registry_action("set", path=REGISTRY_PATH, name=REGISTRY_FUNCTION_RESULT_TEXT, value="Programm wurde beendet", value_type=winreg.REG_SZ)
            self.registry_action("set", path=REGISTRY_PATH, name=REGISTRY_SET_FUNCTION, value=0, value_type=winreg.REG_DWORD)
        except Exception as e:
            print(f"Fehler beim Setzen des Registry-Werts bei Beenden: {e}")
            self.log(f"[ERROR] Fehler beim Setzen des Registry-Werts bei Beenden: {e}")
        self.master.quit()  # Beendet das Tkinter-Fenster


    #Funktion zum abbrechen innerhalb einer Funktion
    def abbruch(self):
        """Setzt den Wert von 'Funktion' auf 4 und schließt nur das OpenCV-Fenster."""
        try:
            result = self.registry_action("get", path=REGISTRY_PATH, name=REGISTRY_SET_FUNCTION)
            self.registry_action("set", path=REGISTRY_PATH, name=REGISTRY_SET_FUNCTION, value=4, value_type=winreg.REG_DWORD)
            self.registry_action("set", path=REGISTRY_PATH, name=REGISTRY_SET_FUNCTION, value=0, value_type=winreg.REG_DWORD)
            self.registry_action("set", path=REGISTRY_PATH, name=REGISTRY_FUNCTION_RESULT_TEXT, value=f"Funktion {result} wurde abgebrochen", value_type=winreg.REG_SZ)
        except Exception as e:
            print(f"Fehler beim Setzen des Registry-Werts beim Abbrechen: {e}")
            self.log(f"[ERROR] Fehler beim Setzen des Registry-Werts beim Abbrechen: {e}")

        # Schließt nur das OpenCV-Fenster mit dem Namen 'Webcam'
        cv2.destroyWindow('Webcam') 



    # Funktion zum Löschen von Kundendaten
    def loesche_kundendaten(self):
        if self.registry_action("get", path=REGISTRY_PATH, name=REGISTRY_AKTUELLER_KUNDE) == "":
            nameLoeschKunde = self.registry_action("get", path=REGISTRY_PATH, name=REGISTRY_KNOWN_CUSTOMER)
        else:
            nameLoeschKunde = self.registry_action("get", path=REGISTRY_PATH, name=REGISTRY_AKTUELLER_KUNDE)

        # Benutzer nach dem Kundennamen fragen
        if not nameLoeschKunde:
            return

        # Vektor in Pinecone löschen
        try:
            index = pc.Index(database_name)
            index.delete(ids=[nameLoeschKunde])
            print(f"Vektor für {nameLoeschKunde} erfolgreich aus Pinecone entfernt.")
            self.log(f"[INFO] Vektor für {nameLoeschKunde} erfolgreich aus Pinecone entfernt.")
            self.registry_action("set", path=REGISTRY_PATH, name=REGISTRY_FUNCTION_RESULT_TEXT, value=f"Kundendaten für '{nameLoeschKunde}' wurden erfolgreich gelöscht.", value_type=winreg.REG_SZ)
            self.registry_action("set", path=REGISTRY_PATH, name=REGISTRY_STATUS, value="Fertig", value_type=winreg.REG_SZ)
        except Exception as e:
            self.registry_action("set", path=REGISTRY_PATH, name=REGISTRY_FUNCTION_RESULT_TEXT, value=f"Fehler beim Löschen des Vektors in Pinecone: {e}", value_type=winreg.REG_SZ)
            self.registry_action("set", path=REGISTRY_PATH, name=REGISTRY_STATUS, value="Fehlgeschlagen", value_type=winreg.REG_SZ)

            
            

        self.master.iconify()

    #Webcamfenster anzeigen lassen für Überprüfung
    def zeige_webcam_fuer_upruefung(self):
        # Registry-Wert setzen
        self.registry_action("set", path=REGISTRY_PATH, name=REGISTRY_SET_FUNCTION, value=1, value_type=winreg.REG_DWORD)

        time.sleep(0.5)

        """Funktion zum Starten der Webcam für die Nutzerüberprüfung."""
        self.video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.video_capture.isOpened():
            self.registry_action("set", path=REGISTRY_PATH, name=REGISTRY_FUNCTION_RESULT_TEXT, value="Webcam konnte nicht geöffnet werden", value_type=winreg.REG_SZ)
            self.log(f"[ERROR] Webcam konnte nicht geöffnet werden")
            self.registry_action("set", path=REGISTRY_PATH, name=REGISTRY_STATUS, value="Fehlgeschlagen", value_type=winreg.REG_SZ)
            return

        # Bildschirmauflösung abrufen
        user32 = ctypes.windll.user32
        screen_width = user32.GetSystemMetrics(0)  # Bildschirmbreite
        screen_height = user32.GetSystemMetrics(1)  # Bildschirmhöhe

        # Dlib Gesichtserkennungsmodell initialisieren
        detector = dlib.get_frontal_face_detector()

        # OpenCV-Fenster vorbereiten
        cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Webcam', cv2.WND_PROP_TOPMOST, 1)  # Fenster bleibt im Vordergrund

        while True:
            for i in range(10):  # 10 Versuche, ein gültiges Bild zu bekommen
                ret, frame = self.video_capture.read()
                if ret and frame is not None and frame.size > 0:
                    break
                print(f"Versuch {i+1}: Kein gültiges Bild erhalten...")
                self.log(f"[ERROR] Kein gültiges Bild erhalten...")
                time.sleep(0.1)  # 100ms warten
            if not ret or frame is None or frame.size == 0:
                print("Webcam-Fehler: Kein Bild abrufbar!")
                self.log("[ERROR] Webcam-Fehler: Kein Bild abrufbar!")
                return

            # Überprüfe, ob das Bild leer ist
            if frame is None or frame.size == 0:
                print("Das aufgerufene Bild ist leer!")  # Debugging-Ausgabe
                self.log("[ERROR] Das aufgerufene Bild ist leer!")
                break

            # Bildhöhe und Breite abrufen
            height, width, _ = frame.shape

            # Gesichtserkennung mit Dlib
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)  # Gesichter in Bild erkennen

            print(f"Anzahl der erkannten Gesichter: {len(faces)}")  # Debugging-Ausgabe: Anzahl der erkannten Gesichter
            self.log(f"[INFO] Anzahl der erkannten Gesichter: {len(faces)}")

            # Rechtecke um die erkannten Gesichter zeichnen
            for face in faces:
                x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Rechteck um Gesicht zeichnen

            # Weiße Leiste am unteren Rand hinzufügen
            white_stripe_height = 50
            new_frame = np.vstack([frame, np.ones((white_stripe_height, width, 3), dtype=np.uint8) * 255])

            # Anweisungen auf der weißen Fläche anzeigen
            cv2.putText(new_frame, "<Esc> Beenden                             <Enter> Vergleichen",
                        (10, height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Fenstergröße und Position berechnen
            window_width = new_frame.shape[1]
            window_height = new_frame.shape[0]
            pos_x = (screen_width - window_width) // 2
            pos_y = (screen_height - window_height) // 2

            # Fenster anpassen und zentrieren
            cv2.resizeWindow('Webcam', window_width, window_height)
            cv2.moveWindow('Webcam', pos_x, pos_y)

            # Fenster immer im Vordergrund halten und unverschiebbar machen
            hwnd = ctypes.windll.user32.FindWindowW(None, "Webcam")
            if hwnd:
                ctypes.windll.user32.SetWindowPos(hwnd, -1, pos_x, pos_y, window_width, window_height, 0x0001 | 0x0040)
                ctypes.windll.user32.SetWindowLongW(hwnd, -16, ctypes.windll.user32.GetWindowLongW(hwnd, -16) & ~0x00040000)

            # OpenCV-Bild anzeigen
            cv2.imshow('Webcam', new_frame)

            # Eingaben abfangen
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter-Taste
                # Starte Vergleich mit der Datenbank
                self.vergleiche_gesicht_mit_pinecone(frame)
                break
            # Beenden des Webcamfensters
            elif key == 27:  # Escape-Taste
                self.abbruch()
                break

            # Registry-Wert abfragen
            current_value = self.registry_action("get", path=REGISTRY_PATH, name=REGISTRY_SET_FUNCTION)
            if current_value == 4:
                self.abbruch()
                break

        self.video_capture.release()
        cv2.destroyAllWindows()
            
    #Funktion um alle Kundenbilder aus dem Deepface_Kunden Ordner zu laden
    def lade_alle_kundenbilder(self):

        
        index = pc.Index(database_name)  # Pinecone-Index initialisieren
        bilder = []
        #Funktion um Embedings aus Bildern zu berechnen und dann in Pinecone zu laden
        def prozessiere_kundenbild(pfad):
            #Versuche Bild zu laden
            try:
                bild = cv2.imread(pfad)
                if bild is None:
                    print(f"Konnte Bild nicht laden: {pfad}")
                    self.log(f"[ERROR] Konnte Bild nicht laden: {pfad}")
                    return None, None

                print(f"Lade Bild: {pfad}")

                # Embedding berechnen
                embedding = self.berechne_embedding(bild)
                if embedding is not None:
                    vector_id = os.path.splitext(os.path.basename(pfad))[0]
                    index.upsert([(vector_id, embedding)])
                    print(f"Vektor für {vector_id} erfolgreich in Pinecone gespeichert.")
                    self.log(f"[INFO] Vektor für {vector_id} erfolgreich in Pinecone gespeichert.")
                    return pfad, embedding
            except Exception as e:
                print(f"Fehler beim Verarbeiten des Bildes {pfad}: {e}")
                self.log(f"[ERROR] Fehler beim Verarbeiten des Bildes {pfad}: {e}")
            
            return None, None  # Falls ein Fehler auftritt oder kein Gesicht erkannt wird

        # Parallele Verarbeitung mit ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(prozessiere_kundenbild, os.path.join(self.deepface_ordner, datei)): datei
                for datei in os.listdir(self.deepface_ordner)
                if datei.lower().endswith(('.jpg', '.jpeg', '.png'))  # Nur Bilddateien verarbeiten
            }

            for future in as_completed(futures):
                pfad, embedding = future.result()
                if embedding is not None:
                    bilder.append(pfad)

        return bilder  # Rückgabe der erfolgreich verarbeiteten Bilder
    
    def berechne_embedding(self, bild):
            """
            Berechnet das Gesichtsembedding für ein gegebenes Bild.
            """
            det = self.dlib_face_detector(bild, 1)  # Gesichtserkennung im Bild durchführen
            if len(det) > 0:  # Falls mindestens ein Gesicht erkannt wurde
                shape = self.dlib_shape_predictor(bild, det[0])  # Gesichtsmerkmale extrahieren
                
                # Gesichtsembedding mit dlib berechnen
                embedding = np.array(self.dlib_face_recognition_model.compute_face_descriptor(bild, shape))
                
                # Falls das Embedding genau 128 Werte hat, auf 1536 skalieren (Größenanpassung)
                if embedding.size == 128:
                    embedding = np.resize(embedding, (1536,))
                
                return embedding  # Das berechnete Embedding zurückgeben
            else:
                print(f"Kein Gesicht erkannt im Bild.")  # Falls kein Gesicht erkannt wurde
                self.log(f"[ERROR] Kein Gesicht erkannt im Bild.")
                return None  # `None` zurückgeben

    #Funktion zum Vergleichen mit der Pineconedatenbank
    def vergleiche_gesicht_mit_pinecone(self, frame):
        try:
            # Wert aus der Registry lesen
            str_Uebereinstimmung = self.registry_action("get", path=REGISTRY_PATH, name=REGISTRY_SCORE)

            # Falls der Wert nicht existiert oder fehlerhaft ist, Standardwert setzen
            try:
                Uebereinstimmung = float(str_Uebereinstimmung) if str_Uebereinstimmung is not None else 0.9
            except ValueError:
                print(f"Fehler: REGISTRY_SCORE enthält keinen gültigen Wert ({str_Uebereinstimmung}). Standardwert 0.9 wird genutzt.")
                self.log(f"[ERROR] REGISTRY_SCORE enthält keinen gültigen Wert ({str_Uebereinstimmung}). Standardwert 0.9 wird genutzt.")
                Uebereinstimmung = 0.9

            # Pinecone-Index initialisieren
            try:
                index = pc.Index(database_name)
            except Exception as e:
                print(f"Fehler beim Initialisieren des Pinecone-Index: {e}")
                self.log(f"[ERROR] Fehler beim Initialisieren des Pinecone-Index: {e}")
                self.registry_action("set_multiple", path=REGISTRY_PATH, value={
                    REGISTRY_KNOWN_CUSTOMER: ("Fehler", winreg.REG_SZ),
                    REGISTRY_FUNCTION_RESULT: ("1", winreg.REG_SZ),
                    REGISTRY_FUNCTION_RESULT_TEXT: ("Pinecone-Index konnte nicht geladen werden",  winreg.REG_SZ),
                    REGISTRY_STATUS: ("Fertig", winreg.REG_SZ)
                })
                return

            # Gesichtsembedding berechnen
            embedding = self.berechne_embedding(frame)
            if embedding is None:
                self.registry_action("set_multiple", path=REGISTRY_PATH, value={
                    REGISTRY_KNOWN_CUSTOMER: ("Fehler", winreg.REG_SZ),
                    REGISTRY_FUNCTION_RESULT: ("1", winreg.REG_SZ),
                    REGISTRY_FUNCTION_RESULT_TEXT: ("Kein Gesicht erkannt", winreg.REG_SZ),
                    REGISTRY_STATUS: ("Fertig", winreg.REG_SZ)
                })
                return

            # Pinecone-Abfrage
            result = index.query(vector=embedding.tolist(), top_k=1, include_values=True, approximate=True)

            if result["matches"]:
                best_match = result["matches"][0]
                erkannter_kunde = best_match["id"]
                score = best_match["score"]

                print(f"Gefundene Übereinstimmung: {erkannter_kunde}, Score: {score:.2f}")
                self.log(f"Gefundene Übereinstimmung: {erkannter_kunde}, Score: {score:.2f}")

                # Vergleich mit der minimalen Übereinstimmung aus der Registry
                if score >= Uebereinstimmung:
                    self.registry_action("set_multiple", path=REGISTRY_PATH, value={
                        REGISTRY_KNOWN_CUSTOMER: (erkannter_kunde, winreg.REG_SZ) ,
                        REGISTRY_FUNCTION_RESULT: ("0", winreg.REG_SZ),
                        REGISTRY_FUNCTION_RESULT_TEXT: ("Kunde erkannt mit hoher Übereinstimmung", winreg.REG_SZ),
                        REGISTRY_STATUS: ("Fertig", winreg.REG_SZ)
                    })
                else:
                    self.registry_action("set_multiple", path=REGISTRY_PATH, value={
                        REGISTRY_KNOWN_CUSTOMER: ("Fehler", winreg.REG_SZ),
                        REGISTRY_FUNCTION_RESULT: ("1", winreg.REG_SZ),
                        REGISTRY_FUNCTION_RESULT_TEXT: (f"Keine perfekte Übereinstimmung. Score: {score:.2f}", winreg.REG_SZ),
                        REGISTRY_STATUS: ("Fertig", winreg.REG_SZ)
                    })
            else:
                self.registry_action("set_multiple", path=REGISTRY_PATH, value={
                    REGISTRY_KNOWN_CUSTOMER: ("Fehler", winreg.REG_SZ),
                    REGISTRY_FUNCTION_RESULT: ("1", winreg.REG_SZ),
                    REGISTRY_FUNCTION_RESULT_TEXT: ("Kein Kunde erkannt", winreg.REG_SZ),
                    REGISTRY_STATUS: ("Fertig", winreg.REG_SZ)
                })

        except Exception as e:
            print(f"Unerwarteter Fehler in vergleiche_gesicht_mit_pinecone: {e}")
            self.registry_action("set_multiple", path=REGISTRY_PATH, value={
                REGISTRY_KNOWN_CUSTOMER: ("Fehler", winreg.REG_SZ),
                REGISTRY_FUNCTION_RESULT: ("1", winreg.REG_SZ),
                REGISTRY_FUNCTION_RESULT_TEXT: (f"Fehler: {str(e)}", winreg.REG_SZ),
                REGISTRY_STATUS: ("Fertig", winreg.REG_SZ)
            })
            
    #Vergleich mit den Bildern im DEEPFACE_KUNDEN Ordner
    def vergleiche_gesicht_mit_alle_kundenbilder(self, frame):
        bilder = self.lade_alle_kundenbilder()
        if bilder:
            print(f"Vergleiche mit {len(bilder)} Kundenbildern")
            self.log(f"Vergleiche mit {len(bilder)} Kundenbildern")
            self.vergleiche_gesicht_mit_pinecone(frame)
        else:
            self.registry_action("set_multiple", path=REGISTRY_PATH, value={
                REGISTRY_KNOWN_CUSTOMER: ("Fehler", winreg.REG_SZ) ,
                REGISTRY_FUNCTION_RESULT: ("1", winreg.REG_SZ),
                REGISTRY_FUNCTION_RESULT_TEXT: ("Keine Kundenbilder gefunden", winreg.REG_SZ),
                REGISTRY_STATUS: ("Fertig", winreg.REG_SZ)
            })
            
    

    def zeige_webcam_fuer_neues_kundenbild(self):
        # Kamera öffnen
        self.video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.video_capture.isOpened():
            self.registry_action("set_multiple", path=REGISTRY_PATH, value={
                REGISTRY_KNOWN_CUSTOMER: ("Fehler", winreg.REG_SZ),
                REGISTRY_FUNCTION_RESULT: ("1", winreg.REG_SZ),
                REGISTRY_FUNCTION_RESULT_TEXT: ("Kamera konnte nicht geöffnet werden", winreg.REG_SZ),
                REGISTRY_STATUS: ("Fertig", winreg.REG_SZ)
            })
            return

        # Bildschirmdimensionen abrufen
        user32 = ctypes.windll.user32
        screen_width = user32.GetSystemMetrics(0)
        screen_height = user32.GetSystemMetrics(1)

        # dlib Gesichtserkennungs-Modell (HOG-Detektor) initialisieren
        detector = dlib.get_frontal_face_detector()

        # OpenCV-Fenster erstellen
        cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Webcam', cv2.WND_PROP_TOPMOST, 1)

        # Trackbars für Helligkeit & Weichzeichnen (anstelle von Rauschunterdrückung)
        def update(_):
            pass

        cv2.createTrackbar("Helligkeit", "Webcam", 10, 30, update)  # Faktor 1.0 - 3.0
        cv2.createTrackbar("Weichzeichnen", "Webcam", 0, 10, update)  # 0 = Kein Blur, 10 = Max

        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            # Werte aus Trackbars holen
            helligkeit = cv2.getTrackbarPos("Helligkeit", "Webcam") / 10.0  # Faktor 1.0 - 3.0
            blur_level = cv2.getTrackbarPos("Weichzeichnen", "Webcam") * 2 + 1  # Immer ungerade Werte für Blur

            # Helligkeit anpassen
            frame = cv2.convertScaleAbs(frame, alpha=helligkeit, beta=0)

            # Weichzeichnen für Live-Vorschau
            if blur_level > 1:
                frame = cv2.GaussianBlur(frame, (blur_level, blur_level), 0)

            # Gesichter erkennen (dlib)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray_frame)

            # Jedes erkannte Gesicht umranden
            for face in faces:
                # Rechteckkoordinaten
                x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Grün für die Umrandung

            # UI-Bereich unten hinzufügen
            height, width, _ = frame.shape
            white_stripe_height = 50
            new_frame = np.vstack([frame, np.ones((white_stripe_height, width, 3), dtype=np.uint8) * 255])

            # Anweisungen anzeigen
            cv2.putText(new_frame, "<Esc> Beenden     <Enter> Speichern", (20, height + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Fenster zentrieren
            pos_x = (screen_width - width) // 2
            pos_y = (screen_height - (height + white_stripe_height)) // 2
            cv2.moveWindow('Webcam', pos_x, pos_y)

            # Bild anzeigen
            cv2.imshow('Webcam', new_frame)

            # Registry-Wert abfragen
            current_value = self.registry_action("get", path=REGISTRY_PATH, value=REGISTRY_SET_FUNCTION)
            if current_value == 4:
                self.abbruch()
                break

            # Tastenaktionen
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter-Taste
                # Beim Speichern echte Rauschunterdrückung anwenden (nur hier!)
                frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
                self.speichere_neues_kundenbild(frame)
                break
            elif key == 27:  # Escape-Taste
                self.abbruch()
                break

        self.video_capture.release()
        cv2.destroyAllWindows()
        self.master.iconify()


    def speichere_neues_kundenbild(self, frame):
            # Holen des Kundennamens aus der Registry
            name = self.registry_action("get", path=REGISTRY_PATH, value=REGISTRY_KNOWN_CUSTOMER)
            if not name:
                print("Kein erkannter Kunde in der Registry.")
                self.log("[ERROR] Kein erkannter Kunde in der Registry.")
                return

            # Speichern des Kundenbildes
            dateiname = os.path.join(self.deepface_ordner, f"{name}.jpg")
            cv2.imwrite(dateiname, frame)
            print(f"Kundenbild für {name} wurde erfolgreich gespeichert.")
            self.log(f"Kundenbild für {name} wurde erfolgreich gespeichert.")
            
            # Optional: Indexieren des Kundenbildes in Pinecone, falls gewünscht
            self.index_kundenbild_in_pinecone(name, frame)

    #Funktion zum Erstellen eines neuen Index für neue Kundenfotos
    def index_kundenbild_in_pinecone(self, name, frame):
        index = pc.Index(database_name)
        embedding = self.berechne_embedding(frame)
        if embedding is not None:
            index.upsert([(name, embedding.tolist())])
            print(f"Vektor für {name} erfolgreich in Pinecone gespeichert.")
            self.log(f"Vektor für {name} erfolgreich in Pinecone gespeichert.")
            self.registry_action("set_multiple", path=REGISTRY_PATH, value={
                REGISTRY_KNOWN_CUSTOMER: (name, winreg.REG_SZ),
                REGISTRY_FUNCTION_RESULT: ("0", winreg.REG_SZ),
                REGISTRY_FUNCTION_RESULT_TEXT: (f"Kundenbild für '{name}' erfolgreich gespeichert.", winreg.REG_SZ),
                REGISTRY_STATUS: ("Fertig", winreg.REG_SZ)

            })
        else:
            self.registry_action("set_multiple", path=REGISTRY_PATH, value={
                REGISTRY_KNOWN_CUSTOMER: (name, winreg.REG_SZ),
                REGISTRY_FUNCTION_RESULT: ("1", winreg.REG_SZ),
                REGISTRY_FUNCTION_RESULT_TEXT: (f"Fehler beim Berechnen des Embeddings für '{name}'", winreg.REG_SZ),
                REGISTRY_STATUS: ("Fertig", winreg.REG_SZ)
                })




            

# Tkinter Hauptprogramm starten
root = tk.Tk()
app = GesichtserkennungApp(root)

 # Fenster immer im Vordergrund
root.wm_attributes("-topmost", True)

    # Fokus setzen
root.focus_force()



root.mainloop()


