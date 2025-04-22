import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import threading
import time
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os  # <- tambahan buat force exit

class DeteksiKematanganPisang:
    def __init__(self, root):
        self.root = root
        self.root.title("Deteksi Kematangan Pisang")
        self.root.geometry("700x850")
        self.root.configure(bg="#f0f0f0")

        # Elemen GUI
        self.label = Label(self.root, text="Deteksi Kematangan Pisang", font=("Arial", 16), bg="#f0f0f0")
        self.label.pack(pady=10)

        self.image_label = Label(self.root, bg="#f0f0f0")
        self.image_label.pack(pady=10)

        self.result_label = Label(self.root, text="Pilih input untuk mendeteksi kematangan", font=("Arial", 12), bg="#f0f0f0")
        self.result_label.pack(pady=10)

        self.webcam_button = Button(self.root, text="Gunakan Webcam", command=self.mulai_webcam, bg="#4CAF50", fg="white", font=("Arial", 12))
        self.webcam_button.pack(pady=5)

        self.file_button = Button(self.root, text="Muat Gambar/Video", command=self.muat_file, bg="#2196F3", fg="white", font=("Arial", 12))
        self.file_button.pack(pady=5)

        self.stop_button = Button(self.root, text="Berhenti", command=self.berhenti_proses, bg="#F44336", fg="white", font=("Arial", 12), state=tk.DISABLED)
        self.stop_button.pack(pady=5)

        self.exit_button = Button(self.root, text="Keluar Program", command=self.keluar_program, bg="#9C27B0", fg="white", font=("Arial", 12))
        self.exit_button.pack(pady=5)

        self.cap = None
        self.running = False
        self.thread = None

        # Window histogram
        self.hist_window = None
        self.canvas = None
        self.fig = None
        self.ax = None
        self.setup_histogram_window()

        self.root.protocol("WM_DELETE_WINDOW", self.keluar_program)

    def setup_histogram_window(self):
        self.hist_window = tk.Toplevel(self.root)
        self.hist_window.title("Histogram Kematangan")
        self.hist_window.geometry("400x300")
        self.hist_window.configure(bg="#f0f0f0")
        self.hist_window.protocol("WM_DELETE_WINDOW", self.close_histogram_window)

        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.hist_window)
        self.canvas.get_tk_widget().pack(pady=10)
        self.update_histogram([0, 0, 0])

    def close_histogram_window(self):
        if self.hist_window:
            self.hist_window.destroy()
            self.hist_window = None
            self.canvas = None
            self.fig = None
            self.ax = None

    def update_histogram(self, percentages):
        if not self.ax or not self.canvas:
            return

        self.ax.clear()
        categories = ['Hijau', 'Kuning', 'Coklat']
        colors = ['green', 'yellow', 'brown']
        self.ax.bar(categories, percentages, color=colors)
        self.ax.set_ylim(0, 100)
        self.ax.set_ylabel('Persentase (%)')
        self.ax.set_title('Distribusi Warna Kematangan')
        for i, v in enumerate(percentages):
            self.ax.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom')
        self.canvas.draw()

    def deteksi_kematangan(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_clahe = clahe.apply(v)
        hsv_clahe = cv2.merge([h, s, v_clahe])

        batas_hijau_bawah = np.array([25, 40, 40]) 
        batas_hijau_atas = np.array([70, 255, 255])
        batas_kuning_bawah = np.array([15, 40, 40])
        batas_kuning_atas = np.array([30, 255, 255])
        batas_coklat_bawah = np.array([0, 40, 0])
        batas_coklat_atas = np.array([15, 255, 100])

        masker_hijau = cv2.inRange(hsv_clahe, batas_hijau_bawah, batas_hijau_atas)
        masker_kuning = cv2.inRange(hsv_clahe, batas_kuning_bawah, batas_kuning_atas)
        masker_coklat = cv2.inRange(hsv_clahe, batas_coklat_bawah, batas_coklat_atas)

        piksel_hijau = cv2.countNonZero(masker_hijau)
        piksel_kuning = cv2.countNonZero(masker_kuning)
        piksel_coklat = cv2.countNonZero(masker_coklat)

        total_piksel = piksel_hijau + piksel_kuning + piksel_coklat
        if total_piksel == 0:
            self.update_histogram([0, 0, 0])
            return "Tidak ada pisang terdeteksi", -1, 0

        persen_hijau = piksel_hijau / total_piksel * 100
        persen_kuning = piksel_kuning / total_piksel * 100
        persen_coklat = piksel_coklat / total_piksel * 100

        self.update_histogram([persen_hijau, persen_kuning, persen_coklat])

        skor = (persen_hijau / 100) * 0 + (persen_kuning / 100) * 5 + (persen_coklat / 100) * 10
        skor = round(skor, 1)

        if persen_hijau / 100 > 0.5:
            label = "Mentah"
            keyakinan = round(persen_hijau, 1)
        elif persen_kuning / 100 > 0.5:
            label = "Matang"
            keyakinan = round(persen_kuning, 1)
        elif persen_coklat / 100 > 0.5:
            label = "Terlalu Matang"
            keyakinan = round(persen_coklat, 1)
        else:
            label = "Campuran"
            persentase = [p / 100 for p in [persen_hijau, persen_kuning, persen_coklat] if p > 0]
            if persentase:
                entropi = -sum(p * math.log2(p) for p in persentase)
                entropi_maks = math.log2(len(persentase))
                keyakinan = round((1 - entropi / entropi_maks) * 100, 1) if entropi_maks > 0 else 0
            else:
                keyakinan = 0

        return label, skor, keyakinan

    def perbarui_frame(self, frame):
        frame = cv2.resize(frame, (400, 400))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk

    def proses_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.result_label.configure(text="Error: Tidak dapat membuka webcam")
            self.berhenti_proses()
            return

        self.running = True
        self.stop_button.configure(state=tk.NORMAL)
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.result_label.configure(text="Error: Webcam terputus")
                break

            label, skor, keyakinan = self.deteksi_kematangan(frame)
            if skor >= 0:
                self.result_label.configure(text=f"Kematangan: {label}, Skor: {skor}/10, Keyakinan: {keyakinan}%")
            else:
                self.result_label.configure(text=label)

            self.perbarui_frame(frame)
            self.root.update()
            time.sleep(0.1)

        self.cap.release()
        self.berhenti_proses()

    def proses_file(self, file_path):
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(file_path)
            if img is None:
                self.result_label.configure(text="Error: Tidak dapat memuat gambar")
                return

            label, skor, keyakinan = self.deteksi_kematangan(img)
            if skor >= 0:
                self.result_label.configure(text=f"Kematangan: {label}, Skor: {skor}/10, Keyakinan: {keyakinan}%")
            else:
                self.result_label.configure(text=label)

            self.perbarui_frame(img)

        elif file_path.lower().endswith(('.mp4', '.avi')):
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                self.result_label.configure(text="Error: Tidak dapat memuat video")
                return

            self.running = True
            self.stop_button.configure(state=tk.NORMAL)
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break

                label, skor, keyakinan = self.deteksi_kematangan(frame)
                if skor >= 0:
                    self.result_label.configure(text=f"Kematangan: {label}, Skor: {skor}/10, Keyakinan: {keyakinan}%")
                else:
                    self.result_label.configure(text=label)

                self.perbarui_frame(frame)
                self.root.update()
                time.sleep(0.033)

            self.cap.release()
            self.berhenti_proses()

    def mulai_webcam(self):
        if not self.running:
            self.thread = threading.Thread(target=self.proses_webcam)
            self.thread.daemon = True
            self.thread.start()

    def muat_file(self):
        if not self.running:
            file_path = filedialog.askopenfilename(filetypes=[("File Gambar/Video", "*.jpg *.jpeg *.png *.mp4 *.avi")])
            if file_path:
                self.thread = threading.Thread(target=self.proses_file, args=(file_path,))
                self.thread.daemon = True
                self.thread.start()

    def berhenti_proses(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.stop_button.configure(state=tk.DISABLED)
        self.result_label.configure(text="Pilih input untuk mendeteksi kematangan")
        self.image_label.configure(image="")
        self.image_label.image = None
        self.update_histogram([0, 0, 0])

    def keluar_program(self):
        self.running = False

        if self.cap:
            self.cap.release()

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

        self.close_histogram_window()

        try:
            cv2.destroyAllWindows()
        except:
            pass

        self.root.destroy()
        os._exit(0)  # force exit Python process tanpa sisa

if __name__ == "__main__":
    root = tk.Tk()
    app = DeteksiKematanganPisang(root)
    root.mainloop()
