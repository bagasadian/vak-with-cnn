import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Scrollbar, Text, ttk
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import shutil
import subprocess

# Load model dan class
try:
    model = tf.keras.models.load_model("model/model_palmistry.h5")
    class_names = ['auditori', 'kinestetik', 'visual']
except:
    model = None
    class_names = ['auditori', 'kinestetik', 'visual']

# Fungsi untuk membuat gradien background
def create_gradient(parent, width, height, color1, color2):
    gradient = tk.Canvas(parent, width=width, height=height, highlightthickness=0)
    for i in range(height):
        ratio = i / height
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        color = f'#{r:02x}{g:02x}{b:02x}'
        gradient.create_line(0, i, width, i, fill=color)
    return gradient

# Fungsi prediksi gambar
def predict(img_path):
    if model is None:
        return "Model tidak tersedia", 0.0
    
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        return predicted_class, confidence
    except Exception as e:
        return "Error dalam prediksi", 0.0

# Global variable untuk menyimpan path gambar
current_image_path = None

# Fungsi untuk menampilkan gambar dalam ukuran penuh
def show_full_image():
    if current_image_path and os.path.exists(current_image_path):
        image_window = Toplevel(window)
        image_window.title("🖼️ Preview Gambar Telapak Tangan")
        image_window.configure(bg="white")
        image_window.resizable(True, True)
        
        try:
            img = Image.open(current_image_path)
            
            screen_width = image_window.winfo_screenwidth()
            screen_height = image_window.winfo_screenheight()
            max_width = min(600, screen_width - 100)
            max_height = min(500, screen_height - 100)
            
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            
            window_width = img.width + 40
            window_height = img.height + 100
            image_window.geometry(f"{window_width}x{window_height}")
            
            x = (screen_width // 2) - (window_width // 2)
            y = (screen_height // 2) - (window_height // 2)
            image_window.geometry(f"+{x}+{y}")
            
            info_frame = tk.Frame(image_window, bg="#E3F2FD", height=50)
            info_frame.pack(fill="x")
            info_frame.pack_propagate(False)
            
            file_name = os.path.basename(current_image_path)
            tk.Label(info_frame, text=f"📁 File: {file_name}", 
                    font=("Segoe UI", 10, "bold"), bg="#E3F2FD", fg="#1976D2").pack(pady=15)
            
            img_label = tk.Label(image_window, image=img_tk, bg="white", relief="solid", bd=1)
            img_label.image = img_tk  # Keep a reference
            img_label.pack(padx=20, pady=10)
            
            tk.Button(image_window, text="✅ Tutup", bg="#4CAF50", fg="white",
                     font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2",
                     command=image_window.destroy, width=15).pack(pady=10)
                     
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menampilkan gambar: {str(e)}")
            image_window.destroy()
    else:
        messagebox.showinfo("Info", "Belum ada gambar yang dipilih!")

# Fungsi upload & klasifikasi
def open_image():
    global current_image_path
    file_path = filedialog.askopenfilename(
        title="Pilih Gambar Telapak Tangan",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )
    if file_path:
        try:
            current_image_path = file_path 
            
            img = Image.open(file_path)
            img.thumbnail((200, 200), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            label_img.configure(image=img_tk)
            label_img.image = img_tk
            
            label_img.configure(text="") 
            
            file_name = os.path.basename(file_path)
            status_label.config(text=f"📁 File dimuat: {file_name}")
            
            result_label.config(text="🔄 Menganalisis...", fg="#FF6B35")
            window.update()
            
            result, prob = predict(file_path)
            
            icons = {'auditori': '🎧', 'kinestetik': '🤸', 'visual': '👁️'}
            icon = icons.get(result, '🧠')
            result_label.config(
                text=f"{icon} Gaya Belajar: {result.upper()}\nTingkat Kepercayaan: {prob:.1f}%\n\n💡 Klik gambar untuk melihat ukuran penuh",
                fg="#2E7D32"
            )
            
            # Simpan ke dataset
            save_to_dataset(file_path, result)
            
            # Log prediksi
            log_prediction(os.path.basename(file_path), result, prob)
            
            # Update status
            status_label.config(text=f"✅ Prediksi selesai: {result.upper()} ({prob:.1f}%)")
            animate_success()
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memproses gambar: {str(e)}")
            status_label.config(text="❌ Gagal memproses gambar")

def save_to_dataset(file_path, result):
    try:
        nama_file = os.path.basename(file_path)
        tujuan_folder = os.path.join("DATASET", result.capitalize())
        os.makedirs(tujuan_folder, exist_ok=True)
        tujuan_file = os.path.join(tujuan_folder, nama_file)

        count = 1
        while os.path.exists(tujuan_file):
            nama_file_split = os.path.splitext(nama_file)
            tujuan_file = os.path.join(tujuan_folder, f"{nama_file_split[0]}_{count}{nama_file_split[1]}")
            count += 1

        shutil.copy(file_path, tujuan_file)
    except Exception as e:
        print(f"Error saving to dataset: {e}")

def log_prediction(nama_file, result, prob):
    try:
        os.makedirs("hasil_prediksi", exist_ok=True)
        with open("hasil_prediksi/log_prediksi.txt", "a", encoding='utf-8') as f:
            f.write(f"{nama_file};{result};{prob:.2f}%\n")
    except Exception as e:
        print(f"Error logging prediction: {e}")

def animate_success():
    original_color = result_label.cget("fg")
    for i in range(3):
        result_label.config(fg="#4CAF50")
        window.update()
        window.after(100)
        result_label.config(fg=original_color)
        window.update()
        window.after(100)

# Untuk menampilkan informasi gambar
def show_image_info():
    if current_image_path and os.path.exists(current_image_path):
        try:
            img = Image.open(current_image_path)
            file_size = os.path.getsize(current_image_path)
            file_size_mb = file_size / (1024 * 1024)
            
            info_text = f"""
📁 Nama File: {os.path.basename(current_image_path)}
📐 Dimensi: {img.width} x {img.height} pixels
🎨 Mode: {img.mode}
📊 Ukuran File: {file_size_mb:.2f} MB
📍 Lokasi: {current_image_path}
"""
            messagebox.showinfo("ℹ️ Informasi Gambar", info_text)
        except Exception as e:
            messagebox.showerror("Error", f"Gagal mendapatkan informasi gambar: {str(e)}")
    else:
        messagebox.showinfo("Info", "Belum ada gambar yang dipilih!")

# UI Tambahan
def lihat_log():
    log_window = Toplevel(window)
    log_window.title("📊 Riwayat Prediksi")
    log_window.geometry("700x500")
    log_window.configure(bg="#F5F5F5")
    
    # Header window
    header_frame = tk.Frame(log_window, bg="#1976D2", height=60)
    header_frame.pack(fill="x")
    header_frame.pack_propagate(False)
    
    tk.Label(header_frame, text="📊 Riwayat Prediksi", 
             font=("Segoe UI", 16, "bold"), fg="white", bg="#1976D2").pack(pady=15)
    
    # Frame untuk treeview
    tree_frame = tk.Frame(log_window, bg="#F5F5F5")
    tree_frame.pack(fill="both", expand=True, padx=20, pady=20)
    
    # Treeview untuk menampilkan data dalam tabel
    columns = ("No", "Nama File", "Gaya Belajar", "Kepercayaan")
    tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=15)
    
    # Konfigurasi kolom
    tree.heading("No", text="No")
    tree.heading("Nama File", text="Nama File")
    tree.heading("Gaya Belajar", text="Gaya Belajar")
    tree.heading("Kepercayaan", text="Kepercayaan")
    
    tree.column("No", width=50, anchor="center")
    tree.column("Nama File", width=300)
    tree.column("Gaya Belajar", width=150, anchor="center")
    tree.column("Kepercayaan", width=100, anchor="center")
    
    # Scrollbar
    scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    
    # Pack tree dan scrollbar
    tree.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Load data
    try:
        with open("hasil_prediksi/log_prediksi.txt", "r", encoding='utf-8') as f:
            lines = f.readlines()
            if lines:
                for i, line in enumerate(lines, 1):
                    parts = line.strip().split(";")
                    if len(parts) >= 3:
                        nama, gaya, persen = parts[0], parts[1], parts[2]
                        icon = {'auditori': '🎧', 'kinestetik': '🤸', 'visual': '👁️'}.get(gaya, '🧠')
                        tree.insert("", "end", values=(i, nama, f"{icon} {gaya.upper()}", persen))
            else:
                tree.insert("", "end", values=("", "Belum ada data prediksi", "", ""))
    except FileNotFoundError:
        tree.insert("", "end", values=("", "File log belum tersedia", "", ""))

def buka_dataset():
    try:
        if os.name == 'nt':
            os.startfile("DATASET")
        else:
            subprocess.run(["open" if os.name == 'posix' else "xdg-open", "DATASET"])
    except:
        messagebox.showinfo("Info", "Folder DATASET akan dibuat saat pertama kali melakukan prediksi")

def jalankan_preprocessing():
    if messagebox.askyesno("Konfirmasi", "Jalankan preprocessing data?\nProses ini mungkin membutuhkan waktu beberapa menit."):
        try:
            subprocess.run(["python", "preprocessing/preprocess.py"])
            messagebox.showinfo("Berhasil", "Preprocessing selesai!")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menjalankan preprocessing: {str(e)}")

def latih_ulang_model():
    if messagebox.askyesno("Konfirmasi", "Latih ulang model CNN?\nProses ini membutuhkan waktu lama dan resource yang cukup."):
        try:
            subprocess.run(["python", "model/train_model.py"])
            messagebox.showinfo("Berhasil", "Training model selesai!")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal melatih model: {str(e)}")

def show_about():
    about_window = Toplevel(window)
    about_window.title("ℹ️ Tentang Aplikasi")
    about_window.geometry("600x550")
    about_window.configure(bg="white")
    about_window.resizable(False, False)
    
    about_window.update_idletasks()
    x = (about_window.winfo_screenwidth() // 2) - (about_window.winfo_width() // 2)
    y = (about_window.winfo_screenheight() // 2) - (about_window.winfo_height() // 2)
    about_window.geometry(f"+{x}+{y}")
    
    header = tk.Frame(about_window, bg="#6A1B9A", height=70)
    header.pack(fill="x")
    header.pack_propagate(False)
    
    tk.Label(header, text="🤚 Palmistry VAK Classifier", 
             font=("Segoe UI", 16, "bold"), fg="white", bg="#6A1B9A").pack(pady=15)
    
    content = tk.Frame(about_window, bg="white")
    content.pack(fill="both", expand=True, padx=25, pady=15)
    
    text_frame = tk.Frame(content, bg="white")
    text_frame.pack(fill="both", expand=True)
    
    text_widget = tk.Text(text_frame, wrap="word", font=("Segoe UI", 10), 
                         bg="white", fg="#424242", relief="flat", 
                         padx=10, pady=10, height=20)
    scrollbar = tk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
    text_widget.configure(yscrollcommand=scrollbar.set)
    
    about_text = """📋 DESKRIPSI:
Aplikasi ini menggunakan teknologi Deep Learning untuk mengklasifikasikan gaya belajar mahasiswa berdasarkan analisis telapak tangan dengan pendekatan VAK (Visual, Auditori, Kinestetik).

🎯 FITUR UTAMA:
• Klasifikasi otomatis gaya belajar (Visual, Auditori, Kinestetik)
• Penyimpanan otomatis hasil prediksi ke dataset
• Riwayat prediksi lengkap dengan tampilan tabel
• Fitur training ulang model CNN
• Preprocessing data gambar otomatis
• Interface pengguna yang modern dan responsif

📊 CARA KERJA SISTEM:
1. Upload gambar telapak tangan
2. Preprocessing gambar (resize, normalisasi)
3. Ekstraksi fitur menggunakan CNN
4. Klasifikasi menggunakan model terlatih
5. Prediksi gaya belajar dengan tingkat kepercayaan

🎓 MANFAAT APLIKASI:
• Membantu identifikasi gaya belajar mahasiswa
• Mendukung personalisasi metode pembelajaran
• Menyediakan data untuk penelitian pendidikan
• Otomatisasi proses klasifikasi yang akurat
"""
    
    text_widget.insert("1.0", about_text)
    text_widget.config(state="disabled")
    
    text_widget.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    close_frame = tk.Frame(about_window, bg="white", height=50)
    close_frame.pack(fill="x", pady=10)
    close_frame.pack_propagate(False)
    
    tk.Button(close_frame, text="✅ Tutup", bg="#4CAF50", fg="white",
              font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2",
              command=about_window.destroy, width=15).pack()


# GUI Setup
window = tk.Tk()
window.title("🤚 Palmistry VAK Classifier")
window.geometry("900x800") 
window.configure(bg="#FAFAFA")
window.resizable(True, True)
window.minsize(800, 700)

# Gradient background
bg_gradient = create_gradient(window, 900, 800, (26, 35, 126), (63, 81, 181))
bg_gradient.place(x=0, y=0)

# Main container
main_container = tk.Frame(window, bg="white", relief="raised", bd=2)
main_container.place(x=50, y=30, width=800, height=740)  # Ukuran container diperbesar

# Header
header = tk.Frame(main_container, bg="#1A237E", height=80)
header.pack(fill="x")
header.pack_propagate(False)

# Title
title_frame = tk.Frame(header, bg="#1A237E")
title_frame.pack(expand=True)

tk.Label(title_frame, text="🤚", font=("Segoe UI", 24), bg="#1A237E").pack(side="left", padx=(20, 5), pady=15)
title_text = tk.Frame(title_frame, bg="#1A237E")
title_text.pack(side="left", pady=15)

tk.Label(title_text, text="VAK Classifier", 
         font=("Segoe UI", 18, "bold"), fg="white", bg="#1A237E").pack(anchor="w")
tk.Label(title_text, text="Sistem Klasifikasi Gaya Belajar Berbasis AI", 
         font=("Segoe UI", 9), fg="#E3F2FD", bg="#1A237E").pack(anchor="w")

# About button
tk.Button(header, text="ℹ️ About", bg="#3F51B5", fg="white", 
          font=("Segoe UI", 9), relief="flat", command=show_about).place(x=720, y=25, width=60, height=30)

# Main content area
content_frame = tk.Frame(main_container, bg="white")
content_frame.pack(fill="both", expand=True, padx=30, pady=20)

# Upload section
upload_frame = tk.Frame(content_frame, bg="#F8F9FA", relief="solid", bd=1)
upload_frame.pack(fill="x", pady=(0, 15))

tk.Label(upload_frame, text="📤 Upload & Analisis", 
         font=("Segoe UI", 14, "bold"), bg="#F8F9FA", fg="#1976D2").pack(pady=(15, 5))

upload_btn = tk.Button(upload_frame, text="🖼️ Pilih Gambar Telapak Tangan", 
                      bg="#2196F3", fg="white", font=("Segoe UI", 11, "bold"),
                      relief="flat", cursor="hand2", command=open_image)
upload_btn.pack(pady=(5, 15))

def on_enter(e):
    upload_btn.config(bg="#1976D2")
def on_leave(e):
    upload_btn.config(bg="#2196F3")

upload_btn.bind("<Enter>", on_enter)
upload_btn.bind("<Leave>", on_leave)

# Image display area
img_frame = tk.Frame(content_frame, bg="white")
img_frame.pack(pady=10)

label_img = tk.Label(img_frame, bg="#FAFAFA", relief="solid", bd=2,
                     text="🖼️\nGambar akan muncul di sini\n\n💡 Klik untuk melihat ukuran penuh", 
                     font=("Segoe UI", 10), fg="#9E9E9E",
                     width=25, height=10, cursor="hand2")
label_img.pack()

# Bind click event untuk menampilkan gambar penuh
label_img.bind("<Button-1>", lambda e: show_full_image())

# Result display
result_label = tk.Label(content_frame, text="", font=("Segoe UI", 12, "bold"), 
                       fg="#424242", bg="white", wraplength=400)
result_label.pack(pady=10)

# Menu buttons
menu_frame = tk.Frame(content_frame, bg="white")
menu_frame.pack(fill="x", pady=15)

button_style = {
    "font": ("Segoe UI", 9, "bold"),
    "relief": "flat",
    "cursor": "hand2",
    "width": 30,
    "height": 2
}

buttons = [
    ("📊 Lihat Riwayat Prediksi", "#4CAF50", lihat_log),
    ("📁 Buka Folder Dataset", "#FF9800", buka_dataset),
    ("ℹ️ Info Gambar", "#2196F3", show_image_info),
    ("⚙️ Jalankan Preprocessing", "#9C27B0", jalankan_preprocessing),
    ("🧠 Latih Ulang Model CNN", "#F44336", latih_ulang_model)
]

# Buat grid 
for i, (text, color, command) in enumerate(buttons):
    if i < 3:  # Row pertama
        row = 0
        col = i
    else:  # Row kedua
        row = 1 
        col = i - 3
    
    btn = tk.Button(menu_frame, text=text, bg=color, fg="white", 
                   command=command, **button_style)
    btn.grid(row=row, column=col, padx=8, pady=8, sticky="ew")

# Grid untuk 3 kolom
menu_frame.grid_columnconfigure(0, weight=1)
menu_frame.grid_columnconfigure(1, weight=1)
menu_frame.grid_columnconfigure(2, weight=1)

# Status bar
status_bar = tk.Frame(main_container, bg="#ECEFF1", height=35)  # Height ditambah
status_bar.pack(fill="x", side="bottom")
status_bar.pack_propagate(False)

status_label = tk.Label(status_bar, text="✅ Siap untuk klasifikasi", 
                       font=("Segoe UI", 9), bg="#ECEFF1", fg="#546E7A")
status_label.pack(side="left", padx=15, pady=8)


# Center window 
window.update_idletasks()
x = (window.winfo_screenwidth() // 2) - (window.winfo_width() // 2)
y = (window.winfo_screenheight() // 2) - (window.winfo_height() // 2)
window.geometry(f"+{x}+{y}")

# Start application
window.mainloop()