# === Preprocessing Data Citra Telapak Tangan ===
# Deskripsi: Cropping, augmentasi, dan deteksi tepi (Canny)

import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Fungsi untuk menerapkan deteksi tepi menggunakan algoritma Canny
def canny_edge(image_path, output_path, low=20, high=60):
    # Membaca gambar dalam format grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Menghaluskan gambar untuk mengurangi noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # Menerapkan algoritma Canny untuk mendeteksi tepi
    edges = cv2.Canny(blurred, low, high)
    # Menyimpan hasil gambar deteksi tepi
    cv2.imwrite(output_path, edges)

# Fungsi untuk melakukan augmentasi gambar (rotasi dan zoom)
def augment_image(image_path, output_folder):
    # Membaca gambar berwarna
    image = cv2.imread(image_path)
    basename = os.path.splitext(os.path.basename(image_path))[0]

    # --- Rotasi gambar sebesar 20 derajat ---
    rows, cols, _ = image.shape
    # Matriks transformasi rotasi
    M_rot = cv2.getRotationMatrix2D((cols / 2, rows / 2), 20, 1)
    # Menerapkan rotasi pada gambar
    rotated = cv2.warpAffine(image, M_rot, (cols, rows))
    # Menyimpan hasil gambar rotasi
    cv2.imwrite(os.path.join(output_folder, f"{basename}_rotated.png"), rotated)

    # --- Zoom: crop bagian tengah lalu resize kembali ke ukuran asli ---
    zoomed = image[30:-30, 30:-30]  # Crop bagian tengah gambar
    zoomed = cv2.resize(zoomed, (cols, rows))  # Resize ke ukuran semula
    # Menyimpan hasil gambar zoom
    cv2.imwrite(os.path.join(output_folder, f"{basename}_zoomed.png"), zoomed)

# Fungsi untuk memproses seluruh folder dataset (per label/kategori)
def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)  # Membuat folder output jika belum ada

    # Iterasi setiap subfolder/label (misalnya: Visual, Auditory, Kinesthetic)
    for label in os.listdir(input_folder):
        input_path = os.path.join(input_folder, label)
        output_path = os.path.join(output_folder, label)
        os.makedirs(output_path, exist_ok=True)  # Membuat folder untuk masing-masing label

        # Iterasi setiap gambar dalam folder label
        for img_name in os.listdir(input_path):
            img_in = os.path.join(input_path, img_name)
            img_out = os.path.join(output_path, img_name)

            # Deteksi tepi menggunakan Canny
            canny_edge(img_in, img_out)

            # Augmentasi gambar (rotasi dan zoom)
            augment_image(img_in, output_path)

# Fungsi utama yang dipanggil saat program dijalankan
if __name__ == "__main__":
    process_folder("dataset", "dataset_processed")
    print("Preprocessing selesai!")
