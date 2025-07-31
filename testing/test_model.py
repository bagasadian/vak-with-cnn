# === Testing dan Prediksi Gambar Baru ===
# Deskripsi: Uji gambar baru dengan model hasil training dan tampilkan hasil klasifikasi

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

# Memuat model CNN yang sudah dilatih sebelumnya
model = tf.keras.models.load_model("model/model_palmistry.h5")

# Daftar nama kelas yang sesuai urutan output model
class_names = ['auditori', 'kinestetik', 'visual']

# Fungsi untuk memprediksi gambar tunggal
def predict_image(img_path):
    try:
        # Membaca gambar dan mengubah ukurannya ke 224x224 piksel (ukuran input model)
        img = image.load_img(img_path, target_size=(224, 224))
        # Mengubah gambar menjadi array dan melakukan normalisasi (skala 0-1)
        img_array = image.img_to_array(img) / 255.0
        # Menambahkan dimensi batch (karena model mengharapkan input dalam bentuk batch)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Melakukan prediksi
        predictions = model.predict(img_array)
        
        # Menentukan kelas dengan probabilitas tertinggi
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100
        
        # Menampilkan hasil prediksi
        print(f"Gambar: {img_path}\nPrediksi: {predicted_class.upper()} ({confidence:.2f}%)")
    
    except Exception as e:
        # Menangani kesalahan jika ada masalah saat memproses gambar
        print(f"Terjadi kesalahan: {e}")

# Fungsi utama saat file dijalankan langsung dari terminal
if __name__ == "__main__":
    # Mengecek apakah path gambar diberikan sebagai argumen
    if len(sys.argv) != 2:
        print("Usage: python testing/test_model.py <path_ke_gambar>")
    else:
        # Memanggil fungsi prediksi dengan path gambar dari argumen
        predict_image(sys.argv[1])
