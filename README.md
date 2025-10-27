# SEGMENTASI CITRA: ALGORITMA WATERSHED
Proyek ini bertujuan untuk mempraktikkan materi Segmentasi Lanjut, khususnya Algoritma Watershed, pada citra ilustrasi/anime yang kompleks menggunakan bahasa pemrograman Python dan pustaka OpenCV. Segmentasi Watershed ideal untuk memisahkan objek yang saling bersentuhan (clumped objects) dan memiliki gradasi warna yang berbeda.

Untuk memahami dan mampu menerapkan Algoritma Watershed yang melibatkan langkah-langkah preprocessing dan morfologi kompleks, meliputi:
1. Thresholding dan Noise Reduction.
2. Penentuan Sure Foreground menggunakan Distance Transform.
3. Penentuan Sure Background menggunakan Dilasi.
4. Marker Labeling untuk area yang pasti dan area yang tidak diketahui (unknown region).
5. Penerapan algoritma Watershed.


#1. Import Library dan Processing Awal
'''python
import cv2
import numpy as np
import matplotlib.pyplot as plt

a. cv2 (OpenCV): Digunakan untuk sebagian besar operasi pemrosesan citra, seperti Thresholding, Morfologi, dan Watershed.
b. numpy: Digunakan untuk manipulasi array, seperti membuat kernel atau structuring element.
c. matplotlib.pyplot: Digunakan untuk menampilkan hasil visualisasi.

#2. Membaca dan Mengonversi Citra
FILE_PATH = 'images/input_image.jpg' 
image_color_bgr = cv2.imread(FILE_PATH)
image_gray = cv2.cvtColor(image_color_bgr, cv2.COLOR_BGR2GRAY)

Membaca citra dari direktori images/ dan mengubahnya:
a. image_gray: Citra Grayscale untuk Thresholding dan Morfologi.
b. image_color_bgr: Citra BGR (3-channel) yang diperlukan sebagai input untuk fungsi cv2.watershed().

#3. Processing (Blur dan Thresholding)
blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

a. Gaussian Blur: Menerapkan filter penghalusan untuk mengurangi noise yang dapat menyebabkan over-segmentation pada Watershed.
b. Thresholding Otsu (Inv): Mengonversi citra menjadi dua tingkat warna (biner). THRESH_BINARY_INV memastikan objek foreground memiliki nilai 255 (putih).

4. Membuat Structuring Element (Kernel)
kernel = np.ones((3,3), np.uint8)

kernel: Structuring element persegi berukuran (3x3) dibuat menggunakan NumPy. Ini berfungsi sebagai acuan bentuk dalam semua operasi morfologi.

#5. OPerasi Morfologi untuk Penentuan Markers (Watershed)
a. Opening
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
Melakukan Erosi diikuti Dilasi untuk menghilangkan noise kecil yang berada di luar objek utama (speckle noise).

b. Sure Background (Dilasi)
sure_bg = cv2.dilate(opening, kernel, iterations=3)
Memperluas area opening. Tujuan Dilasi adalah mendapatkan area yang pasti bersih dari objek, sehingga menjadi Sure Background.

c. Distance Transform
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
Menghitung jarak setiap piksel foreground (putih) ke background terdekat (hitam). Piksel dengan nilai tertinggi berada di inti objek.

d. Sure Foreground 
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
Memilih piksel yang berada di inti objek (nilai tinggi) sebagai markers objek yang pasti. Nilai ambang batas 0.7 * max() digunakan karena memberikan marker yang ketat.

e. Unknowm Region
unknown = cv2.subtract(sure_bg, sure_fg)
Menghasilkan selisih antara Sure Background dan Sure Foreground. Area ini adalah area yang tidak pasti dan akan diberi label 0, tempat Watershed akan membangun batas.

#6. Marker Labeling dan Eksekusi Watershed
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1  
markers[unknown == 255] = 0 
markers_ws = cv2.watershed(image_color_bgr, markers)
image_color_bgr[markers_ws == -1] = [255, 255, 255]

a. cv2.connectedComponents: Memberi label unik (1, 2, 3, ...) pada setiap objek di Sure Foreground.
b. Marker Shifting: Semua label digeser +1. Area unknown disetel ke label 0. Label 0 adalah instruksi bagi Watershed untuk memisahkan objek.
c. cv2.watershed(): Menerapkan algoritma. Batas-batas objek yang dihasilkan diberi label -1.
d. Visualisasi Batas: Piksel berlabel -1 ditandai dengan warna PUTIH [255, 255, 255] pada citra asli untuk hasil akhir.

#7. Menampilkan Hasil
plt.figure(figsize=(15, 8))
# ... (Kode loop plot) ...
plt.tight_layout()
plt.show()

Menampilkan semua tahapan pemrosesan (Asli, Threshold, Distance Transform, Sure FG, Sure BG, dan Hasil Akhir) dalam satu tampilan visual 2x3 untuk memudahkan perbandingan proses.
