# Pipeline CNN+Transformer vs CNN+GNN+Transformer

Dokumen ini menjelaskan pipeline training mulai dari residual matrix, dengan fokus pada perbedaan bentuk tensor dan aliran informasi antara model normal CNN+Transformer dan versi yang ditambah GNN.

## Setup Contoh

Misal:

```text
T = 1000 hari training
N = 500 saham/residual series
lookback = 30 hari
```

Input residual:

```text
residuals = (1000, 500)
```

Setelah sliding window:

```text
windows = (970, 500, 30)
```

Artinya:

```text
970 tanggal keputusan
500 saham
30 hari history residual per saham
```

Satu sample berarti:

```text
1 saham
1 tanggal keputusan
30 hari history residual sebelum tanggal keputusan itu
```

Contoh:

```text
saham A, keputusan hari ke-31, history hari 1-30
saham B, keputusan hari ke-31, history hari 1-30
saham A, keputusan hari ke-32, history hari 2-31
```

## Side-by-Side Pipeline

| Step | Normal CNN+Transformer | CNN+GNN+Transformer |
|---|---|---|
| Residual input | `(1000, 500)` | `(1000, 500)` |
| Sliding window | `(970, 500, 30)` | `(970, 500, 30)` |
| Batch | `(B, 500, 30)` | `(B, 500, 30)` |
| Valid mask | `(B, 500)` | `(B, 500)` |
| Flatten valid sample | `(M, 30)` | sementara `(M, 30)` untuk CNN saja |
| CNN input | `(M, 1, 30)` | `(M, 1, 30)` |
| CNN output | `(M, 8, 30)` | `(M, 8, 30)` |
| Restore stock grid | tidak dilakukan | `(B, 500, 8, 30)` |
| GNN input | tidak ada | `(B, 30, 500, 8)` |
| GNN output | tidak ada | `(B, 30, 500, 8)` |
| Transformer input | `(30, M, 8)` | `(30, B*500, 8)` |
| Transformer attention | antar 30 hari dalam 1 saham-window | antar 30 hari setelah fitur saham diperkaya GNN |
| Output linear | `(M,)` | `(B, 500)` |
| Bobot akhir batch | scatter ke `(B, 500)` | sudah `(B, 500)` |

`M` adalah jumlah pasangan `(tanggal keputusan, saham)` yang valid dalam batch. Jika semua valid:

```text
M = B * 500
```

## Pipeline Normal CNN+Transformer

Model normal memperlakukan setiap saham-window sebagai sample independen.

Contoh batch:

```text
windows_batch = (125, 500, 30)
```

Setelah valid mask:

```text
x = (M, 30)
```

Kalau semua valid:

```text
M = 125 * 500 = 62500
x = (62500, 30)
```

Lalu masuk CNN:

```text
(62500, 30)
-> reshape
(62500, 1, 30)
-> CNN
(62500, 8, 30)
```

Makna:

```text
62500 = jumlah sample valid
1 = channel input residual
8 = fitur/channel hasil CNN
30 = hari dalam lookback
```

Kemudian masuk Transformer:

```text
(62500, 8, 30)
-> permute
(30, 62500, 8)
```

Transformer melihat:

```text
30 token waktu
62500 sample independen
8 embedding per token
```

Jadi attention terjadi antar hari dalam satu residual path, bukan antar saham.

Output:

```text
Transformer output: (30, 62500, 8)
ambil token terakhir: (62500, 8)
Linear(8 -> 1): (62500,)
scatter balik: (125, 500)
```

Intinya:

```text
CNN+Transformer normal belajar pola temporal universal.
Saham berbeda hanya dianggap sample berbeda dalam batch.
Tidak ada interaksi antar saham di dalam model.
```

## Pipeline CNN+GNN+Transformer

Versi GNN tetap boleh memakai valid-only CNN, tapi setelah CNN hasilnya dikembalikan ke grid saham.

Contoh batch:

```text
windows_batch = (32, 500, 30)
valid_mask = (32, 500)
```

Ambil sample valid untuk CNN:

```text
x_valid = windows_batch[valid_mask]
x_valid = (M, 30)
```

CNN:

```text
(M, 30)
-> reshape
(M, 1, 30)
-> CNN
(M, 8, 30)
```

Scatter balik ke struktur batch dan saham:

```text
x_full = zeros(32, 500, 8, 30)
x_full[valid_mask] = x_cnn_valid
```

Jadi:

```text
x_full = (32, 500, 8, 30)
```

Lalu diputar untuk GNN:

```text
(32, 500, 8, 30)
-> (32, 30, 500, 8)
```

Makna:

```text
32 = tanggal keputusan dalam batch
30 = hari lookback
500 = saham sebagai node
8 = fitur node hasil CNN
```

GNN fully connected:

```text
(32, 30, 500, 8)
-> masked mean aggregation antar saham
-> Linear + ReLU
-> (32, 30, 500, 8)
```

Di sini setiap saham bisa menerima informasi dari saham lain pada hari lookback yang sama.

Setelah GNN, masuk Transformer temporal:

```text
(32, 30, 500, 8)
-> reorder
(32, 500, 30, 8)
-> flatten batch dan saham
(32*500, 30, 8)
-> PyTorch Transformer format
(30, 16000, 8)
```

Transformer tetap bekerja secara temporal:

```text
hari 1 sampai hari 30
```

Tapi embedding tiap hari sudah diperkaya oleh informasi cross-sectional dari GNN.

Output:

```text
Transformer output: (30, 16000, 8)
ambil token terakhir: (16000, 8)
Linear(8 -> 1): (16000,)
reshape: (32, 500)
invalid mask -> zero
```

Intinya:

```text
CNN membaca pola lokal temporal per saham.
GNN menambahkan interaksi antar saham.
Transformer membaca dinamika temporal dari embedding yang sudah mengandung informasi antar saham.
```

## Perbedaan Konseptual Utama

| Aspek | CNN+Transformer | CNN+GNN+Transformer |
|---|---|---|
| Identitas saham | Tidak eksplisit | Masih tidak eksplisit, tapi saham menjadi node |
| Relasi antar saham | Tidak dimodelkan | Dimodelkan lewat fully connected GNN |
| Flatten `(B*N, 30)` | permanen selama forward model | hanya sementara untuk CNN |
| Struktur `(B, N, ...)` | hilang di dalam model | dikembalikan sebelum GNN |
| Attention Transformer | antar hari | antar hari setelah GNN |
| Cross-sectional information | hanya muncul saat normalisasi bobot/loss | masuk ke representasi model sebelum output |

## Kenapa GNN Perlu Reshape Balik

Kalau data tetap:

```text
(M, 30)
```

model hanya melihat daftar sample independen:

```text
sample 1 = saham A, tanggal t
sample 2 = saham B, tanggal t
sample 3 = saham C, tanggal t
```

GNN butuh tahu bahwa:

```text
saham A, B, C berada pada tanggal keputusan yang sama
```

Karena itu setelah CNN perlu scatter balik:

```text
(M, 8, 30)
-> (B, N, 8, 30)
```

Baru GNN bisa menganggap:

```text
N saham = N node dalam graph
```

## Ringkasan Pendek

Normal:

```text
(B,N,30)
-> valid flatten
-> (M,30)
-> CNN
-> Transformer temporal
-> bobot
```

Dengan GNN:

```text
(B,N,30)
-> valid flatten
-> CNN
-> scatter balik ke (B,N,8,30)
-> GNN antar saham
-> Transformer temporal
-> bobot
```

Perbedaan paling penting:

```text
Normal CNN+Transformer:
beda saham = beda sample independen

CNN+GNN+Transformer:
beda saham = node dalam graph yang saling bertukar informasi
```
