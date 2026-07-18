# 10. Referensi API

> Sumber: `app/chat/api.py`, `app/kb/api.py`, `app/frontend.py` (diverifikasi langsung terhadap kode, bukan disalin dari ringkasan lain).

Seluruh endpoint JSON di-mount melalui dua router (`kb_router`, `chat_router`) plus satu router halaman HTML (`frontend_router`), didaftarkan di [`app/main.py`](../app/main.py). Tidak ada versi API (`/v1/...`) — seluruh path memakai prefix datar `/api/...`.

## 10.1 Chat (`app/chat/api.py`)

| Method | Path | Fungsi | Catatan |
|---|---|---|---|
| `POST` | `/api/chat/sessions` | Membuat sesi chat baru | Return `201 Created` |
| `GET` | `/api/chat/sessions` | Daftar semua sesi chat | |
| `GET` | `/api/chat/sessions/{session_id}` | Detail satu sesi (termasuk riwayat pesan) | `404` jika tidak ditemukan |
| `DELETE` | `/api/chat/sessions/{session_id}` | Hapus sesi chat | `404` jika tidak ditemukan |
| `POST` | `/api/chat/sessions/{session_id}/stream` | Streaming jawaban LLM (NDJSON) melalui IVM + RAM | Body: `{"message": str}`. Query opsional `skip_guardrails=true` melewati IVM safety/relevance dan RAM (mode baseline untuk eksperimen ablasi) — retrieval tetap berjalan agar LLM tetap punya konteks. Lihat [08-pipeline-chat.md](08-pipeline-chat.md) untuk urutan event NDJSON. |

## 10.2 Knowledge Base / Admin (`app/kb/api.py`)

| Method | Path | Fungsi | Catatan |
|---|---|---|---|
| `GET` | `/api/admin/pdfs` | Daftar seluruh dokumen PDF beserta metadata | |
| `POST` | `/api/admin/pdfs` | Unggah satu PDF, memicu ingestion asinkron | `multipart/form-data`: `title`, `description`, `file`. Return `202 Accepted` — ingestion berjalan di background |
| `POST` | `/api/admin/pdfs/batch` | Unggah banyak PDF sekaligus | `files[]`, `titles[]`, `descriptions[]` dipasangkan berdasarkan index; `422` jika jumlah `files` ≠ jumlah `titles`. Return `202` dengan ringkasan `count`/`failed_count` |
| `PUT` | `/api/admin/pdfs/{pdf_id}/status` | Ubah status aktif/nonaktif dokumen | Body: `{"active": bool}` — soft toggle, tidak menghapus vektor |
| `DELETE` | `/api/admin/pdfs/{pdf_id}` | Hapus dokumen dari knowledge base | |
| `GET` | `/api/admin/pdfs/{pdf_id}/ingestion-status` | Cek status pipeline ingestion dokumen | Untuk polling progres dari Admin UI |
| `POST` | `/api/admin/pdfs/{pdf_id}/reingest` | Jalankan ulang ingestion untuk dokumen yang sudah ada | Return `202`, berguna jika ingestion sebelumnya macet/gagal |
| `GET` | `/api/kb/pdfs/{pdf_id}/download` | Unduh file PDF asli | |
| `GET` | `/api/kb/search` | Pencarian hybrid (dense+sparse dengan RRF fusion) pada knowledge base | Query: `q` (wajib), `top_k` (default 15, 1–100), `session_id` (opsional), `mode` (`hybrid`/`dense`/`sparse`, default `hybrid`). Dipakai oleh chat pipeline dan skrip evaluasi retrieval |
| `GET` | `/api/kb/naive-search` | Pencarian substring judul literal (naif) | Query: `q`. Mereplikasi perilaku portal JDIH konvensional untuk halaman perbandingan `/demo/` — bukan bagian dari jalur RAG asli |

## 10.3 Halaman Frontend (`app/frontend.py`)

Router ini didaftarkan dengan `include_in_schema=False` (tidak muncul di OpenAPI docs) dan me-render template Jinja2 dari `app/templates/pages/`.

| Method | Path | Halaman |
|---|---|---|
| `GET` | `/` | Halaman chat utama |
| `GET` | `/admin/` | Halaman admin (kelola knowledge base) |
| `GET` | `/demo/` | Halaman perbandingan RAG vs pencarian judul naif |

## 10.4 Ringkasan Bentuk Respons

- Endpoint chat/KB administratif mengembalikan JSON standar (kecuali `stream`, yang mengembalikan `application/x-ndjson`).
- `POST /api/admin/pdfs` dan `/batch` sengaja memakai `202 Accepted`, bukan `200`/`201` — ingestion berjalan sebagai `BackgroundTasks`, sehingga response dikirim sebelum pipeline selesai. Klien harus polling `GET /api/admin/pdfs/{pdf_id}/ingestion-status`.
- `GET /api/kb/search` adalah satu-satunya endpoint yang mengekspos hasil retrieval mentah (dengan skor); dipakai baik oleh `ChatService` (secara internal, lewat `SearchService`, bukan lewat HTTP) maupun sebagai alat evaluasi/debug independen.

---
⟵ [09-ivm-ram-keamanan.md](09-ivm-ram-keamanan.md) | [README.md](README.md) | [11-deployment.md](11-deployment.md) ⟶
