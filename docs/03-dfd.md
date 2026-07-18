# 03. Data Flow Diagram (DFD)

> Konten baru — sistem belum memiliki DFD notasi klasik sebelumnya (yang ada di `writing/chapter3.md` adalah flowchart/ERD/class diagram). Diturunkan dari fakta pipeline yang sudah diverifikasi di [06](06-pipeline-ingestion.md)–[09](09-ivm-ram-keamanan.md), digambar mengikuti notasi Gane-Sarson: **persegi panjang** = entitas eksternal, **lingkaran bernomor** = proses, **silinder** = data store.

## 3.1 Diagram Konteks (Level 0)

Level 0 menunjukkan sistem sebagai satu proses tunggal beserta seluruh entitas eksternal yang berinteraksi dengannya.

```mermaid
flowchart TD
    Pengguna["👤 Pengguna"]
    Admin["👤 Admin"]
    LLM["Penyedia LLM\n(OpenRouter / Ollama)"]
    INF["Server Infinity\n(Reranker, PromptGuard, NLI)"]
    UNS["Unstructured API"]

    P0((("0.0\nSistem Chatbot RAG\nDomain-Agnostic")))

    Pengguna -->|Pertanyaan| P0
    P0 -->|Jawaban + Sitasi per kalimat| Pengguna

    Admin -->|Unggah / Kelola Dokumen| P0
    P0 -->|Status Ingestion, Daftar Dokumen| Admin

    P0 -->|Prompt / Kueri LLM| LLM
    LLM -->|Respons LLM| P0

    P0 -->|Teks untuk Klasifikasi / Rerank / NLI| INF
    INF -->|Skor / Label| P0

    P0 -->|File PDF| UNS
    UNS -->|Elemen Terstruktur| P0
```

**Entitas eksternal:**

| Entitas | Peran |
|---|---|
| Pengguna | Mengirim pertanyaan, menerima jawaban + sitasi |
| Admin | Mengunggah & mengelola dokumen KB, memantau status ingestion |
| Penyedia LLM | Menjawab prompt (generasi jawaban, HyDE, LLM-judge relevansi, deskripsi VLM) |
| Server Infinity | Mengklasifikasi keamanan prompt, me-rerank kandidat, mengevaluasi entailment NLI |
| Unstructured API | Mem-parsing PDF mentah menjadi elemen terstruktur |

## 3.2 DFD Level 1 — Dekomposisi Proses

Proses 0.0 didekomposisi menjadi 5 proses bernomor. Aliran eksternal pada level ini **seimbang** dengan Level 0 — tidak ada entitas eksternal baru, hanya diperjelas proses mana yang benar-benar mengonsumsi/menghasilkan tiap aliran.

```mermaid
flowchart TD
    Pengguna["👤 Pengguna"]
    Admin["👤 Admin"]
    LLM["Penyedia LLM"]
    INF["Server Infinity"]
    UNS["Unstructured API"]

    D1[("D1 — PostgreSQL\nDokumen, Chunks, Sesi, Pesan")]
    D2[("D2 — Qdrant\nIndeks Vektor")]
    D3[("D3 — File Storage\nuploads/knowledge_base")]

    P1(("1.0\nValidasi Input\nIVM"))
    P2(("2.0\nRetrieval Konteks"))
    P3(("3.0\nGenerasi Jawaban"))
    P4(("4.0\nPenilaian Respons\nRAM"))
    P5(("5.0\nIngestion Dokumen"))

    Pengguna -->|Pertanyaan| P1
    P1 -->|Pesan Error jika ditolak| Pengguna

    P1 -->|Kueri untuk pre-check| P2
    P2 -->|Konteks awal top-3 + skor| P1
    P1 -->|Kueri tervalidasi| P2

    P1 -->|Teks kueri sliding-window| INF
    INF -->|Verdict aman/berbahaya| P1
    P1 -->|Kueri + konteks untuk relevance judge| LLM
    LLM -->|YES / NO| P1

    P2 <-->|Kueri vektor / hasil hybrid search| D2
    P2 <-->|Fetch child & parent chunk| D1
    P2 -->|Kueri untuk HyDE| LLM
    LLM -->|Dokumen hipotetis| P2

    P2 -->|Konteks terambil top-15| P3
    P3 -->|Prompt tersusun| LLM
    LLM -->|Token jawaban stream| P3

    P3 -->|Kalimat jawaban + konteks| P4
    P4 -->|Kueri rerank + NLI| INF
    INF -->|Skor entailment/contradiction| P4

    P4 -->|Jawaban + badge sitasi stream| Pengguna
    P3 -->|Simpan pesan pengguna & asisten| D1
    P4 -->|Simpan hasil penilaian| D1

    Admin -->|Unggah PDF| P5
    Admin -->|"Perintah kelola\naktif/nonaktif/hapus/reingest"| P5
    P5 -->|Status ingestion & daftar dokumen| Admin

    P5 -->|File PDF| UNS
    UNS -->|Elemen terstruktur| P5
    P5 -->|Permintaan deskripsi gambar/tabel| LLM
    LLM -->|Deskripsi VLM| P5

    P5 -->|Simpan file asli| D3
    P5 -->|Simpan metadata, parent & child chunk| D1
    P5 -->|Upsert vektor dense + sparse| D2
```

**Deskripsi proses:**

| Proses | Deskripsi singkat | Detail lengkap |
|---|---|---|
| 1.0 Validasi Input (IVM) | Memeriksa keamanan prompt (deteksi injection) dan relevansi domain (OOD) sebelum kueri diproses lebih lanjut | [09-ivm-ram-keamanan.md](09-ivm-ram-keamanan.md) |
| 2.0 Retrieval Konteks | Mencari konteks relevan dari knowledge base (opsional HyDE → embed → hybrid search → rerank → hidrasi sibling/cross-ref) | [07-pipeline-retrieval.md](07-pipeline-retrieval.md) |
| 3.0 Generasi Jawaban | Menyusun prompt dari konteks terambil dan men-stream jawaban dari LLM | [08-pipeline-chat.md](08-pipeline-chat.md) |
| 4.0 Penilaian Respons (RAM) | Menilai tiap kalimat jawaban terhadap konteks sumber via NLI, melampirkan badge sitasi | [09-ivm-ram-keamanan.md](09-ivm-ram-keamanan.md) |
| 5.0 Ingestion Dokumen | Mem-parsing, chunking, embedding, dan mengindeks dokumen PDF baru | [06-pipeline-ingestion.md](06-pipeline-ingestion.md) |

**Data store:**

| Data Store | Isi |
|---|---|
| D1 — PostgreSQL | `pdf_documents`, `parent_chunks`, `child_chunks`, `ingestion_tasks`, `sessions`, `messages` |
| D2 — Qdrant | Collection `knowledge_base`: vektor dense (1024-dim) + sparse (BM25) per child chunk |
| D3 — File Storage | File PDF asli (`uploads/knowledge_base/`) |

Skema lengkap tiap data store ada di [05-basis-data.md](05-basis-data.md).

## 3.3 Catatan Leveling

- Proses 1.0 dan 2.0 saling bertukar data dua arah: 1.0 memerlukan hasil pencarian awal (top-3) dari 2.0 untuk memutuskan relevansi, sebelum 2.0 dipanggil ulang untuk pencarian dalam (top-15) — ini disebut *two-stage retrieval* di [08-pipeline-chat.md](08-pipeline-chat.md).
- Proses 5.0 (ingestion) berjalan independen dari 1.0–4.0 dan tidak pernah memanggil `IRelevanceChecker` — tidak ada validasi relevansi dokumen di sisi tulis, murni transformasi struktural (lihat [02-arsitektur.md](02-arsitektur.md) §2.5).
- Flow "Prompt / Kueri LLM" di Level 0 adalah agregasi dari 4 pemakaian LLM berbeda di Level 1 (relevance judge, HyDE, generasi jawaban, deskripsi VLM) — seluruhnya memakai entitas eksternal yang sama (OpenRouter/Ollama, API OpenAI-compatible) sehingga digabung menjadi satu entitas di Level 0.

---
⟵ [02-arsitektur.md](02-arsitektur.md) | [README.md](README.md) (indeks) | [04-diagram-aktivitas.md](04-diagram-aktivitas.md) ⟶
