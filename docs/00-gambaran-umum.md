# 00. Gambaran Umum Sistem

> Sumber utama: `writing/chapter3.md` §3.1 (diadaptasi menjadi dokumen referensi mandiri).

## 0.1 Apa Sistem Ini

Sistem ini adalah **chatbot berbasis Retrieval-Augmented Generation (RAG) yang bersifat domain-agnostic** — dapat beroperasi pada domain pengetahuan apa pun tanpa perubahan kode, hanya dengan mengganti dokumen dalam *knowledge base* (KB). Pengguna mengunggah dokumen PDF (misalnya peraturan, SOP, atau dokumen institusional lain), dan pengguna lain dapat bertanya dalam bahasa natural; sistem menjawab berdasarkan isi dokumen tersebut, lengkap dengan sitasi sumber dan indikator keandalan jawaban per kalimat.

Sistem ini dibangun sebagai bagian dari penelitian skripsi (Universitas Pendidikan Indonesia) yang menyasar tiga masalah umum pada sistem RAG konvensional:

1. **Fragmentasi Struktural** — pemotongan teks berbasis ukuran karakter tetap merusak struktur tabel dan membuang gambar/diagram yang sering memuat informasi paling penting dalam dokumen institusional.
2. **Kehilangan Konteks Hierarkis** — kalimat yang diekstrak dari "Bagian 3.2" kehilangan jalur seksinya; model *embedding* tidak tahu kalimat itu berada di bawah "Bab 3 → Prosedur → Alur Persetujuan".
3. **Konflik Presisi-Konteks** — *chunk* kecil presisi namun minim konteks; *chunk* besar kontekstual namun melemahkan sinyal embedding. Ukuran tetap memaksa satu kompromi tunggal yang tidak optimal.

## 0.2 Tiga Kontribusi Utama

| Kontribusi | Modul | Masalah yang Diselesaikan |
|---|---|---|
| **Hierarchical Small-to-Big Chunking** | `app/thesis/chunking/` | Konflik presisi-konteks + fragmentasi struktural |
| **Input Validation Module (IVM)** | `app/thesis/ivm/` | Keamanan prompt & relevansi domain-agnostic |
| **Response Assessment Module (RAM)** | `app/thesis/ram/` | Deteksi halusinasi per-kalimat dengan NLI |

Detail masing-masing ada di [06-pipeline-ingestion.md](06-pipeline-ingestion.md) (chunking), dan [09-ivm-ram-keamanan.md](09-ivm-ram-keamanan.md) (IVM & RAM).

**Prinsip domain-agnostic**: seluruh keputusan *filtering* dan relevansi diturunkan dari isi knowledge base itu sendiri, bukan dikodekan secara statis. IVM tidak memeriksa apakah kueri "tentang topik X" secara hardcoded; ia memeriksa apakah kueri memiliki konteks yang relevan dalam KB yang sedang aktif. Sistem bekerja identik untuk dokumen hukum, rekam medis, atau domain lain — batas domain *adalah* KB itu sendiri.

## 0.3 Tumpukan Teknologi (Tech Stack)

| Lapisan | Teknologi |
|---|---|
| Bahasa & runtime | Python 3.11 (dikelola via `mise`) |
| Web framework | FastAPI (async), Jinja2 (frontend server-rendered) |
| Basis data relasional | PostgreSQL 17 (async SQLAlchemy) |
| Vector store | Qdrant 1.18 (hybrid dense + sparse search, RRF fusion) |
| Inference server ML | Infinity 0.0.77 — reranker (BGE-reranker-v2-m3), prompt-guard (Llama-Prompt-Guard-2-86M), NLI (indo-roberta-indonli) |
| Embedding | BGE-M3 in-process (dense 1024-dim + sparse BM25) |
| Parsing dokumen | Unstructured API (self-hosted) |
| LLM generasi | OpenAI-compatible via OpenRouter (default) atau Ollama lokal |
| VLM (deskripsi gambar/tabel) | OpenRouter cloud, Ollama lokal, atau fallback heuristik |
| Observability | structlog (JSON) → Vector → Loki → Grafana |

Lihat [11-deployment.md](11-deployment.md) untuk topologi container lengkap dan [02-arsitektur.md](02-arsitektur.md) untuk struktur modul kode.

## 0.4 Glosarium Istilah

| Istilah | Kepanjangan / Arti |
|---|---|
| **RAG** | Retrieval-Augmented Generation — LLM menjawab berdasarkan konteks yang diambil (retrieved) dari knowledge base, bukan hanya dari pengetahuan internal model |
| **IVM** | Input Validation Module — modul dua-lapis: (1) deteksi prompt-injection/konten berbahaya, (2) deteksi relevansi domain (out-of-domain/OOD) |
| **RAM** | Response Assessment Module — modul yang menilai tiap kalimat jawaban LLM terhadap konteks sumber via NLI, menghasilkan badge sitasi Supported/Contradiction |
| **HyDE** | Hypothetical Document Embeddings — teknik query expansion: LLM menulis jawaban hipotetis dari kueri pendek, lalu jawaban hipotetis itu yang di-embed untuk pencarian, bukan kueri asli |
| **RRF** | Reciprocal Rank Fusion — metode menggabungkan hasil pencarian dense dan sparse berbasis peringkat (bukan skor absolut) |
| **NLI** | Natural Language Inference — model yang menilai hubungan logis (entailment/contradiction/neutral) antara dua teks (premise & hypothesis) |
| **VLM** | Vision-Language Model — model yang mendeskripsikan konten visual (gambar, diagram, tabel kompleks) dalam teks |
| **Small-to-Big** | Strategi retrieval: pencarian vektor dilakukan pada teks kecil dan presisi (*child chunk*), tetapi konteks yang diberikan ke LLM adalah teks besar dan lengkap (*parent chunk*) |
| **OOD** | Out-of-Domain — kueri yang tidak relevan dengan isi knowledge base yang tersedia |
| **DDD** | Domain-Driven Design — pola arsitektur yang memisahkan domain/logika inti dari infrastruktur teknis |

---
[README.md](README.md) (indeks) | [01-use-case.md](01-use-case.md) ⟶
