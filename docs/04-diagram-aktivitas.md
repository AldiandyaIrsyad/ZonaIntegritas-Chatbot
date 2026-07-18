# 04. Diagram Aktivitas (Process Flow, Swimlane)

> Konten baru — melengkapi flowchart langkah-demi-langkah di [06](06-pipeline-ingestion.md)/[08](08-pipeline-chat.md) dengan pandangan **lintas-fungsi**: siapa/komponen apa yang menjalankan tiap langkah. Mermaid tidak memiliki primitif swimlane asli, sehingga tiap "lane" digambar sebagai `subgraph` — teknik yang sama yang sudah dipakai `writing/chapter3.md` untuk tahapan pipeline.

## 4.1 Alur Tanya-Jawab Chat

Lane: **Pengguna** / **Sistem (Chat Orchestrator: IVM → Retrieval → Generasi → RAM)** / **LLM & Infinity (eksternal)** / **Basis Data (PostgreSQL + Qdrant)**.

```mermaid
flowchart TD
    subgraph LANE_USER["🧑 Pengguna"]
        U1(["Kirim pesan"])
        U2(["Terima error"])
        U3(["Terima jawaban + sitasi\n(streaming per kalimat)"])
    end

    subgraph LANE_SYS["⚙️ Sistem — Chat Orchestrator"]
        S1["Muat sesi + riwayat"]
        S2{"Safety check\n(IVM)?"}
        S3{"Aman?"}
        S4["Pre-check retrieval\n(top_k=3)"]
        S5{"Relevan?\n(IVM)"}
        S6["Deep retrieval\n(top_k=15)"]
        S7["Susun prompt"]
        S8["Pecah jadi proposisi\n+ nilai tiap kalimat (RAM)"]
        S9["Format sitasi\n& simpan pesan"]
    end

    subgraph LANE_EXT["🌐 LLM & Infinity"]
        E1["PromptGuard\nklasifikasi teks"]
        E2["Judge / NLI / skor RRF\n(sesuai ood_method)"]
        E3["LLM: stream token jawaban"]
        E4["Reranker + NLI\nper proposisi"]
    end

    subgraph LANE_DB["🗄️ Basis Data"]
        DB1[("PostgreSQL — sessions,\nmessages")]
        DB2[("Qdrant + PostgreSQL —\nhybrid search, chunks")]
    end

    U1 --> S1 --> S2
    S1 -.->|simpan pesan pengguna| DB1

    S2 -->|ya| E1
    E1 -->|verdict| S3
    S2 -->|skip_guardrails=true| S4
    S3 -->|tidak aman| ERR1(["Error: diblokir\nsafety filter"])
    ERR1 --> U2
    S3 -->|aman| S4

    S4 -.->|kueri pre-check| DB2
    S4 --> S5
    S5 -->|butuh verifikasi| E2
    E2 -->|hasil| S5
    S5 -->|tidak relevan| ERR2(["Error: tidak ada\nkonteks relevan"])
    ERR2 --> U2
    S5 -->|relevan| S6

    S6 -.->|kueri dalam + hidrasi| DB2
    S6 --> S7 --> E3
    E3 -->|token stream| S8
    S8 --> E4
    E4 -->|label + skor| S8
    S8 --> S9
    S9 -.->|simpan pesan asisten| DB1
    S9 --> U3

    style ERR1 fill:#ff6b6b,color:#fff
    style ERR2 fill:#ff6b6b,color:#fff
    style U3 fill:#51cf66,color:#333
```

Urutan langkah ini identik dengan flowchart detail di [08-pipeline-chat.md](08-pipeline-chat.md) — diagram ini hanya menambahkan dimensi "siapa yang mengerjakan apa" yang tidak eksplisit di flowchart linear.

## 4.2 Alur Unggah & Ingestion Dokumen

Lane: **Admin** / **Sistem (IngestWorker)** / **Unstructured API & VLM (eksternal)** / **Basis Data (PostgreSQL + Qdrant + File Storage)**.

```mermaid
flowchart TD
    subgraph LANE_ADMIN["🧑‍💼 Admin"]
        A1(["Unggah PDF\nvia Admin UI"])
        A2(["Terima 202 Accepted\n(ingestion di background)"])
        A3(["Polling status\ningestion"])
    end

    subgraph LANE_SYS2["⚙️ Sistem — IngestWorker"]
        W1["Buat PDFDocument\n(status: pending)"]
        W2["Jadwalkan background task\n+ balas 202"]
        W3["Flip status → processing"]
        W4["Route & enrich elemen\n(TABLE/FIGURE/TEXT)"]
        W5["Chunking hierarkis\n(parent → child)"]
        W6["Embed BGE-M3\n(in-process, dense+sparse)"]
        W7["Tandai completed\n/ failed"]
    end

    subgraph LANE_EXT2["🌐 Unstructured API & VLM"]
        X1["Parse PDF →\nParsedElement list\n(strategy hi_res)"]
        X2["Deskripsi gambar/tabel\n(OpenRouter / Ollama)"]
    end

    subgraph LANE_DB2["🗄️ Basis Data"]
        DBF[("File Storage —\nuploads/knowledge_base/")]
        DBP[("PostgreSQL — pdf_documents,\nparent/child chunks,\ningestion_tasks")]
        DBQ[("Qdrant —\nvektor dense + sparse")]
    end

    A1 --> W1
    W1 -.->|simpan file asli| DBF
    W1 -.->|insert row| DBP
    W1 --> W2 --> A2

    W2 --> W3
    W3 -.->|update status| DBP
    W3 --> X1
    X1 -->|ParsedElement list| W4

    W4 -->|elemen FIGURE| X2
    X2 -->|deskripsi| W4

    W4 --> W5
    W5 -.->|simpan parent & child chunk| DBP
    W5 --> W6
    W6 --> W7B{"Dokumen masih ada?\n(guard race-condition)"}
    W7B -->|dihapus| ABORT(["Abort silently"])
    W7B -->|ada| W6B["Upsert vektor"]
    W6B -.-> DBQ
    W6B --> W7
    W7 -.->|update status akhir| DBP

    A3 -.->|query status| DBP
    W7 -->|status akhir| A3

    style ABORT fill:#ffa94d,color:#fff
    style W7 fill:#51cf66,color:#333
```

Urutan langkah ini identik dengan flowchart 8-tahap di [06-pipeline-ingestion.md](06-pipeline-ingestion.md); diagram ini menambahkan pemisahan tanggung jawab per komponen (Admin vs Sistem vs layanan eksternal vs data store).

---
⟵ [03-dfd.md](03-dfd.md) | [README.md](README.md) (indeks) | [05-basis-data.md](05-basis-data.md) ⟶
