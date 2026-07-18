# 07. Pipeline Retrieval

> Sumber: `writing/chapter3.md` §3.5 (diadaptasi menjadi dokumen referensi mandiri).

## 7.1 Gambaran Umum

Pipeline retrieval adalah jalur baca (*read-path*). `SearchService` mengorkestrasikan **enam langkah** dari kueri mentah hingga daftar `RetrievedContext` yang siap dikonsumsi `ChatService` (lihat [08-pipeline-chat.md](08-pipeline-chat.md)): (1) HyDE opsional, (2) embed, (3) hybrid search menghasilkan pool kandidat besar (`top_k=50`), (4) fetch *child chunk* langsung dari PostgreSQL berdasarkan id, (5) *rerank* cross-encoder pada level child menjadi top-8, (6) *hydrate* — ambil parent dari child yang lolos rerank, lalu perluas dengan *sibling* dan *cross-reference* chunk sebelum digabung dan di-dedupe.

Dipanggil lewat `GET /api/kb/search` (lihat [10-referensi-api.md](10-referensi-api.md)) untuk debugging/evaluasi, dan secara internal (bukan lewat HTTP) oleh `ChatService`.

## 7.2 Diagram 6-Langkah Retrieval

```mermaid
flowchart TD
    Q["User Query"] --> H1

    subgraph S1["Langkah 1: HyDE Expansion (opsional, IQueryExpander)"]
        H1["HyDEExpander.expand(query)\nLLM → paragraf jawaban hipotetis\ntemperature=0.0, max_tokens=256\nAktif hanya jika hyde_enabled (chat/config.py)"]
        HF{"Berhasil &\nnon-empty?"}
        H1 --> HF
        HF -->|Ya| HO["embed_text = hyde_doc"]
        HF -->|Gagal / disabled| HX["Fallback: embed_text = query\n(fail-open)"]
    end

    HO --> E1
    HX --> E1

    subgraph S2["Langkah 2: Query Embedding"]
        E1["ITextEmbedder.embed_texts([embed_text])\nBGE-M3: dense 1024-dim + sparse BM25"]
    end

    E1 --> VS

    subgraph S3["Langkah 3: Hybrid Vector Search — pool besar"]
        VS["IVectorStore.hybrid_search(\n  top_k=INITIAL_SEARCH_TOP_K (50), mode='hybrid'\n)\nPrefetch bm25 + dense (masing-masing top_k×2) → RRF fusion\nFilter: is_active=True AND session_id IS EMPTY\nHasil: List[SearchResult] (chunk_id, parent_chunk_id, doc_id, score)"]
    end

    VS --> CF

    subgraph S4["Langkah 4: Fetch Child Chunks"]
        CF["kb_repo.get_child_chunks_by_ids(chunk_ids)\n(dari PostgreSQL)"]
        CF --> NOCH{"Ada child\nyang persisten?"}
        NOCH -->|Tidak — fallback lama| FALL["_fallback_parent_search()\nPakai parent.text langsung,\nrerank di level parent"]
        NOCH -->|Ya| CAND[["candidates = [(SearchResult, ChildChunk), ...]"]]
    end

    CAND --> RR

    subgraph S5["Langkah 5: Cross-Encoder Rerank — level child, top-8"]
        RR["IReranker.rerank(\n  query=ORIGINAL_QUERY (bukan HyDE),\n  documents=[child.text, ...],\n  top_k=RERANK_TOP_K (8)\n)\nModel: BGE-reranker-v2-m3\nFail-open: jika error, truncate ke 8 by search score"]
    end

    RR --> HY

    subgraph S6["Langkah 6: Hydrate — Parent + Sibling + Cross-Ref"]
        HY["Fetch parent_chunks untuk child yang lolos rerank\n→ primary contexts (RetrievedContext,\nteks = parent.text, child_text = child.text)"]
        HY --> SIB["_hydrate_siblings()\nUntuk tiap primary: cari parent lain\ndengan parent_id sama (bagian bertetangga)\nvia get_sibling_chunks()"]
        HY --> XREF["_detect_and_fetch_cross_refs()\nRegex pada child_text + parent.text:\n'Pasal N', 'BAB N/IVXLC', 'Ayat N'\n→ path prefix lookup (get_chunks_by_path_prefix)"]
        SIB --> MERGE["_merge_and_dedupe()\nUrutan: primary → siblings → cross-refs\nDedupe by parent_chunk_id"]
        XREF --> MERGE
    end

    FALL --> OUT
    MERGE --> TRUNC["Truncate ke top_k akhir\n(default 15, ditentukan pemanggil)"]
    TRUNC --> OUT(["List[RetrievedContext]"])
```

Catatan desain: Langkah 5 me-rerank **teks child** (presisi, ≤512 char) terhadap kueri asli, bukan teks parent yang sudah dirakit — menghasilkan sinyal rerank yang lebih tajam. Parent (konteks lengkap untuk LLM) baru diambil setelah keputusan rerank final di Langkah 6, mempertahankan prinsip Small-to-Big: presisi match di level child, kelengkapan konteks di level parent.

## 7.3 Desain Small-to-Big Retrieval

```mermaid
flowchart LR
    subgraph Q["Saat Kueri"]
        UQ["User: 'Bagaimana alur persetujuan?'"]
        UQ --> QE["Embed query → dense + sparse"]
        QE --> QDRANT["Qdrant hybrid search\n→ Child chunk C₁ matched\n(score 0.87, ≤512 chars)"]
    end

    subgraph STB["Small-to-Big Lookup"]
        QDRANT --> CID["Fetch ChildChunk C₁ dari PostgreSQL by id"]
        CID --> PID["parent_chunk_id dari C₁"]
        PID --> PG["PostgreSQL SELECT parent_chunks\nWHERE id = parent_chunk_id"]
        PG --> PT["Parent P₁ text (≤4096 chars, pure body text)\n'Alur persetujuan terdiri dari...'\n(breadcrumbs ['BAB III','Pasal 12'] tersedia\nsebagai metadata terpisah, bukan inline)"]
    end

    subgraph LLM["Konteks LLM"]
        PT --> CTX["LLM menerima full section context\nbukan hanya sentence yang matched\n→ Jawaban lebih lengkap dan akurat"]
    end
```

## 7.4 Hybrid Search dengan RRF Fusion

Langkah 3 (hybrid search) menjalankan dua prefetch paralel yang digabungkan dengan **Reciprocal Rank Fusion (RRF)**:

$$\text{score}_{\text{RRF}}(d) = \sum_{r \in \{dense,\, sparse\}} \frac{1}{k + \text{rank}_r(d)}$$

Di mana $k = 60$ (konstanta default Qdrant). RRF berbasis peringkat — bukan skor absolut — sehingga robust terhadap distribusi skor yang berbeda antara cosine similarity dan BM25. Setiap prefetch mengambil `top_k × 2` kandidat untuk memperluas pool sebelum fusion. RRF di sini hanya menentukan pool kandidat awal (Langkah 3, `top_k=50`) — keputusan akhir konteks mana yang benar-benar dipakai LLM ditentukan oleh cross-encoder rerank di Langkah 5, bukan skor RRF.

## 7.5 Hidrasi Sibling dan Cross-Reference

Dokumen institusional (peraturan, SOP) sering merujuk pasal/bagian lain secara eksplisit ("sebagaimana diatur dalam Pasal 5") atau menyimpan informasi terkait di bagian bertetangga secara struktural tanpa disebut ulang secara semantik pada teks yang match kueri. Vector search murni tidak menangkap kebutuhan ini karena kaitannya struktural, bukan semantik. Langkah 6 menambahkan dua sumber konteks tambahan di atas hasil rerank:

- **Sibling hydration** — memanfaatkan hierarki `parent_id`/`path`/`depth` yang sudah dibangun saat chunking (§6.5): untuk setiap konteks utama, ambil parent chunk lain yang berbagi `parent_id` yang sama (bagian-bagian di bawah judul yang sama).
- **Cross-reference detection** — memindai teks konteks utama dengan pola regex untuk gaya rujukan dokumen legal Indonesia (`Pasal N`, `BAB N`/angka Romawi, `Ayat N`), lalu mencari chunk dengan `path` yang berawalan sesuai.

Kedua sumber ini diberi `score=0.0` (tidak punya skor pencarian langsung) dan ditempatkan setelah primary contexts saat *merge+dedupe*, sehingga urutan relevansi hasil rerank tetap diutamakan. Catatan: pola regex cross-reference (`Pasal`/`BAB`/`Ayat`) berbentuk konvensi dokumen legal Indonesia secara spesifik — bagian implementasi yang tidak sepenuhnya domain-agnostic seperti prinsip di [00-gambaran-umum.md](00-gambaran-umum.md), meski tidak memengaruhi jalur retrieval utama jika tidak ada pola yang cocok (regex tidak match hanya berarti tidak ada cross-ref tambahan, bukan kegagalan).

---
⟵ [06-pipeline-ingestion.md](06-pipeline-ingestion.md) | [README.md](README.md) (indeks) | [08-pipeline-chat.md](08-pipeline-chat.md) ⟶
