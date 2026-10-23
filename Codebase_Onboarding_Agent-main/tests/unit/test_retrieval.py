import numpy as np

from onboarding_agent.retrieval.bm25_store import BM25Store
from onboarding_agent.retrieval.embeddings import HashingEmbedder
from onboarding_agent.retrieval.fusion import reciprocal_rank_fusion
from onboarding_agent.retrieval.tokenize import tokenize
from onboarding_agent.retrieval.vector_store import VectorStore


def test_tokenize_splits_camel_and_snake():
    toks = tokenize("getUserName parse_config_file HTTPServer")
    assert "getusername" in toks and "user" in toks and "name" in toks
    assert "parse" in toks and "config" in toks
    assert "server" in toks


def test_hash_embedder_deterministic_and_normalized():
    emb = HashingEmbedder(dim=128)
    v1 = emb.embed(["def authenticate(user): ..."])
    v2 = emb.embed(["def authenticate(user): ..."])
    assert np.allclose(v1, v2)
    assert abs(np.linalg.norm(v1[0]) - 1.0) < 1e-5


def test_hash_embedder_similar_texts_closer():
    emb = HashingEmbedder(dim=256)
    vecs = emb.embed([
        "def login(user, password): authenticate the user",
        "user login authentication with password check",
        "matrix multiplication kernel for GPUs",
    ])
    sim_related = float(vecs[0] @ vecs[1])
    sim_unrelated = float(vecs[0] @ vecs[2])
    assert sim_related > sim_unrelated


def test_vector_store_add_search_remove():
    store = VectorStore(dim=4)
    vecs = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0.9, 0.1, 0, 0]], dtype=np.float32)
    store.add(["a", "b", "c"], vecs)
    hits = store.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=2)
    assert hits[0][0] == "a"
    assert hits[1][0] == "c"
    store.remove({"a"})
    hits = store.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=2)
    assert hits[0][0] == "c"
    assert len(store) == 2


def test_vector_store_persistence(tmp_path):
    store = VectorStore(dim=4)
    store.add(["x"], np.array([[0, 0, 1, 0]], dtype=np.float32))
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path, dim=4)
    assert len(loaded) == 1
    assert loaded.search(np.array([0, 0, 1, 0], dtype=np.float32), 1)[0][0] == "x"


def test_bm25_ranks_matching_doc_first():
    store = BM25Store()
    store.add(
        ["doc1", "doc2"],
        ["def authenticate(user, password): verify credentials",
         "def render_template(html): draw the page"],
    )
    hits = store.search("how does user authentication work", top_k=2)
    assert hits and hits[0][0] == "doc1"


def test_bm25_persistence(tmp_path):
    store = BM25Store()
    store.add(["d"], ["reciprocal rank fusion implementation"])
    store.save(tmp_path)
    loaded = BM25Store.load(tmp_path)
    assert loaded.search("rank fusion", 1)[0][0] == "d"


def test_rrf_fuses_rankings():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    order = [doc for doc, _ in fused]
    # a and b each appear high in both lists; d only once, last.
    assert set(order[:2]) == {"a", "b"}
    assert order[-1] in ("c", "d")
    scores = dict(fused)
    assert scores["a"] > scores["d"]


def test_rrf_weights():
    fused = reciprocal_rank_fusion([["a"], ["b"]], k=1, weights=[2.0, 1.0])
    assert fused[0][0] == "a"
