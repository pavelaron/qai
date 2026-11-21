import numpy as np
import requests
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# 1. Knowledge Base & Query (Raw Text)
documents = [
    "Quantum entanglement links two qubits so the state of one instantly correlates with the other.",
    "To boil pasta properly, use heavily salted water and cook until al dente.",
    "A Hadamard gate puts a qubit into an equal superposition of 0 and 1.",
    "Photosynthesis allows green plants to convert sunlight into chemical energy.",
    "Qiskit is an open-source SDK for working with quantum computers at the circuit level.",
]
user_question = "How do quantum gates create superposition?"

# 2. Generate Real Embeddings & Compress to 4 Dimensions (4 Qubits)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(documents)
query_embedding = embedder.encode([user_question])

# Fit PCA on documents to reduce 384 dims -> 4 dims
n_qubits = 4
pca = PCA(n_components=n_qubits)
doc_vectors_4d = pca.fit_transform(doc_embeddings)
query_vector_4d = pca.transform(query_embedding)[0]

# 3. Multi-Qubit Quantum SWAP Test
def encode_vector(qc, vec, start_qubit):
    # Scale values to [0, pi] for Ry rotations
    norm = np.linalg.norm(vec)
    normalized = vec / norm if norm > 0 else vec
    angles = np.arccos(np.clip(normalized, -1.0, 1.0))
    for i, theta in enumerate(angles):
        qc.ry(2 * theta, start_qubit + i)

def multi_qubit_swap_test(vec_a, vec_b, shots=2048):
    n = len(vec_a)
    # Qubit 0: ancilla | Qubits 1..n: query | Qubits n+1..2n: document
    qc = QuantumCircuit(2 * n + 1, 1)

    encode_vector(qc, vec_a, 1)
    encode_vector(qc, vec_b, n + 1)

    qc.h(0)
    for i in range(n):
        qc.cswap(0, 1 + i, n + 1 + i)
    qc.h(0)
    qc.measure(0, 0)

    sim = AerSimulator()
    counts = sim.run(qc, shots=shots).result().get_counts()
    prob_0 = counts.get("0", 0) / shots
    return max(0.0, 2 * prob_0 - 1)

# 4. Rank Documents via Quantum Similarity
scored_docs = []
for text, vec in zip(documents, doc_vectors_4d):
    score = multi_qubit_swap_test(query_vector_4d, vec)
    scored_docs.append((score, text))

scored_docs.sort(key=lambda x: x[0], reverse=True)
top_context = "\n".join([f"- {text}" for _, text in scored_docs[:2]])

# 5. Generate Answer with Local LLM (Ollama)
prompt = f"""Answer the user's question using ONLY the context below.

Context:
{top_context}

Question: {user_question}
Answer:"""

response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "llama3.2", "prompt": prompt, "stream": False},
)

print("--- Quantum Similarity Rankings ---")
for score, text in scored_docs:
    print(f"[{score:.3f}] {text}")

print("\n--- Local LLM Answer ---")
print(response.json()["response"])
