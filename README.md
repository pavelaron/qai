# qai

A small demo that ranks a hardcoded set of documents against a question using sentence embeddings and a quantum SWAP test, then asks a local Ollama model to answer using only the top matches.

## Requirements

- Python 3.11 or newer (the project was set up with Python 3.12)
- [Ollama](https://ollama.com) with the `llama3.2` model, and the Ollama server running on `localhost:11434`

Pull the model before the first run:

```bash
ollama pull llama3.2
```

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run also downloads the `all-MiniLM-L6-v2` sentence-transformers model.

## Run

With the virtual environment activated and Ollama running:

```bash
python main.py
```

It prints a ranked list of documents (quantum similarity scores) and the local LLM answer.

## Configuration

These values are hardcoded in `main.py`:

- Ollama endpoint: `http://localhost:11434/api/generate`
- Model: `llama3.2`
- Embedding model: `all-MiniLM-L6-v2`
- Documents and the user question are also defined in `main.py`

There is no config file or environment-variable layer.

## How it works

• Classical feature compression: Because quantum simulators handle only a few qubits efficiently, you first compress text into tiny 2D or 4D numerical vectors (for example, using TF-IDF or a small embedding model reduced via PCA).
• Quantum state encoding: Each normalized vector is mapped onto qubit rotation angles (such as \(R_y(\theta)\) gates) so the quantum state represents the direction of the text vector.
• SWAP test circuit: A circuit uses an ancillary qubit, a Hadamard gate, and a controlled-SWAP (Fredkin) gate to compare the query state against a document state.
• Similarity measurement: Measuring the ancillary qubit gives a probability \(P(0) = \frac{1}{2} + \frac{1}{2}\vert{}\langle \psi_{\text{query}} \vert{} \psi_{\text{doc}} \rangle\vert{}^2\). Higher probability of measuring 0 means the two texts are more similar.


## Connecting It to an LLM

• Retrieve: Run the quantum similarity loop over your small document set and pick the top-1 or top-2 document strings.
• Generate: Pass those winning text strings into a standard LLM prompt (via Ollama, OpenAI, or Hugging Face) as context so the model can answer your question.


## How the End-to-End Flow Works

• Quantum ranking: The Qiskit circuit evaluates the inner product between your query vector and each document vector, returning similarity scores between 0 and 1.
• Context injection: Python formats the highest-scoring text chunks into a strict system prompt so the local model relies on your retrieved facts rather than guessing.
• Local inference: The requests.post call sends that prompt to Ollama's local REST endpoint on port 11434, and your local GPU or CPU generates the final answer offline.


## Key Pipeline Details

• PCA compression: Since all-MiniLM-L6-v2 outputs 384 numbers, PCA(n_components=4) keeps the 4 strongest semantic axes so the circuit needs only 9 total qubits (1 ancilla + 4 query + 4 document).
• Multi-qubit SWAP test: Applying cswap across each pair of query and document qubits computes the combined inner product across all 4 dimensions at once.
• Scaling note: You can increase n_qubits to 6 or 8 for better semantic accuracy while still running fast on AerSimulator.


## Troubleshooting

**`Connection refused` on port 11434.** The script posts to `http://localhost:11434/api/generate` and expects Ollama to be listening there. Start the Ollama server, confirm `llama3.2` is pulled (`ollama pull llama3.2`), and run the script again.
