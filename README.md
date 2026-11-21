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

1. Embed the documents and the question with `SentenceTransformer("all-MiniLM-L6-v2")`.
2. Reduce the 384-dimensional embeddings to 4 dimensions with PCA (one dimension per qubit).
3. For each document, encode the query and document vectors on a Qiskit circuit and estimate their similarity with a multi-qubit SWAP test on `AerSimulator` (2048 shots).
4. Sort the documents by that score and keep the top two as context.
5. POST a prompt to Ollama asking it to answer the question using only that context.

## Troubleshooting

**`Connection refused` on port 11434.** The script posts to `http://localhost:11434/api/generate` and expects Ollama to be listening there. Start the Ollama server, confirm `llama3.2` is pulled (`ollama pull llama3.2`), and run the script again.
