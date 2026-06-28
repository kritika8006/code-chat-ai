This project is a private, local-first Retrieval-Augmented Generation (RAG) tool designed to function as a Senior Security Engineer AI Assistant.
It indexes your codebase locally and provides security-conscious insights, risk analysis, and code reasoning without ever sending your sensitive
data to external servers.


Key Features

    Privacy-First: Operates entirely offline using your local machine.

    Context-Aware: Uses FAISS to create a searchable index of your project files.

    Security Analysis: Uses Gemma 3 (via Ollama) to act as a Senior Security Engineer, specifically trained to look for input sanitization issues, improper error handling, and other vulnerabilities.

    Citations: Every response references the specific files and code blocks being analyzed.
