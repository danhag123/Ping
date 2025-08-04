# 🏓 Ping

## 🔎 Overview
Ping is a Retrieval Augmented Generation (RAG) project focused on serving Swedish table tennis enthusiasts. It provides accurate, up-to-date insights about table tennis rules and regulations.

---

## 📋 Table of Contents
1. [Features](#features)
2. [Sources](#sources)
3. [Getting Started](#getting-started)
4. [Poetry](#poetry)
5. [Docker](#docker)
6. [Contributing](#contributing)

---

## Features
- **RAG-based Solution**: Utilizes the latest version of the Swedish table tennis rulebook.
- **Flexible Architecture**: Easy to integrate with various NLP/LLM tools.
- **Scalable**: Leverage containerization for deployment.

---

## Sources
- SBTF:s Spelregler (Swedish Table Tennis Federation Rules) - Translation of ITTF Statutes with Swedish additions and comments

---

## Getting Started

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop) or a compatible Docker environment.
- [Poetry](https://python-poetry.org/docs/) for Python dependency management.
- [Git](https://git-scm.com/) for cloning and version management.

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/danhag123/Ping.git
   cd Ping
   ```

2. **Check Environment**
   - Ensure Docker is running.
   - Verify Poetry is installed.

### Configuration
- Create or update any `.env` files or config files required by the application.
- Confirm environment variables for database connections or external APIs.

#### Required Environment Variables
Below is an example of the environment variables you may need in your `.env` file:
```
OPENAI_API_KEY=<your_openai_api_key>
QDRANT_URL=<your_qdrant_url>
QDRANT_API_KEY=<your_qdrant_api_key>
QDRANT_COLLECTION=<your_qdrant_collection_name>
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=<your_langsmith_endpoint>
LANGSMITH_API_KEY=<your_langsmith_api_key>
LANGSMITH_PROJECT=<your_langsmith_project>
```
Make sure to replace these placeholder values with your actual credentials and settings.

---

## Poetry
Before building your Docker containers, make sure that all Python dependencies are up to date. Typically, you'll install dependencies and lock them using Poetry:

```bash
poetry install
poetry lock
```
---

## Docker

1. 🐙 **Install Docker Desktop** (or confirm it is running).
2. 📁 Make sure that `dockerfile.rag` & `dockerfile.rag.dockerignore` are present inside the **Docker** folder (or adjust your Docker context accordingly).
   - If making a new dockerfile, use the following structure `dockerfile.yourfeature` & `dockerfile.yourfeature.dockerignore`.
3. ⚙️ Ensure there is a `docker-compose.yaml` in the root directory.
4. 🎼 Ensure your [Poetry](#poetry) dependencies are up to date locally, or plan to install them inside the container.
5. 💻 **Build and run** with Docker Compose:
   - **Build** the images:
     ```bash
     docker-compose build
     ```
     or with a specific tag:
     ```bash
     docker-compose build -t your-project-name
     ```
   - **Start** the containers:
     ```bash
     docker-compose up
     ```
     or do both in one step:
     ```bash
     docker-compose up --build
     ```

---
