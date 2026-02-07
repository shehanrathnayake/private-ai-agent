# Private AI Agent

A lightweight, Dockerized AI agent designed to run on a private LAN. It uses OpenRouter to access various LLMs (like Gemini 2.0) and features a dual-layer memory system (SQLite + Markdown) for persistent, long-term context.

## 🚀 Features

- **LAN Accessible**: Exposes a FastAPI endpoint reachable over your local network.
- **Dockerized**: Easy deployment with a single command.
- **Dual-Stream Memory**:
  - **SQLite**: Stores exact message history for short-term context.
  - **Markdown Summaries**: Automatically generates periodic summaries for long-term "narrative" memory.
- **OpenRouter Integration**: Seamlessly switch between any model provided by OpenRouter.

## 🛠️ Prerequisites

- Docker and Docker Compose
- An [OpenRouter API Key](https://openrouter.ai/)

## 📦 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd private-ai-agent
   ```

2. **Configure Environment Variables**:
   Copy the example environment file and fill in your details:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your `OPENROUTER_API_KEY`.

3. **Start the Agent**:
   ```bash
   docker compose up -d --build
   ```

4. **Identify Server IP**:
   Find the IP address of the machine running the agent:
   - Windows: `ipconfig`
   - Linux/Mac: `ifconfig` or `ip addr`

## 🧪 Usage

You can talk to the agent from another computer on the same WiFi using the provided test client:

```bash
python test_client.py <AGENT_IP> "Hello! Can you hear me?" my-session-id
```

## 📂 Project Structure

- `app/`: Core application logic (FastAPI, Memory, LLM wrappers).
- `memory/`: Persistent storage for history (DB) and summaries (MD).
- `test_client.py`: CLI tool for interacting with the agent remotely.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
