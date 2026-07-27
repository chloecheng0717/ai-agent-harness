# AI Agent Harness

A reusable framework for building multi-agent AI systems with tool use, memory, and self-improvement capabilities.

## Features

- **Multi-Agent Architecture**: Specialized agents for planning, execution, research, security, and review
- **Tool System**: Plugin-based architecture for API integrations, file operations, web search, and code execution
- **Memory System**: Short-term conversation memory, long-term knowledge memory with vector database support
- **Event-Driven Communication**: Agent-to-agent messaging via Redis Pub/Sub
- **OpenAI-Compatible**: Support for OpenAI, Anthropic, and other LLM providers
- **Production Ready**: FastAPI backend with PostgreSQL, Redis, and Docker support

## Architecture

The framework follows Clean Architecture principles with:

- **Agent Core Engine**: Orchestrator, task analyzer, reasoning loop, execution loop, and reflection
- **Agent Components**: Specialized agents (Planner, Executor, Researcher, Security, Reviewer)
- **Memory System**: Short-term (Redis), long-term (Vector DB), episodic (PostgreSQL)
- **Tool System**: Plugin architecture with dynamic tool registration
- **Communication**: Event-driven architecture with Redis Pub/Sub

## Quick Start

### Prerequisites

- Python 3.12+
- PostgreSQL
- Redis
- OpenAI API key (or compatible provider)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-org/ai-agent-harness.git
cd ai-agent-harness
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r backend/requirements/base.txt
pip install -r backend/requirements/dev.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Start services with Docker:
```bash
docker-compose up -d
```

6. Run the application:
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Development

### Code Quality

```bash
# Linting
ruff check .

# Formatting
black .

# Type checking
mypy .

# Testing
pytest
```

### Project Structure

```
ai-agent-harness/
├── backend/
│   ├── app/                    # Main application
│   │   ├── agents/            # Agent components
│   │   ├── memory/            # Memory system
│   │   ├── tools/             # Tool system
│   │   ├── communication/     # Event system
│   │   ├── api/               # API routes
│   │   └── services/          # Business logic
│   ├── tests/                 # Test suite
│   └── alembic/               # Database migrations
├── frontend/                  # Optional web interface
├── docs/                      # Documentation
└── scripts/                   # Utility scripts
```

## Configuration

All configuration is managed via environment variables. See `.env.example` for available options.

Key configuration areas:
- **Database**: PostgreSQL connection settings
- **Redis**: Cache and pub/sub configuration
- **LLM**: OpenAI API keys and model selection
- **Vector DB**: ChromaDB connection settings
- **Security**: CORS, authentication settings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.