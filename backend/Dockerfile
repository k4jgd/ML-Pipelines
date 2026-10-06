FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY backend ./backend
COPY inference ./inference
COPY pipeline ./pipeline
COPY CustomerIntelligence.csv .
COPY Demand&Operations.csv .
COPY MachineIntelligence.csv .
COPY mlruns ./mlruns
COPY outputs ./outputs

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]