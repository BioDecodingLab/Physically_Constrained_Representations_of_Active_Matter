FROM python:3.12-slim
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124
ENV PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg CUBLAS_WORKSPACE_CONFIG=:4096:8
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/active-matter
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir torch==2.6.0 --index-url ${TORCH_INDEX} && python -m pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -m pip install --no-deps -e components/representation -e components/publication
ENTRYPOINT ["python", "scripts/study.py"]
