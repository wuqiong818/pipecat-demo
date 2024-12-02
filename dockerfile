FROM python:3.12.7-bookworm

RUN apt-get update && \
    apt-get install --no-install-recommends -y curl && \
    pip install --no-cache -U pip poetry && \
    rm -rf /var/cache/apt/archives /var/lib/apt/lists/*


RUN poetry config virtualenvs.create false && \
    poetry config experimental.system-git-client true && \
    mkdir /app && \
    git clone https://github.com/wuqiong818/pipecat-demo.git /app && \
    cd /app && \
    poetry install --no-cache --no-root --only main && \
    rm -rf ~/.cache

CMD python /app/pipecat-demo/25daily-openai4o-openaitts.py