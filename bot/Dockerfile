FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 可通过 compose build args 覆盖，缓解网络不稳定导致的安装超时。
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_EXTRA_INDEX_URL=
ARG PIP_TRUSTED_HOST=
ARG PIP_DEFAULT_TIMEOUT=240
ARG PIP_RETRIES=10

WORKDIR /app

RUN apt-get update \
	&& apt-get install -y --no-install-recommends ca-certificates \
	&& update-ca-certificates \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel \
	&& pip install --no-cache-dir \
		--index-url ${PIP_INDEX_URL} \
		${PIP_EXTRA_INDEX_URL:+--extra-index-url ${PIP_EXTRA_INDEX_URL}} \
		${PIP_TRUSTED_HOST:+--trusted-host ${PIP_TRUSTED_HOST}} \
		--default-timeout ${PIP_DEFAULT_TIMEOUT} \
		--retries ${PIP_RETRIES} \
		-r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
