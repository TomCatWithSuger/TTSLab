FROM python:3.10-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    HOME=/workspace/models/pretrained \
    HF_HOME=/workspace/models/pretrained/huggingface \
    TORCH_HOME=/workspace/models/pretrained/torch \
    NLTK_DATA=/usr/local/share/nltk_data

WORKDIR /workspace

COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1+cpu torchaudio==2.5.1+cpu
RUN python -m pip install -r requirements.txt
RUN python -m pip install --no-build-isolation parallel-wavegan==0.6.1
RUN mkdir -p "$NLTK_DATA/taggers" "$NLTK_DATA/corpora" \
    && curl --fail --location --retry 3 \
        https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger.zip \
        --output /tmp/averaged_perceptron_tagger.zip \
    && curl --fail --location --retry 3 \
        https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger_eng.zip \
        --output /tmp/averaged_perceptron_tagger_eng.zip \
    && curl --fail --location --retry 3 \
        https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/cmudict.zip \
        --output /tmp/cmudict.zip \
    && unzip -q /tmp/averaged_perceptron_tagger.zip -d "$NLTK_DATA/taggers" \
    && unzip -q /tmp/averaged_perceptron_tagger_eng.zip -d "$NLTK_DATA/taggers" \
    && unzip -q /tmp/cmudict.zip -d "$NLTK_DATA/corpora" \
    && rm /tmp/averaged_perceptron_tagger.zip \
        /tmp/averaged_perceptron_tagger_eng.zip /tmp/cmudict.zip

COPY src/ src/
COPY data/ data/

CMD ["python", "src/run_experiment.py"]
