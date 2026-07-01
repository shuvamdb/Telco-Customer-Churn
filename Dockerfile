# 1) Base Image
FROM python:3.11-slim

# 2) Working directory
WORKDIR /app

# 3) Copy dependency file first, . represents copy everything inside the file
COPY requirements.txt .

# 4) Install all the libraries/dependencies, apt-get-clean - cleans the downloaded package caches
    # rm -rf (remove recurve files - i.e. everything inside folder forcefully), /var/lib/apt/lists/* - contains downloaded package info 
RUN pip install --upgrade pip \
&& pip install -r requirements.txt \
&& apt-get clean && rm -rf /var/lib/apt/lists/*

# 5)  Copy rest of the co
COPY . .

# Explicitly copy model -- below 4 lines is additional, I finalized this model to be used explicitly
COPY src/serving/model  /app/src/serving/model

COPY src/serving/model/911aacb584b545bb850df55e1dc5244a/artifacts/model /app/model
COPY src/serving/model/911aacb584b545bb850df55e1dc5244a/artifacts/feature_columns.txt /app/model
COPY src/serving/model/911aacb584b545bb850df55e1dc5244a/artifacts/preprocessing.pkl /app/model

# Ensure logs are shown
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# 6) Expose fastapi port
EXPOSE 8000

# 7) Command to run fastapi using uvicorn - This is the command Docker runs when the container starts
#    Python, run the uvicorn package (-m means python module) -- Start the FastAPI object called app inside src/app/main.py
#    -- --host 0.0.0.0 lstens to all network interfaces -- port 8000 means run the api on port 8000 (fastapi)
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]