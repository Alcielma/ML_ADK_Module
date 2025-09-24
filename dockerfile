# Dockerfile simples para API ML_ADK_Module
FROM python:3.10-slim

WORKDIR /app

# Copia requirements e instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código do módulo ML
COPY ML/ ./ML/

# Copia o código da API
COPY src/ ./src/

# Cria diretórios necessários
RUN mkdir -p ./models ./data/processed

# Expõe a porta
EXPOSE 8080

# Rodar a aplicação
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]