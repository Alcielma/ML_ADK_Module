FROM python:3.10-slim

# Instala a biblioteca do sistema libgomp, necessária para o LightGBM.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho dentro do contêiner.
WORKDIR /app

# Copia e instala as dependências.
# Copiar requirements.txt primeiro e instalá-lo permite que o Docker use o cache
# de camadas, o que agiliza o processo em builds futuros, se as dependências não mudarem.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código da sua aplicação para o contêiner.
# Copiar o projeto inteiro simplifica a configuração.
COPY . .

# Expõe a porta que o Uvicorn vai usar.
# Embora o Cloud Run injete a variável de ambiente $PORT, é uma boa prática
# expor a porta que o servidor vai usar.
EXPOSE 8080

# Comando para iniciar a aplicação.
# O comando usa a variável de ambiente PORT, garantindo que o Uvicorn
# escute na porta correta fornecida pelo Cloud Run.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

