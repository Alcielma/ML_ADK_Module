import os
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from src.preprocessing import gold_transformer
from src.model import train_lgbm, lgb_predictor
from src.evaluate import evaluate_lgbm

# Configurações
warnings.filterwarnings("ignore")

# Criar aplicação FastAPI
app = FastAPI(
    title="ML ADK API",
    description="API para treinamento e predição de modelos LightGBM",
    version="1.0.0"
)

# Modelos Pydantic para requests/responses
class TrainingResponse(BaseModel):
    status: str
    message: str
    results: Dict[str, Any]

class PredictionRequest(BaseModel):
    context_window: List[List[float]]
    horizon: int = 7
    n_lags: int = 45
    n_features: int = 8

class PredictionResponse(BaseModel):
    predictions: List[float]
    status: str

# Variáveis globais
training_data = None
models_trained = False

@app.get("/")
async def root():
    """Endpoint raiz da API"""
    return {
        "message": "🚀 ML ADK API",
        "version": "1.0.0",
        "status": "API rodando com sucesso!",
        "endpoints": {
            "/": "GET - Informações da API",
            "/health": "GET - Status de saúde",
            "/train": "POST - Treinar modelos",
            "/predict": "POST - Fazer predições",
            "/status": "GET - Status dos modelos",
            "/docs": "GET - Documentação automática"
        }
    }

@app.get("/health")
async def health_check():
    """Health check para Cloud Run"""
    return {"status": "healthy", "service": "ml-adk-api"}

@app.get("/status")
async def get_status():
    """Verifica status dos dados e modelos"""
    global training_data, models_trained
    
    # Verificar se modelos existem
    model_files_exist = all(
        os.path.exists(f"lgbm_model_t{i}.pkl") for i in range(1, 8)
    )
    
    # Verificar se dados existem
    data_file_exists = os.path.exists("dados_normalizados.csv")
    
    return {
        "data_loaded": training_data is not None,
        "data_file_exists": data_file_exists,
        "models_trained": models_trained,
        "model_files_exist": model_files_exist,
        "available_models": list(range(1, 8)) if model_files_exist else []
    }

@app.post("/train", response_model=TrainingResponse)
async def train_models():
    """Treinar modelos LightGBM"""
    global training_data, models_trained
    
    try:
        print("🚀 Iniciando treinamento...")
        
        # Carregar dados
        if training_data is None:
            if os.path.exists("dados_normalizados.csv"):
                print("📊 Carregando dados...")
                training_data = pd.read_csv("dados_normalizados.csv", sep=";")
                print(f"✅ Dataset carregado: {training_data.shape}")
            else:
                raise HTTPException(
                    status_code=400, 
                    detail="Arquivo 'dados_normalizados.csv' não encontrado"
                )
        
        # Pré-processamento
        print("🔄 Pré-processando dados...")
        X_train, y_train, X_test, y_test = gold_transformer(training_data)
        print(f"✅ Treino: {X_train.shape}, Teste: {X_test.shape}")
        
        # Treinamento
        print("🤖 Treinando modelos...")
        models = train_lgbm(X_train, y_train)
        print("✅ Modelos treinados!")
        
        # Avaliação
        print("📈 Avaliando modelos...")
        results, preds, trues = evaluate_lgbm("models", X_test, y_test)
        print("✅ Avaliação concluída!")
        
        models_trained = True
        
        return TrainingResponse(
            status="success",
            message=f"Modelos treinados com sucesso! {len(models)} modelos criados.",
            results=results
        )
        
    except Exception as e:
        print(f"❌ Erro no treinamento: {e}")
        raise HTTPException(status_code=500, detail=f"Erro durante treinamento: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def make_predictions(request: PredictionRequest):
    """Fazer predições com modelos treinados"""
    try:
        print(f"🔮 Fazendo predições (horizon={request.horizon})...")
        
        # Verificar se modelos existem
        model_files_exist = all(
            os.path.exists(f"lgbm_model_t{i}.pkl") for i in range(1, request.horizon + 1)
        )
        
        if not model_files_exist:
            raise HTTPException(
                status_code=400,
                detail="Modelos não encontrados. Execute /train primeiro."
            )
        
        # Converter dados
        context_window = np.array(request.context_window)
        
        # Validar dimensões
        expected_shape = (request.n_lags, request.n_features)
        if context_window.shape != expected_shape:
            raise HTTPException(
                status_code=400,
                detail=f"Dimensões incorretas. Esperado: {expected_shape}, Recebido: {context_window.shape}"
            )
        
        # Fazer predições
        predictions = lgb_predictor(
            context_window,
            horizon=request.horizon,
            n_lags=request.n_lags,
            n_features=request.n_features
        )
        
        print(f"✅ Predições geradas: {predictions}")
        
        return PredictionResponse(
            predictions=predictions.tolist(),
            status="success"
        )
        
    except Exception as e:
        print(f"❌ Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=f"Erro durante predição: {str(e)}")

@app.post("/train-and-predict")
async def train_and_predict():
    """Pipeline completo: treina e retorna exemplo de predição"""
    try:
        # Treinar modelos
        train_result = await train_models()
        
        # Fazer predição de exemplo
        if training_data is not None:
            # Usar últimos dados como exemplo
            X_train, _, _, _ = gold_transformer(training_data)
            if len(X_train) > 0:
                example_context = X_train[-1]  # Último contexto
                
                pred_request = PredictionRequest(
                    context_window=example_context.tolist(),
                    horizon=7
                )
                pred_result = await make_predictions(pred_request)
                
                return {
                    "status": "success",
                    "message": "Pipeline completo executado",
                    "training": train_result.dict(),
                    "example_prediction": pred_result.dict()
                }
        
        return {
            "status": "success", 
            "message": "Treinamento concluído",
            "training": train_result.dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no pipeline: {str(e)}")

if __name__ == "__main__":
    # Obter porta do ambiente (Cloud Run injeta PORT)
    port = int(os.environ.get("PORT", 8080))
    
    print("🚀 ML ADK API - FastAPI Server")
    print("=" * 40)
    print(f"🌐 Porta: {port}")
    print(f"📱 Acesse: http://localhost:{port}")
    print(f"📚 Docs: http://localhost:{port}/docs")
    print("=" * 40)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False
    )
