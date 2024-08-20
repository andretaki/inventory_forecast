import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from predict import make_prediction

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)

app = FastAPI()

class PredictionRequest(BaseModel):
    sku: str
    days: int = 30

class PredictionResponse(BaseModel):
    sku: str
    predictions: list[float]

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    logger.info(f"Received prediction request for SKU: {request.sku}, Days: {request.days}")
    try:
        logger.info(f"Initiating prediction for SKU: {request.sku}")
        prediction = make_prediction(request.sku, request.days)
        logger.info(f"Prediction successful for SKU: {request.sku}")
        logger.debug(f"Prediction results for SKU {request.sku}: {prediction.tolist()}")
        return PredictionResponse(sku=request.sku, predictions=prediction.tolist())
    except Exception as e:
        logger.error(f"Error during prediction for SKU {request.sku}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the API server")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down the API server")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting the API server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
