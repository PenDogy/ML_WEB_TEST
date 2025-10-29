from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import io, base64, os

app = FastAPI()

# =============================
# ✅ CORS (อนุญาตทุก Origin สำหรับทดสอบ)
# =============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# ✅ Serve Frontend (ถ้ามี)
# =============================
frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
def index():
    if os.path.exists(os.path.join(frontend_path, "index.html")):
        return FileResponse(os.path.join(frontend_path, "index.html"))
    return {"message": "FastAPI ML API is running 🚀"}


# =============================
# ✅ Global Variables
# =============================
model = None
model_features = None
feature_names = None


# =============================
# 📤 Upload Excel & Train Model
# =============================
@app.post("/upload_excel")
async def upload_excel(file: UploadFile = File(...)):
    global model, model_features, feature_names
    try:
        contents = await file.read()

        # Read file
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        else:
            df = pd.read_excel(io.BytesIO(contents))

        # Check target
        if 'target' not in df.columns:
            return JSONResponse({"status": "error", "message": "ไม่มี column 'target'"})

        # Distribution
        target_dist = df['target'].value_counts().to_dict()
        total = len(df)
        target_0_pct = (target_dist.get(0, 0) / total) * 100
        target_1_pct = (target_dist.get(1, 0) / total) * 100

        # Separate features
        X = df.drop(columns=['target'])
        y = df['target']
        feature_names = X.columns.tolist()
        X = pd.get_dummies(X)
        model_features = X.columns.tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
        )

        # Train XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        scale_pos_weight = target_dist.get(0, 1) / max(target_dist.get(1, 1), 1)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 4,
            "learning_rate": 0.1,
            "scale_pos_weight": scale_pos_weight,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        }

        evals = [(dtrain, 'train'), (dtest, 'test')]
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=False
        )

        # Evaluate
        y_pred_proba = model.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)

        # Feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        xgb.plot_importance(model, ax=ax, max_num_features=10)
        plt.title('Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return JSONResponse({
            "status": "success",
            "rows": len(df),
            "features": feature_names,
            "encoded_features": model_features,
            "target_distribution": {
                "0": f"{target_0_pct:.1f}%",
                "1": f"{target_1_pct:.1f}%"
            },
            "accuracy": f"{accuracy:.2%}",
            "scale_pos_weight": f"{scale_pos_weight:.2f}",
            "image": image_base64
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})


# =============================
# 📋 Get Features
# =============================
@app.get("/get_features")
def get_features():
    global feature_names
    if feature_names is None:
        return JSONResponse({"status": "error", "message": "กรุณาอัปโหลดข้อมูลก่อน"})
    return JSONResponse({"status": "success", "features": feature_names})


# =============================
# 🔮 Predict Endpoint
# =============================
@app.post("/predict")
async def predict(data: dict):
    global model, model_features, feature_names
    if model is None:
        return JSONResponse({"status": "error", "message": "กรุณาอัปโหลดข้อมูลและฝึกโมเดลก่อน"})

    try:
        X = pd.DataFrame([data])
        X = pd.get_dummies(X)
        for col in model_features:
            if col not in X.columns:
                X[col] = 0
        X = X[model_features]

        dmatrix = xgb.DMatrix(X)
        pred_proba = model.predict(dmatrix)[0]
        pred_class = int(pred_proba > 0.5)
        confidence = max(pred_proba, 1 - pred_proba) * 100

        message = (
            f"ผลการทำนาย: Positive (1) | ความมั่นใจ: {confidence:.1f}%"
            if pred_class == 1 else
            f"ผลการทำนาย: Negative (0) | ความมั่นใจ: {confidence:.1f}%"
        )

        return JSONResponse({
            "status": "success",
            "prediction": pred_class,
            "probability": float(pred_proba),
            "confidence": f"{confidence:.1f}%",
            "message": message
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})


# =============================
# 🚀 Run app (สำหรับ Railway)
# =============================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
