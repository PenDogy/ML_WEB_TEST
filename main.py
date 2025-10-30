from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')  # ใช้ non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64, os, json, pickle
import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

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
# ✅ Serve Frontend
# =============================
frontend_path = os.path.join(os.path.dirname(__file__), "frontend")

# Mount CSS and JS folders
css_path = os.path.join(frontend_path, "css")
js_path = os.path.join(frontend_path, "js")

if os.path.exists(css_path):
    app.mount("/css", StaticFiles(directory=css_path), name="css")
if os.path.exists(js_path):
    app.mount("/js", StaticFiles(directory=js_path), name="js")
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
label_encoders = {}
original_data = None
X_train_global = None
X_test_global = None
y_train_global = None
y_test_global = None
y_pred_global = None
y_pred_proba_global = None


# =============================
# 📊 Generate Various Charts
# =============================
def generate_chart(chart_type: str, color_scheme: str = 'blue'):
    global model, original_data, X_train_global, y_train_global, y_test_global, y_pred_global, y_pred_proba_global
    
    # Color schemes
    color_maps = {
        'blue': ['#2196F3', '#1976D2', '#0D47A1'],
        'green': ['#4CAF50', '#388E3C', '#1B5E20'],
        'purple': ['#9C27B0', '#7B1FA2', '#4A148C'],
        'orange': ['#FF9800', '#F57C00', '#E65100'],
        'red': ['#F44336', '#D32F2F', '#B71C1C']
    }
    
    colors = color_maps.get(color_scheme, color_maps['blue'])
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if chart_type == 'feature_importance':
            xgb.plot_importance(model, ax=ax, max_num_features=15, color=colors[0])
            plt.title('Top 15 Feature Importance', fontsize=14, fontweight='bold')
            
        elif chart_type == 'target_distribution':
            target_counts = original_data['target'].value_counts()
            ax.bar(target_counts.index.astype(str), target_counts.values, color=colors[:len(target_counts)])
            ax.set_xlabel('Target Class', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Target Distribution', fontsize=14, fontweight='bold')
            for i, v in enumerate(target_counts.values):
                ax.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
                
        elif chart_type == 'confusion_matrix':
            cm = confusion_matrix(y_test_global, y_pred_global)
            sns.heatmap(cm, annot=True, fmt='d', cmap=f'{color_scheme}s' if color_scheme != 'blue' else 'Blues', ax=ax)
            ax.set_xlabel('Predicted', fontsize=12)
            ax.set_ylabel('Actual', fontsize=12)
            ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
            
        elif chart_type == 'roc_curve':
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_test_global, y_pred_proba_global)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[0], lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
            ax.legend(loc="lower right")
            ax.grid(alpha=0.3)
            
        elif chart_type == 'feature_correlation':
            numeric_cols = original_data.select_dtypes(include=[np.number]).columns
            corr_data = original_data[numeric_cols].corr()
            sns.heatmap(corr_data, annot=False, cmap=f'{color_scheme}s' if color_scheme != 'blue' else 'coolwarm', 
                       center=0, ax=ax, cbar_kws={'label': 'Correlation'})
            ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
        elif chart_type == 'prediction_distribution':
            ax.hist(y_pred_proba_global, bins=30, color=colors[0], alpha=0.7, edgecolor='black')
            ax.set_xlabel('Prediction Probability', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
            ax.axvline(0.5, color='red', linestyle='--', label='Decision Threshold')
            ax.legend()
            ax.grid(alpha=0.3)
            
        elif chart_type == 'class_distribution':
            numeric_features = original_data.select_dtypes(include=[np.number]).columns[:6]
            n_features = len(numeric_features)
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for idx, feature in enumerate(numeric_features):
                if idx < 6:
                    for target_val in original_data['target'].unique():
                        data = original_data[original_data['target'] == target_val][feature]
                        axes[idx].hist(data, alpha=0.6, label=f'Class {target_val}', 
                                     color=colors[int(target_val) % len(colors)], bins=20)
                    axes[idx].set_xlabel(feature, fontsize=10)
                    axes[idx].set_ylabel('Frequency', fontsize=10)
                    axes[idx].legend()
                    axes[idx].grid(alpha=0.3)
            
            plt.suptitle('Feature Distribution by Class', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        
        return image_base64
        
    except Exception as e:
        print(f"Chart generation error: {e}")
        return None


# =============================
# 📤 Upload File & Train Model (Enhanced)
# =============================
@app.post("/upload_excel")
async def upload_excel(file: UploadFile = File(...)):
    global model, model_features, feature_names, label_encoders, original_data
    global X_train_global, X_test_global, y_train_global, y_test_global, y_pred_global, y_pred_proba_global
    
    try:
        contents = await file.read()
        label_encoders = {}

        # Read various file formats
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        elif file.filename.endswith(".json"):
            df = pd.read_json(io.StringIO(contents.decode("utf-8")))
        elif file.filename.endswith(".parquet"):
            df = pd.read_parquet(io.BytesIO(contents))
        else:
            return JSONResponse({"status": "error", "message": "ไฟล์ไม่รองรับ (รองรับ: CSV, Excel, JSON, Parquet)"})

        # Check target
        if 'target' not in df.columns:
            return JSONResponse({"status": "error", "message": "ไม่มี column 'target'"})

        original_data = df.copy()
        
        # Distribution
        target_dist = df['target'].value_counts().to_dict()
        total = len(df)
        target_0_pct = (target_dist.get(0, 0) / total) * 100
        target_1_pct = (target_dist.get(1, 0) / total) * 100

        # Separate features
        X = df.drop(columns=['target'])
        y = df['target']
        feature_names = X.columns.tolist()
        
        # Encode categorical variables
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
        
        # One-hot encoding for remaining categorical
        X = pd.get_dummies(X)
        model_features = X.columns.tolist()

        # Split data
        X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
        )

        # Train XGBoost
        dtrain = xgb.DMatrix(X_train_global, label=y_train_global)
        dtest = xgb.DMatrix(X_test_global, label=y_test_global)
        scale_pos_weight = target_dist.get(0, 1) / max(target_dist.get(1, 1), 1)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 5,
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
            num_boost_round=150,
            evals=evals,
            early_stopping_rounds=15,
            verbose_eval=False
        )

        # Evaluate
        y_pred_proba_global = model.predict(dtest)
        y_pred_global = (y_pred_proba_global > 0.5).astype(int)
        accuracy = accuracy_score(y_test_global, y_pred_global)

        # Generate default chart
        image_base64 = generate_chart('feature_importance', 'blue')

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
            "image": image_base64,
            "available_charts": [
                "feature_importance",
                "target_distribution", 
                "confusion_matrix",
                "roc_curve",
                "feature_correlation",
                "prediction_distribution",
                "class_distribution"
            ]
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})


# =============================
# 📊 Get Specific Chart
# =============================
@app.get("/get_chart/{chart_type}")
def get_chart(chart_type: str, color: str = 'blue'):
    if model is None:
        return JSONResponse({"status": "error", "message": "กรุณาอัปโหลดข้อมูลก่อน"})
    
    image_base64 = generate_chart(chart_type, color)
    
    if image_base64:
        return JSONResponse({"status": "success", "image": image_base64})
    else:
        return JSONResponse({"status": "error", "message": "ไม่สามารถสร้างกราฟได้"})


# =============================
# 💾 Download Model
# =============================
@app.get("/download_model")
def download_model():
    global model, model_features, feature_names, label_encoders
    
    if model is None:
        return JSONResponse({"status": "error", "message": "ไม่มีโมเดลให้ดาวน์โหลด"})
    
    try:
        # Save model and metadata
        model_data = {
            'model': model,
            'model_features': model_features,
            'feature_names': feature_names,
            'label_encoders': label_encoders
        }
        
        buffer = io.BytesIO()
        pickle.dump(model_data, buffer)
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=xgboost_model.pkl"}
        )
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})


# =============================
# 📤 Upload Trained Model
# =============================
@app.post("/upload_model")
async def upload_model(file: UploadFile = File(...)):
    global model, model_features, feature_names, label_encoders
    
    try:
        if not file.filename.endswith('.pkl'):
            return JSONResponse({"status": "error", "message": "รองรับเฉพาะไฟล์ .pkl เท่านั้น"})
        
        contents = await file.read()
        model_data = pickle.loads(contents)
        
        # โหลดข้อมูลจาก model
        model = model_data['model']
        model_features = model_data['model_features']
        feature_names = model_data['feature_names']
        label_encoders = model_data.get('label_encoders', {})
        
        return JSONResponse({
            "status": "success",
            "message": "โหลด Model สำเร็จ!",
            "features": feature_names,
            "encoded_features": len(model_features),
            "label_encoders": len(label_encoders)
        })
        
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"ไม่สามารถโหลด Model: {str(e)}"})


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
# 🔮 Predict Endpoint (Enhanced)
# =============================
@app.post("/predict")
async def predict(data: dict):
    global model, model_features, feature_names, label_encoders
    if model is None:
        return JSONResponse({"status": "error", "message": "กรุณาอัปโหลดข้อมูลและฝึกโมเดลก่อน"})

    try:
        X = pd.DataFrame([data])
        
        # Apply label encoding
        for col, le in label_encoders.items():
            if col in X.columns:
                try:
                    X[col] = le.transform(X[col].astype(str))
                except:
                    X[col] = 0
        
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