let modelTrained = false;
let currentColor = 'blue';
let currentChartType = 'feature_importance';

// Color picker event listeners
document.querySelectorAll('.color-option').forEach(option => {
    option.addEventListener('click', function() {
        document.querySelectorAll('.color-option').forEach(o => o.classList.remove('active'));
        this.classList.add('active');
        currentColor = this.dataset.color;
    });
});

// Chart type change listener
document.getElementById('chart-type').addEventListener('change', function() {
    currentChartType = this.value;
});

// Update chart button
document.getElementById('update-chart-btn').addEventListener('click', async function() {
    const button = this;
    const btnText = button.querySelector('.btn-text');
    const originalText = btnText.textContent;
    
    btnText.innerHTML = '<span class="loading"></span>Loading...';
    button.disabled = true;

    try {
        const res = await fetch(`/get_chart/${currentChartType}?color=${currentColor}`);
        const data = await res.json();
        
        if(data.status === "success") {
            const img = document.getElementById("feature-img");
            img.src = "data:image/png;base64," + data.image;
            img.classList.add('show');
            
            const chartTitles = {
                'feature_importance': '📊 Feature Importance Analysis',
                'target_distribution': '📈 Target Distribution',
                'confusion_matrix': '🎯 Confusion Matrix',
                'roc_curve': '📉 ROC Curve Analysis',
                'feature_correlation': '🔗 Feature Correlation Heatmap',
                'prediction_distribution': '📊 Prediction Probability Distribution',
                'class_distribution': '📊 Feature Distribution by Class'
            };
            
            document.getElementById('chart-title').textContent = chartTitles[currentChartType] || '📊 Chart';
        }
    } catch(err) {
        alert('Error loading chart: ' + err.message);
    } finally {
        btnText.textContent = originalText;
        button.disabled = false;
    }
});

// Download model button
document.getElementById('download-model-btn').addEventListener('click', async function() {
    try {
        const response = await fetch('/download_model');
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'xgboost_model.pkl';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    } catch(err) {
        alert('Error downloading model: ' + err.message);
    }
});

// Upload form submission
document.getElementById("upload-form").addEventListener("submit", async function(e){
    e.preventDefault();
    const button = this.querySelector('button[type="submit"]');
    const btnText = button.querySelector('.btn-text');
    const originalText = btnText.textContent;
    
    btnText.innerHTML = '<span class="loading"></span>Processing...';
    button.disabled = true;
    
    const formData = new FormData();
    const fileField = document.getElementById('data-file');
    formData.append("file", fileField.files[0]);

    try {
        const res = await fetch("/upload_excel", { method: "POST", body: formData });
        const data = await res.json();
        
        // Remove old alerts
        this.querySelectorAll('.alert').forEach(el => el.remove());
        
        if(data.status === "success"){
            processSuccessfulUpload(data, this);
        } else {
            showAlert(this, 'error', data.message);
        }
    } catch(err) {
        showAlert(this, 'error', err.message);
    } finally {
        btnText.textContent = originalText;
        button.disabled = false;
    }
});

// Upload pre-trained model
document.getElementById("upload-model-form").addEventListener("submit", async function(e){
    e.preventDefault();
    const button = this.querySelector('button[type="submit"]');
    const btnText = button.querySelector('.btn-text');
    const originalText = btnText.textContent;
    
    btnText.innerHTML = '<span class="loading"></span>Loading Model...';
    button.disabled = true;
    
    const formData = new FormData();
    const fileField = document.getElementById('model-file');
    formData.append("file", fileField.files[0]);

    try {
        const res = await fetch("/upload_model", { method: "POST", body: formData });
        const data = await res.json();
        
        // Remove old alerts
        this.querySelectorAll('.alert').forEach(el => el.remove());
        
        if(data.status === "success"){
            // Enable prediction form
            modelTrained = true;
            const predictSection = document.getElementById('predict-section');
            predictSection.classList.add('active');
            predictSection.querySelector('p').textContent = 'Enter feature values to make a prediction';
            
            // Create dynamic form
            createDynamicForm(data.features);
            
            // Enable download button
            document.getElementById('download-model-btn').disabled = false;
            
            // Success alert
            const alert = document.createElement('div');
            alert.className = 'alert alert-success';
            alert.innerHTML = `✅ <strong>${data.message}</strong><br>
                Features: ${data.features.join(', ')}<br>
                Encoded Features: ${data.encoded_features}<br>
                Label Encoders: ${data.label_encoders}`;
            this.appendChild(alert);
            
            // Hide chart section (no training data to show charts)
            document.getElementById('chart-section').style.display = 'none';
            
            setTimeout(() => alert.remove(), 8000);
        } else {
            showAlert(this, 'error', data.message);
        }
    } catch(err) {
        showAlert(this, 'error', err.message);
    } finally {
        btnText.textContent = originalText;
        button.disabled = false;
    }
});

function processSuccessfulUpload(data, form) {
    // Show feature importance image
    const img = document.getElementById("feature-img");
    img.src = "data:image/png;base64," + data.image;
    img.classList.add('show');
    
    // Show chart section
    document.getElementById('chart-section').style.display = 'block';
    
    // Show model statistics
    const statsDiv = document.getElementById('model-stats');
    const statsGrid = statsDiv.querySelector('.stats-grid');
    statsGrid.innerHTML = `
        <div class="stat-card">
            <div class="label">Total Rows</div>
            <div class="value">${data.rows}</div>
        </div>
        <div class="stat-card">
            <div class="label">Accuracy</div>
            <div class="value">${data.accuracy}</div>
        </div>
        <div class="stat-card">
            <div class="label">Negative Cases</div>
            <div class="value">${data.target_distribution['0']}</div>
        </div>
        <div class="stat-card">
            <div class="label">Positive Cases</div>
            <div class="value">${data.target_distribution['1']}</div>
        </div>
        <div class="stat-card">
            <div class="label">Features</div>
            <div class="value">${data.features.length}</div>
        </div>
    `;
    statsDiv.style.display = 'block';
    
    // Success alert
    const alert = document.createElement('div');
    alert.className = 'alert alert-success';
    alert.innerHTML = `✅ <strong>Model trained successfully!</strong><br>
        Features: ${data.features.join(', ')}<br>
        Scale Pos Weight: ${data.scale_pos_weight}`;
    form.appendChild(alert);
    
    // Enable download button
    document.getElementById('download-model-btn').disabled = false;
    
    // Enable prediction form
    modelTrained = true;
    const predictSection = document.getElementById('predict-section');
    predictSection.classList.add('active');
    predictSection.querySelector('p').textContent = 'Enter feature values to make a prediction';
    
    // Create dynamic form
    createDynamicForm(data.features);
    
    setTimeout(() => alert.remove(), 8000);
}

function showAlert(form, type, message) {
    const alert = document.createElement('div');
    alert.className = `alert alert-${type}`;
    alert.textContent = `${type === 'error' ? '❌' : '⚠️'} ${message}`;
    form.appendChild(alert);
    setTimeout(() => alert.remove(), 5000);
}

function createDynamicForm(features) {
    const container = document.getElementById('dynamic-form-container');
    container.innerHTML = '';
    
    features.forEach(feature => {
        const formGroup = document.createElement('div');
        formGroup.className = 'form-group';
        formGroup.innerHTML = `
            <label>${feature}:</label>
            <input type="text" name="${feature}" placeholder="Enter ${feature}" required>
        `;
        container.appendChild(formGroup);
    });
    
    // Enable submit button
    document.querySelector('#predict-form button').disabled = false;
}

function getConfidenceLevel(confidence) {
    const conf = parseFloat(confidence);
    if (conf >= 90) return { level: 'Very High', class: 'positive' };
    if (conf >= 70) return { level: 'High', class: 'positive' };
    if (conf >= 60) return { level: 'Moderate', class: 'uncertain' };
    return { level: 'Low', class: 'uncertain' };
}

function getRecommendation(prediction, confidence) {
    const conf = parseFloat(confidence);
    
    if (prediction === 1) {
        if (conf >= 80) return '⚠️ High probability detected. Immediate action recommended.';
        if (conf >= 60) return '⚕️ Moderate risk detected. Consider further evaluation.';
        return '📋 Low confidence. Monitor and retest if needed.';
    } else {
        if (conf >= 80) return '✅ Low risk detected. Continue normal monitoring.';
        if (conf >= 60) return '📊 Moderate confidence. Stay vigilant.';
        return '⚠️ Uncertain result. Additional evaluation may be helpful.';
    }
}

// Predict form submission
document.getElementById("predict-form").addEventListener("submit", async function(e){
    e.preventDefault();
    
    if (!modelTrained) {
        alert('Please upload data and train the model first');
        return;
    }
    
    const button = this.querySelector('button');
    const btnText = button.querySelector('.btn-text');
    const originalText = btnText.textContent;
    
    btnText.innerHTML = '<span class="loading"></span>Predicting...';
    button.disabled = true;
    
    const formData = new FormData(e.target);
    let data = {};
    formData.forEach((value, key) => { 
        const numValue = parseFloat(value);
        data[key] = isNaN(numValue) ? value : numValue;
    });

    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: {"Content-Type":"application/json"},
            body: JSON.stringify(data)
        });
        const result = await res.json();
        
        const resultDiv = document.getElementById("prediction-result");
        resultDiv.classList.remove('positive', 'negative', 'uncertain');
        
        if(result.status === "success"){
            const confidence = parseFloat(result.confidence);
            const confLevel = getConfidenceLevel(result.confidence);
            const recommendation = getRecommendation(result.prediction, result.confidence);
            
            resultDiv.classList.add('show');
            resultDiv.classList.add(result.prediction === 1 ? confLevel.class : 'negative');
            
            resultDiv.innerHTML = `
                <div style="font-size: 2em; margin-bottom: 15px;">
                    ${result.prediction === 1 ? '✅ POSITIVE' : '❌ NEGATIVE'}
                </div>
                <div style="font-size: 1em; opacity: 0.95; margin-bottom: 10px;">
                    Confidence Level: <strong>${confLevel.level}</strong> (${result.confidence})
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${confidence}%"></div>
                </div>
                <div style="font-size: 0.9em; opacity: 0.9; margin-top: 10px;">
                    Probability Score: ${(result.probability * 100).toFixed(1)}%
                </div>
                <div class="recommendation">
                    ${recommendation}
                </div>
            `;
        } else {
            resultDiv.classList.add('show', 'negative');
            resultDiv.textContent = `⚠️ ${result.message}`;
        }
    } catch(err) {
        const resultDiv = document.getElementById("prediction-result");
        resultDiv.classList.add('show', 'negative');
        resultDiv.textContent = `❌ Error: ${err.message}`;
    } finally {
        btnText.textContent = originalText;
        button.disabled = false;
    }
});