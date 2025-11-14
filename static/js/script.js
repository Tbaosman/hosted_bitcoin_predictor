// static/js/script.js

// Global variables for charts
let priceChart, sentimentChart, performanceChart, featureChart, confidenceChart;
let predictionHistory = [];

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    checkStatus();
    loadHistoricalData();
});

// Tab switching function
function switchTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected tab content
    document.getElementById(tabName).classList.add('active');
    
    // Add active class to clicked tab
    event.currentTarget.classList.add('active');
    
    // Refresh charts when switching to analytics tab
    if (tabName === 'analytics') {
        setTimeout(() => {
            if (featureChart) featureChart.update();
            if (confidenceChart) confidenceChart.update();
        }, 100);
    }
}

// Initialize all charts
function initializeCharts() {
    // Price History Chart
    const priceCtx = document.getElementById('priceChart').getContext('2d');
    priceChart = new Chart(priceCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Bitcoin Price (USD)',
                data: [],
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Recent Price Trend'
                }
            },
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });
    
    // Sentiment Chart
    const sentimentCtx = document.getElementById('sentimentChart').getContext('2d');
    sentimentChart = new Chart(sentimentCtx, {
        type: 'bar',
        data: {
            labels: ['Positive', 'Neutral', 'Negative'],
            datasets: [{
                label: 'Wikipedia Sentiment',
                data: [0, 0, 0],
                backgroundColor: [
                    'rgba(40, 167, 69, 0.7)',
                    'rgba(108, 117, 125, 0.7)',
                    'rgba(220, 53, 69, 0.7)'
                ],
                borderColor: [
                    'rgb(40, 167, 69)',
                    'rgb(108, 117, 125)',
                    'rgb(220, 53, 69)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Sentiment Distribution'
                }
            }
        }
    });
    
    // Performance Chart
    const performanceCtx = document.getElementById('performanceChart').getContext('2d');
    performanceChart = new Chart(performanceCtx, {
        type: 'doughnut',
        data: {
            labels: ['Correct Predictions', 'Incorrect Predictions'],
            datasets: [{
                data: [0, 0],
                backgroundColor: [
                    'rgba(40, 167, 69, 0.7)',
                    'rgba(220, 53, 69, 0.7)'
                ],
                borderColor: [
                    'rgb(40, 167, 69)',
                    'rgb(220, 53, 69)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Prediction Accuracy'
                }
            }
        }
    });
    
    // Feature Importance Chart
    const featureCtx = document.getElementById('featureChart').getContext('2d');
    featureChart = new Chart(featureCtx, {
        type: 'bar',
        data: {
            labels: ['Price Ratio 7D', 'Trend 60D', 'Sentiment', 'Edit Count', 'Price Ratio 365D'],
            datasets: [{
                label: 'Feature Importance',
                data: [0.22, 0.18, 0.15, 0.12, 0.10],
                backgroundColor: 'rgba(102, 126, 234, 0.7)',
                borderColor: 'rgb(102, 126, 234)',
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Most Important Prediction Factors'
                }
            }
        }
    });
    
    // Confidence Distribution Chart
    const confidenceCtx = document.getElementById('confidenceChart').getContext('2d');
    confidenceChart = new Chart(confidenceCtx, {
        type: 'polarArea',
        data: {
            labels: ['High (70-100%)', 'Medium (50-70%)', 'Low (0-50%)'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: [
                    'rgba(40, 167, 69, 0.7)',
                    'rgba(255, 193, 7, 0.7)',
                    'rgba(220, 53, 69, 0.7)'
                ],
                borderColor: [
                    'rgb(40, 167, 69)',
                    'rgb(255, 193, 7)',
                    'rgb(220, 53, 69)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Confidence Level Distribution'
                }
            }
        }
    });
}

// Load historical data for charts
function loadHistoricalData() {
    // In a real app, this would fetch from your API
    // For now, we'll simulate with sample data
    
    // Sample price data (last 30 days)
    const samplePrices = [];
    let basePrice = 90000;
    for (let i = 30; i >= 0; i--) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        const price = basePrice + (Math.random() - 0.5) * 10000;
        samplePrices.push({
            date: date.toISOString().split('T')[0],
            price: Math.round(price)
        });
        basePrice = price;
    }
    
    // Update price chart
    priceChart.data.labels = samplePrices.map(item => {
        const date = new Date(item.date);
        return `${date.getMonth()+1}/${date.getDate()}`;
    });
    priceChart.data.datasets[0].data = samplePrices.map(item => item.price);
    priceChart.update();
    
    // Update sentiment chart with sample data
    sentimentChart.data.datasets[0].data = [45, 35, 20];
    sentimentChart.update();
    
    // Update performance chart with sample data
    performanceChart.data.datasets[0].data = [65, 35];
    performanceChart.update();
    
    // Update stats with sample data
    document.getElementById('accuracyStat').textContent = '65%';
    document.getElementById('upAccuracy').textContent = '68%';
    document.getElementById('downAccuracy').textContent = '62%';
    document.getElementById('avgConfidence').textContent = '71%';
    
    // Update confidence distribution
    confidenceChart.data.datasets[0].data = [12, 8, 5];
    confidenceChart.update();
    
    // Load sample prediction history
    loadSampleHistory();
}

// Load sample prediction history
function loadSampleHistory() {
    const sampleHistory = [
        { date: '2025-11-13', prediction: 'UP', confidence: 68, actual: 'UP', correct: true },
        { date: '2025-11-12', prediction: 'DOWN', confidence: 72, actual: 'DOWN', correct: true },
        { date: '2025-11-11', prediction: 'UP', confidence: 54, actual: 'DOWN', correct: false },
        { date: '2025-11-10', prediction: 'DOWN', confidence: 61, actual: 'DOWN', correct: true },
        { date: '2025-11-09', prediction: 'UP', confidence: 78, actual: 'UP', correct: true },
        { date: '2025-11-08', prediction: 'UP', confidence: 49, actual: 'DOWN', correct: false },
        { date: '2025-11-07', prediction: 'DOWN', confidence: 65, actual: 'DOWN', correct: true },
    ];
    
    predictionHistory = sampleHistory;
    updateHistoryDisplay();
}

// Update history display
function updateHistoryDisplay() {
    const historyList = document.getElementById('historyList');
    historyList.innerHTML = '';
    
    predictionHistory.forEach(item => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        
        const date = new Date(item.date);
        const formattedDate = `${date.getMonth()+1}/${date.getDate()}/${date.getFullYear()}`;
        
        historyItem.innerHTML = `
            <div>
                <strong>${formattedDate}</strong>
                <div style="font-size: 0.8em; color: #6c757d;">Actual: ${item.actual}</div>
            </div>
            <div style="text-align: right;">
                <span class="prediction-badge ${item.prediction === 'UP' ? 'badge-up' : 'badge-down'}">
                    ${item.prediction} (${item.confidence}%)
                </span>
                <div style="margin-top: 5px;">
                    ${item.correct ? 
                        '<i class="fas fa-check" style="color: #28a745;"></i>' : 
                        '<i class="fas fa-times" style="color: #dc3545;"></i>'
                    }
                </div>
            </div>
        `;
        
        historyList.appendChild(historyItem);
    });
    
    // Update performance summary
    const correctCount = predictionHistory.filter(item => item.correct).length;
    const totalCount = predictionHistory.length;
    const accuracy = Math.round((correctCount / totalCount) * 100);
    
    document.getElementById('performanceSummary').innerHTML = `
        <p><strong>Overall Accuracy:</strong> ${accuracy}% (${correctCount}/${totalCount})</p>
        <p><strong>Recent Trend:</strong> ${accuracy >= 60 ? 'Improving' : 'Stable'}</p>
        <p><strong>Best Streak:</strong> 5 consecutive correct predictions</p>
        <p><strong>Avg Confidence:</strong> ${Math.round(predictionHistory.reduce((sum, item) => sum + item.confidence, 0) / predictionHistory.length)}%</p>
    `;
}

// Core application functions
async function getPrediction() {
    showLoading('Analyzing current market data...');
    disableButtons(true);
    
    try {
        const response = await fetch('/predict');
        const data = await response.json();
        
        hideLoading();
        disableButtons(false);
        
        if (data.error) {
            showError(data.error);
            return;
        }
        
        updatePredictionDisplay(data);
        checkStatus();
        
        // Add to history
        addToHistory(data);
        
    } catch (error) {
        hideLoading();
        disableButtons(false);
        showError('Failed to get prediction: ' + error.message);
    }
}

async function updateData() {
    showLoading('Updating Wikipedia and price data...');
    disableButtons(true);
    
    try {
        const response = await fetch('/update', { method: 'POST' });
        const data = await response.json();
        
        hideLoading();
        disableButtons(false);
        
        if (data.status === 'success') {
            // Show success message with animation
            const updateBtn = document.getElementById('updateBtn');
            updateBtn.innerHTML = '<i class="fas fa-check"></i> Updated!';
            updateBtn.style.background = 'linear-gradient(135deg, #28a745, #20c997)';
            
            setTimeout(() => {
                updateBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Update Data';
                updateBtn.style.background = '';
            }, 2000);
            
            checkStatus();
        } else {
            showError('Update failed: ' + data.message);
        }
        
    } catch (error) {
        hideLoading();
        disableButtons(false);
        showError('Update failed: ' + error.message);
    }
}

async function checkStatus() {
    try {
        const response = await fetch('/status');
        const data = await response.json();
        
        document.getElementById('lastUpdate').textContent = data.last_update;
        document.getElementById('statusInfo').innerHTML = `
            Model: ${data.model_loaded ? '✅ Loaded' : '❌ Missing'}<br>
            Data: ${data.data_loaded ? '✅ Loaded' : '❌ Missing'}<br>
            Last Update: ${data.last_update}<br>
            Current Time: ${data.current_time}
        `;
    } catch (error) {
        document.getElementById('statusInfo').textContent = 'Unable to check status';
    }
}

function updatePredictionDisplay(data) {
    const card = document.getElementById('predictionCard');
    const predictionText = document.getElementById('predictionText');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceFill = document.getElementById('confidenceFill');
    const priceText = document.getElementById('priceText');
    const dateText = document.getElementById('predictionDate');
    
    // Update styling based on prediction
    card.className = 'prediction-card ' + 
        (data.prediction === 'UP' ? 'prediction-up' : 'prediction-down');
    
    // Add animation for price update
    priceText.classList.add('price-update');
    setTimeout(() => {
        priceText.classList.remove('price-update');
    }, 1500);
    
    predictionText.innerHTML = data.prediction === 'UP' ? 
        '<i class="fas fa-arrow-up"></i> PRICE WILL GO UP' : 
        '<i class="fas fa-arrow-down"></i> PRICE WILL GO DOWN';
    
    predictionText.style.color = data.prediction === 'UP' ? '#28a745' : '#dc3545';
    
    // Update confidence meter with animation
    confidenceValue.textContent = `${data.confidence}%`;
    setTimeout(() => {
        confidenceFill.style.width = `${data.confidence}%`;
        
        // Set confidence color based on level
        if (data.confidence >= 70) {
            confidenceFill.className = 'confidence-fill confidence-high';
        } else if (data.confidence >= 50) {
            confidenceFill.className = 'confidence-fill confidence-medium';
        } else {
            confidenceFill.className = 'confidence-fill confidence-low';
        }
    }, 100);
    
    priceText.textContent = `$${data.current_price.toLocaleString()}`;
    dateText.textContent = `Prediction for: ${data.prediction_date}`;
    
    // Add pulse animation for high confidence predictions
    if (data.confidence >= 80) {
        card.classList.add('pulse');
    } else {
        card.classList.remove('pulse');
    }
}

function addToHistory(predictionData) {
    // In a real app, this would be saved to a database
    // For now, we'll just add to the frontend array
    const historyItem = {
        date: new Date().toISOString().split('T')[0],
        prediction: predictionData.prediction,
        confidence: predictionData.confidence,
        // In a real app, we would check against actual price later
        actual: 'Pending',
        correct: null
    };
    
    predictionHistory.unshift(historyItem);
    if (predictionHistory.length > 10) {
        predictionHistory.pop();
    }
    
    updateHistoryDisplay();
}

function showLoading(message) {
    document.getElementById('loading').style.display = 'block';
    document.getElementById('loadingText').textContent = message;
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

function disableButtons(disabled) {
    document.getElementById('predictBtn').disabled = disabled;
    document.getElementById('updateBtn').disabled = disabled;
}

function showError(message) {
    const predictionText = document.getElementById('predictionText');
    predictionText.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Error';
    predictionText.style.color = '#dc3545';
    document.getElementById('confidenceText').textContent = message;
    document.getElementById('priceText').textContent = '$--';
    document.getElementById('confidenceValue').textContent = '0%';
    document.getElementById('confidenceFill').style.width = '0%';
}