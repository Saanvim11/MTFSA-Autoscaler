# MTFSA-Autoscaler  
**Meta + Few-Shot + Transfer Learning Hybrid Autoscaler for FaaS Cold Start Prediction**  

**Predicts function-level cold starts using only 3 hours of historical data — no fine-tuning required.**  
Achieved **RMSE 0.117 | R² 0.876 | 100% accuracy within ±0.5** on 1,272 sparse functions across 5 regions.

**Live Dashboard** → [https://mtfsa-autoscaler.onrender.com ](https://mtfsa-autoscaler-2.onrender.com/) 

---

## Key Features
- **3-shot prediction** with Meta-Few-Shot LSTM pre-trained on dense region (R1)  
- **Random Forest refinement layer** for sub-cold-start precision  
- Real-time interactive dashboard with scaling alerts  
- Tiny models: **77 KB** `.keras` + **41 KB** `.pkl`  
- Fully cloud-native deployment on Render (free tier)

## Results (31-day Huawei Cloud traces)
| Metric                       | Value     |
|------------------------------|-----------|
| RMSE                         | **0.117** |
| R² Score                     | **0.876** |
| Accuracy (±0.5 cold starts)  | **100%**  |
| Prediction Horizon           | Next hour |
| Training Data Needed         | ~3 hours  |

## Architecture


                     ┌─────────────────────┐
                     │  Dense Region (R1)   │
                     │   377 functions      │
                     └─────────▲───────────┘
                               │
                   Meta-Few-Shot Pre-training (LSTM)
                               │
                               ▼
                Transfer to Sparse 1,272 Sparse Functions (R2–R5)
                               │
                               ▼
                        3-shot Inference
                               │
                               ▼
                         LSTM Prediction
                               │
                               ▼
                      Random Forest Refiner
                               │
                               ▼
                   Final Forecast + Auto-Scaling Alert

## Project Structure
```text
├── app_final.py
├── models/
│   ├── final_model2.keras    (77 KB)
│   └── rf_refiner.pkl        (41 KB)
├── data/processed/           (Parquet traces)
├── requirements.txt
└── render.yaml               (One-click Render deploy)
```

## Quick Start 
- git clone https://github.com/Saanvim11/MTFSA-Autoscaler.git
- cd MTFSA-Autoscaler
- pip install -r requirements.txt
- Render run app_final.py
