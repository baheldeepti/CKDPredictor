#!/usr/bin/env python3
import requests
import json

BASE_URL = "http://127.0.0.1:8000"  # or your deployed URL

def pretty(d):
    return json.dumps(d, indent=2)

def check_health():
    print("🔍 Checking /health ...")
    r = requests.get(f"{BASE_URL}/health")
    print("✅ Status:", r.status_code)
    print(pretty(r.json()))

def test_predict():
    print("\n🧪 Testing /predict ...")
    sample = {
        "age": 55,
        "gender": 1,
        "systolicbp": 130,
        "diastolicbp": 85,
        "serumcreatinine": 1.2,
        "bunlevels": 15,
        "gfr": 60,
        "acr": 25,
        "serumelectrolytessodium": 140,
        "serumelectrolytespotassium": 4.5,
        "hemoglobinlevels": 13.5,
        "hba1c": 5.6,
        "pulsepressure": 45,
        "ureacreatinineratio": 12.5,
        "ckdstage": 2,
        "albuminuriacat": 1,
        "bp_risk": 1,
        "hyperkalemiaflag": 0,
        "anemiaflag": 0
    }
    r = requests.post(f"{BASE_URL}/predict", json=sample)
    print("✅ Status:", r.status_code)
    print(pretty(r.json()))

def check_metrics():
    print("\n📊 Checking /metrics endpoints ...")
    r1 = requests.get(f"{BASE_URL}/metrics/last_inferences")
    print("🧪 /metrics/last_inferences:", r1.status_code)
    print(pretty(r1.json()))
    
    r2 = requests.get(f"{BASE_URL}/metrics/retrain_report")
    print("🧪 /metrics/retrain_report:", r2.status_code)
    print(pretty(r2.json()))

def trigger_retrain():
    print("\n⚙️ Triggering /admin/retrain (async) ...")
    r = requests.post(f"{BASE_URL}/admin/retrain")
    print("✅ Status:", r.status_code)
    print(pretty(r.json()))

if __name__ == "__main__":
    check_health()
    test_predict()
    check_metrics()
    trigger_retrain()
