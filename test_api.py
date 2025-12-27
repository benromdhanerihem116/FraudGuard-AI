import requests
import json

# 1. CONFIGURATION
URL = "http://127.0.0.1:5000/predict"

# Prepare "normal" features (close to 0) and "abnormal" features (extreme)
NORMAL_FEATURES = [0.1] * 28
FRAUD_FEATURES = [-30.0] * 28

# 2. LIST OF SCENARIOS TO TEST
test_cases = [
    {
        "name": "TEST 1: Normal Case (LEGIT)",
        "payload": {
            "time": 80,
            "amount": 25.50,
            "features": NORMAL_FEATURES
        },
        "expected_status": 200,
        "expected_result": "APPROVED"
    },
    {
        "name": "TEST 2: Fraud Case (FRAUD)",
        "payload": {
            "time": 450,
            "amount": 9999.99,
            "features": FRAUD_FEATURES
        },
        "expected_status": 200,
        "expected_result": "BLOCKED"
    },
    {
        "name": "TEST 3: Error - Missing Amount",
        "payload": {
            "time": 450,
            # "amount" is intentionally removed
            "features": NORMAL_FEATURES
        },
        "expected_status": 400, # We expect an error
        "expected_result": "error"
    },
    {
        "name": "TEST 4: Error - Missing Time",
        "payload": {
            "amount": 500.0,
            # "time" is intentionally removed
            "features": NORMAL_FEATURES
        },
        "expected_status": 400,
        "expected_result": "error"
    },
    {
        "name": "TEST 5: Error - Both Missing",
        "payload": {
            # Neither time nor amount
            "features": NORMAL_FEATURES
        },
        "expected_status": 400,
        "expected_result": "error"
    },
    {
        "name": "TEST 6: Error - Wrong Type (Text)",
        "payload": {
            "time": 450,
            "amount": "TEN EUROS", # This is not a number
            "features": NORMAL_FEATURES
        },
        "expected_status": 400,
        "expected_result": "error"
    }
]

# 3. TEST ENGINE
print(f" STARTING TESTS ON {URL} \n")

for test in test_cases:
    print(f"Running: {test['name']}...")
    
    try:
        # Sending the request
        response = requests.post(URL, json=test['payload'])
        
        # Verifying HTTP Status Code (200 or 400)
        if response.status_code == test['expected_status']:
            
            # If success (200) was expected, check content (APPROVED/BLOCKED)
            if test['expected_status'] == 200:
                data = response.json()
                if data['status'] == test['expected_result']:
                    print("  SUCCESS")
                else:
                    print(f"  FAIL: Expected {test['expected_result']}, Received {data['status']}")
            
            # If an error (400) was expected, it's a success because the API blocked it correctly
            else:
                print("  SUCCESS (API rejected the request as expected)")
                
        else:
            print(f"  FAIL: Expected Code {test['expected_status']}, Received {response.status_code}")
            print(f"    Server Response: {response.text}")

    except Exception as e:
        print(f" CRITICAL ERROR: Could not contact the server. {e}")

    print("-" * 30)

print("\n END OF TESTS ")