import time
import requests
import random
import concurrent.futures
import json
from datetime import datetime

API_URL = "http://localhost:8000/detect"

def generate_log():
    # Simple log generator
    events = [
        "Failed password for user root",
        "Accepted publickey for user admin",
        "Connection closed by 192.168.1.1",
        "sudo: a password is required",
        "Invalid user test from 10.0.0.1"
    ]
    return {
        "log_id": f"log-{random.randint(1000, 9999)}",
        "log_text": f"{datetime.now().isoformat()} {random.choice(events)}",
        "metadata": {
            "source": "load_test",
            "ip_address": f"192.168.1.{random.randint(1, 255)}"
        }
    }

def send_request(session):
    data = generate_log()
    start = time.time()
    try:
        resp = session.post(API_URL, json=data)
        latency = (time.time() - start) * 1000
        return resp.status_code, latency
    except Exception as e:
        print(f"Request failed: {e}")
        return 500, 0

def run_load_test(num_requests=1000, concurrency=10):
    print(f"Starting load test with {num_requests} requests, concurrency {concurrency}...")
    
    session = requests.Session()
    latencies = []
    status_codes = {}
    
    start_total = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request, session) for _ in range(num_requests)]
        
        for future in concurrent.futures.as_completed(futures):
            status, latency = future.result()
            latencies.append(latency)
            status_codes[status] = status_codes.get(status, 0) + 1
            
            if len(latencies) % 100 == 0:
                print(f"Processed {len(latencies)} requests...")
                
    end_total = time.time()
    duration = end_total - start_total
    
    avg_latency = sum(latencies) / len(latencies)
    throughput = num_requests / duration
    
    print("\n" + "="*40)
    print("LOAD TEST RESULTS")
    print("="*40)
    print(f"Total Requests: {num_requests}")
    print(f"Total Duration: {duration:.2f}s")
    print(f"Throughput: {throughput:.2f} req/s")
    print(f"Average Latency: {avg_latency:.2f}ms")
    print(f"Status Codes: {status_codes}")
    print("="*40)

if __name__ == "__main__":
    run_load_test()
