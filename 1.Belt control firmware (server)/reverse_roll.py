import requests
import time

# URLs
START_URL = "http://192.168.1.198/continuous?speed=200&dir=ccw"
STOP_URL = "http://192.168.1.198/stop"

try:
    print("Starting motor...")
    response = requests.get(START_URL, timeout=5)
    print(f"Response: {response.status_code}")
    print("Press Ctrl+C to stop.")
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\nStopping motor...")
    try:
        response = requests.get(STOP_URL, timeout=5)
        print(f"Response: {response.status_code}")
    except Exception as e:
        print(f"Error sending stop command: {e}")
    print("Stopped successfully.")