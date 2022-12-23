import requests
import json
import time

if __name__ == "__main__":
    text = "The Easter bunny riding a motorcycle in New York City"
    t0 = time.time()
    for i in range(50):
        print("Iteration: ", i)
        out = requests.post(
            "http://localhost:5000/predict/", data=json.dumps({"prompt": [text]})
        )
    t1 = time.time()
    print("Inference time is: ", (t1 - t0) / 50)
    with open("output_api.png", "wb") as f:
        f.write(out.content)
