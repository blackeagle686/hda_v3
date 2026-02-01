from fastapi.testclient import TestClient
from app import app
import os
from PIL import Image
import io

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_chat_endpoint():
    response = client.post(
        "/api/chat",
        data={"message": "Hello", "context": ""}
    )
    assert response.status_code == 200
    json_data = response.json()
    assert "response" in json_data
    assert len(json_data["response"]) > 0

def test_analyze_endpoint():
    # Create a dummy image
    img = Image.new('RGB', (100, 100), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    response = client.post(
        "/api/analyze",
        files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "classification" in data
    assert "heatmap_url" in data
    assert "advice" in data
    
    # Check if files were actually created in static/uploads
    # Note: TestClient runs in same process, so file writes happen
    heatmap_url = data["heatmap_url"]
    filename = heatmap_url.split("/")[-1]
    assert os.path.exists(f"static/uploads/{filename}")

if __name__ == "__main__":
    # helper to run if pytest not installed
    try:
        test_read_main()
        print("test_read_main PASSED")
        test_chat_endpoint()
        print("test_chat_endpoint PASSED")
        test_analyze_endpoint()
        print("test_analyze_endpoint PASSED")
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
    except Exception as e:
        print(f"ERROR: {e}")
