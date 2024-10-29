import requests

def generate_content(api_key, prompt):
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}'
    response = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]})
    return response.json() if response.status_code == 200 else None

if __name__ == "__main__":
    API_KEY = 'AIzaSyBEeRHaeIg2gEo5vRHKP7xW2k33blZS2p8'  
    prompt = "Explain how AI works"
    result = generate_content(API_KEY, prompt)
    text = result['candidates'][0]['content']['parts'][0]['text']
    print(text)
