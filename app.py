import requests

def process_text_with_ai(text, api_key):
    if not api_key:
        return "Enter API key"
    if text.strip() == "":
        return "No gesture detected"

    try:
        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": "Convert these letters into a meaningful English sentence."},
                {"role": "user", "content": text}
            ]
        }

        response = requests.post(url, headers=headers, json=data)
        result = response.json()

        return result["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Error: {e}"
