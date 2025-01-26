import pytest
import requests

def test_generate_test_input():
    """Fixture to generate test input using Ollama API."""
    def _generate(prompt):
        ollama_url = "http://localhost:11434/api/generate"
        try:
            response = requests.post(
                ollama_url,
                json={"model": "llama3.2:latest", "prompt": "why is the sky blue", "stream": "false"},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json().get("generated_text", "")
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Error generating data from Ollama: {e}")
            return None
    return _generate

def test_sentiment_api():
    """Fixture to test sentiment prediction API."""
    def _test(input_text):
        api_url = "http://localhost:11434/api/generate"
        try:
            response = requests.post(
                api_url,
                json={"text": input_text},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Error testing the API: {e}")
            return None
    return _test

def test_positive_sentiment(generate_test_input, test_sentiment_api):
    positive_prompt = "Generate a sentence expressing happiness."
    positive_input = generate_test_input(positive_prompt)
    assert positive_input, "Failed to generate positive input"

    result = test_sentiment_api(positive_input)
    assert result, "API call failed"
    assert result["sentiment"] == "positive", "Unexpected sentiment"
    assert result["confidence"] > 0.7, "Confidence too low"

def test_negative_sentiment(generate_test_input, test_sentiment_api):
    negative_prompt = "Generate a sentence expressing disappointment."
    negative_input = generate_test_input(negative_prompt)
    assert negative_input, "Failed to generate negative input"

    result = test_sentiment_api(negative_input)
    assert result, "API call failed"
    assert result["sentiment"] == "negative", "Unexpected sentiment"
    assert result["confidence"] > 0.7, "Confidence too low"


