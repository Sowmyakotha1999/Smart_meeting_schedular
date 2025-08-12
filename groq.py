import requests
import time


# Set your Groq API key
API_KEY = "Your Groq API key"

def summarize_with_llama(text, summary_length=30):
    """Summarizes a conversation and extracts agenda, summary, and action items using LLaMA on Groq API."""
    
    # Define the improved prompt format
    prompt = f"""
    Summarize the following conversation in {summary_length} words.
    - A **one-line agenda** summarizing the main topic.
    - Then, extract and list key action items.
    - Each action item must clearly assign a task to a speaker.
    - If a task applies to multiple people, mention them explicitly.
    - If the speaker is unclear, use "The team" instead of omitting the name.
    Transcript:
    {text}

    Output format:
    Agenda: [One-line agenda]
    
    Summary: 
    [Concise summary here]
    
    Action Items:
    1. [Action item 1]
    2. [Action item 2]
    3. [Action item 3]
    """

    # Groq API endpoint
    url = "https://api.groq.com/openai/v1/chat/completions"  # ✅ Corrected endpoint
    
    # Request payload
    payload = {
        "model": "llama3-8b-8192",  # change model name here other Options: "llama3-70b-8192", "mixtral-8x7b","llama-3.1-8b-instant"
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.3
    }

    # Set headers
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # Measure request start time
    start_time = time.time()

    # Make API request
    response = requests.post(url, json=payload, headers=headers)

    # Measure request end time
    end_time = time.time()
    latency = (end_time - start_time)   

    # Print full response for debugging
    # print("\n🔹 Full API Response:", response.json())

    # Check if "choices" exist in response
    result = response.json()
    if "choices" not in result:
        return f"Error: Unexpected response format - {result}"

    # Extract and return summary with latency
    summary_text = result["choices"][0]["message"]["content"]
    return summary_text, latency

# Example usage: Read transcript from a text file
with open("RadisysTranscript.txt", "r", encoding="utf-8") as file:
    transcript_text = file.read()

# Get summarized output & latency
summary_result, response_time = summarize_with_llama(transcript_text, 50)

# Print agenda, summary, and action items along with latency
print("\n=== 🚀 LLaMA 3 8b Summarization & Action Items ===\n")
print(summary_result)
print(f"\n⏱️ API Latency: {response_time:.2f} s")
