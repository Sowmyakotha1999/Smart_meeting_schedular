import requests
import time

# Set your Groq API key
API_KEY = "Your Groq API key"

# Directly input the transcript in the code
transcript_text = """
Sowmya: Hi everyone, can you both hear me clearly?
Nathan: Yes, loud and clear! Good morning, Sowmya.
Arya: Good morning! Excited to discuss today's topic.
Sowmya: Great! So, let’s dive straight in. We are talking about AI Retrieval-Augmented Generation (RAG) versus Fine-tuning. I know both have their use cases, but I’d love to hear your thoughts on which approach works best in different scenarios.
Nathan: Absolutely! I think the key distinction is that RAG integrates external knowledge retrieval at runtime, while fine-tuning modifies the model’s internal weights based on new training data.
Arya: Right, and that’s a big difference. Fine-tuning is great when you need a model to adapt to domain-specific knowledge permanently, whereas RAG is useful for dynamically pulling in updated information without retraining the model.
Sowmya: That makes sense. But Nathan, do you think RAG is more efficient since it doesn’t require costly retraining?
Nathan: Definitely! With RAG, the model stays lightweight. You don’t need to store every detail inside the model’s parameters, which makes it more scalable. Instead, it retrieves relevant documents or data when answering queries.
Arya: True, but fine-tuning has its advantages too. When you fine-tune, the model can generalize better within a specific domain. It doesn't rely on external retrieval, so it can provide faster and more contextually consistent responses.
Sowmya: That’s an interesting point. What about handling sensitive or proprietary data? Wouldn’t fine-tuning be safer since you can train on internal datasets without exposing them externally?
Nathan: Good question. Yes, fine-tuning can be more secure because the data remains within the model’s parameters. But if you use an in-house retrieval system for RAG, you can also maintain data privacy while keeping the model up-to-date with the latest information.
Arya: Agreed. Another thing to consider is maintenance. Fine-tuning requires ongoing model retraining as new data comes in, whereas RAG can just update its document store without needing to retrain the model itself.
Sowmya: That’s a strong point for RAG. But what about accuracy? Does RAG sometimes struggle to retrieve the right information?
Nathan: It can, especially if the retrieval system isn’t optimized. If the document store lacks relevant data or the retrieval mechanism isn't fine-tuned, responses might not be as accurate.
Arya: That’s where hybrid models come in. Some companies combine RAG with fine-tuned models to get the best of both worlds—dynamic knowledge retrieval plus domain-specific accuracy.
Sowmya: That sounds like a promising approach. So, if you had to choose, which one would you go for?
Nathan: I’d say it depends on the use case. If you need real-time updates and broad coverage, RAG is the way to go. But for highly specialized applications like legal or medical AI, fine-tuning might be better.
Arya: I agree. If the knowledge is relatively stable and domain-specific, fine-tuning is worth the effort. But if the domain is constantly evolving, RAG is much more flexible.
Sowmya: This has been a great discussion. I think I now have a much clearer understanding of when to use RAG versus fine-tuning.
Nathan: Glad to help! Always enjoy geeking out over AI strategies.
Arya: Same here! Looking forward to our next deep dive.
Sowmya: Absolutely! Thanks, both of you. Let’s catch up soon.
Nathan & Arya: Sounds good! Bye for now.
"""

def summarize_with_llama(text, summary_length=50):

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
    url = "https://api.groq.com/openai/v1/chat/completions"  
    
    # Request payload
    payload = {
        "model": "llama3-70b-8192",  # Options: "llama3-70b-8192", "mixtral-8x7b"
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
    latency_seconds = end_time - start_time  # Convert to seconds

    # Print full API response for debugging
    # print("\n🔹 Full API Response:", response.json())

    # Check if "choices" exist in response
    result = response.json()
    if "choices" not in result:
        return f"Error: Unexpected response format - {result}"

    # Extract and return summary with latency
    summary_text = result["choices"][0]["message"]["content"]
    return summary_text, latency_seconds

# Get summarized output & latency
summary_result, response_time = summarize_with_llama(transcript_text, 50)

# Print agenda, summary, and action items along with latency
print("\n=== 🚀 LLaMA 3 70b Summarization & Action Items ===\n")
print(summary_result)
print(f"\n⏱️ API Latency: {response_time:.2f} s")
