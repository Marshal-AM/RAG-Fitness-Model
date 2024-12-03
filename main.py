from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from groq import Groq
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set up your API key and base URL
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")
client = Groq(api_key=API_KEY)

# Initialize FastAPI
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def retrieve_relevant_chunks(texts, query, top_k=5):
    """Retrieve top-k relevant chunks using TF-IDF and cosine similarity."""
    corpus = [query] + texts
    vectorizer = TfidfVectorizer().fit_transform(corpus)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)
    similarity_scores = cosine_matrix[0][1:]  # Exclude query itself
    ranked_indices = np.argsort(similarity_scores)[-top_k:]
    relevant_texts = [texts[idx] for idx in ranked_indices]
    return relevant_texts

def generate_response(chunks, query, system_prompt, model="llama-3.1-70b-versatile"):
    """Generate a response using the Groq API."""
    context = " ".join(chunks)
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context} Query: {query}"}
        ],
        model=model,
        stream=False,
        temperature=0.3,
        max_tokens=7000,
    )
    return response.choices[0].message.content

# Hardcoded system prompt
SYSTEM_PROMPT = """1.) The PDF consists of the workouts for various muscles such as Chest, Shoulder, Biceps, Triceps, Lat & Abs, Leg
2.) You will receive an input consisting of the current body condition of a Male along with the Gain/Loss value
3.) If the Gain/Loss value is "+" then it means the user wants to gain weight, if the Gain/Loss value is "-" then the user wants to lose weight and if the Gain/Loss value is the same as the user's weight, then it means the user wants to retain the same weight
4.) Your job is to give a 2 week workout plan based on the user's goal and taking into consideration the current body condition of the Male.
5.) The daily workout should be done in a time period of 1 hour with 10 minutes of cardio, 40 minutes of workout and 10 minutes of warmdown
6.) The workout plan should be generated in such a way that it targets Chest on Monday, Shoulder on Tuesday, Biceps on Wednesday, Triceps on Thursday, Lat & Abs on Friday and Leg on Saturday
7.) Example output format:
Monday: Chest
exercise1: sets x reps
targeted muscle: 
machine:
.....
exercise n:
targeted muscle: 
machine:

similarly upto Saturday for two weeks should be generated
8.) MOST IMPORTANT: Always remember that you should give a 1 hour workout plan consisting of "warmup for 10 minutes", "workouts for 40 minutes", "warmdown for 10 minutes"
This split HAS TO BE MAINTAINED at ALL COSTS!!!
9.) Even the warmup and cool down should be included in the daily workout plan in the EXACT format given. 
10.) Note: The exact workout has to be given for WARMUP and COOLDOWN remember NEVER EVER tell the user to decide and do the exercise on his own.
IMPORTANT: No notes or additional text should EVER BE generated!!!
11.) NEVER EVER fail to genrate the entire Monday to Saturday workplan. It is ABSOLUTELY NECESSARY to generate the content for all the days AT ANY COSTS!!!
"""

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    try:
        # Specify the path to the PDF file
        pdf_path = "data/RAG.pdf"

        # Extract text from the PDF
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The file {pdf_path} does not exist.")

        text = extract_text_from_pdf(pdf_path)

        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks([text], request.query)

        # Generate a response
        response = generate_response(relevant_chunks, request.query, SYSTEM_PROMPT)

        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app with `uvicorn filename:app --reload`
