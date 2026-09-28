import chromadb
import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from flask import Flask, request, render_template_string

app = Flask(__name__)

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

with open("my_document.txt", "r", encoding="utf-8") as f:
    document = f.read()

chunks = [p.strip() for p in document.strip().split("\n\n") if p.strip()]
def get_embedding(text):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values



# Persistent = saved to disk automatically, survives between runs
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="cat_facts")

# Only populate it once — check if it's already got data
if collection.count() == 0:
    print("Populating the database...")
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    embeddings = [get_embedding(chunk) for chunk in chunks]
    
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )
else:
    print("Database already populated, skipping.")

def retrieve(question, top_k=2):
    question_vector = get_embedding(question)
    
    results = collection.query(
        query_embeddings=[question_vector],
        n_results=top_k
    )
    
    return results["documents"][0]  # list of matching chunk texts


def answer_question(question):
    top_matches = retrieve(question, top_k=2)
    context = "\n".join(top_matches)
    
    prompt = f"""Answer the question using ONLY the information below.
If the answer isn't in the information, say "I don't know based on this information."

Information:
{context}

Question: {question}"""
    
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )
    return response.text

# A simple HTML page with a text box and a submit button
PAGE = """
<!DOCTYPE html>
<html>
<head><title>Coffee shop RAG</title></head>
<body>
    <h1>Ask about coffee shop</h1>
    <form method="POST">
        <input type="text" name="question" style="width:300px" placeholder="Ask a question...">
        <button type="submit">Ask</button>
    </form>
    {% if answer %}
        <h3>Answer:</h3>
        <p>{{ answer }}</p>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    answer = None
    if request.method == "POST":
        question = request.form["question"].strip()
        
        if not question:
            answer = "Please type a question."
        else:
            try:
                answer = answer_question(question)
            except Exception as e:
                print(f"Error: {e}")  # goes to YOUR terminal, for debugging
                answer = "Sorry, something went wrong answering that. Please try again."
    
    return render_template_string(PAGE, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
