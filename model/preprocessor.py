from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
import requests
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import os

load_dotenv()

# ✅ NEWS API KEY
API_KEY = "b7e6c4a2784d4360a655f50b2eaa008b"

# ✅ HUGGINGFACE TOKEN (IMPORTANT)
HF_TOKEN = os.getenv("HF_TOKEN")

# -------------------------------
# 🔹 STEP 1: FETCH NEWS
# -------------------------------
def build_articles(query, max_results=15):
    url = f"https://newsapi.org/v2/everything?q={query}&pageSize={max_results}&apiKey={API_KEY}"
    response = requests.get(url).json()
    return response.get("articles", [])


# -------------------------------
# 🔹 STEP 2: CREATE DOCUMENTS
# -------------------------------
def build_documents(articles):
    return [
        Document(
            page_content=article.get("content") or article.get("description") or "",
            metadata={
                "title": article.get("title"),
                "url": article.get("url"),
                "source": article.get("source", {}).get("name"),
            },
        )
        for article in articles
    ]


# -------------------------------
# 🔹 STEP 3: VECTOR STORE
# -------------------------------
def build_vectorstore(docs):
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(docs, embedding)


# -------------------------------
# 🔹 STEP 4: LLM (FIXED)
# -------------------------------
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-large",   # ✅ safer model (no access issues)
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN
)

chat_model = ChatHuggingFace(llm=llm)


# -------------------------------
# 🔹 STEP 5: PROMPT
# -------------------------------
prompt = PromptTemplate(
    template="""
You are an expert news analyst.

Analyze the following articles about: {query}

Context:
{context}

Provide:
1. A short summary
2. Key insights
3. Whether the news seems reliable or suspicious
""",
    input_variables=["query", "context"],
)


parser = StrOutputParser()


# -------------------------------
# 🔹 STEP 6: FAKE NEWS DETECTION
# -------------------------------
classifier = pipeline(
    "text-classification",
    model="roberta-base-openai-detector"
)


def detect_fake(text):
    # ✅ Avoid crash due to long input
    text = text[:1000]

    result = classifier(text)[0]

    return {
        "verdict": result["label"],  # REAL / FAKE
        "confidence": round(result["score"] * 100, 2),
        "reason": "Based on language pattern analysis",
    }


# -------------------------------
# 🔹 STEP 7: SUMMARIZATION
# -------------------------------
def summarize(query, retriever):
    docs = retriever.get_relevant_documents(query)

    context = "\n\n".join(doc.page_content for doc in docs)

    chain_inputs = RunnableParallel(
        {
            "query": lambda x: x["query"],
            "context": lambda x: context,
        }
    )

    chain = chain_inputs | prompt | chat_model | parser

    result = chain.invoke({"query": query})

    detection = detect_fake(context)

    return {
        "summary": result,
        "fake_news_detection": detection,
    }