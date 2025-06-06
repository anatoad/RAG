from flask import Blueprint, jsonify, request
from flask_cors import cross_origin
from pathlib import Path
CHATBOT_DIR = Path(__file__).resolve().parent
SRC_DIR = (CHATBOT_DIR.parent.parent / "src").as_posix()

import sys
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from chatbot import Chatbot
from retriever import Retriever
from reranker import Reranker
import settings
from langchain_openai import ChatOpenAI

main = Blueprint('main', __name__)

retr = Retriever()
reranker = Reranker()
llm = ChatOpenAI(model=settings.OPENAI_MODEL, temperature=settings.TEMPERATURE)
chat = Chatbot(retriever=retr, reranker=reranker, llm=llm)

@main.route('/')
def index():
    return jsonify({"Hello": "World"})

@main.route('/chat', methods=["POST"])
@cross_origin()
def invoke():
    data = request.json
    content = data["messages"][-1]["content"]
    query = content[-1]["text"]
    chat.run(query)

    return jsonify({"text": chat.get_response()}) 

@main.route('/restart')
@cross_origin()
def restart():
    global chat, retr, reranker
    retr = Retriever()
    reranker = Reranker()
    chat = Chatbot(retriever=retr, reranker=reranker, llm=llm)

    return jsonify({"Status": "Restarted"}), 200