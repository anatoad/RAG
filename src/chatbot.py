from utils import get_text_wrapper, get_logger
from retriever import Retriever
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Any
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from reranker import Reranker
from pydantic import BaseModel, ValidationError
import json
from logging import Logger

class State(TypedDict):
    conversation_history: List[dict]
    query: Optional[str]
    refined_question: Optional[str]
    documents: Optional[List[Document]]
    response: Optional[str]
    model: Optional[Any]

class Source(BaseModel):
    source: str
    url: str
    page: Optional[int] = None

class LLMResponse(BaseModel):
    response: str
    sources: List[Source]

class Chatbot:
    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        llm: Any,
        k: int = 5,
        logger: Logger | None = None,
        prompt_template: str = None,
        question_refiner_prompt_template: str = None,
    ) -> None:
        if not prompt_template:
            prompt_template = (
            "Ești un asistent care răspunde la întrebări pe baza contextului din documentele relevante.\n"
            "Generează un răspuns formal pe baza datelor primite. Ia în considerare întreaga întrebare, nu ignora detalii sau părți din întrebare.\n"
            "Pentru fiecare afirmație furnizează explicit sursa extrasă din metadatele documentului.\n"
            "Dacă nu poți formula un răspuns pe baza datelor primite, spune că nu ai suficiente informații pentru a răspunde la întrebare, nu încerca să inventezi un răspuns.\n"
            "Documente relevante:\n"
            "{relevant_docs}\n\n"
            "Întrebarea utilizatorului:\n"
            "{query_text}\n\n"
            "Formatează răspunsul ca JSON astfel:\n"
            "{{\n"
            "    \"response\": \"<Răspunsul tău detaliat formatat ca Markdown>\",\n"
            "    \"sources\": [\n"
            "        {{\n"
            "            \"source\": \"<sursă>\",\n"
            "            \"url\": \"<url>\",\n"
            "            \"page\": <pagina>  # pagina este opțională, poate fi null\n"
            "        }}\n"
            "    ]\n"
            "}}\n\n"
            "Respectă formatul JSON chiar dacă nu ai informații suficiente pentru a răspunde la întrebare.\n"
            )
        if not question_refiner_prompt_template:
            question_refiner_prompt_template = (
            "Ești un asistent specializat în reformularea întrebărilor din conversații.\n"
            "Analizează întrebarea utilizatorului și conversația anterioară pentru a înțelege contextul.\n"
            "Identifică ambiguitățile și reformulează întrebarea astfel încât să fie complet clară și înțeleasă fără a fi nevoie de context suplimentar.\n"
            "Păstrează cât mai mult din structura și formularea originală, dar adaugă explicit detaliile și informațiile contextuale relevante, dacă sunt necesare.\n\n"
            "Contextul conversației:\n"
            "{conversation_history}\n\n"
            "Întrebarea:\n"
            "{query_text}\n\n"
            )
        self.prompt = PromptTemplate(
            input_variables=["relevant_docs", "query_text"],
            template=prompt_template
        )
        self.question_refiner_prompt = PromptTemplate(
            input_variables=["conversation_history", "query_text"],
            template=question_refiner_prompt_template
        )
        self.formatted_prompt = None
        self.formatted_question_refiner_prompt = None
        self.refined_question = None
        self.retriever = retriever
        self.reranker = reranker
        self.logger = logger or get_logger(__name__)
        self.state = None
        self.llm = llm
        self.k = k
        self.response = None
        self.rewrite_query = True
        self.rerank = True
        # Define the chatbot as a state machine
        self.graph = StateGraph(State)
        self._build_graph()
        self.executor = self.graph.compile()
        self._wrapper = get_text_wrapper()

    def _initialize_state(self, query: str) -> State:
        """
        Initialize the chatbot state with the user's query.
        """
        state = {
            "conversation_history": [],
            "query": query,
            "documents": None,
        }
        return state
    
    def delete_last_message(self) -> State:
        if not self.state["conversation_history"]:
            return
        self.state["conversation_history"].pop()
        self.state["conversation_history"].pop()
    
    def delete_conversation_history(self) -> None:
        self.logger.info("Deleted conversation history")
        if not self.state or not self.state["conversation_history"]:
            return
        self.state["conversation_history"].clear()

    def _format_conversation_history(self, conversation_history) -> str:
        return "\n".join(
            [
                f"{msg["role"]}: {msg["content"]}"
                for msg in conversation_history[:5]
            ]
        )

    def refine_question(self, state: State) -> State:
        """
        Use the pretrained LLM for new question generation based on the conversational history.
        Use a prompt that includes previous conversation turns along with the current ambiguous question.
        """
        if not self.rewrite_query or not state["conversation_history"]:
            self.refined_question = state["query"]
            state["refined_question"] = self.refined_question
            return state

        formatted_conversation_history = self._format_conversation_history(state["conversation_history"])

        # Format the prompt
        self.formatted_question_refiner_prompt = self.question_refiner_prompt.format(
            conversation_history=formatted_conversation_history,
            query_text=state["query"]
        )

        response = self.llm.invoke(self.formatted_question_refiner_prompt)
        self.refined_question = response.content
        state["refined_question"] = response.content

        self.logger.info("------ Query Refiner ------")
        self.logger.info(f"Initial query: {state["query"]}")
        self.logger.info(f"Refined query: {state["refined_question"]}")

        return state

    def retrieve_documents(self, state: State) -> State:
        """
        Retrieve documents relevant to the user's query.
        Return updated state with retrieved documents.
        """
        state["documents"] = self.retriever.retrieve_documents(state["refined_question"])
        return state

    def rerank_documents(self, state: State) -> State:
        """
        Rerank documents and keep top_k with the highest score.
        """
        if not self.rerank:
            return state
        state["documents"] = self.reranker.rerank_documents(state["refined_question"], state["documents"])
        state["documents"] = state["documents"][:self.k]
        return state

    def generate_response(self, state: State) -> State:
        """
        Invoke LLM to generate a response using the language model and the formatted prompt.
        """
        formatted_docs = self.retriever.format_documents(state["documents"])

        # Format the prompt
        self.formatted_prompt = self.prompt.format(
            relevant_docs=formatted_docs,
            query_text=state["refined_question"]
        )

        self.response = self.llm.invoke(self.formatted_prompt)
        self.response = self.response.content
        self.response = self.response[self.response.find("{"):self.response.rfind("}")+1]

        # Parse and validate the response
        try:
            response_obj = LLMResponse(**json.loads(self.response))
        except (ValidationError, json.JSONDecodeError) as e:
            # Create a fallback response object
            response_obj = LLMResponse(response="Nu a putut fi generat un răspuns.", sources=[])
        
        # Update the state with the validated response object
        state["response"] = response_obj.response
        if response_obj.sources:
            state["response"] += (
                "\nSurse:\n" +
                '\n'.join([
                    f"- {source.source}, URL: {source.url}, pagina: {source.page}"
                    if getattr(source, 'page', None) is not None or source.page != source.url
                    else f"- URL: {source.url}"
                    for source in response_obj.sources
                ])
            )

        return state
    
    def update_conversation_history(self, state: State) -> State:
        """Update chat history with the latest user query and assistant response."""
        # Append new messages
        state["conversation_history"].append({"role": "Utilizator", "content": state["query"]})
        state["conversation_history"].append({"role": "Asistent", "content": state["response"]})
        return state
    
    def _build_graph(self) -> None:
        """
        Build the execution graph for the RAG pipeline.
        """
        # Nodes that represent the llm and functions the chatbot can call
        self.graph.add_node("refine_question", self.refine_question)
        self.graph.add_node("retrieve_documents", self.retrieve_documents)
        self.graph.add_node("rerank_documents", self.rerank_documents)
        self.graph.add_node("generate_response", self.generate_response)
        self.graph.add_node("update_conversation_history", self.update_conversation_history)

        # Define execution flow: edges define how the chatbot should transition between nodes
        self.graph.set_entry_point("refine_question")
        self.graph.add_edge("refine_question", "retrieve_documents")
        self.graph.add_edge("retrieve_documents", "rerank_documents")
        self.graph.add_edge("rerank_documents", "generate_response")
        self.graph.add_edge("generate_response", "update_conversation_history")
        self.graph.add_edge("update_conversation_history", END)

    def run(self, query: str, rewrite_query: bool = True) -> None:
        """
        Run the RAG pipeline for the provided query text.
        """
        if not self.state:
            self.state = self._initialize_state(query)
        else:
            self.state["query"] = query
        self.rewrite_query = rewrite_query
        final_state = self.executor.invoke(self.state)
        self.state = final_state
    
    def get_response(self) -> str:
        return self.state.get("response")

    def get_context(self) -> List[str]:
        return [doc.page_content for doc in self.state.get("documents", [])]
    
    def get_documents_ids(self) -> List[str]:
        return [doc.metadata["id"] for doc in self.state.get("documents", [])]

    def get_documents(self) -> List[Document]:
        return self.state.get("documents", [])

    def _print_conversation_history(self):
        for conversation in self.state["conversation_history"]:
            print(self._wrapper.fill(f"{conversation["role"].upper()}:"))
            print(self._wrapper.fill(conversation["content"]))
            print('-' * self._wrapper.width)
    
    def _print_response(self):
        for conversation in self.state["conversation_history"][-2:]:
            print(self._wrapper.fill(f"{conversation["role"].upper()}:"))
            print(self._wrapper.fill(conversation["content"]))
            print('-' * self._wrapper.width)        
