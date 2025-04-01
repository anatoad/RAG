from utils import get_text_wrapper
from retriever import Retriever
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Any
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

class State(TypedDict):
    conversation_text: Optional[str]
    chat_history: List[dict]
    query: Optional[str]
    documents: Optional[List[Document]]
    response: Optional[str]
    model: Optional[Any]

class Chatbot:
    def __init__(
        self,
        retriever: Retriever,
        llm: Any,
        prompt_template: str = None
    ) -> None:
        if not prompt_template:
            prompt_template = """
            Ești un asistent care răspunde la întrebări pe baza contextului din documentele relevante.
            Generează un răspuns formal pe baza datelor primite. Pentru fiecare afirmație furnizează explicit sursa extrasă din metadatele documentului.
            Dacă nu poți formula un răspuns pe baza datelor primite, spune că nu știi, nu încerca să inventezi un răspuns.

            Documente relevante:
            ---------------------
            {relevant_docs}

            Întrebarea utilizatorului:
            --------------
            {query_text}

            Formatează răspunsul astfel:
            <Răspunsul tău detaliat>
            Surse:
            - <sursă>: <url> <filename> și <page_number> dacă există
            """
        self.prompt = PromptTemplate(
            input_variables=["chat_history", "relevant_docs", "query_text"],
            template=prompt_template
        )
        self.formatted_prompt = None
        self.retriever = retriever
        self.state = None
        self.llm = llm
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
            "conversation_text": "",
            "chat_history": [],
            "query": query,
            "retrieved_documents": None,
            "bot_response": None,
            "llm_model": None,
        }
        return state
    
    def load_chat_history(self, state: State) -> State:
        """
        Load and format the conversation history from memory.
        Return updated state with formatted conversation text.
        """
        # Store formatted chat history
        state["conversation_text"] = "\n".join(
            [
                f"{msg["role"]}: {msg["content"]}"
                for msg in state["chat_history"]
            ]
        )
        return state

    def retrieve_documents(self, state: State) -> State:
        """
        Retrieve documents relevant to the user's query.
        Return updated state with retrieved documents.
        """
        state["documents"] = self.retriever.retrieve_documents(state["query"])
        return state
    
    def generate_response(self, state: State) -> State:
        """
        Invoke LLM to generate a response using the language model and the formatted prompt.
        """
        formatted_docs = self.retriever.format_documents(state["documents"])

        # Format the prompt
        self.formatted_prompt = self.prompt.format(
            relevant_docs=formatted_docs,
            query_text=state["query"]
        )

        response = self.llm.invoke(self.formatted_prompt)
        state["response"] = response.content

        return state
    
    def update_chat_history(self, state: State) -> State:
        """Update chat history with the latest user query and assistant response."""
        # Append new messages
        state["chat_history"].append({"role": "Utilizator", "content": state["query"]})
        state["chat_history"].append({"role": "Asistent", "content": state["response"]})
        return state
    
    def _build_graph(self) -> None:
        """
        Build the execution graph for the RAG pipeline.
        """
        # Nodes that represent the llm and functions the chatbot can call
        self.graph.add_node("load_chat_history", self.load_chat_history)
        self.graph.add_node("retrieve_documents", self.retrieve_documents)
        self.graph.add_node("generate_response", self.generate_response)
        self.graph.add_node("update_chat_history", self.update_chat_history)

        # Define execution flow: edges define how the chatbot should transition between nodes
        self.graph.set_entry_point("load_chat_history")
        self.graph.add_edge("load_chat_history", "retrieve_documents")
        self.graph.add_edge("retrieve_documents", "generate_response")
        self.graph.add_edge("generate_response", "update_chat_history")
        self.graph.add_edge("update_chat_history", END)

    def run(self, query: str):
        """
        Run the RAG pipeline for the provided query text.
        """
        if not self.state:
            self.state = self._initialize_state(query)
        else:
            self.state["query"] = query
        
        final_state = self.executor.invoke(self.state)
        self.state = final_state
    
    def _print_conversation_history(self):
        for conversation in self.state["chat_history"]:
            print(self._wrapper.fill(f"{conversation["role"].upper()}:"))
            print(self._wrapper.fill(conversation["content"]))
            print('-' * self._wrapper.width)
    
    def _print_response(self):
        for conversation in self.state["chat_history"][-2:]:
            print(self._wrapper.fill(f"{conversation["role"].upper()}:"))
            print(self._wrapper.fill(conversation["content"]))
            print('-' * self._wrapper.width)        
