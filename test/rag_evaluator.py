import os
import sys
if "../src" not in sys.path:
    sys.path.append("../src")
import json
import numpy as np
import settings
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from ragas.metrics import Metric, Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from langchain.embeddings import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

class RAGEvaluator:
    def __init__(self, k: int = 5, llm: ChatOpenAI | None = None):
        self.k = k
        self.data = None
        self.queries = None
        self.answers = None
        self.docs = None
        self.doc_ids = None
        self.dataset = None
        self.judge_llm = llm or ChatOpenAI(model="gpt-4.1", temperature=0, top_p=1)
        self.ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1", temperature=0, top_p=1))
        self.embedding_model = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
        # Prompts translated to Romanian from:
        # from ragas.metrics import ContextRelevance
        # ContextRelevance.template_relevance1
        # ContextRelevance.template_relevance2
        self.template_relevance1 = (
            "### Instrucțiuni\n\n"
            "Ești un expert de clasă mondială conceput să evalueze scorul de relevanță al unui Context "
            "pentru a răspunde la Întrebare.\n"
            "Sarcina ta este să determini dacă Contextul conține informații corecte pentru a răspunde la Întrebare.\n"
            "Nu te baza pe cunoștințele tale anterioare despre Întrebare.\n"
            "Folosește doar ceea ce este scris în Context și în Întrebare.\n"
            "Urmează instrucțiunile de mai jos:\n"
            "0. Dacă Contextul nu conține nicio informație relevantă pentru a răspunde la întrebare, spune 0.\n"
            "1. Dacă Contextul conține parțial informații relevante pentru a răspunde la întrebare, spune 1.\n"
            "2. Dacă Contextul conține informații relevante pentru a răspunde la întrebare, spune 2.\n"
            "Trebuie să furnizezi scorul de relevanță 0, 1 sau 2, nimic altceva.\n"
            "Nu explica.\n"
            "### Întrebare: {query}\n\n"
            "### Context: {context}\n\n"
            "Nu încerca să explici.\n"
            "Analizând Contextul și Întrebarea, scorul de relevanță este "
        )
        self.template_relevance2 = (
            "În calitate de expert special conceput să evalueze scorul de relevanță al unui Context dat în raport cu o Întrebare, "
            "sarcina mea este să determin măsura în care Contextul oferă informațiile necesare pentru a răspunde la Întrebare. "
            "Mă voi baza exclusiv pe informațiile furnizate în Context și în Întrebare, fără a utiliza cunoștințe anterioare.\n\n"
            "Iată instrucțiunile pe care le voi urma:\n"
            "* Dacă Contextul nu conține nicio informație relevantă pentru a răspunde la Întrebare, voi răspunde cu scorul de relevanță 0.\n"
            "* Dacă Contextul conține parțial informații relevante pentru a răspunde la Întrebare, voi răspunde cu scorul de relevanță 1.\n"
            "* Dacă Contextul conține informații relevante pentru a răspunde la Întrebare, voi răspunde cu scorul de relevanță 2.\n\n"
            "### Întrebare: {query}\n\n"
            "### Context: {context}\n\n"
            "Nu încerca să explici.\n"
            "Pe baza Întrebării și Contextului furnizate, scorul de relevanță este "
        )
        # Prompt translated and adapted from the RAGAs paper (http://arxiv.org/abs/2309.15217)
        self.template_faithfulness = (
            "### Instrucțiuni\n\n"
            "Fidelitatea măsoară consistența informațională a răspunsului în raport cu contextul oferit.\n"
            "Orice afirmații din răspuns care nu pot fi deduse din context trebuie penalizate.\n"
            "Având un Răspuns și un Context, acordă un scor pentru acuratețe în intervalul 0–10.\n\n"
            "Trebuie să furnizezi scorul de fidelitate din intervalul 0-10, nimic altceva.\n"
            "Nu explica.\n"
            "### Context: {context}\n\n"
            "### Răspuns: {answer}\n"
            "Nu încerca să explici.\n"
            "Pe baza Contextului și Răspunsului furnizate, scorul de fidelitate este "
        )
        self.template_answer_relevance = (
            "### Instrucțiuni\n\n"
            "Relevanța răspunsului măsoară gradul în care un răspuns se adresează direct și este adecvat pentru o întrebare dată.\n"
            "Ea penalizează prezența informațiilor redundante sau a răspunsurilor incomplete în raport cu întrebarea.\n"
            "Având o Întrebare și un Răspuns, acordă un scor pentru relevanță în intervalul 0–10.\n\n"
            "Trebuie să furnizezi scorul de relevanță din intervalul 0-10, nimic altceva.\n"
            "Nu explica.\n"
            "### Întrebare: {question}\n\n"
            "### Răspuns: {answer}\n"
            "Nu încerca să explici.\n"
            "Pe baza Întrebării și Răspunsului furnizate, scorul de relevanță este "
        )
        self.logs_dir = os.path.join(settings.BASE_DIR, "test", "logs")
        self.results_dir = os.path.join(settings.BASE_DIR, "test", "results")
        self.prompts_dir = os.path.join(settings.BASE_DIR, "test", "prompts")

    def load_data(self, data: List[Dict[str, Any]]) -> None:
        """
        Load a JSON-style list of dictionaries and convert to internal dataset format.
        JSON format:
        [
            {
                "user_input": "<query>",
                "response": "<answer>"
                "retrieved_contexts":  ["<doc1>", "<doc2>", ...],
                "retrieved_ids": ["<doc1_id>", "<doc2_id">, ...]
            },
            ...
        ]
        """
        self.data = data
        self.dataset = self._convert_to_dataset(data)
        self.queries = [item["user_input"] for item in data]
        self.answers = [item["response"] for item in data]
        self.docs = [item["retrieved_contexts"] for item in data]
        self.doc_ids = [item["retrieved_ids"] for item in data]

    def _load_json(self, filepath: str) -> List[dict]:
        with open(filepath, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
        return data
    
    def _convert_to_dataset(self, data: List[Dict[str, Any]]):
        """ Convert JSON to a RAGAs-compatible dataset. """
        return EvaluationDataset.from_dict(data)

    def _append_to_json_file(self, filepath: str, obj: dict) -> None:
        with open(filepath, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
    
        data.append(obj)

        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    def _llm_judge_context_relevance(self, query: str, document: str, document_id: int) -> int:
        """
        Based on RAGAs' context relevance metric, this function uses two independent LLMs to judge the relevance of a document.
        https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/nvidia_metrics/#context-relevance
        """
        results_filepath = f"{self.logs_dir}/llm_judge_context_relevance.json"
        data = self._load_json(results_filepath)

        item = list(filter(lambda entry: entry["query"] == query and entry["document_id"] == document_id, data))
        if item:
            return item[0]["relevance_score"]

        # two independent llm as a judge prompt calls to rate relevance on a scale of 0, 1, or 2
        llm_response1 = self.judge_llm.invoke(
            self.template_relevance1.format(query=query, context=document)
        ).content
        llm_response2 = self.judge_llm.invoke(
            self.template_relevance2.format(query=query, context=document)
        ).content

        # convert the llm responses to integers
        try:
            score1 = int(llm_response1.strip())
            score2 = int(llm_response2.strip())
        except ValueError:
            print(f"Invalid response from LLM: {llm_response1}, {llm_response2}")
            return 0

        # convert scores to [0,1] scale
        score1 = score1 / 2
        score2 = score2 / 2

        # average the scores
        avg_score = (score1 + score2) / 2
        final_score = 1 if avg_score >= 0.25 else 0

        # append llm responses to a log file
        self._append_to_json_file(results_filepath, {
            "query": query,
            "document_id": document_id,
            "llm_score_1": int(llm_response1.strip()),
            "llm_score_2": int(llm_response2.strip()),
            "relevance_score": final_score
        })

        return final_score

    def _average_precision(self, relevance_scores: List[int], k: int) -> float:
        """
        Average precision@k for a single query
        """
        hits = 0
        score = 0.0
        for i in range(k):
            if relevance_scores[i] == 1:
                hits += 1
                score += hits / (i + 1)
        return score / hits if hits > 0 else 0.0

    def _llm_judge_faithfulness(self, query: str, answer: str, docs: List[str], doc_ids: List[str]) -> int:
        # check if the pair (answer, context) has already been judged
        results_filepath = f"{self.logs_dir}/llm_judge_faithfulness.json"
        data = self._load_json(results_filepath)

        item = list(filter(lambda entry: entry["answer"] == answer and entry["context"] == doc_ids, data))
        if item:
            return item[0]["faithfulness_score"]
        
        # format context as a single string
        context = "\n\n".join(docs)

        llm_response = self.judge_llm.invoke(
            self.template_faithfulness.format(answer=answer, context=context)
        ).content

        # convert the llm response to integer
        try:
            score = int(llm_response.strip())
        except ValueError:
            print(f"Invalid response from LLM: {llm_response}")
            return 0

        # append llm responses to a log file
        self._append_to_json_file(results_filepath, {
            "query": query,
            "answer": answer,
            "context": doc_ids,
            "faithfulness_score": score,
        })

        return score

    def _llm_judge_answer_relevance(self, query: str, answer: str) -> int:
        # check if the pair (query, answer) has already been judged
        results_filepath = f"{self.logs_dir}/llm_judge_answer_relevance.json"
        data = self._load_json(results_filepath)

        item = list(filter(lambda entry: entry["answer"] == answer and entry["query"] == query, data))
        if item:
            return item[0]["relevance_score"]

        llm_response = self.judge_llm.invoke(
            self.template_answer_relevance.format(answer=answer, question=query)
        ).content

        # convert the llm response to integer
        try:
            score = int(llm_response.strip())
        except ValueError:
            print(f"Invalid response from LLM: {llm_response}")
            return 0

        # append llm responses to a log file
        self._append_to_json_file(results_filepath, {
            "query": query,
            "answer": answer,
            "relevance_score": score,
        })

        return score

    def compute_map_at_k(self) -> float:
        """ Compute MAP@k over the dataset. """
        relevance_scores = [
            [self._llm_judge_context_relevance(query, doc, doc_id) for doc, doc_id in zip(query_docs, query_doc_ids)]
            for query, query_docs, query_doc_ids in zip(self.queries, self.docs, self.doc_ids)
        ]
        ap_scores = [self._average_precision(query_scores, self.k) for query_scores in relevance_scores]
        return float(np.mean(ap_scores))

    def compute_mrr_at_k(self) -> float:
        """
        Compute MRR@k over the dataset.
        """
        relevance_scores = [
            [self._llm_judge_context_relevance(query, doc, doc_id) for doc, doc_id in zip(query_docs, query_doc_ids)]
            for query, query_docs, query_doc_ids in zip(self.queries, self.docs, self.doc_ids)
        ]
        reciprocal_ranks = []
        for i in range(len(relevance_scores)):
            reciprocal_rank = 0
            if 1 in relevance_scores[i]:
                # find the first occurrence of relevance score 1
                reciprocal_rank = 1 / (relevance_scores[i].index(1) + 1)
            reciprocal_ranks.append(reciprocal_rank)

        return float(np.mean(reciprocal_ranks))

    def compute_faithfulness_gpt_score(self) -> float:
        """
        Evaluate faithfulness of generated answers to retrieved documents.
        """
        faithfulness_scores = [
            self._llm_judge_faithfulness(query, answer, query_docs, query_doc_ids)
            for query, answer, query_docs, query_doc_ids in zip(self.queries, self.answers, self.docs, self.doc_ids)
        ]
        return float(np.mean([score / 10 for score in faithfulness_scores])) # scale to [0, 1], average over all queries

    def compute_answer_relevance_gpt_score(self) -> float:
        """
        Evaluate how relevant the generated answers are to the original queries.
        """
        answer_relevance_scores = [
            self._llm_judge_answer_relevance(query, answer)
            for query, answer in zip(self.queries, self.answers)
        ]
        return float(np.mean([score / 10 for score in answer_relevance_scores])) # scale to [0, 1], average over all queries

    async def _initialize_ragas_metric(self, metric: Metric, **kwargs) -> Metric:
        """
        Set up metric with adapted prompts and save them.
        Prompts are adapted to Romanian:
        https://docs.ragas.io/en/latest/howtos/customizations/metrics/_metrics_language_adaptation/
        """
        metric = metric(llm=self.ragas_llm, **kwargs)
        try:
            prompts = await metric.adapt_prompts(language="romanian", llm=self.ragas_llm)
            metric.set_prompts(**prompts)
            metric.save_prompts(self.prompts_dir)
        except Exception as e:
            print(f"Error saving prompts: {e}")
        return metric
    
    def _run_ragas_evaluation(self, metric: Metric, metric_name: str, results_filepath: str | None = None) -> float:
        """Run the RAGAS evaluation for a given metric."""
        # first load existing results if available
        existing_results = self._load_json(results_filepath) if results_filepath else []

        # identify which entries to evaluate
        existing_lookup = {item["user_input"]: item for item in existing_results}
        entries_to_eval, new_entries_to_add = [], []

        for i, entry in enumerate(self.dataset):
            existing_entry = existing_lookup.get(entry.user_input)

            if not existing_entry or existing_entry.get(metric_name, "nan") == "nan":
                entries_to_eval.append(i)
                new_entries_to_add.append(entry)

        if not entries_to_eval:
            print("All entries already evaluated.")
            return float(np.mean([float(item.get(metric_name)) for item in existing_results]))

        # select only the entries that need evaluation
        dataset_entries = [self.dataset[i] for i in entries_to_eval]
        dataset = EvaluationDataset(samples=dataset_entries)

        result = evaluate(
            dataset=dataset,
            metrics=[metric],
            llm=self.ragas_llm
        )
    
        # update or add new results
        for entry, score_dict in zip(new_entries_to_add, result.scores):
            user_input = entry.user_input
            score = str(score_dict.get(metric_name, "nan"))

            updated_entry = {
                "user_input": user_input,
                "retrieved_contexts": entry.retrieved_contexts,
                "response": entry.response,
                metric_name: score
            }

            if user_input in existing_lookup:
                # Update the entry
                for i, item in enumerate(existing_results):
                    if item["user_input"] == user_input:
                        existing_results[i] = updated_entry
                        break
            else:
                existing_results.append(updated_entry)

        # Save updated results
        with open(results_filepath, "w", encoding="utf-8") as f:
            json.dump(existing_results, f, indent=4, ensure_ascii=False)

        valid_scores = [float(item.get(metric_name)) for item in existing_results if item[metric_name] != "nan"]
        return float(np.mean(valid_scores))

    async def compute_faithfulness_ragas(self) -> float:
        """
        Compute faithfulness of an answer to the context using RAGAs. The response is broken down
        into claims, each claim is checked to see if it can be inferred from the context.
        The faithfulness score is computed as:
        faithfulness = (# of claims that can be inferred from context) / (total # of claims in the answer)
        """
        faithfulness = await self._initialize_ragas_metric(Faithfulness)
        result = self._run_ragas_evaluation(
            metric=faithfulness,
            metric_name="faithfulness",
            results_filepath=f"{self.logs_dir}/ragas_faithfulness_results.json"
        )
        return result

    async def compute_answer_relevance_ragas(self) -> float:
        """
        Compute the relevance of an answer to the user query using RAGAs. Generates a set of (3) questions
        for each answer, compute cosine similarity between each generated question and the query.
        The answer relevance score is the average score over all the queries.
        answer_relevance = (avg cosine similarity between query and generated questions) / (# of queries)

        https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/answer_relevance/
        """
        answer_relevance = await self._initialize_ragas_metric(ResponseRelevancy, embeddings=self.embedding_model)
        result = self._run_ragas_evaluation(
            metric=answer_relevance,
            metric_name="answer_relevancy",
            results_filepath=f"{self.logs_dir}/ragas_answer_relevance_results.json"
        )
        return result

    def evaluate_all(self) -> Dict[str, float]:
        """
        Run all evaluation metrics and return a dictionary of results.
        """
        results =  {
            "MAP@k": self.compute_map_at_k(),
            "MRR@k": self.compute_mrr_at_k(),
            "Faithfulness GPT Score": self.compute_faithfulness_gpt_score(),
            "Answer Relevance GPT Score": self.compute_answer_relevance_gpt_score()
        }
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        return results
