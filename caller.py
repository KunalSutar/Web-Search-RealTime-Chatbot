# caller.py
from config import *
import os
from dotenv import load_dotenv, find_dotenv
import json
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.url import UnstructuredURLLoader
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Weaviate
from langchain_community.embeddings import HuggingFaceEmbeddings
import weaviate
from langchain.docstore.document import Document

load_dotenv(find_dotenv())

# Attempt to import llama_cpp runtime
try:
    from llama_cpp import Llama
except Exception as e:
    Llama = None

class Caller:
    """
    Caller holds:
      - a single live LLM session (self.llm_session) which is created once and reused,
        so the model's internal KV cache/session remains active across prompts.
      - retrieval components (Weaviate + embeddings)
      - web search (Serper)
    IMPORTANT: This requires a runtime (llama_cpp or equivalent) that supports session persistence.
    """

    def __init__(self):
        self.serper_api_key = os.getenv("SERPER_API_KEY")
        self.weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=450,
            chunk_overlap=190
        )
        self.prompt_template = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

        self.hfembeddings = HuggingFaceEmbeddings(model_name=EMBEDDER, model_kwargs={"device": "cuda"})
        auth_config = weaviate.AuthApiKey(api_key=self.weaviate_api_key)
        self.client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=auth_config)
        try:
            self.vectorstore = Weaviate(client=self.client, index_name=WEAVIATE_INDEX, embedding=self.hfembeddings, text_key="text")
        except Exception:
            self.vectorstore = None

        # Live LLM session creation
        self.llm_session = None
        if Llama is not None:
            self.llm = Llama(model_path=MODEL_PATH, n_ctx=CONTEXT_LENGTH, temperature=TEMPERATURE)
            try:
                self.llm_session = self.llm.create_session()  
            except Exception:
                # If create_session does not exist, we will keep the llm object alive and use streaming calls
                # which should keep internal state if the runtime supports it.
                self.llm_session = None
        else:
            raise RuntimeError("llama_cpp not installed. Install llama-cpp-python or provide a compatible runtime.")

    # Serper web search
    def find_pages(self, query):
        url = "https://google.serper.dev/search"
        data = json.dumps({"q": query})
        headers = {
            'X-API-KEY': self.serper_api_key,
            'Content-Type': 'application/json'
        }
        resp = requests.request("POST", url, headers=headers, data=data, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # Load web content
    def get_content(self, urls):
        if not urls:
            return []
        loader = UnstructuredURLLoader(urls=urls, show_progress_bar=False)
        return loader.load()

    # Append documents to Weaviate (stateful knowledge store)
    def add_docs_to_vectorstore(self, docs):
        if not self.vectorstore:
            return
        chunks = self.text_splitter.split_documents(docs)
        try:
            self.vectorstore.add_documents(chunks)
        except Exception:
            # fallback or log error
            pass

    def build_prompt(self, context_text, question):
        return PROMPT_TEMPLATE.format(context=context_text, question=question)

    # Call the live LLM session to get an answer
    def call_llm_session(self, prompt_text):
        """
        Use the live LLM session to generate a response.
        This function MUST use the same underlying `self.llm` or `self.llm_session`
        so the runtime keeps the internal context (KV cache) between calls.
        The exact code below uses llama_cpp API shape — adjust to your installed binding.
        """
        try:
            if self.llm_session is not None:
                out = self.llm.generate(prompt_text, session=self.llm_session, max_tokens=MAX_NEW_TOKENS)
                text = "".join(choice["text"] for choice in out["choices"])
                return text
            else:
                res = self.llm(prompt_text, max_tokens=MAX_NEW_TOKENS)
                # llama_cpp returns something like {'choices': [{'text': '...'}], ...}
                if isinstance(res, dict) and "choices" in res:
                    return res["choices"][0]["text"]
                # some versions return a string directly
                return str(res)
        except Exception as e:
            # If session API not available, try a safe call and return error if unsupported
            return f"[LLM ERROR] {repr(e)}"

    # ------------------------
    # Main search function used by the app
    def search(self, query):
        # 1) web search
        articles = self.find_pages(query)
        urls, titles = [], []
        try:
            urls, titles = [], []
            for i in range(0, min(3, len(articles.get("organic", [])))):
                urls.append(articles["organic"][i]["link"])
                titles.append(articles["organic"][i]["title"])
        except Exception:
            pass

        content_docs = self.get_content(urls)  # list of Document-like objects


        if content_docs:
            self.add_docs_to_vectorstore(content_docs)

        retrieved_text = ""
        if self.vectorstore:
            try:
                results = self.vectorstore.similarity_search(query, k=SEARCH_KWARGS.get("k", 2))
                retrieved_text = "\n".join([r.page_content for r in results])
            except Exception:
                retrieved_text = ""

        prompt_text = self.build_prompt(retrieved_text, query)

        answer = self.call_llm_session(prompt_text)

        return answer, urls, titles
