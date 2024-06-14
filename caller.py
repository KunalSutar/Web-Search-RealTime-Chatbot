from config import *
import os
from dotenv import load_dotenv, find_dotenv
import json
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders.url import UnstructuredURLLoader
from langchain.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Weaviate
from langchain_groq import ChatGroq
from langchain.llms import CTransformers
import weaviate
import os
load_dotenv(find_dotenv())

class Caller:
    def __init__(self):
        
        self.serper_api_key = os.getenv("SERPER_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
        
        self.prompt_template = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=INPUT_VARIABLES
        )
        
        auth_config = weaviate.AuthApiKey(api_key=self.weaviate_api_key)
        
        self.client = weaviate.Client(
            url="https://test1-kqciybsp.weaviate.network",
            auth_client_secret=auth_config
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=SEPARATORS,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        self.llm = CTransformers(
            model = MODEL,
            model_type = MODEL_TYPE,
            config =  {'max_new_tokens': MAX_NEW_TOKENS,
                       'temperature': TEMPERATURE,
                       'context_length': CONTEXT_LENGTH
            }
        ) 
        
        self.hfembeddings = HuggingFaceEmbeddings(
                            model_name = EMBEDDER, 
                            model_kwargs = MODEL_KWARGS
                        )

    def find_pages(self, query):

        url = "https://google.serper.dev/search"
        data = json.dumps({"q":query})

        headers = {
            'X-API-KEY': self.serper_api_key,
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=data)
        return response.json()
    
    def research_answerer(self):
    
        research_qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type=CHAIN_TYPE,
                retriever= self.db.as_retriever(search_kwargs=SEARCH_KWARGS),
                return_source_documents=True,
                verbose=True,
                chain_type_kwargs={"prompt": self.prompt_template}
            )
        
        return research_qa_chain

    def get_urls(self, articles):
        urls = []
        titles = []
        # We will only be extracting the top 2 links from the API result
        for i in range(0, min(3, len(articles["organic"]))):
            urls.append(articles["organic"][i]["link"])
            titles.append(articles["organic"][i]["title"])
            
        return urls, titles
    
    def get_content(self, urls):
        
        loader = UnstructuredURLLoader(
            urls=urls,
            show_progress_bar=True
        )
        
        research_content = loader.load()
        return research_content
    
    def get_answer(self, query, research_content):
        
        docs = self.text_splitter.split_documents(research_content)
        self.db = Weaviate.from_documents(
            docs, 
            self.hfembeddings, 
            client=self.client, 
            by_text=False
        )
        
        bot = self.research_answerer()
        answer = bot({"query": query})
       
        return answer["result"]
    
    def search(self, query):
        
        articles = self.find_pages(query)
        urls, titles = self.get_urls(articles)
        
        content = self.get_content(urls)
        answer = self.get_answer(query, content)
        
        return answer, urls, titles