from haystack import Pipeline
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.builders import PromptBuilder, ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.components.converters import MarkdownToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter, RecursiveDocumentSplitter
from haystack.dataclasses.byte_stream import ByteStream
from haystack.components.writers import DocumentWriter
from utils.litellm.core import llm
import requests
import nltk

class pytract_rag:
    def __init__(self, db='pinecone',chunking_strategy='word-400-overlap-40'):
        nltk.download('punkt')
        self.db=db.lower()
        self.cs=chunking_strategy
        self.namespace = {'sentence-5':'nvidia_cs_1', 'word-400-overlap-40':'nvidia_cs_2', 'char-1200-overlap-120':'nvidia_cs_3'}.get(chunking_strategy)
        self.document_store = PineconeDocumentStore(index="nvidia-vectors", namespace=self.namespace, dimension=1536) 
                    
    def run_nvidia_text_generation_pipeline(self, search_params, query, model='gpt-4o-mini-2024-07-18'):
        master_documents=[]
        for param in search_params:
            year, qtr = param['year'], param['qtr']
            filters = { "operator": "AND",
                        "conditions": [
                            {"field": "meta.year", "operator": "==", "value": year},
                            {"field": "meta.qtr", "operator": "==", "value": qtr},
                                ]
                        }
            text_embedder = OpenAITextEmbedder(model="text-embedding-3-small", dimensions=1536)
            retriever = PineconeEmbeddingRetriever(document_store=self.document_store, filters=filters) 
            rag_pipeline = Pipeline()
            rag_pipeline.add_component("text_embedder", text_embedder)
            rag_pipeline.add_component("retriever", retriever)
            rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
            result = rag_pipeline.run(data={"text_embedder": {"text": query}})
            master_documents.extend(result['retriever']['documents'])
            prompt_template = [
                    ChatMessage.from_user(
                        """
                        Given the following documents, answer the question in **Markdown format**.
                        
                        ## Documents:
                        {% for doc in documents %}
                        - **Document Year:{{ doc.meta['year'] }}, Quarter: {{ doc.meta['qtr'] }}**  
                        {{ doc.content }}
                        
                        {% endfor %}
                        
                        ## Question:
                        **{{query}}**
                        
                        ---
                        
                        ## Answer:
                        Format the response in the markdown format using:
                        - Headings (`##`, `###`)
                        - Bullet points (`-`, `*`)
                        - Tables (if necessary)
                        - Properly formatted code blocks for technical content (` ``` `)
                        
                        Rememeber the following instructions: 
                        - Each document is indicated as Document Year: <Year>, Quarter: <Quarter>
                        - Analyze all document contents and answer the question
                        """
                    )
                ]
        prompt_builder = ChatPromptBuilder(template=prompt_template)
        prompt = prompt_builder.run(documents=master_documents, query=query)['prompt'][0]._content[0].text
        response = llm(model, prompt)
        return response
    