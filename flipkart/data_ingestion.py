from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from flipkart.config import Config
from flipkart.data_converter import DataConverter 

class data_ingestion:
    def __init__(self):
        self.embeddings=HuggingFaceEndpointEmbeddings(model=Config.EMBEDDING_MODEL)
        self.v_store=AstraDBVectorStore(
            collection_name="vector_one",
            embedding=self.embeddings,
            api_endpoint=Config.ASTRA_DB_API_ENDPOINT,
            token=Config.ASTRA_DB_APPLICATION_TOKEN,
            namespace=Config.ASTRA_DB_KEYSPACE,
        )

    def ingest(self, load_existing=True):
        if load_existing==True:
            return self.v_store 
        docs=DataConverter("data/flipkart_product_review.csv").convert()
        self.v_store.add_documents(documents=docs) 
        return self.v_store
    
if __name__=="__main__":
    ingestor=data_ingestion()
    ingestor.ingest(load_existing=False) 