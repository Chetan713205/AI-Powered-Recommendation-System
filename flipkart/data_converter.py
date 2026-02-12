import pandas as pd
from langchain_core.documents import Document

class DataConverter:
    def __init__(self, file_path:str):
        self.file_path=file_path 

    # This will create document for each item in the fow
    def convert(self):
        df = pd.read_csv(self.file_path)
        doc=[
            Document(page_content=row["review"], metadata={"product_name" : row["product_title"]})
            for _, row in df.iterrows()
        ]

        return doc 