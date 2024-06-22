from fastapi import (FastAPI, Form, File, 
                     UploadFile, HTTPException)
from fastapi.responses import JSONResponse
from typing import List
from dotenv import load_dotenv
from langchain.chains import MapReduceDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAI
import uvicorn
import os

load_dotenv()

app = FastAPI()

SMITH_API_KEY = os.getenv("LANGCHAIN_SMITH_API_KEY")

llm = OpenAI(api_key=SMITH_API_KEY)

"""
The Map-Reduce method involves summarizing each document individually (map step) and then combining these summaries into a final summary (reduce step). 
This approach is more scalable and can handle larger volumes of text.
The map_reduce technique is designed for summarizing large documents that exceed the token limit of the language model. 
It involves dividing the document into chunks, generating summaries for each chunk, and then combining these summaries to create a final summary. 
This method is efficient for handling large files and significantly reduces processing time.
"""

app.post("/summarize")
async def summarize(
    text: str = Form(...),
    files: List[UploadFile] = File(...),
    ):
    try:
        
        final_text: str = ""
        
        if files:
            files_storage: List = []
            for file in files:
                file_data = await file.read()
                files_storage.append(file_data.decode("utf-8"))
            final_text += "\n".join(files_storage)
            
        if text:
            final_text += f"\n{text}"
        
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=0
            )
        
        split_docs = text_splitter.split_text(final_text)
        
        map_template = f"""
        The following is a set of text
        {final_text}
        Based on this text, please identify the main themes 
        """
        
        reduce_template = f"""
        The following is set of summaries:
        {final_text}
        Take these and distill it into a final, consolidated summary of the main themes. 
        """
        
        map_prompt = PromptTemplate.from_template(
            map_template
            )
        
        reduce_prompt = PromptTemplate.from_template(
            reduce_template
            )
        
        map_chain = LLMChain(llm=llm, promt=map_prompt)
        
        reduce_chain = LLMChain(llm=llm, promt=reduce_prompt)
    
        map_reduce_chain = MapReduceDocumentsChain(
            map_chain=map_chain, 
            reduce_chain=reduce_chain
            )
        
        result = map_reduce_chain.run(split_docs)
        
        return JSONResponse(content={
            "result": result
            })
    except HTTPException as http_e:
        raise http_e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)