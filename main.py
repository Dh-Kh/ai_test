from fastapi import (FastAPI, Form, File, 
                     UploadFile, HTTPException, Request)
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


templates = Jinja2Templates(directory="templates")


@app.get("/summarize/")
async def root(request: Request):
    return templates.TemplateResponse(
        request=request, name="main.html"
        )

@app.post("/summarize/")
async def summarize(
    text: str = Form(None),
    files: List[UploadFile] = File(None)
    ):
   try:
        """
        More deterministic and focused responses. (temperature=0)
        """
        
        llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

        final_text: str = ""
        
        if text:
            final_text += f" {text}"
        
        if files:
            files_storage: List = []
            for file in files:
                file_data = await file.read()
                files_storage.append(file_data.decode("utf-8"))
            final_text += " ".join(files_storage)

        if not final_text:
            return JSONResponse(content={
                "result": None
            })
                
        """
        Document loaders provide a "load" method for loading data as documents from a configured source.
        """
        
        document = Document(page_content=final_text)
        
        """
        The Map-Reduce method involves summarizing each document individually (map step) and then combining these summaries into a final summary (reduce step). 
        This approach is more scalable and can handle larger volumes of text.
        The map_reduce technique is designed for summarizing large documents that exceed the token limit of the language model. 
        It involves dividing the document into chunks, generating summaries for each chunk, and then combining these summaries to create a final summary. 
        This method is efficient for handling large files and significantly reduces processing time.
        """
        
        chain = load_summarize_chain(llm, chain_type="map_reduce")  
        
        result = chain.run([document])

        return JSONResponse(content={
            "result": result
        })
    
   except HTTPException as http_e:
        raise http_e
   except Exception as e:            
        raise HTTPException(status_code=500, detail=f"{str(e)}")

@app.exception_handler(404)
async def custom_404_handler(_, __):
    return RedirectResponse("/summarize/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)