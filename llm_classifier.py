from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

import json

def classifier(input: str) -> dict: 
    try:
        llm=ChatNVIDIA(model="mistralai/mistral-7b-instruct-v0.3", 
              nvidia_api_key="",
              temperature=0)
        
        vector_store = create_vector_store()
        docs = vector_store.similarity_search(query=input, k=1)

        prompt=ChatPromptTemplate.from_template("""
        You are a compliance assistant. Determine whether the input violates any of the listed policies.
        Classify each example as either: Scamming, Selling, Drugs, or Safe. No other exceptions are allowed for labels.
        RESPOND STRICTLY IN THIS JSON FORMAT AND NOTHING ELSE. DON'T INCLUDE EXTRA VERBAGE:

        {{"label": "The classified label given to the input.","reason": "Explanation for why the label was assigned to the input."}}

        User Input: {input}
        Policies:
        {policies}
        """
        )

        format_prompt = prompt.format_prompt(input=input, policies=docs[0].page_content)
        result = llm.invoke(input=format_prompt)

        return json.loads(result.content)

    except Exception as e:
        raise e
    
def create_vector_store():
    # Load the text file and break each policy into chunks
    policies=[]
    with open("policy.txt") as file:
        content = file.read()

        #Very rudemntary way to chunk, but this gets each policy in policy.txt file and
        #breaks them into Documents to later be embeded and added to vector store
        policies = content.replace("\n\n","").split("#")
        policies = [Document(txt.replace("#", "")) for txt in policies][:4]

    # Initialize the embedding model
    embedder = NVIDIAEmbeddings(model="ai-embed-qa-4", nvidia_api_key="")
    # Embed the policies
    embeddings = embedder.embed_documents([pol.page_content for pol in policies])

    # Initialize an in-memory vector store and add the embeddings
    vector_store = InMemoryVectorStore.from_documents(policies, embedder)
    return vector_store