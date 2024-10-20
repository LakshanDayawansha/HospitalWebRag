from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
import os

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory,RunnableLambda
from langchain_google_genai import GoogleGenerativeAI,GoogleGenerativeAIEmbeddings


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class ChatInput(BaseModel):
    message: str
    userId: str 


load_dotenv()
api_key=os.getenv("API_KEY")
db_path="D:\\subject projects\\RAG\\database"
collection_name="hospital_documents_langchain"


    
llm = GoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.3
        )

embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
            task_type="retrieval_document"
        )

retriever = Chroma(
            persist_directory=os.path.join(db_path, collection_name),
            embedding_function=embeddings,
            collection_name=collection_name
        ).as_retriever()



### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


### Answer question ###
qa_system_prompt = """You are a friendly customer service agent working for Horizon Hospitals Lanka PLC. 
            Your goal is to assist with any questions using the most relevant and up-to-date information provided in the context below. 
            When responding, ensure you:
            
            Previous conversation history:

            - Keep your tone warm, professional, and helpful, just as a caring hospital representative would.
            - Provide detailed and accurate answers, incorporating only relevant data from the context.
            - If the information doesn't directly address the question, acknowledge that politely and offer a general response if appropriate.
            - Avoid making up answers if the data does not apply. It's better to admit that the information is not available than to provide inaccurate information and mention to contact hospital via phone.
            -Make sure to greet in first message. After that it is not nessasary to greet again.
            
            Context: {context}

            Based on the context, craft a thoughtful, precise, and helpful response:
            """

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

@app.post("/chat")
async def get_response(chat_input: ChatInput):
    query = chat_input.message
    session_id = chat_input.userId
    result = conversational_rag_chain.invoke(
        {"input": query},
        config={
            "configurable": {"session_id": session_id}
        }
    )["answer"]
    
    return {"response": {"message": result}}

@app.get("/")
async def root():
    return {"message": "Welcome to the Chatbot API. Send POST requests to /chat to interact with the bot."}

    
if __name__ == "__main__":
    import uvicorn
    print("Starting server... Please wait until you see 'Application startup complete' message.")
    uvicorn.run(app)
