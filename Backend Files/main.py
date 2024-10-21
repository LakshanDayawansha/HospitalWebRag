from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import json
import os
from dotenv import load_dotenv

# Import your existing LangChain imports
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json({
                "message": message,
                "type": "bot_message"
            })

app = FastAPI()
manager = ConnectionManager()

# Your existing CORS middleware and environment setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your existing LangChain setup code here
load_dotenv()
# ... (keep all your existing chain setup code)

api_key=os.getenv("API_KEY")
db_path="./database"
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
            -avoid use emojies when answering.
            
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


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process message using your existing RAG chain
            result = conversational_rag_chain.invoke(
                {"input": message_data["message"]},
                config={
                    "configurable": {"session_id": client_id}
                }
            )["answer"]
            
            # Send response back to client
            await manager.send_message(result, client_id)
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        print(f"Error processing message: {e}")
        await manager.send_message(f"An error occurred: {str(e)}", client_id)

# Keep your existing HTTP endpoint as fallback
@app.get("/")
async def root():
    return {"message": "WebSocket Chat Server"}

if __name__ == "__main__":
    import uvicorn
    print("Starting WebSocket server... Please wait until you see 'Application startup complete' message.")
    uvicorn.run(app, host="0.0.0.0", port=8000)