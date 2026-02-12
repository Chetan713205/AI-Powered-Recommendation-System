from langchain_community.chat_models import ChatOpenAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from flipkart.config import Config


class RAGChainBuilder:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.model = ChatOpenAI(
            model=Config.RAG_MODEL,
            temperature=0.4,
            openai_api_key=Config.OPENROUTER_API_KEY,
            openai_api_base=Config.OPENROUTER_BASE_URL,
        )
        self.history_store={}   ## will store the session history

    ## Making a private method
    def _get_history(self, session_id:str) -> BaseChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id]=ChatMessageHistory()
        return self.history_store[session_id] 
    
    def build_chain(self):
        retriever=self.vector_store.as_retriever(search_kwargs={"k":3}) 
        context_prompt=ChatPromptTemplate.from_messages([
            ("system", "Given the chat history and user query, rewrite it as a standalone question"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        qa_prompt=ChatPromptTemplate.from_messages([
            ("system", """You are an e-commerce bot answering product related queries using reviews and titles.
            Stick to the context. Bec concise and helpful.\n{context}\n\nQUESTION: {input}"""),
            MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}")
        ])

        history_aware_retriever=create_history_aware_retriever(
            self.model, retriever, context_prompt
        )

        question_answer_chain=create_stuff_documents_chain(
            self.model, qa_prompt
        )

        rag_chain=create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

        return RunnableWithMessageHistory(
            rag_chain, 
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

#-----------------------------------------------------------------------------
"""
## 1. The Prompts: `context_prompt` vs. `qa_prompt`

In a conversational bot, you actually need two different "conversations" with the LLM to get one good answer.

* **`context_prompt` (The Rewriter):** If a user says "Tell me more," the vector database won't know what "more" refers to. This prompt tells the LLM to look at the **chat history** and the **latest input** and rewrite it into a "standalone question" (e.g., "Tell me more about the Nike Air Max shoes").
* **`qa_prompt` (The Answerer):** This is the final prompt. It takes the **context** (the product reviews/titles found by the retriever) and the **user's question** to generate the final helpful response.

### What is `MessagesPlaceholder` and `variable_name`?

Think of a standard Python string format like `f"Hello {name}"`. A `MessagesPlaceholder` is the same thing, but for **lists of messages**.

* **`MessagesPlaceholder`**: Tells LangChain, "Insert the entire list of previous chat messages here."
* **`variable_name="chat_history"`**: This is the ID or key. It tells the chain to look into the data for a key named `"chat_history"` and swap this placeholder with those specific messages.

---

## 2. The Core Functions

Your code uses three high-level LangChain "helper" functions to glue everything together:

### `create_history_aware_retriever`

This combines your `model`, your `retriever` (vector store), and the `context_prompt`.

* **Logic:** Input -> LLM rewriter -> Standalone search query -> Vector Store -> Relevant Documents.
* **Result:** It returns documents that are relevant to the *entire conversation*, not just the last message.

### `create_stuff_documents_chain`

The "Stuff" chain is the simplest way to handle documents. It literally "stuffs" all the retrieved text into the `{context}` variable of your `qa_prompt`.

* **Logic:** Documents + Prompt -> LLM -> Final Text Answer.

### `create_retrieval_chain`

This is the final "master" link. It connects the **History Aware Retriever** (which finds the right info) to the **Question Answer Chain** (which writes the final response).

---

## 3. `RunnableWithMessageHistory`

This is the "wrapper" that automates your database work. Without this, you would have to manually load the chat history from a database and save it back every time the user speaks.

* **`self._get_history`**: This is the function you wrote earlier! The wrapper calls this automatically using the `session_id` to fetch the right user's history.
* **`input_messages_key="input"`**: Tells the wrapper which part of the user's input to save to the history.
* **`history_messages_key="chat_history"`**: Tells the wrapper which key in the prompt template should be populated with the history.
* **`output_messages_key="answer"`**: Tells the wrapper to save the LLM's final response into the history so the bot remembers what *it* said in the next turn.

---

### Summary Table

| Component                        | Purpose                                                         |
| ---------------------------------| --------------------------------------------------------------- |
| **`context_prompt`**             | Turns "It" or "That" into specific product names using history. |
| **`qa_prompt`**                  | The "Persona" (e-commerce bot) and the actual instructions.     |
| **`retriever`**                  | The tool that searches your Flipkart/product database.          |
| **`rag_chain`**                  | The end-to-end logic of "Search -> Read -> Answer."             |
| **`RunnableWithMessageHistory`** | The "Memory Manager" that saves and loads chats automatically.  |

"""