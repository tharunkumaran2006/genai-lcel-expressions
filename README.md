# Design and Implementation of LangChain Expression Language (LCEL) Expressions

## AIM:
To design and implement a LangChain Expression Language (LCEL) expression that utilizes at least two prompt parameters and three key components (prompt, model, and output parser), and to evaluate its functionality by analyzing relevant examples of its application in real-world scenarios.

## PROBLEM STATEMENT: 
Design an LCEL pipeline using LangChain with at least two dynamic prompt parameters.  Integrate prompt, model, and output parser components to form a complete expression.  Evaluate its functionality through real-world query-response scenarios.

## DESIGN STEPS:

#### STEP 1: Setup API and Environment: Load environment variables using dotenv and set openai.api_key from the local environment.

#### STEP 2: Create Prompt and Model: Use LangChain to define a ChatPromptTemplate and initialize ChatOpenAI for text generation.

#### STEP 3: Build a Retrieval System: Store predefined texts in DocArrayInMemorySearch with OpenAIEmbeddings and create a retriever.

#### STEP 4: Define Question-Answering Chain: Use RunnableMap to fetch relevant documents and pass them to a chat model for responses.

#### STEP 5: Invoke the Chain: Run chain.invoke() with a question to retrieve context-based answers using the LangChain pipeline.

## PROGRAM:
<h3> Name: THARUN V K</h3>
<H3> Register Number: 212223230231</H3>

### LangChain Expression Language (LCEL)
```python

import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

#!pip install pydantic==1.10.8

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
```
### Simple Chain
```python
prompt = ChatPromptTemplate.from_template(
    "tell me a short joke about {topic}"
)
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser

chain.invoke({"topic": "Engineering college"})
```

### Complex Chain
```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch

vectorstore = DocArrayInMemorySearch.from_texts(
    ["Tharun lives at Poonamallee", "SEC stands for Saveetha Engineering College"],
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

retriever.get_relevant_documents("where does Tharun live?")

retriever.get_relevant_documents("what does SEC stand for?")

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

from langchain.schema.runnable import RunnableMap

chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | output_parser

chain.invoke({"question": "where does Tharun live?"})

inputs = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
})

inputs.invoke({"question": "what does SEC stand for?"})
```


## OUTPUT:
### Simple Chain 
<img width="1063" height="293" alt="Screenshot 2025-09-19 112608" src="https://github.com/user-attachments/assets/875173cc-92ca-483b-9e1a-4e308c465ea6" />


### Complex Chain
<img width="1072" height="83" alt="Screenshot 2025-09-19 112651" src="https://github.com/user-attachments/assets/2d09d22f-fc27-4f37-b313-afbec009db75" />



## RESULT: 
The implemented LCEL expression takes at least two prompt parameters, processes them using a model, and formats the output with a parser, demonstrating its effectiveness through real-world examples.
