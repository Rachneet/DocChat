from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, List
from langchain.schema import Document
from config.settings import settings
import logging
from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.foundation_models import ModelInference

logger = logging.getLogger(__name__)

credentials = Credentials(
                   url = "https://us-south.ml.cloud.ibm.com",
                   # api_key = "<YOUR_API_KEY>" # Normally you'd put an API key here, but we've got you covered here
                  )
client = APIClient(credentials)
project_id = "skills-network"

class ResearchAgent:
    def __init__(self):
        """Initialize the research agent with the OpenAI model."""
        # Initialize the WatsonX model
        model = ModelInference(
            model_id="meta-llama/llama-3-2-3b-instruct",
            credentials=credentials,
            project_id=project_id,
            params={"temperature": 0.3}
        )
        # self.llm = ChatOpenAI(
        #     model="gpt-4-turbo",
        #     temperature=0.3,
        # )
        self.llm = model

        self.prompt = ChatPromptTemplate.from_template(
            """Answer the following question based on the provided context. Be precise and factual.
            
            Question: {question}
            
            Context:
            {context}
            
            If the context is insufficient, respond with: "I cannot answer this question based on the provided documents."
            """
        )
        
    def generate(self, question: str, documents: List[Document]) -> Dict:
        """Generate an initial answer using the provided documents."""
        context = "\n\n".join([doc.page_content for doc in documents])
        
        chain = self.prompt | self.llm | StrOutputParser()
        try:
            answer = chain.invoke({
                "question": question,
                "context": context
            })
            logger.info(f"Generated answer: {answer}")
            logger.info(f"Context used: {context}")
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
        
        return {
            "draft_answer": answer,
            "context_used": context
        }