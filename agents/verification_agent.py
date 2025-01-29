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

class VerificationAgent:
    def __init__(self):
        model = ModelInference(
            model_id="ibm/granite-3-8b-instruct",
            credentials=credentials,
            project_id=project_id,
            params={"temperature": 0}
        )
        # self.llm = ChatOpenAI(
        #     model="gpt-4-turbo",
        #     temperature=0,
        # )
        self.llm = model
        self.prompt = ChatPromptTemplate.from_template(
            """Verify the following answer against the provided context. Check for:
            1. Direct factual support (YES/NO)
            2. Unsupported claims (list)
            3. Contradictions (list)
            4. Relevance to the question (YES/NO)
            
            Respond in this format:
            Supported: YES/NO
            Unsupported Claims: [items]
            Contradictions: [items]
            Relevant: YES/NO
            
            Answer: {answer}
            Context: {context}
            """
        )
        
    def check(self, answer: str, documents: List[Document]) -> Dict:
        """Verify the answer against the provided documents."""
        context = "\n\n".join([doc.page_content for doc in documents])
        
        chain = self.prompt | self.llm | StrOutputParser()
        try:
            verification = chain.invoke({
                "answer": answer,
                "context": context
            })
            logger.info(f"Verification report: {verification}")
            logger.info(f"Context used: {context}")
        except Exception as e:
            logger.error(f"Error verifying answer: {e}")
            raise
        
        return {
            "verification_report": verification,
            "context_used": context
        }