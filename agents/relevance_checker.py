"""
The RelevanceChecker is responsible for determining whether retrieved documents contain relevant information to answer a given question. 
It uses an ensemble retriever to fetch document chunks and then leverages LLM for classification. The goal is to categorize relevance into three possible labels:

* "CAN_ANSWER" - The documents provide sufficient information for a full answer.
* "PARTIAL" - The documents mention the topic but lack complete details.
* "NO_MATCH" - The documents do not discuss the question at all.

This classification helps filter out irrelevant queries, ensuring that further processing is only performed on useful data.
"""
from langchain_openai import ChatOpenAI
from config.settings import settings
import re
import logging

logger = logging.getLogger(__name__)


class RelevanceChecker:
    def __init__(self):
        # Initialize the WatsonX ModelInference
        self.model = ChatOpenAI(
            model="llama-3.3-70b-versatile",
            base_url=settings.GROQ_BASE_URL,
            api_key=settings.GROQ_API_KEY,
        )

    def check(self, question: str, retriever, k=3) -> str:
        """
        1. Retrieve the top-k document chunks from the global retriever.
        2. Combine them into a single text string.
        3. Pass that text + question to the LLM for classification.

        Returns: "CAN_ANSWER", "PARTIAL", or "NO_MATCH".
        """

        logger.debug(f"RelevanceChecker.check called with question='{question}' and k={k}")

        # Retrieve doc chunks from the ensemble retriever
        top_docs = retriever.invoke(question)
        if not top_docs:
            logger.debug("No documents returned from retriever.invoke(). Classifying as NO_MATCH.")
            return "NO_MATCH"

        # Combine the top k chunk texts into one string
        document_content = "\n\n".join(doc.page_content for doc in top_docs[:k])

        # Create a prompt for the LLM to classify relevance
        prompt = f"""
        You are an AI relevance checker between a user's question and provided document content.

        **Instructions:**
        - Classify how well the document content addresses the user's question.
        - Respond with only one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH.
        - Do not include any additional text or explanation.

        **Labels:**
        1) "CAN_ANSWER": The passages contain enough explicit information to fully answer the question.
        2) "PARTIAL": The passages mention or discuss the question's topic but do not provide all the details needed for a complete answer.
        3) "NO_MATCH": The passages do not discuss or mention the question's topic at all.

        **Important:** If the passages mention or reference the topic or timeframe of the question in any way, even if incomplete, respond with "PARTIAL" instead of "NO_MATCH".

        **Question:** {question}
        **Passages:** {document_content}

        **Respond ONLY with one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH**
        """

        # Call the LLM
        try:
            response = self.model.invoke(prompt)
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            return "NO_MATCH"

        # Extract the content from the response
        try:
            llm_response = response.content.strip().upper()
            logger.debug(f"LLM response: {llm_response}")
        except (IndexError, KeyError) as e:
            logger.error(f"Unexpected response structure: {e}")
            return "NO_MATCH"

        print(f"Checker response: {llm_response}")

        # Validate the response
        valid_labels = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}
        if llm_response not in valid_labels:
            logger.debug("LLM did not respond with a valid label. Forcing 'NO_MATCH'.")
            classification = "NO_MATCH"
        else:
            logger.debug(f"Classification recognized as '{llm_response}'.")
            classification = llm_response

        return classification


if __name__ == "__main__":
    # Example usage
    checker = RelevanceChecker()
    # Mock retriever with an invoke method for demonstration purposes
    class MockRetriever:
        def invoke(self, question):
            return [
                type('Doc', (object,), {'page_content': 'The capital of France is Paris.'})()
            ]
    retriever = MockRetriever()
    result = checker.check("What is the capital of France?", retriever)
    print(f"Relevance classification: {result}")
