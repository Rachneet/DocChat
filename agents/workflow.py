from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
from .research_agent import ResearchAgent
from .verification_agent import VerificationAgent
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    question: str
    documents: List[Document]
    draft_answer: str
    verification_report: str

class AgentWorkflow:
    def __init__(self):
        self.researcher = ResearchAgent()
        self.verifier = VerificationAgent()
        self.compiled_workflow = self.build_workflow()  # Compile once during initialization
        
    def build_workflow(self):
        """Create and compile the multi-agent workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("research", self._research_step)
        workflow.add_node("verify", self._verification_step)
        
        # Define edges
        workflow.set_entry_point("research")
        workflow.add_edge("research", "verify")
        
        # Conditional edge for re-research if verification fails
        workflow.add_conditional_edges(
            "verify",
            self._decide_next_step,
            {
                "re_research": "research",
                "end": END
            }
        )
        
        return workflow.compile()
    
    def full_pipeline(self, question: str, retriever):
        """Execute the full workflow pipeline."""
        try:
            # Retrieve relevant documents
            documents = retriever.invoke(question)
            logger.info(f"Retrieved {len(documents)} relevant documents")
            
            # Initialize workflow state
            initial_state = AgentState(
                question=question,
                documents=documents,
                draft_answer="",
                verification_report=""
            )
            
            # Execute the workflow
            final_state = self.compiled_workflow.invoke(initial_state)
            
            return {
                "draft_answer": final_state["draft_answer"],
                "verification_report": final_state["verification_report"]
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise
    
    def _research_step(self, state: AgentState) -> Dict:
        """Generate an initial answer."""
        result = self.researcher.generate(state["question"], state["documents"])
        return {"draft_answer": result["draft_answer"]}
    
    def _verification_step(self, state: AgentState) -> Dict:
        """Verify the generated answer."""
        result = self.verifier.check(state["draft_answer"], state["documents"])
        return {"verification_report": result["verification_report"]}
    
    def _decide_next_step(self, state: AgentState) -> str:
        """Decide the next step based on the verification report."""
        verification_report = state["verification_report"]
        if "Supported: NO" in verification_report or "Relevant: NO" in verification_report:
            logger.info("Verification failed, re-researching...")
            return "re_research"
        else:
            logger.info("Verification successful, ending workflow.")
            return "end"