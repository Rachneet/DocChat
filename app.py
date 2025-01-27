import gradio as gr
import hashlib
from typing import List, Dict
from document_processor.file_handler import DocumentProcessor
from retriever.builder import RetrieverBuilder
from agents.workflow import AgentWorkflow
from config import constants, settings
from utils.logging import logger

def main():
    processor = DocumentProcessor()
    retriever_builder = RetrieverBuilder()
    workflow = AgentWorkflow()

    with gr.Blocks(title="Enterprise Document QA") as demo:
        gr.Markdown("## Multi-Document QA with Session Caching")
        
        session_state = gr.State({
            "file_hashes": frozenset(),
            "retriever": None
        })
        
        with gr.Row():
            with gr.Column():
                files = gr.Files(label="Upload Documents", 
                               file_types=constants.ALLOWED_TYPES)
                question = gr.Textbox(label="Question", lines=3)
                submit_btn = gr.Button("Submit")
                
            with gr.Column():
                answer_output = gr.Textbox(label="Answer", interactive=False)
                verification_output = gr.Textbox(label="Verification Report")

        def process_question(question_text: str, uploaded_files: List, state: Dict):
            """Handle questions with document caching"""
            try:
                if not question_text.strip():
                    raise ValueError("Question cannot be empty")
                if not uploaded_files:
                    raise ValueError("No documents uploaded")

                current_hashes = _get_file_hashes(uploaded_files)
                
                if state["retriever"] is None or current_hashes != state["file_hashes"]:
                    logger.info("Processing new/changed documents...")
                    chunks = processor.process(uploaded_files)
                    retriever = retriever_builder.build_hybrid_retriever(chunks)
                    
                    state.update({
                        "file_hashes": current_hashes,
                        "retriever": retriever
                    })
                
                result = workflow.full_pipeline(
                    question=question_text,
                    retriever=state["retriever"]
                )
                
                return result["draft_answer"], result["verification_report"], state
                    
            except Exception as e:
                logger.error(f"Processing error: {str(e)}")
                return f"Error: {str(e)}", "", state

        submit_btn.click(
            fn=process_question,
            inputs=[question, files, session_state],
            outputs=[answer_output, verification_output, session_state]
        )

    demo.launch(server_port=7860, server_name="0.0.0.0")

def _get_file_hashes(uploaded_files: List) -> frozenset:
    """Generate SHA-256 hashes for uploaded files"""
    hashes = set()
    for file in uploaded_files:
        with open(file.name, "rb") as f:
            hashes.add(hashlib.sha256(f.read()).hexdigest())
    return frozenset(hashes)

if __name__ == "__main__":
    main()