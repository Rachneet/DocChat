import gradio as gr
from document_processor.file_handler import DocumentProcessor
from retriever.builder import RetrieverBuilder
from agents.workflow import AgentWorkflow
from config.settings import settings
from utils.logging import logger

def main():
    """Main function to run the Gradio interface for the Enterprise Document QA application."""
    processor = DocumentProcessor()
    retriever_builder = RetrieverBuilder()
    workflow = AgentWorkflow()

    with gr.Blocks(title="Enterprise Document QA") as demo:
        gr.Markdown("## Multi-Document QA with Hallucination Checks")
        
        with gr.Row():
            with gr.Column():
                files = gr.Files(label="Upload Documents", 
                               file_types=settings.ALLOWED_TYPES)
                question = gr.Textbox(label="Question", lines=3)
                submit_btn = gr.Button("Submit")
                
            with gr.Column():
                answer_output = gr.Textbox(label="Answer", interactive=False)
                verification_output = gr.Textbox(label="Verification Report")

        def process_question(question_text, uploaded_files):
            """Process the question and uploaded files to generate an answer and verification report."""
            try:
                if not question_text.strip():
                    raise ValueError("Question cannot be empty.")
                if not uploaded_files:
                    raise ValueError("No documents uploaded.")
                
                logger.info(f"Processing question: {question_text}")
                logger.info(f"Uploaded files: {[f.name for f in uploaded_files]}")
                
                # Process documents
                processed_docs = processor.process(uploaded_files)
                logger.info(f"Processed {len(processed_docs)} document chunks.")
                
                # Build retriever
                retriever = retriever_builder.build_hybrid_retriever(processed_docs)
                logger.info("Hybrid retriever built successfully.")
                
                # Run workflow
                result = workflow.full_pipeline(question_text, retriever)
                logger.info("Workflow completed successfully.")
                
                return result["draft_answer"], result["verification_report"]
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                return f"Error: {str(e)}", ""

        submit_btn.click(
            fn=process_question,
            inputs=[question, files],
            outputs=[answer_output, verification_output]
        )

    demo.launch(server_port=7860, server_name="0.0.0.0")

if __name__ == "__main__":
    main()