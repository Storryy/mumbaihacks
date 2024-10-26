from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
from groq import Groq
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Optional
import voyageai
from langchain_core.embeddings import Embeddings
import numpy as np
import json
from pydantic import Field
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import json

# Set API keys
os.environ["GROQ_API_KEY"] = "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"
os.environ["VOYAGE_API_KEY"] = "pa-6ntBXP_c5t1kjqk1ufWeYLKrfiHJdGqh2D2vPRUovP8"

def check_api_keys():
    """Check and validate API keys"""
    voyage_key = os.getenv("VOYAGE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if not voyage_key or not groq_key:
        st.error("API keys not properly configured.")
        st.stop()
    
    return voyage_key, groq_key

class VoyageAIEmbeddings(Embeddings):
    """Embeddings class for legal document processing"""
    
    def __init__(self, voyage_client, model_name="voyage-law-2", batch_size=128):
        self.client = voyage_client
        self.model_name = model_name
        self.dimension = None
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed legal documents with specialized focus on legal content"""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            try:
                embeddings = self.client.embed(batch, model=self.model_name, input_type="document")
                if hasattr(embeddings, 'embeddings'):
                    numpy_embeddings = [np.array(emb) for emb in embeddings.embeddings]
                elif isinstance(embeddings, list):
                    numpy_embeddings = [np.array(emb) for emb in embeddings]
                else:
                    raise ValueError(f"Unexpected embeddings type: {type(embeddings)}")
                
                all_embeddings.extend(numpy_embeddings)
            except Exception as e:
                st.error(f"Error processing batch: {str(e)}")
                raise

        return [emb.tolist() for emb in all_embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed user queries with legal context awareness"""
        try:
            embedding = self.client.embed([text], model=self.model_name, input_type="query")
            if hasattr(embedding, 'embeddings'):
                numpy_embedding = np.array(embedding.embeddings[0])
            elif isinstance(embedding, list):
                numpy_embedding = np.array(embedding[0])
            else:
                raise ValueError(f"Unexpected query embedding type: {type(embedding)}")
            
            return numpy_embedding.tolist()
        except Exception as e:
            st.error(f"Error embedding query: {str(e)}")
            raise

class FIRAssistantLLM(LLM):
    """Enhanced FIR Assistant with conversation capabilities"""
    
    client: Any = Field(default=None)
    model_name: str = Field(default="llama3-70b-8192")
    vectorstore: Any = Field(default=None)
    memory: Any = Field(default=None)
    collected_info: dict = Field(default_factory=dict)
    information_complete: bool = Field(default=False)
    
    def __init__(self, vectorstore, **data):
        super().__init__(**data)
        _, groq_key = check_api_keys()
        self.client = Groq(api_key=groq_key)
        self.vectorstore = vectorstore
        self.memory = ConversationBufferMemory()
        self.collected_info = {
            "fir_details": {},
            "complainant_details": {},
            "accused_details": {},
            "witnesses": [],
            "evidence_details": {},
            "incident_description": "",
            "legal_analysis": {
                "applicable_sections": [],
                "maximum_punishment": "",
                "offense_category": "",
                "bail_status": "",
                "recommended_immediate_actions": []
            }
        }
        self.information_complete = False
    
    @property
    def _llm_type(self) -> str:
        return "groq"

    @property
    def chat_system_prompt(self):
        return """You are a helpful FIR Assistant engaged in gathering information about an incident. Your goal is to:
        1. Collect all necessary details for filing an FIR through natural conversation
        2. Ask relevant follow-up questions to gather missing information
        3. Be empathetic and professional while dealing with sensitive information
        4. Store important details shared during the conversation
        
        Important guidelines:
        - Ask one question at a time
        - Acknowledge the information shared
        - Show empathy when discussing sensitive topics
        - Keep track of what information has been collected
        - Guide the conversation to gather all necessary FIR details
        - Do NOT generate the final FIR until the user explicitly confirms they have shared all information"""

    @property
    def fir_system_prompt(self):
        return """You are a legal expert tasked with generating a detailed FIR report with legal analysis. Your responsibilities include:
        1. Analyzing the collected information against the legal context
        2. Identifying and citing specific applicable IPC sections
        3. Providing detailed analysis of each applicable section
        4. Determining offense category and bail status
        5. Recommending immediate actions based on case severity
        
        For each applicable section, provide:
        - Section number and title
        - Specific clauses that apply
        - How the facts of the case align with the section
        - Maximum punishment under that section
        - Any relevant case law or precedents
        
        The legal analysis must be comprehensive and actionable."""

    def extract_information(self, text: str) -> dict:
        """Extract relevant information from text and categorize it"""
        try:
            # Create a template for expected information structure
            info_template = {
                "fir_details": {
                    "date": "",
                    "time": "",
                    "location": "",
                    "incident_type": ""
                },
                "complainant_details": {
                    "name": "",
                    "contact": "",
                    "address": "",
                    "id_proof": ""
                },
                "accused_details": {
                    "name": "",
                    "description": "",
                    "known_address": "",
                    "identifying_marks": ""
                },
                "witnesses": [],
                "evidence_details": {
                    "physical_evidence": [],
                    "digital_evidence": [],
                    "documents": []
                },
                "incident_description": "",
                "legal_analysis": {
                    "applicable_sections": [],
                    "maximum_punishment": "",
                    "offense_category": "",
                    "bail_status": "",
                    "recommended_immediate_actions": []
                }
            }
            
            extraction_prompt = f"""Extract and categorize information from this text into appropriate FIR categories.
            Only extract factual information that is explicitly stated in the text.
            
            Text to analyze: {text}
            
            Format the response as a valid JSON object using this structure:
            {json.dumps(info_template, indent=2)}
            
            Rules:
            1. Only include information explicitly mentioned in the text
            2. Leave fields empty if no relevant information is found
            3. Don't make assumptions or infer information
            4. For arrays, add new items only if explicitly mentioned
            5. Return the JSON structure even if most fields are empty"""

            messages = [
                {
                    "role": "system", 
                    "content": """You are an information extraction expert. 
                    Extract only explicitly stated information and format it as valid JSON.
                    Do not make assumptions or infer information not present in the text."""
                },
                {"role": "user", "content": extraction_prompt}
            ]
            
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"}  # Ensure JSON response
            )
            
            # Parse the JSON response
            extracted_info = json.loads(response.choices[0].message.content)
            
            # Validate the extracted information
            return self.validate_extracted_info(extracted_info, info_template)
            
        except json.JSONDecodeError as e:
            st.error(f"Error parsing extracted information: {str(e)}")
            return {}
        except Exception as e:
            st.error(f"Error extracting information: {str(e)}")
            return {}

    def update_collected_info(self, new_info: dict):
        """Update the collected information with new details"""
        try:
            for category, info in new_info.items():
                if category not in self.collected_info:
                    continue
                    
                if isinstance(self.collected_info[category], list):
                    if isinstance(info, list):
                        # Add only new items to avoid duplicates
                        self.collected_info[category].extend([
                            item for item in info 
                            if item not in self.collected_info[category]
                        ])
                elif isinstance(self.collected_info[category], dict):
                    # Update only non-empty values
                    for field, value in info.items():
                        if value and value.strip():  # Check if value is non-empty
                            self.collected_info[category][field] = value
                else:
                    if info and str(info).strip():  # Update only if new info is non-empty
                        self.collected_info[category] = info
                        
            # Check if critical information is present
            self.check_information_completeness()
                
        except Exception as e:
            st.error(f"Error updating information: {str(e)}")

    def analyze_legal_context(self, incident_details: str) -> dict:
        """Perform legal analysis based on incident details with robust error handling"""
        try:
            relevant_context = self.get_relevant_context(incident_details)
            
            # Create a structured template for the expected response
            analysis_template = {
                "applicable_sections": [],
                "maximum_punishment": "",
                "offense_category": "",
                "bail_status": "",
                "recommended_immediate_actions": []
            }
            
            analysis_prompt = f"""Analyze this incident and provide a detailed legal assessment.

    Incident Details:
    {incident_details}

    Legal Context:
    {relevant_context}

    Provide a comprehensive legal analysis following this exact structure:
    {json.dumps(analysis_template, indent=2)}

    Guidelines:
    1. For applicable_sections, provide an array of objects with structure:
    {{"section": "section_number", "description": "description", "relevance": "explanation", "severity": "level"}}
    2. For maximum_punishment, provide the highest possible punishment considering all sections
    3. For offense_category, specify whether it's cognizable/non-cognizable and bailable/non-bailable
    4. For bail_status, provide detailed justification based on offense type
    5. For recommended_immediate_actions, provide an array of specific actions police should take

    IMPORTANT: Return ONLY a valid JSON object matching the template structure. No additional text."""

            messages = [
                {
                    "role": "system",
                    "content": "You are a legal expert. Provide analysis in valid JSON format exactly matching the specified template structure."
                },
                {"role": "user", "content": analysis_prompt}
            ]
            
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            # Parse and validate response
            try:
                analysis = json.loads(response.choices[0].message.content)
                
                # Validate required fields
                for key in analysis_template.keys():
                    if key not in analysis:
                        analysis[key] = analysis_template[key]
                
                # Validate applicable sections structure
                if isinstance(analysis["applicable_sections"], list):
                    validated_sections = []
                    for section in analysis["applicable_sections"]:
                        if isinstance(section, dict):
                            validated_section = {
                                "section": section.get("section", ""),
                                "description": section.get("description", ""),
                                "relevance": section.get("relevance", ""),
                                "severity": section.get("severity", "")
                            }
                            validated_sections.append(validated_section)
                    analysis["applicable_sections"] = validated_sections
                else:
                    analysis["applicable_sections"] = []
                
                return analysis
                
            except json.JSONDecodeError as je:
                st.warning("Error parsing legal analysis response. Using default structure.")
                return analysis_template
                
        except Exception as e:
            st.error(f"Error in legal analysis: {str(e)}")
            return {
                "applicable_sections": [],
                "maximum_punishment": "Error analyzing punishment",
                "offense_category": "Error determining category",
                "bail_status": "Error determining bail status",
                "recommended_immediate_actions": []
            }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        mode: str = "chat"
    ) -> str:
        try:
            if mode == "chat":
                return self._handle_chat(prompt)
            else:
                if not self.information_complete:
                    raise ValueError("Cannot generate FIR until user confirms information is complete")
                return self._handle_fir_generation(prompt)
        except Exception as e:
            raise

    def _handle_chat(self, user_input: str) -> str:
        # Check if user is confirming information completion
        if any(phrase in user_input.lower() for phrase in [
            "this is all the information",
            "that's all the information",
            "i have shared everything",
            "this is everything i have"
        ]):
            self.information_complete = True
            # Perform legal analysis before confirming completion
            legal_analysis = self.analyze_legal_context(json.dumps(self.collected_info))
            if legal_analysis:
                self.collected_info["legal_analysis"] = legal_analysis
            return ("Thank you for providing all the information. I have performed a preliminary legal analysis. "
                   "Would you like me to generate the complete FIR report now?")

        # Regular chat handling
        history = self.memory.load_memory_variables({})
        
        chat_prompt = f"""Based on the conversation history and current input, continue gathering information for the FIR.
        
        Conversation History:
        {history.get('history', '')}
        
        Current Input: {user_input}
        
        Already Collected Information:
        {json.dumps(self.collected_info, indent=2)}
        
        Remember to:
        1. Acknowledge the user's input
        2. Extract and store any relevant information
        3. Ask a relevant follow-up question if needed
        4. Be empathetic and professional
        5. Focus on missing information"""

        messages = [
            {"role": "system", "content": self.chat_system_prompt},
            {"role": "user", "content": chat_prompt}
        ]
        
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            temperature=0.7,
            max_tokens=2000
        )
        
        # Extract and update information
        new_info = self.extract_information(user_input)
        self.update_collected_info(new_info)
        
        # Update memory
        self.memory.save_context(
            {"input": user_input},
            {"output": response.choices[0].message.content}
        )
        
        return response.choices[0].message.content

    def _handle_fir_generation(self, prompt: str) -> dict:
        """Handle FIR generation with proper response formatting"""
        try:
            relevant_context = self.get_relevant_context(prompt)
            
            # Create FIR template structure
            fir_template = {
                "fir_details": {
                    "fir_number": "",
                    "police_station": "",
                    "registration_date": "",
                    "registration_time": "",
                    "incident_date": "",
                    "incident_time": "",
                    "location": "",
                    "incident_type": ""
                },
                "complainant_details": self.collected_info.get("complainant_details", {}),
                "accused_details": self.collected_info.get("accused_details", {}),
                "witnesses": self.collected_info.get("witnesses", []),
                "evidence_details": self.collected_info.get("evidence_details", {}),
                "incident_description": self.collected_info.get("incident_description", ""),
                "investigation_requirements": {
                    "immediate_actions": [],
                    "evidence_collection": [],
                    "witness_statements": [],
                    "forensic_requirements": []
                },
                "legal_analysis": {
                    "applicable_sections": [],
                    "maximum_punishment": "",
                    "offense_category": "",
                    "bail_status": "",
                    "recommended_immediate_actions": []
                }
            }

            fir_prompt = f"""Generate a complete FIR report using this information and legal context.

    Legal Context:
    {relevant_context}

    Collected Information:
    {json.dumps(self.collected_info, indent=2)}

    Requirements:
    1. Format the response as a valid JSON object
    2. Follow this exact structure:
    {json.dumps(fir_template, indent=2)}

    Guidelines:
    1. Include all mandatory FIR fields
    2. Use collected information where available
    3. Format dates as "YYYY-MM-DD"
    4. Format times as "HH:MM"
    5. For legal analysis, include:
    - Section numbers with descriptions
    - Maximum punishment under each section
    - Clear offense categorization
    - Bail status with justification
    - Specific immediate actions needed
    6. Mark any missing required information as "Not provided"

    Return only the JSON object, no additional text or formatting."""

            messages = [
                {
                    "role": "system",
                    "content": """You are a legal expert generating FIR reports.
                    Always return properly formatted JSON following the exact template structure.
                    Include all mandatory fields and mark missing information as "Not provided"."""
                },
                {"role": "user", "content": fir_prompt}
            ]
            
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=0.1,
                max_tokens=4000,
                response_format={"type": "json_object"}  # Ensure JSON response
            )
            
            # Parse and validate the response
            return self.format_legal_response(response.choices[0].message.content)
            
        except Exception as e:
            st.error(f"Error generating FIR: {str(e)}")
            raise

    def format_legal_response(self, response_text: str) -> dict:
        """Format and validate legal response with enhanced error handling"""
        try:
            # Default structure for legal analysis
            default_legal_structure = {
                "applicable_sections": [],
                "maximum_punishment": "",
                "offense_category": "",
                "bail_status": "",
                "recommended_immediate_actions": []
            }
            
            # Attempt to parse JSON response
            try:
                fir_data = json.loads(response_text)
            except json.JSONDecodeError:
                st.error("Invalid JSON format in legal response")
                return {"legal_analysis": default_legal_structure}
            
            # Ensure legal_analysis exists and has required structure
            if "legal_analysis" not in fir_data:
                fir_data["legal_analysis"] = {}
            
            # Validate each field in legal analysis
            for key, default_value in default_legal_structure.items():
                if key not in fir_data["legal_analysis"]:
                    fir_data["legal_analysis"][key] = default_value
                elif not isinstance(fir_data["legal_analysis"][key], type(default_value)):
                    fir_data["legal_analysis"][key] = default_value
            
            # Validate applicable sections structure
            if isinstance(fir_data["legal_analysis"]["applicable_sections"], list):
                validated_sections = []
                for section in fir_data["legal_analysis"]["applicable_sections"]:
                    if isinstance(section, dict):
                        validated_section = {
                            "section": section.get("section", ""),
                            "description": section.get("description", ""),
                            "relevance": section.get("relevance", ""),
                            "severity": section.get("severity", "")
                        }
                        validated_sections.append(validated_section)
                fir_data["legal_analysis"]["applicable_sections"] = validated_sections
            
            return fir_data
            
        except Exception as e:
            st.error(f"Error formatting legal response: {str(e)}")
            return {"legal_analysis": default_legal_structure}

    def generate_fir(self) -> dict:
        """Generate complete FIR with validation"""
        try:
            if not self.information_complete:
                raise ValueError("Cannot generate FIR until all required information is provided")
            
            # Update legal analysis before generation
            self.collected_info["legal_analysis"] = self.analyze_legal_context(
                json.dumps(self.collected_info)
            )
            
            # Generate FIR with all collected information
            fir_prompt = f"""Generate a complete FIR report using:

    Collected Information:
    {json.dumps(self.collected_info, indent=2)}

    Include:
    1. All FIR details
    2. Complete legal analysis
    3. Investigation requirements
    4. Evidence details"""

            return self._handle_fir_generation(fir_prompt)
            
        except Exception as e:
            st.error(f"Error generating FIR: {str(e)}")
            raise

    def get_relevant_context(self, query: str, k: int = 5) -> str:
        """Search for relevant legal context based on the query"""
        if not self.vectorstore:
            return ""
            
        documents = self.vectorstore.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in documents])
    def validate_extracted_info(self, extracted_info: dict, template: dict) -> dict:
        """Validate and clean extracted information against the template"""
        validated_info = {}
        
        try:
            for category, structure in template.items():
                if category not in extracted_info:
                    validated_info[category] = structure
                    continue
                    
                if isinstance(structure, dict):
                    validated_info[category] = {}
                    for field, _ in structure.items():
                        if field in extracted_info[category]:
                            validated_info[category][field] = extracted_info[category][field]
                        else:
                            validated_info[category][field] = ""
                            
                elif isinstance(structure, list):
                    validated_info[category] = extracted_info[category] if isinstance(extracted_info[category], list) else []
                else:
                    validated_info[category] = extracted_info[category]
            
            return validated_info
            
        except Exception as e:
            st.error(f"Error validating extracted information: {str(e)}")
            return template
    def check_information_completeness(self):
        """Check if all critical information has been collected"""
        critical_fields = {
            "fir_details": ["date", "location", "incident_type"],
            "complainant_details": ["name", "contact"],
            "incident_description": ""
        }
        
        try:
            for category, fields in critical_fields.items():
                if isinstance(fields, list):
                    if category not in self.collected_info:
                        return
                    for field in fields:
                        if not self.collected_info[category].get(field):
                            return
                else:
                    if not self.collected_info.get(category):
                        return
            
            # If we reach here, all critical fields are present
            if not self.information_complete:
                st.success("All critical information has been collected!")
                
        except Exception as e:
            st.error(f"Error checking information completeness: {str(e)}")


def create_fir_assistant(vectorstore):
    """Create enhanced FIR assistant with chat capabilities"""
    try:
        llm = FIRAssistantLLM(vectorstore=vectorstore)
        return llm
    except Exception as e:
        st.error(f"Error creating FIR assistant: {str(e)}")
        st.stop()


def setup_vector_store(text_chunks):
    """Setup vector store with Voyage AI embeddings"""
    voyage_key, _ = check_api_keys()
    
    try:
        st.info("Setting up Voyage AI embeddings...")
        vo = voyageai.Client(api_key=voyage_key)
        embeddings = VoyageAIEmbeddings(vo)
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error setting up vector store: {str(e)}")
        st.stop()

def process_legal_documents(pdf_docs):
    """Process both penal code and reference documents"""
    if not pdf_docs:
        st.error("No documents uploaded. Please upload the penal code PDF and any reference documents.")
        st.stop()
    
    try:
        penal_code_text = ""
        reference_text = ""
        
        # Process the first document as penal code
        penal_code = pdf_docs[0]
        pdf_reader = PdfReader(penal_code)
        for page in pdf_reader.pages:
            penal_code_text += page.extract_text()
            
        # Process remaining documents as references
        for pdf in pdf_docs[1:]:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                reference_text += page.extract_text()
                
        return penal_code_text, reference_text
    except Exception as e:
        st.error(f"Error processing PDF documents: {str(e)}")
        st.stop()

def create_legal_text_chunks(text):
    """Create specialized chunks for legal document processing"""
    try:
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = splitter.split_text(text)
        if not chunks:
            st.warning("No text chunks were created. Please check your PDF content.")
            st.stop()
        return chunks
    except Exception as e:
        st.error(f"Error creating text chunks: {str(e)}")
        st.stop()

def create_fir_assistant(vectorstore):
    """Create FIR assistant with legal analysis capabilities"""
    try:
        llm = FIRAssistantLLM(vectorstore=vectorstore)
        return llm  # Return the LLM directly instead of creating a chain
    except Exception as e:
        st.error(f"Error creating FIR assistant: {str(e)}")
        st.stop()

def display_legal_analysis(legal_analysis):
    """Display legal analysis in a structured format"""
    st.subheader("Legal Analysis")
    
    # Display applicable sections
    st.write("**Applicable Sections:**")
    for section in legal_analysis["applicable_sections"]:
        with st.expander(f"Section {section['section']}"):
            st.write(f"**Description:** {section['description']}")
            st.write(f"**Relevance:** {section['relevance']}")
            st.write(f"**Severity:** {section['severity']}")
    
    # Display punishment and category information
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Maximum Punishment:**")
        st.write(legal_analysis["maximum_punishment"])
        st.write("**Offense Category:**")
        st.write(legal_analysis["offense_category"])
    
    with col2:
        st.write("**Bail Status:**")
        st.write(legal_analysis["bail_status"])
        st.write("**Recommended Actions:**")
        for action in legal_analysis["recommended_immediate_actions"]:
            st.write(f"- {action}")

def main():
    st.set_page_config(page_title="FIR Assistant System", layout="wide")
    st.title("FIR Assistant System with Legal Analysis")
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    if "show_fir" not in st.session_state:
        st.session_state.show_fir = False
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("Document Upload")
        st.markdown("""
        **Upload Order:**
        1. Penal Code PDF (first)
        2. Additional reference documents (optional)
        """)
        pdf_docs = st.file_uploader(
            "Upload legal documents",
            accept_multiple_files=True,
            type=['pdf']
        )
        
        process_docs = st.button("Process Documents", key="process_docs")
        
        if process_docs and pdf_docs:
            with st.spinner("Processing legal documents..."):
                try:
                    # Process documents
                    penal_code_text, reference_text = process_legal_documents(pdf_docs)
                    legal_chunks = create_legal_text_chunks(penal_code_text)
                    reference_chunks = create_legal_text_chunks(reference_text) if reference_text else []
                    
                    all_chunks = legal_chunks + reference_chunks
                    vectorstore = setup_vector_store(all_chunks)
                    
                    # Create conversation with memory
                    st.session_state.conversation = create_fir_assistant(vectorstore=vectorstore)
                    st.session_state.documents_processed = True
                    
                    st.success("Documents processed successfully! You can now start the conversation.")
                    
                    # Add initial assistant message to chat
                    if not st.session_state.chat_history:
                        initial_message = {
                            "role": "assistant",
                            "content": "Hello! I'm your FIR Assistant. I'll help you file a First Information Report. Could you start by telling me what incident you'd like to report?"
                        }
                        st.session_state.chat_history.append(initial_message)
                        
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
                    st.stop()
        
        # Help section
        with st.expander("Help & Instructions"):
            st.markdown("""
            ### How to Use
            1. **Upload Documents**
               - First, upload the Penal Code PDF
               - Then upload any additional reference documents
               - Click 'Process Documents'
            
            2. **Chat with Assistant**
               - Have a natural conversation about the incident
               - Answer the assistant's questions
               - Provide as much detail as possible
            
            3. **Generate FIR**
               - Click 'Generate FIR' when you've provided all information
               - Review the generated report
               - Download the report if needed
            
            ### Tips for Better Results
            - Be specific with incident details
            - Include dates, times, and locations
            - Mention any witnesses or evidence
            - Answer all follow-up questions
            - Review the information before generating the FIR
            
            ### Support
            If you encounter any issues:
            - Check that documents are properly uploaded
            - Ensure PDFs are readable
            - Try providing more detailed responses
            - Clear chat and start over if needed
            """)

    # Main content area - Two columns layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        st.header("Chat with FIR Assistant")
        
        # Chat container with fixed height and scrolling
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        
        # Chat input
        if user_input := st.chat_input(
            "Type your message here...",
            disabled=not st.session_state.documents_processed
        ):
            # Add user message to chat
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)
            
            if st.session_state.conversation:
                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            response = st.session_state.conversation(user_input, mode="chat")
                            st.write(response)
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                        except Exception as e:
                            st.error(f"Error generating response: {str(e)}")
            else:
                st.warning("Please upload and process the legal documents first.")
    
    with col2:
        st.header("FIR Generation")
        
        # Show collected information
        if st.session_state.conversation and hasattr(st.session_state.conversation, 'collected_info'):
            with st.expander("Currently Collected Information", expanded=True):
                st.json(st.session_state.conversation.collected_info)
        
        if st.button("Generate FIR from Chat", 
                disabled=not st.session_state.documents_processed):
            if st.session_state.conversation:
                if not st.session_state.conversation.information_complete:
                    st.warning("Please confirm that you have shared all the information by typing 'This is all the information I have' or similar phrase.")
                else:
                    with st.spinner("Generating FIR from our conversation..."):
                        try:
                            fir_data = st.session_state.conversation.generate_fir()
                            st.session_state.show_fir = True
                            st.session_state.fir_data = fir_data
                            st.success("FIR generated successfully!")
                        except Exception as e:
                            st.error(f"Error generating FIR: {str(e)}")
            else:
                st.warning("Please upload documents and have a conversation first.")
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            if st.session_state.conversation:
                st.session_state.conversation.memory.clear()
                st.session_state.conversation.collected_info = {
                    "fir_details": {},
                    "complainant_details": {},
                    "accused_details": {},
                    "witnesses": [],
                    "evidence_details": {},
                    "incident_description": ""
                }
            st.success("Chat history cleared!")
            st.rerun()

    # Display FIR if generated
    if st.session_state.show_fir and hasattr(st.session_state, 'fir_data'):
        st.header("Generated FIR Report")
        fir_data = st.session_state.fir_data
        
        # Display FIR with tabs
        tab1, tab2, tab3 = st.tabs(["Legal Analysis", "FIR Details", "Raw JSON"])
        
        with tab1:
            if "legal_analysis" in fir_data:
                display_legal_analysis(fir_data["legal_analysis"])
        
        with tab2:
            fir_without_analysis = fir_data.copy()
            if "legal_analysis" in fir_without_analysis:
                del fir_without_analysis["legal_analysis"]
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Incident Information")
                st.json(fir_without_analysis["fir_details"])
                
                st.subheader("Complainant Details")
                st.json(fir_without_analysis["complainant_details"])
            
            with col2:
                st.subheader("Accused Details")
                st.json(fir_without_analysis["accused_details"])
                
                st.subheader("Witnesses")
                st.json(fir_without_analysis["witnesses"])
            
            st.subheader("Evidence Details")
            st.json(fir_without_analysis["evidence_details"])
            
            st.subheader("Investigation Requirements")
            st.json(fir_without_analysis["investigation_requirements"])
        
        with tab3:
            st.json(fir_data)
        
        # Download options
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            fir_str = json.dumps(fir_data, indent=2)
            st.download_button(
                label="Download Complete FIR Report",
                data=fir_str,
                file_name="fir_report_complete.json",
                mime="application/json"
            )
        
        with col2:
            if "legal_analysis" in fir_data:
                legal_analysis_str = json.dumps(fir_data["legal_analysis"], indent=2)
                st.download_button(
                    label="Download Legal Analysis Only",
                    data=legal_analysis_str,
                    file_name="legal_analysis.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()
