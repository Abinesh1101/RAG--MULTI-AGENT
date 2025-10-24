"""
Reasoning Agent - Synthesizes final answers using Ollama Mistral 7B
OPTIMIZED VERSION - Faster responses
"""

import ollama
from typing import List, Dict, Optional


class ReasoningAgent:
    """
    Uses Ollama Mistral 7B LLM to generate intelligent answers by combining:
    - Retrieved text chunks
    - Image descriptions from Vision Agent
    - User query context
    """
    
    def __init__(self, model_name: str = "mistral"):
        """
        Initialize the Reasoning Agent with Ollama
        
        Args:
            model_name: Ollama model to use (default: mistral)
        """
        self.model_name = model_name
        print(f"✅ Reasoning Agent initialized with Ollama model: {model_name}")
        
        # Verify Ollama is running
        try:
            ollama.list()
            print("✅ Ollama is running and accessible")
        except Exception as e:
            print(f"⚠️ WARNING: Could not connect to Ollama!")
            print(f"   Make sure Ollama is installed and running.")
            print(f"   Run: ollama serve")
            print(f"   Error: {e}")
    
    
    def answer(self, query: str, retrieved_docs: List[Dict]) -> str:
        """
        Generate answer for 'fact' intent using retrieved documents
        
        Args:
            query: User's question
            retrieved_docs: List of retrieved document chunks
        
        Returns:
            Generated answer
        """
        # Prepare context from retrieved documents
        context = self._prepare_text_context(retrieved_docs, max_chunks=3)  # ⚡ Reduced from 5
        
        # Create prompt
        prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context.

Context from documents:
{context}

User Question: {query}

Instructions:
- Provide a clear, concise answer (2-3 sentences maximum)
- If the context doesn't contain enough information, say so
- Be factual and accurate

Answer:"""
        
        # Generate response using Ollama
        response = self._generate_with_ollama(prompt)
        return response
    
    
    def reason(self, query: str, retrieved_docs: List[Dict]) -> str:
        """
        Generate analytical answer for 'analysis' intent
        Performs multi-document reasoning and comparison
        
        Args:
            query: User's analytical question
            retrieved_docs: List of retrieved document chunks
        
        Returns:
            Analytical answer
        """
        context = self._prepare_text_context(retrieved_docs, max_chunks=4)  # ⚡ Reduced from 5
        
        prompt = f"""You are an expert analyst. Analyze and synthesize information from multiple documents to answer the question.

Context from documents:
{context}

User Question: {query}

Instructions:
- Compare and contrast information across documents (3-4 sentences)
- Identify key patterns and insights
- Be concise and specific

Analysis:"""
        
        response = self._generate_with_ollama(prompt)
        return response
    
    
    def summarize(self, all_docs: List[Dict]) -> str:
        """
        Generate summary for 'summary' intent
        
        Args:
            all_docs: All relevant document chunks
        
        Returns:
            Summary of documents
        """
        context = self._prepare_text_context(all_docs, max_chunks=6)  # ⚡ Reduced from more
        
        prompt = f"""You are a professional summarizer. Create a brief summary of the following documents.

Documents:
{context}

Instructions:
- Highlight 3-5 key points
- Keep the summary clear and concise (4-5 sentences)
- Focus on most important information

Summary:"""
        
        response = self._generate_with_ollama(prompt)
        return response
    
    
    def answer_with_visual(
        self, 
        query: str, 
        text_context: List[Dict], 
        image_descriptions: List[Dict]
    ) -> str:
        """
        Generate answer for 'visual' intent using both text and image context
        
        Args:
            query: User's question about visual content
            text_context: Retrieved text chunks
            image_descriptions: Image descriptions from Vision Agent
        
        Returns:
            Answer combining text and visual information
        """
        # Prepare text context
        text_info = self._prepare_text_context(text_context, max_chunks=3)  # ⚡ Reduced
        
        # Prepare image context
        image_info = self._prepare_image_context(image_descriptions)
        
        prompt = f"""You are a multimodal AI assistant analyzing both text and visual content.

Text Context:
{text_info}

Visual Context (Image Descriptions):
{image_info}

User Question: {query}

Instructions:
- Combine insights from both text and visual content (3-4 sentences)
- Explain what the images/charts show
- Be specific and concise

Answer:"""
        
        response = self._generate_with_ollama(prompt)
        return response
    
    
    def _prepare_text_context(self, docs, max_chunks: int = 5) -> str:
        """
        Prepare text context from retrieved documents (OPTIMIZED)
        
        Args:
            docs: List of document chunks or dict with results
            max_chunks: Maximum number of chunks to include
        
        Returns:
            Formatted context string
        """
        # Handle dict response from retrieval agent
        if isinstance(docs, dict):
            docs = docs.get('results', [])
        
        if not docs:
            return "No relevant documents found."
        
        # Convert to list if not already
        if not isinstance(docs, list):
            docs = [docs]
        
        context_parts = []
        for i, doc in enumerate(docs[:max_chunks], 1):
            if isinstance(doc, dict):
                text = doc.get('text', doc.get('content', doc.get('metadata', {}).get('text', '')))
                source = doc.get('source', doc.get('metadata', {}).get('source', 'Unknown'))
                
                # ⚡ Truncate long text to first 300 characters
                if text and len(text) > 300:
                    text = text[:300] + "..."
                    
            else:
                text = str(doc)[:300]  # ⚡ Truncate
                source = 'Unknown'
            
            if text:
                context_parts.append(f"[Doc {i} - {source}]\n{text}\n")
        
        return "\n".join(context_parts) if context_parts else "No relevant text found."
    
    
    def _prepare_image_context(self, image_descriptions: List[Dict]) -> str:
        """
        Prepare context from image descriptions
        
        Args:
            image_descriptions: List of image description dictionaries
        
        Returns:
            Formatted image context string
        """
        if not image_descriptions:
            return "No relevant images found."
        
        context_parts = []
        for i, img_desc in enumerate(image_descriptions, 1):
            image_name = img_desc.get('image_name', f'Image {i}')
            description = img_desc.get('description', 'No description available')
            context_parts.append(f"[{image_name}]: {description}")
        
        return "\n".join(context_parts)
    
    
    def _generate_with_ollama(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generate response using Ollama (OPTIMIZED for speed)
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-1.0)
        
        Returns:
            Generated text
        """
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': temperature,
                    'num_predict': 250,     # ⚡ Reduced from 512 for 3x faster responses
                    'num_ctx': 2048,        # Context window
                    'top_k': 40,            # Sampling parameter
                    'top_p': 0.9,           # Nucleus sampling
                }
            )
            
            # Extract the generated text
            answer = response['response'].strip()
            return answer
            
        except Exception as e:
            error_msg = f"Error generating response with Ollama: {str(e)}"
            print(f"❌ {error_msg}")
            return f"I apologize, but I encountered an error. Please make sure Ollama is running ('ollama serve'). Error: {str(e)}"


# Test the Reasoning Agent
if __name__ == "__main__":
    print("="*60)
    print("🧪 TESTING REASONING AGENT (Ollama Mistral 7B - OPTIMIZED)")
    print("="*60)
    
    # Initialize agent
    agent = ReasoningAgent()
    
    # Test 1: Simple factual answer
    print("\n" + "="*60)
    print("TEST 1: Fact-based Query")
    print("="*60)
    
    test_docs = [
        {
            'text': 'Tesla delivered 1.8 million vehicles in 2023, representing a 38% increase from 2022.',
            'source': 'Tesla Report 2023'
        },
        {
            'text': 'The company achieved record revenue of $96.8 billion in 2023.',
            'source': 'Tesla Report 2023'
        }
    ]
    
    query = "How many vehicles did Tesla deliver in 2023?"
    print(f"\n📝 Query: {query}")
    answer = agent.answer(query, test_docs)
    print(f"\n✅ Answer:\n{answer}")
    
    # Test 2: Visual context
    print("\n" + "="*60)
    print("TEST 2: Visual Query (Text + Image)")
    print("="*60)
    
    image_descs = [
        {
            'image_name': 'emissions_chart.png',
            'description': 'A bar chart showing CO2 emissions declining from 45 million tons in 2022 to 38 million tons in 2023.'
        }
    ]
    
    query = "What does the emissions chart show?"
    print(f"\n📝 Query: {query}")
    answer = agent.answer_with_visual(query, test_docs, image_descs)
    print(f"\n✅ Answer:\n{answer}")
    
    print("\n" + "="*60)
    print("✅ REASONING AGENT TEST COMPLETE!")
    print("="*60)