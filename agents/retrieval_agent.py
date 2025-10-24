"""
Retrieval Agent - Enhanced with Better Chunking and Query Expansion
"""

import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import fitz  # PyMuPDF
from PIL import Image
import re


class RetrievalAgent:
    """Handles retrieval of text and image embeddings from FAISS vector store"""
    
    def __init__(self, vector_store_path: str = "data"):
        """Initialize the Retrieval Agent"""
        self.vector_store_path = vector_store_path
        self.model = None
        self.text_index = None
        self.image_index = None
        self.text_metadata = []
        self.image_metadata = []
        
        print("🔄 Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded!")
        
        self.initialize_vector_store()
    
    def initialize_vector_store(self):
        """Initialize or load existing vector store"""
        vector_store_file = os.path.join(self.vector_store_path, "faiss_store.npz")
        if os.path.exists(vector_store_file):
            print("📂 Loading existing vector store...")
            self.load_vector_store(self.vector_store_path)
        else:
            print("\n" + "="*60)
            print("🚀 INITIALIZING VECTOR STORE")
            print("="*60)
            self.build_vector_store()
    
    def build_vector_store(self):
        """Build vector store from PDFs and images"""
        pdf_dir = os.path.join(self.vector_store_path, "pdfs")
        image_dir = os.path.join(self.vector_store_path, "images")
        
        print(f"\n📄 Extracting text from PDFs in {pdf_dir}...")
        text_chunks = self.extract_text_from_pdfs(pdf_dir)
        print(f"✅ Extracted {len(text_chunks)} text chunks from PDFs")
        
        print(f"\n🖼️  Processing images from {image_dir}...")
        image_data = self.process_images(image_dir, max_images=None)
        print(f"✅ Processed {len(image_data)} images")
        
        print("\n🔢 Creating embeddings...")
        self.create_embeddings(text_chunks, image_data)
        print("✅ Embeddings created!")
        
        print("\n🗃️  Building FAISS indexes...")
        self.build_faiss_indexes()
        self.save_vector_store(self.vector_store_path)
        
        print("\n" + "="*60)
        print("✅ VECTOR STORE INITIALIZED!")
        print(f"   📄 Text chunks: {len(self.text_metadata)}")
        print(f"   🖼️  Images: {len(self.image_metadata)}")
        print("="*60)
    
    def extract_text_from_pdfs(self, pdf_dir: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """
        ⚡ ENHANCED: Better chunking with smaller sizes and smart boundaries
        """
        text_chunks = []
        
        if not os.path.exists(pdf_dir):
            print(f"⚠️ PDF directory not found: {pdf_dir}")
            return text_chunks
        
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            print(f"   Processing: {pdf_file}")
            
            try:
                doc = fitz.open(pdf_path)
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text()
                    
                    # Clean up text
                    text = text.replace('\n', ' ').replace('\r', ' ')
                    text = ' '.join(text.split())  # Remove extra whitespace
                    
                    # Skip empty pages
                    if len(text) < 50:
                        continue
                    
                    # ⚡ ENHANCED: Split by sentences first for better semantic units
                    sentences = re.split(r'(?<=[.!?])\s+', text)
                    
                    current_chunk = ""
                    for sentence in sentences:
                        # If adding this sentence exceeds chunk_size, save current chunk
                        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                            text_chunks.append({
                                'text': current_chunk.strip(),
                                'source': pdf_file,
                                'page': page_num + 1
                            })
                            # ⚡ Keep last sentence for overlap
                            current_chunk = sentence + " "
                        else:
                            current_chunk += sentence + " "
                    
                    # Add remaining chunk
                    if current_chunk.strip():
                        text_chunks.append({
                            'text': current_chunk.strip(),
                            'source': pdf_file,
                            'page': page_num + 1
                        })
                
                doc.close()
                
            except Exception as e:
                print(f"   ⚠️ Error processing {pdf_file}: {e}")
        
        return text_chunks

    def expand_query(self, query: str) -> List[str]:
        """
        ⚡ NEW: Expand query with synonyms for better retrieval
        """
        # Synonym mapping for common query terms
        expansions = {
            'mission': ['mission', 'goal', 'purpose', 'objective', 'vision', 'aim'],
            'statement': ['statement', 'declaration', 'commitment', 'pledge'],
            'emissions': ['emissions', 'carbon', 'CO2', 'greenhouse gas', 'GHG'],
            'revenue': ['revenue', 'income', 'earnings', 'sales', 'financial performance'],
            'deliveries': ['deliveries', 'delivered', 'production', 'vehicles produced', 'units sold'],
            'sustainability': ['sustainability', 'environmental', 'green', 'eco-friendly', 'climate'],
            'chart': ['chart', 'graph', 'figure', 'diagram', 'visualization', 'data'],
            'goals': ['goals', 'targets', 'objectives', 'aims', 'commitments'],
            'achievements': ['achievements', 'accomplishments', 'milestones', 'progress', 'results']
        }
        
        query_lower = query.lower()
        expanded_terms = set(query.split())
        
        # Add synonyms for words found in query
        for key, synonyms in expansions.items():
            if key in query_lower:
                expanded_terms.update(synonyms)
        
        # Create multiple query variations
        queries = [
            query,  # Original
            ' '.join(expanded_terms)[:200],  # Expanded with synonyms
        ]
        
        return queries

    def retrieve(self, query: str, search_type: str = 'text', top_k: int = 5) -> Dict[str, List]:
        """
        ⚡ ENHANCED: Multi-query retrieval with query expansion
        """
        print(f"\n🔍 Searching for: '{query}'")
        print(f"   Search type: {search_type}, Top-K: {top_k}")
        
        text_results, image_results = [], []
        
        if search_type in ['text', 'both'] and self.text_index is not None:
            # ⚡ Expand query for better matching
            expanded_queries = self.expand_query(query)
            
            all_results = {}  # Use dict to deduplicate by index
            
            for exp_query in expanded_queries:
                query_emb = self.model.encode([exp_query])[0].astype('float32')
                
                # Search with expanded query
                D, I = self.text_index.search(np.array([query_emb]), top_k * 2)  # Get more candidates
                
                for idx, dist in zip(I[0], D[0]):
                    if idx < len(self.text_metadata):
                        if idx not in all_results:  # Avoid duplicates
                            r = self.text_metadata[idx].copy()
                            # ⚡ Better similarity score (higher is better)
                            r['similarity_score'] = float(1.0 / (1.0 + dist))
                            all_results[idx] = r
                        else:
                            # Keep best score
                            new_score = float(1.0 / (1.0 + dist))
                            if new_score > all_results[idx]['similarity_score']:
                                all_results[idx]['similarity_score'] = new_score
            
            # Sort by similarity and take top_k
            text_results = sorted(all_results.values(), 
                                 key=lambda x: x['similarity_score'], 
                                 reverse=True)[:top_k]
        
        if search_type in ['image', 'both']:
            image_results = self.retrieve_images(query, top_k=top_k)
        
        print(f"   ✅ Found {len(text_results)} text results, {len(image_results)} image results")
        
        # ⚡ Debug: Show top result
        if text_results:
            print(f"   📝 Top result similarity: {text_results[0]['similarity_score']:.4f}")
            print(f"   📄 From: {text_results[0]['source']} (Page {text_results[0]['page']})")
        
        return {'text_results': text_results, 'image_results': image_results}

    def process_images(self, image_dir: str, max_images: int = None) -> List[Dict]:
        """Process images and create captions"""
        image_data = []
        if not os.path.exists(image_dir):
            print(f"⚠️ Image directory not found: {image_dir}")
            return image_data
        
        image_files = [f for f in os.listdir(image_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:max_images]
        
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            caption = img_file.replace('_', ' ').replace('-', ' ').rsplit('.', 1)[0]
            image_data.append({
                'image_path': img_path,
                'image_name': img_file,
                'caption': caption
            })
        return image_data
    
    def create_embeddings(self, text_chunks: List[Dict], image_data: List[Dict]):
        """Create embeddings for text and images"""
        if text_chunks:
            print(f"   Embedding {len(text_chunks)} text chunks...")
            texts = [chunk['text'] for chunk in text_chunks]
            self.text_embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
            self.text_metadata = text_chunks
        else:
            self.text_embeddings = np.array([])
            self.text_metadata = []
        
        if image_data:
            print(f"   Embedding {len(image_data)} image captions...")
            captions = [img['caption'] for img in image_data]
            self.image_embeddings = self.model.encode(captions, show_progress_bar=True, batch_size=32)
            self.image_metadata = image_data
        else:
            self.image_embeddings = np.array([])
            self.image_metadata = []
    
    def build_faiss_indexes(self):
        """Build FAISS indexes"""
        if len(self.text_embeddings) > 0:
            d = self.text_embeddings.shape[1]
            self.text_index = faiss.IndexFlatL2(d)
            self.text_index.add(self.text_embeddings.astype('float32'))
            print(f"   ✅ Text index built with {self.text_index.ntotal} vectors")
        else:
            self.text_index = None
        
        if len(self.image_embeddings) > 0:
            d = self.image_embeddings.shape[1]
            self.image_index = faiss.IndexFlatL2(d)
            self.image_index.add(self.image_embeddings.astype('float32'))
            print(f"   ✅ Image index built with {self.image_index.ntotal} vectors")
        else:
            self.image_index = None
    
    def save_vector_store(self, save_path: str):
        """Save vector store"""
        np.savez_compressed(os.path.join(save_path, "faiss_store.npz"),
                            text_embeddings=self.text_embeddings,
                            image_embeddings=self.image_embeddings,
                            text_metadata=self.text_metadata,
                            image_metadata=self.image_metadata)
        print(f"\n💾 Vector store saved to {save_path}/faiss_store.npz")
    
    def load_vector_store(self, load_path: str):
        """Load vector store"""
        try:
            data = np.load(os.path.join(load_path, "faiss_store.npz"), allow_pickle=True)
            self.text_embeddings = data['text_embeddings']
            self.image_embeddings = data['image_embeddings']
            self.text_metadata = list(data['text_metadata'])
            self.image_metadata = list(data['image_metadata'])
            
            if self.text_embeddings.size > 0:
                d = self.text_embeddings.shape[1]
                self.text_index = faiss.IndexFlatL2(d)
                self.text_index.add(self.text_embeddings.astype('float32'))
                print(f"   ✅ Loaded {len(self.text_metadata)} text chunks")
            if self.image_embeddings.size > 0:
                d = self.image_embeddings.shape[1]
                self.image_index = faiss.IndexFlatL2(d)
                self.image_index.add(self.image_embeddings.astype('float32'))
                print(f"   ✅ Loaded {len(self.image_metadata)} image captions")
        except Exception as e:
            print(f"⚠️ Error loading vector store: {e}")
            print("   Building new vector store...")
            self.build_vector_store()

    def retrieve_images(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant images with hybrid matching"""
        print(f"\n🔍 Image search for query: '{query}' (Top-{top_k})")
        query_lower = query.lower()
        chart_keywords = ['chart', 'graph', 'diagram', 'figure', 'data', 'visualization',
                          'emissions', 'revenue', 'trend', 'performance', 'impact',
                          'energy', 'carbon', 'sustainability', 'environmental']
        
        image_results = []
        if self.image_index is not None and len(self.image_embeddings) > 0:
            query_emb = self.model.encode([query])[0].astype('float32')
            D, I = self.image_index.search(np.array([query_emb]), top_k * 2)
            for idx, distance in zip(I[0], D[0]):
                if idx < len(self.image_metadata):
                    meta = self.image_metadata[idx].copy()
                    meta['similarity_score'] = float(1 / (1 + distance))
                    image_results.append(meta)
        
        scored_images = []
        for img in self.image_metadata:
            score = 0
            img_name_lower = img['image_name'].lower()
            caption_lower = img['caption'].lower()
            for kw in chart_keywords:
                if kw in query_lower and (kw in img_name_lower or kw in caption_lower):
                    score += 2
            
            try:
                im = Image.open(img['image_path'])
                w, h = im.size
                if w > 800 and h > 600:
                    score += 3
                elif w > 400 and h > 400:
                    score += 2
            except:
                pass
            if score > 0:
                img_copy = img.copy()
                img_copy['similarity_score'] = img_copy.get('similarity_score', 0) + score * 0.05
                scored_images.append(img_copy)
        
        all_images = {img['image_path']: img for img in (image_results + scored_images)}
        relevant_images = sorted(all_images.values(),
                                 key=lambda x: x.get('similarity_score', 0),
                                 reverse=True)[:top_k]
        
        print(f"🖼️  Retrieval Agent: Found {len(relevant_images)} relevant images")
        return relevant_images

    def get_all_docs(self) -> List[Dict]:
        """Get all document chunks"""
        return self.text_metadata


# Test
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 TESTING ENHANCED RETRIEVAL AGENT")
    print("="*60)
    
    agent = RetrievalAgent("data")
    
    # Test mission statement query
    print("\n" + "="*60)
    print("TEST: Mission Statement Query")
    print("="*60)
    
    result = agent.retrieve("What is Tesla's mission statement?", top_k=3)
    
    print("\n📋 Top 3 Results:")
    for i, doc in enumerate(result['text_results'], 1):
        print(f"\n{i}. Similarity: {doc['similarity_score']:.4f}")
        print(f"   Source: {doc['source']}, Page: {doc['page']}")
        print(f"   Text: {doc['text'][:200]}...")
    
    print("\n✅ TEST COMPLETE!")