"""
Controller Agent - Enhanced Version (October 2025)
---------------------------------------------------
Improved multi-agent orchestration for:
✅ Fact-based queries
✅ Analytical queries
✅ Summarization queries (enhanced retrieval & fallback)
✅ Visual queries (context filtering using similarity)
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
from sentence_transformers import SentenceTransformer, util  # ✅ used for similarity filter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.intent_agent import IntentAgent
from agents.retrieval_agent import RetrievalAgent
from agents.vision_agent import VisionAgent
from agents.reasoning_agent import ReasoningAgent


class ControllerAgent:
    """Main orchestrator of all sub-agents."""

    def __init__(
        self,
        vector_store_path: str = "data",
        intent_model_path: str = "models/intent_model.pkl",
        vectorizer_path: str = "models/vectorizer.pkl"
    ):
        print("=" * 60)
        print("🧠 INITIALIZING CONTROLLER AGENT")
        print("=" * 60)

        # Initialize core agents
        print("\n1️⃣ Loading Intent Agent...")
        self.intent_agent = IntentAgent(intent_model_path, vectorizer_path)

        print("\n2️⃣ Loading Retrieval Agent...")
        self.retrieval_agent = RetrievalAgent(vector_store_path)

        print("\n3️⃣ Loading Vision Agent...")
        self.vision_agent = VisionAgent()

        print("\n4️⃣ Loading Reasoning Agent...")
        self.reasoning_agent = ReasoningAgent()

        # ✅ New: embedding model for cross-modal similarity
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        print("\n" + "=" * 60)
        print("✅ CONTROLLER AGENT READY")
        print("=" * 60)

    # ===============================================================
    # Core processing
    # ===============================================================
    def process_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Main orchestration pipeline"""
        print("\n" + "=" * 60)
        print(f"📝 PROCESSING QUERY: {query}")
        print("=" * 60)

        # Step 1: Classify intent
        print("\n🎯 Step 1: Classifying Intent...")
        intent_result = self.intent_agent.classify(query)
        if isinstance(intent_result, dict):
            intent = intent_result.get('intent', 'fact')
            confidence = intent_result.get('confidence', 0)
            print(f"🎯 Intent Agent: Classified as '{intent}' (Confidence: {confidence:.1f}%)")
        else:
            intent = intent_result
            print(f"🎯 Intent Agent: Classified as '{intent}'")

        result = {
            'query': query,
            'intent': intent,
            'answer': None,
            'retrieved_docs': None,
            'image_descriptions': None
        }

        # Step 2: Route by intent
        if intent == "fact":
            result = self._handle_fact_query(query, top_k)
        elif intent == "analysis":
            result = self._handle_analysis_query(query, top_k)
        elif intent == "summary":
            result = self._handle_summary_query(query, top_k)
        elif intent == "visual":
            result = self._handle_visual_query(query, top_k)
        else:
            result['answer'] = f"Unknown intent '{intent}'. Please rephrase your query."

        result['intent'] = intent
        result['query'] = query
        return result

    # ===============================================================
    # Helper extraction
    # ===============================================================
    def _extract_text_results(self, retrieved_results):
        """Normalize text retrieval output"""
        if isinstance(retrieved_results, list):
            return retrieved_results
        if isinstance(retrieved_results, dict):
            if 'text_results' in retrieved_results:
                return retrieved_results['text_results']
            if 'results' in retrieved_results:
                inner = retrieved_results['results']
                if isinstance(inner, dict) and 'text_results' in inner:
                    return inner['text_results']
                if isinstance(inner, list):
                    return inner
        return []

    # ===============================================================
    # FACT
    # ===============================================================
    def _handle_fact_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        print("\n📚 Step 2: Retrieving Documents (Fact-based)...")
        retrieved_results = self.retrieval_agent.retrieve(query, top_k=top_k)
        retrieved_docs = self._extract_text_results(retrieved_results)
        print(f"   ➜ Retrieved {len(retrieved_docs)} documents")
        if retrieved_docs:
            print(f"   📝 Sample doc keys: {retrieved_docs[0].keys() if isinstance(retrieved_docs[0], dict) else type(retrieved_docs[0])}")

        print("\n🧠 Step 3: Generating Answer...")
        answer = self.reasoning_agent.answer(query, retrieved_docs)
        return {'answer': answer, 'retrieved_docs': retrieved_docs, 'image_descriptions': None}

    # ===============================================================
    # ANALYSIS
    # ===============================================================
    def _handle_analysis_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        print("\n📚 Step 2: Retrieving Documents (Multi-document Analysis)...")
        retrieved_results = self.retrieval_agent.retrieve(query, top_k=top_k)
        retrieved_docs = self._extract_text_results(retrieved_results)
        print(f"   ➜ Retrieved {len(retrieved_docs)} documents for analysis")

        print("\n🧠 Step 3: Performing Analysis...")
        answer = self.reasoning_agent.reason(query, retrieved_docs)
        return {'answer': answer, 'retrieved_docs': retrieved_docs, 'image_descriptions': None}

    # ===============================================================
    # SUMMARY  (Enhanced Retrieval)
    # ===============================================================
    def _handle_summary_query(self, query: str, top_k: int = 8) -> Dict[str, Any]:
        print("\n📚 Step 2: Retrieving Documents (Summary)...")
        retrieved_results = self.retrieval_agent.retrieve(query, top_k=top_k)
        docs = self._extract_text_results(retrieved_results)
        print(f"   ➜ Retrieved {len(docs)} initial documents")

        # ✅ Fallback: Expand query with keywords if too few retrieved
        if len(docs) < 3:
            expanded_query = query + " achievements progress results milestones goals impact 2023 performance summary"
            print("   ⚠️ Few docs found. Expanding query...")
            fallback_results = self.retrieval_agent.retrieve(expanded_query, top_k=top_k)
            fallback_docs = self._extract_text_results(fallback_results)
            docs.extend(fallback_docs)
            print(f"   ➜ After expansion: {len(docs)} total documents")

        # ✅ Combine text chunks for summarization
        combined_text = "\n".join([d.get("text", "") for d in docs if isinstance(d, dict)])
        if not combined_text.strip():
            print("   ⚠️ No relevant text found for summarization.")
            return {'answer': "No relevant text available to summarize.", 'retrieved_docs': docs}

        print("\n🧠 Step 3: Generating Summary...")
        answer = self.reasoning_agent.summarize(combined_text)
        return {'answer': answer, 'retrieved_docs': docs, 'image_descriptions': None}

    # ===============================================================
    # VISUAL  (Enhanced Context Filtering)
    # ===============================================================
    def _handle_visual_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        print("\n🖼️  Step 2a: Describing Related Images...")
        retrieved_results = self.retrieval_agent.retrieve(query, search_type='image', top_k=3)
        image_results = []
        if isinstance(retrieved_results, dict):
            image_results = retrieved_results.get('image_results', [])

        image_descriptions = []
        for img_info in image_results:
            image_path = None
            if isinstance(img_info, dict):
                image_path = img_info.get("image_path")
                image_name = img_info.get("image_name", Path(image_path).name if image_path else "unknown_image")
            else:
                image_path = img_info
                image_name = Path(image_path).name

            if not image_path:
                continue

            try:
                desc = self.vision_agent.describe_image(image_path)
                # ✅ Compute semantic similarity between query and image caption
                sim = util.cos_sim(
                    self.embedding_model.encode(query, convert_to_tensor=True),
                    self.embedding_model.encode(desc, convert_to_tensor=True)
                ).item()
                if sim >= 0.35:
                    image_descriptions.append({"image_name": image_name, "description": desc})
                else:
                    print(f"   ⚠️ Skipping irrelevant image ({image_name}), similarity={sim:.2f}")
            except Exception as e:
                print(f"   ⚠️ Error describing image {image_name}: {e}")

        print(f"   ➜ Found {len(image_descriptions)} relevant images")

        print("\n📚 Step 2b: Retrieving Text Context...")
        text_results = self.retrieval_agent.retrieve(query, search_type='text', top_k=3)
        text_context = self._extract_text_results(text_results)
        print(f"   ➜ Retrieved {len(text_context)} text documents")

        # ✅ Fallback: text-only reasoning if no relevant images
        if not image_descriptions:
            print("   ⚠️ No relevant images found. Using text-only reasoning.")
            answer = self.reasoning_agent.answer(query, text_context)
        else:
            print("\n🧠 Step 3: Generating Answer with Visual Context...")
            answer = self.reasoning_agent.answer_with_visual(query, text_context, image_descriptions)

        return {'answer': answer, 'retrieved_docs': text_context, 'image_descriptions': image_descriptions}

    # ===============================================================
    # Result Printer
    # ===============================================================
    def print_result(self, result: Dict[str, Any]) -> None:
        print("\n" + "=" * 60)
        print("📊 FINAL RESULT")
        print("=" * 60)
        print(f"\n🎯 Intent: {result['intent'].upper()}")
        print(f"\n❓ Query: {result['query']}")
        print(f"\n✅ Answer:\n{result['answer']}")
        if result.get('retrieved_docs'):
            print(f"\n📚 Retrieved {len(result['retrieved_docs'])} documents")
        if result.get('image_descriptions'):
            print(f"\n🖼️  Analyzed {len(result['image_descriptions'])} images")
        print("\n" + "=" * 60)


# ===============================================================
# Standalone Testing
# ===============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 TESTING CONTROLLER AGENT")
    print("=" * 60)

    if not os.path.exists("models"):
        print("❌ models/ directory not found!")
        sys.exit(1)
    if not os.path.exists("data"):
        print("❌ data/ directory not found!")
        sys.exit(1)

    input("\nPress Enter to continue with initialization...")

    controller = ControllerAgent(vector_store_path="data")

    test_queries = [
        "How many vehicles did Tesla deliver in 2023?",
        "What does the revenue chart show?",
        "Summarize Tesla's key achievements",
        "Compare Tesla's performance across years"
    ]

    print("\n" + "=" * 60)
    print("🧪 RUNNING TEST QUERIES")
    print("=" * 60)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"TEST QUERY {i}/{len(test_queries)}")
        print(f"{'='*60}")
        result = controller.process_query(query)
        controller.print_result(result)
        if i < len(test_queries):
            input("\nPress Enter for next test...")

    print("\n" + "=" * 60)
    print("✅ CONTROLLER AGENT TESTING COMPLETE!")
    print("=" * 60)
