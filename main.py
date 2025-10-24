

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.controller_agent import ControllerAgent


def print_banner():
    """Print welcome banner"""
    print("\n" + "=" * 70)
    print("🤖 MULTI-AGENT RAG SYSTEM")
    print("=" * 70)
    print("Vision-Enhanced, Intent-Aware AI Assistant")
    print("Supports: Fact, Analysis, Summary, and Visual queries")
    print("=" * 70 + "\n")


def print_help():
    """Print usage instructions"""
    print("\n📖 USAGE INSTRUCTIONS:")
    print("-" * 70)
    print("Ask questions about your documents in natural language!")
    print("\nExample queries:")
    print("  • Fact:     'How many vehicles did Tesla deliver in 2023?'")
    print("  • Analysis: 'Compare Google and Tesla sustainability efforts'")
    print("  • Summary:  'Summarize the key environmental goals'")
    print("  • Visual:   'What does the emissions chart show?'")
    print("\nCommands:")
    print("  • 'help'  - Show this help message")
    print("  • 'exit'  - Exit the program")
    print("  • 'quit'  - Exit the program")
    print("-" * 70 + "\n")


def interactive_mode(controller):
    """Run interactive query mode"""
    print("\n🎯 INTERACTIVE MODE")
    print("Type your questions below (or 'help' for instructions, 'exit' to quit)\n")
    
    while True:
        try:
            # Get user input
            query = input("❓ Your question: ").strip()
            
            # Handle empty input
            if not query:
                continue
            
            # Handle commands
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye! Thank you for using the Multi-Agent RAG System!")
                break
            
            if query.lower() in ['help', 'h', '?']:
                print_help()
                continue
            
            # Process query
            print("\n" + "⏳ Processing your query...\n")
            result = controller.process_query(query)
            
            # Print result
            controller.print_result(result)
            
            print("\n" + "-" * 70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again with a different query.\n")


def demo_mode(controller):
    """Run demonstration with pre-defined queries"""
    print("\n🎬 DEMO MODE")
    print("Running demonstration with sample queries...\n")
    
    demo_queries = [
        {
            'query': 'How many vehicles did Tesla deliver in 2023?',
            'description': 'Factual query about Tesla deliveries'
        },
        {
            'query': 'Compare Tesla and Google environmental initiatives',
            'description': 'Analytical comparison across documents'
        },
        {
            'query': 'Summarize the key sustainability achievements',
            'description': 'Summary of main points'
        },
        {
            'query': 'What does the emissions data show?',
            'description': 'Visual query about charts/data'
        }
    ]
    
    for i, demo in enumerate(demo_queries, 1):
        print(f"\n{'=' * 70}")
        print(f"DEMO QUERY {i}/{len(demo_queries)}")
        print(f"{'=' * 70}")
        print(f"📝 Description: {demo['description']}")
        print(f"❓ Query: {demo['query']}")
        print(f"{'=' * 70}\n")
        
        try:
            result = controller.process_query(demo['query'])
            controller.print_result(result)
        except Exception as e:
            print(f"❌ Error processing query: {str(e)}")
        
        if i < len(demo_queries):
            input("\n⏸️  Press Enter to continue to next demo query...")
    
    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE!")
    print("=" * 70)


def single_query_mode(controller, query):
    """Process a single query"""
    print(f"\n🔍 Processing single query: {query}\n")
    
    try:
        result = controller.process_query(query)
        controller.print_result(result)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0


def main():
    """Main entry point"""
    print_banner()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(
        description='Multi-Agent RAG System - Vision-Enhanced AI Assistant'
    )
    parser.add_argument(
        '--mode',
        choices=['interactive', 'demo', 'query'],
        default='interactive',
        help='Execution mode: interactive (default), demo, or single query'
    )
    parser.add_argument(
        '--query',
        type=str,
        help='Single query to process (requires --mode query)'
    )
    parser.add_argument(
        '--vector-store',
        type=str,
        default='data',
        help='Path to vector store directory (default: data)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'query' and not args.query:
        print("❌ Error: --query argument required when using --mode query")
        return 1
    
    # Check if required directories exist
    print("🔍 Checking system requirements...")
    
    required_dirs = ['models', 'data', 'agents']
    missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
    
    if missing_dirs:
        print(f"❌ Error: Missing required directories: {', '.join(missing_dirs)}")
        print("\nPlease ensure the following structure exists:")
        print("  - models/     (contains intent_model.pkl, vectorizer.pkl)")
        print("  - data/       (contains PDFs and images)")
        print("  - agents/     (contains agent Python files)")
        return 1
    
    # Check for required model files
    required_models = ['models/intent_model.pkl', 'models/vectorizer.pkl']
    missing_models = [m for m in required_models if not os.path.exists(m)]
    
    if missing_models:
        print(f"❌ Error: Missing required model files: {', '.join(missing_models)}")
        print("\nPlease train the intent model first:")
        print("  python models/train_intent_model.py")
        return 1
    
    print("✅ All requirements satisfied!\n")
    
    # Initialize controller
    print("🚀 Initializing Multi-Agent System...")
    print("This may take 1-2 minutes...\n")
    
    try:
        controller = ControllerAgent(vector_store_path=args.vector_store)
    except Exception as e:
        print(f"\n❌ Failed to initialize system: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n✅ System initialized successfully!\n")
    
    # Run selected mode
    if args.mode == 'interactive':
        print_help()
        interactive_mode(controller)
    elif args.mode == 'demo':
        demo_mode(controller)
    elif args.mode == 'query':
        return single_query_mode(controller, args.query)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())