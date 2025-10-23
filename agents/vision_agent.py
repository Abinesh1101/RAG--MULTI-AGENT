from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from pathlib import Path

class VisionAgent:
    """
    Agent responsible for understanding and describing images
    Uses BLIP-2 Vision-Language Model
    """
    
    def __init__(self):
        """Initialize the Vision Agent with BLIP model"""
        print("🔄 Loading Vision Model (BLIP)...")
        
        # Load BLIP model and processor
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        print(f"✅ Vision Agent: Model loaded on {self.device}")
    
    def describe_image(self, image_path):
        """
        Generate description for a single image
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Description of the image
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Process image
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption
            outputs = self.model.generate(**inputs, max_length=50)
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            print(f"🖼️  Vision Agent: Described image - {Path(image_path).name}")
            return caption
            
        except Exception as e:
            print(f"⚠️  Error processing image {image_path}: {str(e)}")
            return f"Image from {Path(image_path).name}"
    
    def describe_related_images(self, query, image_paths, max_images=3):
        """
        Describe multiple images related to a query
        
        Args:
            query (str): User's question
            image_paths (list): List of image paths
            max_images (int): Maximum number of images to describe
            
        Returns:
            str: Combined description of all images
        """
        if not image_paths:
            return "No relevant images found."
        
        descriptions = []
        
        # Limit to max_images
        selected_images = image_paths[:max_images]
        
        print(f"🔍 Vision Agent: Analyzing {len(selected_images)} images...")
        
        for img_path in selected_images:
            description = self.describe_image(img_path)
            descriptions.append(f"- {Path(img_path).name}: {description}")
        
        combined = "\n".join(descriptions)
        
        print(f"✅ Vision Agent: Generated descriptions for {len(selected_images)} images")
        
        return combined
    
    def analyze_chart(self, image_path, context_query=None):
        """
        Specialized method for analyzing charts and graphs
        
        Args:
            image_path (str): Path to chart image
            context_query (str): Optional query for context
            
        Returns:
            str: Analysis of the chart
        """
        # Use conditional generation with context
        try:
            image = Image.open(image_path).convert('RGB')
            
            if context_query:
                # Use query as prompt for more relevant description
                prompt = f"a chart showing {context_query}"
                inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
            else:
                inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(**inputs, max_length=100)
            description = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            return description
            
        except Exception as e:
            return f"Chart from {Path(image_path).name}"


# Test the agent if run directly
if __name__ == "__main__":
    print("🧪 Testing Vision Agent...\n")
    
    agent = VisionAgent()
    
    # Test with first few images
    from pathlib import Path
    image_folder = Path("data/images")
    
    test_images = list(image_folder.glob("*.png"))[:3]
    
    if test_images:
        print(f"\n📸 Testing with {len(test_images)} sample images:\n")
        
        for img_path in test_images:
            print(f"Image: {img_path.name}")
            description = agent.describe_image(str(img_path))
            print(f"Description: {description}\n")
    else:
        print("⚠️  No images found in data/images folder")