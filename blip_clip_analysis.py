import os
import json
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import open_clip
import torchvision.transforms as transforms
import torch.nn.functional as F

# Load configuration from config.json
with open('config.json') as f:
    config = json.load(f)

image_folder = config['image_folder']
image_files = config['image_files']
text_prompts = config['text_prompts']

# BLIP Setup (for captioning)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# CLIP Setup (for zero-shot classification)
clip_model = open_clip.create_model('ViT-B-32', pretrained='openai')
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Define the image preprocessing steps
clip_preprocess = transforms.Compose([
    transforms.Resize(224),  # Resize the image to 224x224
    transforms.CenterCrop(224),  # Crop the center part of the image
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])  # Normalize with CLIP's mean and std
])

# Function to generate captions using BLIP with longer captions
def generate_caption(image_path, max_new_tokens=50, num_beams=5):
    raw_image = Image.open(image_path).convert('RGB')
    inputs = blip_processor(raw_image, return_tensors="pt")
    # Generate caption with longer length and beam search for better quality
    caption_ids = blip_model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)
    caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)
    return caption

# Function to calculate cosine similarity
def cosine_similarity(a, b):
    return F.cosine_similarity(a, b.unsqueeze(0))

# Function to perform zero-shot classification using CLIP
def zero_shot_classification(image_path, text_prompts):
    image = clip_preprocess(Image.open(image_path)).unsqueeze(0)  # Preprocess image
    text_inputs = clip_tokenizer(text_prompts)  # Tokenize text prompts
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text_inputs)

        # Calculate cosine similarities between the image and text features
        similarities = [cosine_similarity(image_features, text_feature) for text_feature in text_features]
        similarities = torch.cat(similarities)

    return similarities

# Process each image
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    
    # Generate caption with BLIP (longer captions)
    caption = generate_caption(image_path, max_new_tokens=50, num_beams=5)
    print(f"Caption for {image_file}: {caption}")
    
    # Zero-shot classification with CLIP
    logits = zero_shot_classification(image_path, text_prompts)
    best_prompt_idx = logits.argmax().item()
    best_prompt = text_prompts[best_prompt_idx]
    confidence = logits.max().item()

    print(f"Best match for {image_file}: '{best_prompt}' with confidence {confidence:.4f}")
