import base64
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

def image_to_base64(image_path: str) -> str:
    """Convert image to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return ""
    except Exception as e:
        print(f"Error reading image file: {e}")
        return ""

def recognize_plate_local(image_path: str):
    print("Loading model...")
    # default: Load the model on the available device(s)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map="auto"
    )
    print("Model loaded, loading processor...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    print("Processor loaded.")

    print(f"Processing image: {image_path}...")
    image_base64 = image_to_base64(image_path)
    if not image_base64:
        print("Failed to convert image to base64. Exiting.")
        return None

    # This is the prompt for license plate recognition
    prompt = """This is a Turkish license plate image. Carefully read the license plate number.
    The format is typically: 2 digits, then 1-3 uppercase letters, then 2-4 digits.
                
    Strict rules for the output:
    1. ONLY the license plate number. No other text, punctuation, or descriptions.
    2. All characters must be alphanumeric (A-Z, 0-9).
    3. All letters must be uppercase.
    4. No spaces between characters.
    
    If you are unable to detect a plate number or are unsure, return an empty string."""
            

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"data:image/jpeg;base64,{image_base64}", # Use the local image base64
                },
                {"type": "text", "text": prompt}, # Your license plate prompt
            ],
        }
    ]

    # Preparation for inference
    print("Applying chat template and moving to device...")
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    print("Generating response...")
    # Increased max_new_tokens for potentially longer responses and removed unsupported flags
    generated_ids = model.generate(**inputs, max_new_tokens=50) # Adjusted max_new_tokens

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip() # [0] because batch_decode returns a list, and we expect one output
    
    print(f"Raw model output: {output_text}")

    # Clean up the output to get only the plate number
    plate_text = ''.join(c.upper() for c in output_text if c.isalnum())
    print(f"Cleaned plate: {plate_text}")
    return plate_text

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python your_script_name.py <image_path>")
        sys.exit(1)
        
    image_file_path = sys.argv[1]
    
    try:
        detected_plate = recognize_plate_local(image_file_path)
        print(f"\nFinal Detected Plate: {detected_plate}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()