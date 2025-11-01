#!/usr/bin/env python3
"""Simple main script for Turkish License Plate Recognition."""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from eplatereader.core.pipeline import PlateReaderPipeline
from eplatereader.utils.logger import setup_logger

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Turkish License Plate Recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py image.jpg
  python main.py image.jpg --verbose
  python main.py image.jpg --device cuda
        """
    )
    
    parser.add_argument('image', help='Image file path')
    parser.add_argument('--device', '-d', default=None, help='Device (cuda/cpu)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--output', '-o', help='Output file (JSON/CSV)')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO" if args.verbose else "WARNING"
    setup_logger(log_level=log_level)
    
    try:
        # Initialize pipeline
        print("🚀 Initializing pipeline...")
        pipeline = PlateReaderPipeline(
            device=args.device, 
            debug=args.debug
        )
        print("✅ Pipeline ready\n")
        
        # Process image
        image_path = Path(args.image)
        
        if not image_path.exists():
            print(f"❌ Error: Image not found: {args.image}")
            return 1
        
        print(f"📸 Processing: {image_path.name}")
        result = pipeline.process_image(image_path)
        
        # Display results
        print("\n" + "="*60)
        if result.success:
            status = "✅ VALID" if result.is_valid else "⚠️  INVALID FORMAT"
            print(f"🚗 Plate Number: {result.plate_text}")
            print(f"📊 Confidence: {result.confidence:.2%}")
            print(f"✓  Status: {status}")
            print(f"🔍 Detection Method: {result.detection_method}")
            print(f"⏱️  Processing Time: {result.processing_time:.2f}s")
            
            if args.output:
                save_result(result, args.output)
                print(f"💾 Saved to: {args.output}")
        else:
            print(f"❌ Failed: {result.error}")
            return 1
        
        print("="*60)
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 130
    
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def save_result(result, output_path):
    """Save result to file."""
    output_path = Path(output_path)
    
    if output_path.suffix == '.json':
        import json
        data = {
            'plate': result.plate_text,
            'confidence': result.confidence,
            'valid': result.is_valid,
            'processing_time': result.processing_time,
            'detection_method': result.detection_method
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    else:  # CSV
        with open(output_path, 'w') as f:
            f.write(f"{result.plate_text},{result.confidence:.4f},{result.is_valid},{result.processing_time:.2f}\n")


if __name__ == "__main__":
    sys.exit(main())
