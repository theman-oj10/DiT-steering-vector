"""
Configuration settings for Flux Autosteering
"""

from pathlib import Path
from google.colab import drive
import random


class Config:    
    def __init__(self):
        # Model settings
        self.repo_id = "black-forest-labs/FLUX.1-schnell"
        
        # Default generation settings
        self.default_height = 512
        self.default_width = 768
        self.default_steps = 4
        self.default_seed = 42
        self.max_sequence_length = 256
        self.guidance_scale = 0.0
        
        # Vector generation settings
        self.use_cached_vectors = False
        self.vector_extraction_layers = [18]
        self.seeds_for_averaging = 10
        self.streams_to_extract = ["image", "text"]
        
        # Sweep settings
        self.steering_weights = [0.0, 32.0, 64.0, 128.0, 256.0, 512.0]
        self.steering_targets = ["image", "text", "both"]
        self.sweep_layers = [18]  # Both start and end layers
        
        # File paths (will be set up after drive mount)
        self.base_dir = None
        self.cache_dir = None
        self.results_csv = None
        self.images_dir = None
        self.baseline_path = None
        self.manifest_path = None
        
        # Concept definitions
        self.concept_suite = {
        "age": ("a photo of a person", "an elderly person", "a young person"),
        "material": ("an object", "a metallic object", "a wooden object"),
        # "gender": ("a photo of a person", "a man", "a woman"),
        # "happiness": ("a photo of a person", "a person smiling", "a person frowning"), 
        # "lighting": ("a photo", "a photo taken in bright sunlight", "a photo taken in dim light"),
        }
        
        # Processing settings
        self.base_prompt = "a photo of a person"
        self.save_steered_images = True
        self.save_baseline_image = True
        
        # Run control for sweep
        self.run_mode = "restart"  # "restart" or "continue"
        
    def setup_paths(self, base_path="/content/drive/MyDrive/Flux_Autosteering/results/"):
        """Setup file paths after drive mount"""
        drive.mount("/content/drive", force_remount=False)
        
        self.base_dir = Path(base_path)
        self.cache_dir = self.base_dir / "concept_vectors"
        self.results_csv = self.base_dir / "concept_sweep_results.csv"
        self.images_dir = self.base_dir / "images"
        self.baseline_path = self.base_dir / "baseline.png"
        self.manifest_path = self.cache_dir / "manifest.jsonl"
        
        # Create directories
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.save_steered_images:
            self.images_dir.mkdir(parents=True, exist_ok=True)
            
    def get_seed_list(self, base_seed=None):    
      if base_seed is None:
          base_seed = self.default_seed
      
      # Use base_seed to seed the random generator for reproducibility
      random.seed(base_seed)
      
      # Generate random seeds in a reasonable range
      # Using a wide range to avoid correlation between seeds
      seed_list = []
      for _ in range(self.seeds_for_averaging):
          # Generate random seeds between 0 and 999999
          seed_list.append(random.randint(0, 999999))
      return seed_list
        
    def build_sweep_combinations(self, concept_vectors, concept_meta):
        """
        Build all sweep combinations from available vectors.
        
        Creates three types of combinations:
        1. "image" target: uses image stream vector
        2. "text" target: uses text stream vector
        3. "both" target: uses dict with both vectors (only when both exist)
        """
        combinations = []
        
        # Group vectors by concept and layer
        grouped = {}
        for k, vec in concept_vectors.items():
            if vec is None:
                continue
            meta = concept_meta.get(k, {})
            concept = meta.get("concept")
            extracted_layer = meta.get("layer")
            extracted_stream = meta.get("stream")
            
            group_key = (concept, extracted_layer)
            if group_key not in grouped:
                grouped[group_key] = {}
            grouped[group_key][extracted_stream] = (k, vec)
        
        # Build combinations
        for (concept, extracted_layer), extracted_streams in grouped.items():
            for start_layer in self.sweep_layers:
                for end_layer in self.sweep_layers:
                    if end_layer < start_layer:
                        continue
                    for weight in self.steering_weights:
                        # Add "image" target if image vector exists
                        if "image" in extracted_streams:
                            img_key, img_vec = extracted_streams["image"]
                            combinations.append((
                                concept,
                                extracted_layer,
                                start_layer,
                                end_layer,
                                weight,
                                "image",
                                img_key  # concept_key
                            ))
                        
                        # Add "text" target if text vector exists
                        if "text" in extracted_streams:
                            txt_key, txt_vec = extracted_streams["text"]
                            combinations.append((
                                concept,
                                extracted_layer,
                                start_layer,
                                end_layer,
                                weight,
                                "text",
                                txt_key  # concept_key
                            ))
                        
                        # Add "both" target if BOTH vectors exist
                        if "image" in extracted_streams and "text" in extracted_streams:
                            # Create a special key for "both" combination
                            both_key = f"{concept}_L{extracted_layer}_both"
                            combinations.append((
                                concept,
                                extracted_layer,
                                start_layer,
                                end_layer,
                                weight,
                                "both",
                                both_key  # special key to identify this needs both vectors
                            ))
        
        return combinations



# Default configuration instance
default_config = Config()
