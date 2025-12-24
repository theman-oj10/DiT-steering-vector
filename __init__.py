"""
Flux Autosteering - A toolkit for steering diffusion models using concept vectors
"""

from model_utils import setup_device, load_flux_model, model_embed_dim
from hook_utils import get_layer, make_steering_hook, register_steering_range
from vector_utils import (
    random_steering_vector, 
    contrastive_steering_vector, 
    generate_steered_image
)
from cache_utils import VectorCache
from image_utils import evaluate_image_edits, pil_to_uint8
from file_utils import (
    build_image_filename, 
    save_image_unique, 
    setup_directories,
    _row_key
)
from visualization_utils import (
    create_concept_grid, 
    create_line_charts, 
    create_scatter_plots
)
from config import Config, default_config

__version__ = "1.0.0"
__all__ = [
    # Model utilities
    "setup_device",
    "load_flux_model", 
    "model_embed_dim",
    
    # Hook utilities
    "get_layer",
    "make_steering_hook",
    "register_steering_range",
    
    # Vector utilities
    "random_steering_vector",
    "contrastive_steering_vector", 
    "generate_steered_image",
    
    # Cache utilities
    "VectorCache",
    
    # Image utilities
    "evaluate_image_edits",
    "pil_to_uint8",
    
    # File utilities
    "build_image_filename",
    "save_image_unique",
    "setup_directories",
    "_row_key",
    
    # Visualization utilities
    "create_concept_grid",
    "create_line_charts",
    "create_scatter_plots",
    
    # Configuration
    "Config",
    "default_config",
]
