"""
Caching utilities for storing and loading steering vectors
"""
import json
import hashlib
import torch
from pathlib import Path


class VectorCache:
    """Manages caching of steering vectors with manifest tracking"""
    def __init__(self, cache_dir: Path, manifest_path: Path = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.manifest_path = manifest_path or (self.cache_dir / "manifest.jsonl")
        self._manifest_keys = set()
        
        # Load existing manifest keys
        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            self._manifest_keys.add(json.loads(line).get("key"))
                        except Exception:
                            pass

    def _append_manifest_line(self, key_hash: str, meta: dict):
        """Append a line to the manifest file"""
        rec = {"key": key_hash, **meta, "path": f"{key_hash}.pt"}
        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._manifest_keys.add(key_hash)


    def vector_key(
        self,
        concept: str,
        layer_idx: int,
        stream: str,                 # 'image' | 'text'
        pos_prompt: str,
        neg_prompt: str,
        steps: int,
        H: int,
        W: int,
        model_tag: str | None = None,
        seed_avg: dict | None = None,
    ):
        """
        Generate a stable hash key for a stream-specific steering vector.
        
        Returns:
            tuple: (hash_key, metadata_dict)
        """
        payload = {
            "concept": concept,
            "layer_idx": int(layer_idx),
            "stream": stream,                     # <— IMPORTANT: disambiguates image/text
            "pos": pos_prompt,
            "neg": neg_prompt,
            "steps": int(steps),
            "H": int(H),
            "W": int(W),
            "model_tag": model_tag or "unknown_model",
            "seed_avg": seed_avg or {"base": None, "n": 1},
        }
        stable_str = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        hashed_key = hashlib.sha1(stable_str.encode("utf-8")).hexdigest()[:16]  # keep sha1 to match existing scheme
        return hashed_key, payload


    def save_vector(self, vec: torch.Tensor, key_hash: str, meta: dict):
        """Save a vector to cache with metadata"""
        path = self.cache_dir / f"{key_hash}.pt"
        torch.save({"vec": vec.cpu().to(torch.float16), "meta": meta}, path)
        self._append_manifest_line(key_hash, meta)
        return path


    def load_vector_if_exists(self, key_hash: str):
        """
        Load a vector if it exists in cache
        
        Returns:
            tuple: (vector, metadata) or (None, None) if not found
        """
        path = self.cache_dir / f"{key_hash}.pt"
        if path.exists():
            obj = torch.load(path, map_location="cpu")
            return obj["vec"], obj.get("meta", {})
        return None, None


    def load_all_vectors(self):
        """
        Load all vectors from manifest
        
        Returns:
            tuple: (concept_vectors_dict, concept_meta_dict)
        """
        concept_vectors = {}
        concept_meta = {}
        
        if not self.manifest_path.exists():
            return concept_vectors, concept_meta
            
        with open(self.manifest_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                    
                rec = json.loads(line)
                vec_path = self.cache_dir / rec["path"]
                
                if vec_path.exists():
                    data = torch.load(vec_path, map_location="cpu")
                    concept = rec.get("concept")
                    layer_field = rec.get("layer", rec.get("layer_idx"))
                    layer = int(layer_field) if layer_field is not None else None
                    stream = rec.get("stream", "")
                    
                    key = f"{concept}_L{layer}_{stream}" if stream else f"{concept}_L{layer}"
                    concept_vectors[key] = data["vec"]
                    concept_meta[key] = {
                        "concept": concept,
                        "layer": layer,
                        "stream": stream,
                        "path": str(rec.get("path"))
                    }
        
        return concept_vectors, concept_meta


    def load_layer_range(self, concept: str, start_layer: int, end_layer: int, stream: str = "both"):
        """
        Load steering vectors for a range of layers as a dict for layer-specific steering.
        Reads from manifest to find vectors that match the concept, layer range, and stream(s).
        
        Args:
            concept: The concept name (e.g., "Van_Gogh", "age", "material")
            start_layer: Starting layer index
            end_layer: Ending layer index (inclusive)
            stream: "image", "text", or "both"
            
        Returns:
            dict: Mapping layer_idx -> steering_vector
                  If stream="both", each value is {'image': tensor, 'text': tensor}
        """
        if not self.manifest_path.exists():
            return {}
        
        layer_vectors = {}
        
        # Read manifest and find matching vectors
        with open(self.manifest_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    rec = json.loads(line)
                    
                    # Check if this record matches our criteria
                    if rec.get("concept") != concept:
                        continue
                    
                    layer_field = rec.get("layer", rec.get("layer_idx"))
                    if layer_field is None:
                        continue
                    
                    layer_idx = int(layer_field)
                    if not (start_layer <= layer_idx <= end_layer):
                        continue
                    
                    vec_stream = rec.get("stream", "")
                    
                    # Skip if stream doesn't match what we're looking for
                    if stream != "both" and vec_stream != stream:
                        continue
                    if stream == "both" and vec_stream not in ["image", "text"]:
                        continue
                    
                    # Load the vector
                    vec_path = self.cache_dir / rec["path"]
                    if vec_path.exists():
                        data = torch.load(vec_path, map_location="cpu")
                        vec = data["vec"]
                        
                        # Organize by layer and stream
                        if stream == "both":
                            if layer_idx not in layer_vectors:
                                layer_vectors[layer_idx] = {"image": None, "text": None}
                            layer_vectors[layer_idx][vec_stream] = vec
                        else:
                            layer_vectors[layer_idx] = vec
                
                except (json.JSONDecodeError, KeyError) as e:
                    continue
        
        return layer_vectors
