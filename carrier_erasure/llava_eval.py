"""LLaVA-1.6 judge: 10-way/20-way classifier, yes/no probes and the four-way audit.

The class is copied verbatim from notebooks/projection_erasure_v6.ipynb (cell 13).
Prompts and decoding match paper App. A.
"""
from __future__ import annotations

import gc
import re

import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

COHERENCE_QUESTION = "Is this a normal, coherent image (not noise, not blank, not broken)?"


class LLaVA:
    def __init__(self, mid="llava-hf/llava-v1.6-vicuna-7b-hf"):
        self.mid, self.model, self.proc = mid, None, None
    def load(self):
        if self.model is None:
            self.proc = LlavaNextProcessor.from_pretrained(self.mid)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.mid, torch_dtype=torch.float16, device_map="auto")
    def unload(self):
        if self.model is not None:
            del self.model, self.proc; self.model = self.proc = None
            gc.collect(); torch.cuda.empty_cache()
    def ask(self, image, prompt, max_new=10):
        self.load()
        if isinstance(image, str): image = Image.open(image).convert("RGB")
        conv = [{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt}]}]
        t = self.proc.apply_chat_template(conv, add_generation_prompt=True)
        inp = self.proc(images=image, text=t, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        return self.proc.decode(out[0][inp["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
    def classify(self, image, options, domain):
        opts = "\n".join(f"{i+1}. {o.replace('_',' ')}" for i, o in enumerate(options))
        head = ("You are an image classifier. Classify the artistic style of the given image."
                if domain == "style" else "Classify the object depicted in this image.")
        r = self.ask(image, f"{head}\nChoose exactly one option from the numbered list "
                            f"below. Respond with only the number.\nOptions:\n{opts}")
        m = re.findall(r"\d+", r)
        if m and 1 <= int(m[0]) <= len(options): return options[int(m[0]) - 1]
        rl = r.lower()
        for o in options:
            if o.lower().replace("_", " ") in rl: return o
        return None
    def yes(self, image, q):
        return "yes" in self.ask(image, q + " Answer with only yes or no.").lower()
    def bucket(self, image, concept, kind):
        c = concept.replace("_", " ")
        if kind == "object":
            if self.yes(image, f"Is there a {c} in this image?"): return "target_remains"
            if not self.yes(image, "Is this a normal, coherent image (not noise, "
                                   "not blank, not broken)?"):    return "destroyed"
            if self.yes(image, "Is there another clearly visible main object or "
                               "animal in this image?"):           return "replaced"
            return "clean_removal"
        if self.yes(image, f"Is this image in {c} style?"):        return "target_remains"
        if not self.yes(image, "Is this a normal, coherent image (not noise, "
                               "not blank, not broken)?"):         return "destroyed"
        return "clean_removal"
