import re

with open("src/lightning_modules.py", "r") as f:
    content = f.read()

# Remove old method _compiled_forward_and_loss
start = content.find("    def _compiled_forward_and_loss")
if start != -1:
    end = content.find("    def training_step(", start)
    content = content[:start] + content[end:]

# Update training_step to use self.compiled_wrapper
old_block = """        # Ensure compile is setup (do lazily to avoid graph breaking on self initialization)
        if not hasattr(self, "_compiled_fn"):
            if bool(self.cfg.training.generator.compile_model) and hasattr(torch, "compile"):
                self._compiled_fn = torch.compile(self._compiled_forward_and_loss)
            else:
                self._compiled_fn = self._compiled_forward_and_loss

        outs = self._compiled_fn(
            x, num_bins, num_chunks, dark_threshold, 
            guidance_blur_k, guidance_blur_s,
            w_edge, w_tv, w_range, w_wass, 
            w_guide_blur, w_guide_sharp
        )"""

new_block = """        outs = self.compiled_wrapper(
            x, num_bins, num_chunks, dark_threshold, 
            guidance_blur_k, guidance_blur_s,
            w_edge, w_tv, w_range, w_wass, 
            w_guide_blur, w_guide_sharp
        )"""

content = content.replace(old_block, new_block)

with open("src/lightning_modules.py", "w") as f:
    f.write(content)
