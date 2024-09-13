from torch import Tensor, nn
from transformers import (CLIPTextModel, CLIPTokenizer, T5EncoderModel,
                          T5Tokenizer)


class HFEmbedder(nn.Module):
    def __init__(self, version: str, max_length: int, local_path:str = None, **hf_kwargs):
        super().__init__()
        self.is_clip = version.startswith("openai")
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            # 判断是否提供了本地路径，如果为空，则从 Hugging Face 仓库下载
            if local_path:
                print(f"Loading CLIP model from local path: {local_path}")
                self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(local_path, local_files_only=True, max_length=max_length)
                self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(local_path, local_files_only=True, **hf_kwargs)
            else:
                print("Downloading CLIP model from Hugging Face Hub...")
                self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14",
                                                                              max_length=max_length)
                self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14",
                                                                              **hf_kwargs)
        else:
            # 判断是否提供了本地路径，如果为空，则从 Hugging Face 仓库下载
            if local_path:
                print(f"Loading T5 model from local path: {local_path}")
                self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(local_path, local_files_only=True, max_length=max_length)
                self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(local_path, local_files_only=True, **hf_kwargs)
            else:
                print("Downloading T5 model from Hugging Face Hub...")
                self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl", max_length=max_length)
                self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained("google/t5-v1_1-xxl", **hf_kwargs)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]
