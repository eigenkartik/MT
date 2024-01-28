import Infer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

input="You're obstinate."

output=Infer.greedy_decode("opus_books_weights",source=input, source_mask="[PAD]", tokenizer_src="tokenizer_en.json",tokenizer_tgt="tokenizer_it.json", max_len=512, device=device )
print(output)