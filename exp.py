from transformers import PreTrainedTokenizerFast

tokenizer=PreTrainedTokenizerFast(tokenizer_file="tokenizer_it.json")

a=tokenizer.convert_tokens_to_ids('[SOS]')
print(a)