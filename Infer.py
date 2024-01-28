from transformers import PreTrainedTokenizerFast
import torch
from utils import segregate_to_sentence_level_src, segregate_to_sentence_level_tgt
import dataset

sentences_with_context=[]

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    tokenizer_tgt=PreTrainedTokenizerFast(tokenizer_file=tokenizer_tgt)
    tokenizer_src=PreTrainedTokenizerFast(tokenizer_file=tokenizer_src)
    sos_idx = tokenizer_tgt.convert_tokens_to_ids('[SOS]')
    eos_idx = tokenizer_tgt.convert_tokens_to_ids('[EOS]')

    sentences=segregate_to_sentence_level_src(source)
    # Concatenate src sentences with context
    if len(sentences) > 1:
        for j in range(1, len(sentences)):
            concatenated_sentence = sentences[j-1] + "[SEP]" + sentences[j]
            sentences_with_context.append(concatenated_sentence)
    else:
        sentences_with_context.append(sentences[0])

    for sentence in sentences_with_context:
        # Precompute the encoder output and reuse it for every step
        # print(type(sentence))
        # tokenized_input=tokenizer_src.encode(sentence)
        # print(tokenized_input)
        encoder_output = model.encode(sentence, source_mask)
        # Initialize the decoder input with the sos token
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
        while True:
            if decoder_input.size(1) == max_len:
                break

            # build mask for target
            decoder_mask = dataset.causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

            # calculate output
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

            # get next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
            )

            if next_word == eos_idx:
                break

            decoder_input= decoder_input.squeeze(0).split("[SEP]")[1] if "[SEP]" in decoder_input.squeeze(0) else decoder_input.squeeze(0)

    return decoder_input
