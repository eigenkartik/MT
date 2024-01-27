import spacy
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, random_split
from config import get_config, get_weights_file_path
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from dataset import BilingualDataset

# Initialize spacy for sentence segmentation
nlp = spacy.load("en_core_web_sm")

def segregate_to_sentence_level(input_text):
    # Use spacy to segment the input text into sentences
    sentences = [sentence.text.strip() for sentence in nlp(input_text).sents]
    return sentences


# def get_all_sentences(ds, lang):
#     for item in ds:
#         yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config.get_config()['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]","[SEP]"], min_frequency=1) # Amended on 26 Jan 2024
        tokenizer.train_from_iterator(ds, trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    # Load dataset
    ds_raw = load_dataset(f"{config.get_config()['datasource']}", f"{config.get_config()['lang_src']}-{config.get_config()['lang_tgt']}", split='train')

    # Prepare data for training
    sentences_with_context_source = []  # List to store sentences with context
    sentences_with_context_target = []
    for i in range(len(ds_raw)):
        # Segregate each document into sentences
        sentences_src = segregate_to_sentence_level(ds_raw[i]['translation'][config.get_config()['lang_src']])
        sentences_tgt = segregate_to_sentence_level(ds_raw[i]['translation'][config.get_config()['lang_tgt']])
        
        # Concatenate src sentences with context
        if len(sentences_src) > 1:
            for j in range(1, len(sentences_src)):
                concatenated_sentence = sentences_src[j-1] + "[SEP]" + sentences_src[j]
                sentences_with_context_source.append(concatenated_sentence)
        else:
            sentences_with_context_source.append(sentences_src[0])

         # Concatenate tgt sentences with context
        if len(sentences_tgt) > 1:
            for j in range(1, len(sentences_tgt)):
                concatenated_sentence = sentences_tgt[j-1] + " [SEP] " + sentences_tgt[j]
                sentences_with_context_source.append(concatenated_sentence)
        else:
            sentences_with_context_source.append(sentences_tgt[0])

        

    # Tokenizer for source language
    tokenizer_src = get_or_build_tokenizer(config, sentences_with_context_source, config.get_config()['lang_src'])
    
    # Tokenizer for target language
    tokenizer_tgt = get_or_build_tokenizer(config, sentences_with_context_target, config.get_config()['lang_tgt'])
    
    # # Train-test split
    # train_ds_size = int(0.9 * len(ds_raw))
    # val_ds_size = len(ds_raw) - train_ds_size
    # train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # Create BilingualDataset for training
    train_ds = BilingualDataset(sentences_with_context_source, sentences_with_context_target, tokenizer_src, tokenizer_tgt,config['seq_len'])
    # val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = max(len(tokenizer_src.encode(sentence).ids) for sentence in sentences_with_context_source)
    max_len_tgt = max(len(tokenizer_tgt.encode(sentence).ids) for sentence in sentences_with_context_target)
    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    # Prepare DataLoaders
    train_dataloader = DataLoader(train_ds, batch_size=config.get_config()['batch_size'], shuffle=True)
    # val_dataloader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=True)

    # return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
    return train_dataloader, tokenizer_src, tokenizer_tgt
