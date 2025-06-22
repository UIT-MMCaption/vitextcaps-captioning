import os
import json
import torch
from typing import List

from builders.vocab_builder import META_VOCAB
from data_utils.vocabs import Vocab
from data_utils.utils import preprocess_sentence, is_Vietnamese, composing_word
from typing import *

@META_VOCAB.register()
class ViWordVocab(Vocab):
    def __init__(self, config):
        self.tokenizer = config.TOKENIZER

        self.initialize_special_tokens(config)
        phonemes = self.make_vocab(config.JSON_PATH)
        phonemes = list(phonemes)
        self.itos = {
            i: tok for i, tok in enumerate(self.specials + phonemes)
        }

        self.stoi = {
            tok: i for i, tok in enumerate(self.specials + phonemes)
        }

        onsets = ['ngh', 'tr', 'th', 'ph', 'nh', 'ng', 'kh', 
              'gi', 'gh', 'ch', 'q', 'đ', 'x', 'v', 't', 
              's', 'r', 'n', 'm', 'l', 'k', 'h', 'g', 'd', 
              'c', 'b', '<unk>']
        medial = ['o', 'u']
        nucleuses = ['oo', 'ươ', 'ưa', 'uô', 'ua', 'iê', 'yê', 
                 'ia', 'ya', 'e', 'ê', 'u', 'ư', 'ô', 'i', 
                 'y', 'o', 'ơ', 'â', 'a', 'o', 'ă']
        codas = ['ng', 'nh', 'ch', 'u', 'n', 'o', 'p', 'c', 'm', 'y', 'i', 't']
        
        onsets_idx = [self.stoi[char] for char in onsets]
        medial_idx = [self.stoi[char] for char in medial]
        nucleuses_idx = [self.stoi[char] for char in nucleuses]
        codas_idx = [self.stoi[char] for char in codas]

        # only padding token is not allowed to be shown
        self.specials = [self.padding_token]

        all_idx = list(self.stoi.values())
        onset_ignore = list(set(all_idx) - set(onsets_idx))
        medial_ignore = list(set(all_idx) - set(medial_idx))
        nucleuses_ignore = list(set(all_idx) - set(onsets_idx))
        codas_ignore = list(set(all_idx) - set(codas_idx))

        self.ignore_index = [onset_ignore, medial_ignore, nucleuses_ignore, codas_ignore]

    def initialize_special_tokens(self, config) -> None:
        self.padding_token = config.PAD_TOKEN
        self.bos_token = config.BOS_TOKEN
        self.eos_token = config.EOS_TOKEN
        self.unk_token = config.UNK_TOKEN
        
        self.specials = [self.padding_token, self.bos_token, self.eos_token, self.unk_token]

        self.padding_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

    def make_vocab(self, config):
        json_paths = [config.TRAIN, config.DEV, config.TEST]
        phonemes = set()
        self.max_question_length = 2
        self.max_answer_length = 410
        # Collect token stats from each JSON
        for path in json_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"JSON path not found: {path}")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for key in data:
                item = data[key]
                caption = item["caption"]
                words = preprocess_sentence(caption)
                for word in words:
                    is_Vietnamese_word, components = is_Vietnamese(word)
                    if is_Vietnamese_word:
                        phonemes.update([phoneme for phoneme in components if phoneme])

        return phonemes

    def encode_caption(self, caption: List[str]) -> torch.Tensor:
        syllables = [
            (self.bos_idx, self.padding_idx, self.padding_idx, self.padding_idx)
        ]
        for word in caption:
            is_Vietnamese_word, components = is_Vietnamese(word)
            if is_Vietnamese_word:
                syllables.append([
                    self.stoi[phoneme] if phoneme else self.padding_idx for phoneme in components
                ])
            else:
                syllables.append(
                    (self.unk_idx, self.padding_idx, self.padding_idx, self.padding_idx)
                )

        syllables.append(
            (self.eos_idx, self.padding_idx, self.padding_idx, self.padding_idx)
        )

        vec = torch.tensor(syllables).long()

        return vec

    def decode_caption(self, caption_vec: torch.Tensor, join_words=True):
        assert caption_vec.dim() == 2
        syllable_ids = caption_vec.tolist()
        syllables = [
            (
                self.itos[phoneme_idx] for phoneme_idx in phoneme_ids
            ) for phoneme_ids in syllable_ids
        ]
        sentence = []
        for phonemes in syllables:
            onset, medial, nucleus, coda = phonemes
            # turn phoneme into None if they are in special tokens
            onset = None if onset in self.specials else onset
            medial = None if medial in self.specials else medial
            nucleus = None if nucleus in self.specials else nucleus
            coda = None if coda in self.specials else coda
            word, is_Vietnamese_word = composing_word(onset, medial, nucleus, coda)
            if is_Vietnamese_word:
                sentence.append(word)
            else:
                if onset == self.bos_token:
                    sentence.append(self.bos_token)
                    continue
                
                if onset == self.eos_token:
                    sentence.append(self.eos_token)
                    continue
                
                sentence.append(self.unk_token)

        # remove the <bos> and <eos> token
        if sentence[0] == self.bos_token:
            sentence = sentence[1:]
        if sentence[-1] == self.eos_token:
            sentence = sentence[:-1]

        if join_words:
            return " ".join(sentence)
        else:
            return sentence

    def decode_batch_caption(self, caption_batch: torch.Tensor, join_words=True):
        assert caption_batch.dim() == 3
        captions = [
            self.decode_caption(caption_vec, join_words) for caption_vec in caption_batch
        ]

        return captions