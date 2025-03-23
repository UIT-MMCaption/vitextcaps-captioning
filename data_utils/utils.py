import torch
import re
import unicodedata
from typing import List

from utils.instance import Instance, InstanceList

def get_tokenizer(tokenizer):
    if callable(tokenizer):
        return tokenizer
    elif tokenizer is None:
        return lambda s: s 
    elif tokenizer == "pyvi":
        try:
            from pyvi import ViTokenizer
            return ViTokenizer.tokenize
        except ImportError:
            print("Please install PyVi package. "
                  "See the docs at https://github.com/trungtv/pyvi for more information.")
    elif tokenizer == "spacy":
        try:
            from spacy.lang.vi import Vietnamese
            return Vietnamese()
        except ImportError:
            print("Please install SpaCy and the SpaCy Vietnamese tokenizer. "
                  "See the docs at https://gitlab.com/trungtv/vi_spacy for more information.")
            raise
        except AttributeError:
            print("Please install SpaCy and the SpaCy Vietnamese tokenizer. "
                  "See the docs at https://gitlab.com/trungtv/vi_spacy for more information.")
            raise
    elif tokenizer == "vncorenlp":
        try:
            from vncorenlp import VnCoreNLP
            # before using vncorenlp, please run this command in your terminal:
            # vncorenlp -Xmx500m data_utils/vncorenlp/VnCoreNLP-1.1.1.jar -p 9000 -annotators wseg &
            annotator = VnCoreNLP(address="http://127.0.0.1", port=9000, max_heap_size='-Xmx500m')

            def tokenize(s: str):
                words = annotator.tokenize(s)[0]
                return " ".join(words)

            return tokenize
        except ImportError:
            print("Please install VnCoreNLP package. "
                  "See the docs at https://github.com/vncorenlp/VnCoreNLP for more information.")
            raise
        except AttributeError:
            print("Please install VnCoreNLP package. "
                  "See the docs at https://github.com/vncorenlp/VnCoreNLP for more information.")
            raise

    elif tokenizer == 'mbert':
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
            return tokenizer
        except ImportError:
            print("Please install transformers package. "
                  "See the docs at https://github.com/huggingface/transformers for more information.")
            raise

    elif tokenizer == 'vit5-base':
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('VietAI/vit5-base')
            return tokenizer
        except ImportError:
            print("Please install transformers package. "
                  "See the docs at https://github.com/huggingface/transformers for more information.")
            raise

def preprocess_sentence(sentence: str, tokenizer: str = None):
    sentence = sentence.lower()
    sentence = unicodedata.normalize("NFC", sentence)
    sentence = re.sub(r"[“”]", " ", sentence)
    sentence = re.sub(r"!", " ", sentence)
    sentence = re.sub(r"\?", " ", sentence)
    sentence = re.sub(r":", " ", sentence)
    sentence = re.sub(r";", " ", sentence)
    sentence = re.sub(r",", " ", sentence)
    sentence = re.sub(r"\"", " ", sentence)
    sentence = re.sub(r"'", " ", sentence)
    sentence = re.sub(r"\(", " ", sentence)
    sentence = re.sub(r"\[", " ", sentence)
    sentence = re.sub(r"\)", " ", sentence)
    sentence = re.sub(r"\]", " ", sentence)
    sentence = re.sub(r"/", " ", sentence)
    sentence = re.sub(r"\.", " ", sentence)
    sentence = re.sub(r"-", " ", sentence)
    sentence = re.sub(r"\$", " ", sentence)
    sentence = re.sub(r"\&", " ", sentence)
    sentence = re.sub(r"\*", " ", sentence)
    # tokenize the sentence
    if tokenizer is not None:
        tokenizer = get_tokenizer(tokenizer)
        sentence = tokenizer(sentence)
    sentence = " ".join(sentence.strip().split()) # remove duplicated spaces
    tokens = sentence.strip().split()
    
    return tokens


def reporthook(t):
    """
    https://github.com/tqdm/tqdm.
    """
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optional
        Number of blocks just transferred [default: 1].
        bsize: int, optional
        Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner


def unk_init(token, dim):
    '''
        For default:
            + <pad> is 0
            + <sos> is 1
            + <eos> is 2
            + <unk> is 3
    '''

    if token in ["<pad>", "<p>"]:
        return torch.zeros(dim)
    if token in ["<sos>", "<bos>", "<s>"]:
        return torch.ones(dim)
    if token in ["<eos>", "</s>"]:
        return torch.ones(dim) * 2
    return torch.ones(dim) * 3


def default_value():
    return None


def collate_fn(samples: List[Instance]):
    return InstanceList(samples)


def is_japanese_sentence(text: str):
    # REFERENCE UNICODE TABLES: 
    # http:#www.rikai.com/library/kanjitables/kanji_codes.unicode.shtml
    # http:#www.tamasoft.co.jp/en/general-info/unicode.html
    #
    # TEST EDITOR:
    # http:#www.gethifi.com/tools/regex
    #
    # UNICODE RANGE : DESCRIPTION
    # 
    # 3000-303F : punctuation
    # 3040-309F : hiragana
    # 30A0-30FF : katakana
    # FF00-FFEF : Full-width roman + half-width katakana
    # 4E00-9FAF : Common and uncommon kanji
    # 
    # Non-Japanese punctuation/formatting characters commonly used in Japanese text
    # 2605-2606 : Stars
    # 2190-2195 : Arrows
    # u203B     : Weird asterisk thing
    pattern = r"[\u3000-\u303F]|[\u3040-\u309F]|[\u30A0-\u30FF]|[\uFF00-\uFFEF]|[\u4E00-\u9FAF]|[\u2605-\u2606]|[\u2190-\u2195]|\u203B"
    return re.search(pattern, text) is not None

def get_tone(word: str):
    tone_map = {
        '\u0300': '<`>',
        '\u0301': '</>',
        '\u0303': '<~>',
        '\u0309': '<?>',
        '\u0323': '<.>',
    }
    decomposed_word = unicodedata.normalize('NFD', word)
    tone = None
    remaining_word = ''
    for char in decomposed_word:
        if char in tone_map:
            tone = tone_map[char]
        else:
            remaining_word += char
    remaining_word = unicodedata.normalize('NFC', remaining_word)
    
    return tone, remaining_word

def get_onset(word: str) -> tuple[str, str]:
    onsets = ['ngh', 'tr', 'th', 'ph', 'nh', 'ng', 'kh', 
              'gi', 'gh', 'ch', 'q', 'đ', 'x', 'v', 't', 
              's', 'r', 'n', 'm', 'l', 'k', 'h', 'g', 'd', 
              'c', 'b']
    
    # get the onset
    for onset in onsets:
        if word.startswith(onset):
            if onset != "q":
                word = word.removeprefix(onset)
            return onset, word

    return None, word

def get_medial(word: str) -> tuple[str, str]:
    O_MEDIAL = "o"
    U_MEDIAL = "u"

    if word.startswith("q"):
        # in Vietnamese, words starting with "q" always has "u" as the medial
        word = word.removeprefix("qu")
        return U_MEDIAL, word
    
    o_medial_cases = ["oa", "oă", "oe"]
    for o_medial_case in o_medial_cases:
        if word.startswith(o_medial_case):
            word = word.removeprefix("o")
            return O_MEDIAL, word
        
    if word.startswith("ua") or word.startswith("uô"):
        return None, word
    
    nucleuses = ['ê', 'y', 'ơ', 'a', 'â', 'ya']
    for nucleus in nucleuses:
        component = U_MEDIAL + nucleus
        if word.startswith(component):
            word = word.removeprefix("u")
            return U_MEDIAL, word
        
    return None, word

def get_nucleus(word: str) -> tuple[str, str]:
    nucleuses = ['oo', 'ươ', 'ưa', 'uô', 'ua', 'iê', 'yê', 
                 'ia', 'ya', 'e', 'ê', 'u', 'ư', 'ô', 'i', 
                 'y', 'o', 'ơ', 'â', 'a', 'o', 'ă']
    
    for nucleus in nucleuses:
        if word.startswith(nucleus):
            word = word.removeprefix(nucleus)
            return nucleus, word
        
    return None, word
    
def get_coda(word: str) -> str:
    codas = ['ng', 'nh', 'ch', 'u', 'n', 'o', 'p', 'c', 'm', 'y', 'i', 't']
    
    if word in codas:
        return word
    
    return None

def split_phoneme(word: str) -> list[str, str, str]:
    onset, word = get_onset(word)
    
    medial, word = get_medial(word)

    nucleus, word = get_nucleus(word)

    coda = get_coda(word)
    
    return onset, medial, nucleus, coda

def is_Vietnamese(word: str) -> tuple[bool, tuple]:
    tone, word = get_tone(word)
    if not re.match(r"[a-zA-Zăâđưôơê]", word):
        return False, None

    # handling for special cases
    special_words_to_words = {
        "gin": "giin",     # gìn after being removed the tone 
        "giêng": "giiêng", # giếng after being removed the tone
        "giêt": "giiêt",   # giết after being removed the tone
        "giêc": "giiêc",   # giếc (diếc) after being removed the tone
        "gi": "gii"        # gì after removing the tone 
    }

    if word in special_words_to_words:
        word = special_words_to_words[word]

    # check the total number of nucleus in word
    vowels = ['oo', 'ươ', 'ưa', 'uô', 'ua', 'iê', 'yê', 
              'ia', 'ya', 'e', 'ê', 'u', 'ư', 'ô', 'i', 
              'y', 'o', 'ơ', 'â', 'a', 'o', 'ă']
    currentCharacterIsVowels = False
    previousCharacterIsVowels = word[0] in vowels
    foundVowels = 0
    
    for character in word[1:]:
        if character in vowels:
            currentCharacterIsVowels = True
        else:
            currentCharacterIsVowels = False
        
        if currentCharacterIsVowels and not previousCharacterIsVowels:
            foundVowels += 1

        # in Vietnamese, each word has only one syllable    
        if foundVowels > 2:
            return False, None
            
        previousCharacterIsVowels = currentCharacterIsVowels
    
    # in case the word has the structure of a Vietnamese word, we check whether it satisfies the rule of phoneme combination
    onset, medial, nucleus, coda = split_phoneme(word)

    if nucleus is None:
        return False, None
    
    former_word = ""
    for component in [onset, medial, nucleus, coda]:
        if component is not None:
            former_word += component
    if former_word != word:
        return False, None
    
    if onset == "k" and medial is None and nucleus not in ["i", "y", "e", "ê", "iê", "yê", "ia", "ya"]:
        return False, None
    
    if onset == "c" and medial is None and nucleus in ["i", "y", "e", "ê", "iê", "yê", "ia", "ya"]:
        return False, None
    
    if onset == "q" and not medial == "u":
        return False, None
    
    if onset == "gh" and medial is None and nucleus not in ["i", "e", "ê", "iê"]:
        return False, None
    
    if onset == "g" and medial is None and nucleus in ["i", "e", "ê", "iê"]:
        return False, None
    
    if onset == "ngh" and medial is None and nucleus not in ["i", "e", "ê", "iê", "yê", "ia", "ya"]:
        return False, None
    
    if onset == "ng" and medial is None and nucleus in ["i", "e", "ê", "iê", "yê", "ia", "ya"]:
        return False, None
    
    if onset in ["r", "gi"] and medial is not None:
        return False, None
    
    if medial == "o" and nucleus not in ["a", "ă", "e"]:
        return False, None
    
    if medial == "u" and nucleus not in ['yê', 'ya', 'e', 'ê', 'y', 'ơ', "ô", 'a', 'â', 'ă']:
        return False, None
    
    if nucleus == "oo" and coda not in ["ng", "c"]:
        return False, None
    
    if nucleus == "ua" and coda is not None:
        return False, None
    
    if nucleus == "ia" and coda is not None:
        return False, None
    
    if nucleus == "ya" and coda is not None:
        return False, None
    
    if nucleus in ["ua", "uô"] and coda == "ph":
        return False, None
    
    if nucleus in ["yê", "iê"] and coda is None:
        return False, None
    
    if nucleus in ["ă", "â"] and coda is None:
        return False, None
    
    if medial == "o" and nucleus in ["iê", "yê", "ia", "ya"]:
        return False, None
    
    if medial is not None:
        if nucleus in ["u", "oo", "o", "ua", "uô", "ươ", "ưa", "ư"]:
            return False, None
        
        if nucleus in ["i", "e", "ê", "ia", "ya", "iê", "yê"] and coda in ["m", "ph"]:
            return False, None
        
    if coda == "o" and nucleus not in ["a", "e"]:
        return False, None
    
    if coda == "y" and nucleus not in ["a", "â"]:
        return False, None
    
    if coda == "i" and nucleus in ["ă", "â", "i", "e", "iê", "yê", "ia", "ya"]:
        return False, None
    
    if coda == "nh" and nucleus not in ["a", "i", "y", "ê"]:
        return False, None
    
    if coda == "ng" and nucleus not in ["a", "o", "ô", "u", "ư", "e", "iê", "ươ", "â", "ă", "uô", "oo"]:
        return False, None

    if coda == "ch" and nucleus not in ["i", "a", "ê", "y"]:
        return False, None

    if coda == "c" and nucleus in ["i", "ê", "e", "ơ"]:
        return False, None

    if nucleus == coda:
        return False, None

    onset, medial, nucleus, coda = assign_tone(onset, medial, nucleus, coda, tone)

    return True, (onset, medial, nucleus, coda)

def assign_tone(onset: str, medial: str, nucleus: str, coda: str, tone: str):
    tone_map = {
        None: "",
        '<`>': '\u0300',
        '</>': '\u0301',
        '<~>': '\u0303',
        '<?>': '\u0309',
        '<.>': '\u0323',
    }
    tone = tone_map[tone]

    if medial == "u" and nucleus == "ơ" and tone == "\u0309":
        nucleus += tone
        nucleus = unicodedata.normalize("NFC", nucleus)
        return onset, medial, nucleus, coda
    
    if onset == "gi" and nucleus == "i":
        onset += tone
        onset = unicodedata.normalize("NFC", onset)
        nucleus = None
        return onset, medial, nucleus, coda
    
    if onset == "gi" and nucleus == "iê":
        onset = onset[0]
        nucleus += tone
        nucleus = unicodedata.normalize("NFC", nucleus)
        return onset, medial, nucleus, coda
    
    if onset == "q":
        nucleus += tone
        nucleus = unicodedata.normalize("NFC", nucleus)
        return onset, medial, nucleus, coda
    
    if medial == "u" and nucleus == "ê":
        nucleus += tone
        nucleus = unicodedata.normalize("NFC", nucleus)
        return onset, medial, nucleus, coda
    
    if medial and coda is None:
        medial += tone
        medial = unicodedata.normalize("NFC", medial)

        return onset, medial, nucleus, coda
    
    if len(nucleus) == 1:
        nucleus += tone
    else:
        if coda:
            nucleus += tone
        else:
            nucleus = nucleus[0] + tone + nucleus[1]
    
    nucleus = unicodedata.normalize("NFC", nucleus)

    return onset, medial, nucleus, coda

def composing_word(onset: str, medial: str, nucleus: str, coda: str):
    if not (onset or medial or nucleus or coda):
        return None, False
    
    word = ""
    if onset:
        word += onset
    if medial:
        word += medial
    if nucleus:
        word += nucleus
    if coda:
        word += coda
        
    word = unicodedata.normalize("NFC", word)

    return word, True

