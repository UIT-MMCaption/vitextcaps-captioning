from data_utils.utils import preprocess_sentence
from data_utils.datasets.base_dataset import BaseDataset
from utils.instance import Instance
from builders.dataset_builder import META_DATASET

from typing import Dict, List

@META_DATASET.register()
class DictionaryDataset(BaseDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super(DictionaryDataset, self).__init__(json_path, vocab, config)

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for k, v in json_data.items():
            # find the appropriate image
            question = preprocess_sentence(" ", self.vocab.tokenizer)
            answers = preprocess_sentence(v['caption'], self.vocab.tokenizer)
            # answers = [" ".join(answer) for answer in answers]
            annotation = {
                "question_id": " ",
                "type": " ",
                "question": question,
                "answers": answers,
                "image_id": v["image_id"],
                "filename": k
            }

            annotations.append(annotation)

        return annotations

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        image_id = item["image_id"]
        filename = item["filename"]
        features = self.load_features(image_id)
        question = item["question"]
        question_tokens = self.vocab.encode_question(question)
        answers = item["answers"]

        return Instance(
            question_id=item["question_id"],
            type=item["type"],
            image_id=image_id,
            filename=filename,
            question=question,
            question_tokens=question_tokens,
            answers=answers,
            **features
        )
