import torch

from data_utils.datasets.feature_dataset import FeatureDataset
from data_utils.datasets.dictionary_dataset import DictionaryDataset
from utils.instance import Instance
from builders.dataset_builder import META_DATASET

import os
import numpy as np
from typing import Dict, Any, List
from data_utils.utils import get_tokenizer, preprocess_sentence


@META_DATASET.register()
class ViTCFeatureDataset(FeatureDataset):
    def __init__(self,
                 json_path: str,
                 vocab,
                 config) -> None:
        super().__init__(json_path, vocab, config)
        self.fasttext_path = config.FEATURE_PATH.FASTTEXT

        # scene text features
        self.scene_text_features_path = config.FEATURE_PATH.SCENE_TEXT
        self.scene_text_threshold = config.SCENE_TEXT_THRESHOLD
        self.max_scene_text = config.MAX_SCENE_TEXT
        self.tokenizer = get_tokenizer(config.TOKENIZER.PRETRAINED_NAME)

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    for answer in ann["answers"]:
                        question = preprocess_sentence(ann["question"], None)
                        answer = preprocess_sentence(answer, None)
                        annotation = {
                            "question": question,
                            "answer": answer,
                            "image_id": ann["image_id"],
                            "filename": image["filename"]
                        }
                        annotations.append(annotation)
                    break
        return annotations

    def load_image_features(self, image_id: int) -> Dict[str, Any]:
        feature_file = os.path.join(self.image_features_path, f"{image_id}.npy")
        features = np.load(feature_file, allow_pickle=True)[()]
        for key, feature in features.items():
            if isinstance(feature, np.ndarray):
                features[key] = torch.tensor(feature, dtype=torch.float32)

        return features
  
    def load_fasttext_features(self, image_id: int) -> Dict[str, Any]:
        feature_file = os.path.join(self.fasttext_path, f"{image_id}.npy")
        features = np.load(feature_file, allow_pickle=True)[()]

        if features.shape[0] < 1:
            features = np.zeros([1, 300])
                
        return torch.tensor(features, dtype=torch.float32)

    def load_scene_text_features(self, image_id: int) -> Dict[str, Any]:
        feature_file = os.path.join(self.scene_text_features_path, f"{image_id}.npy")
        features = np.load(feature_file, allow_pickle=True)[()]
        for key, feature in features.items():
            if key == 'det_features':
                if features[key].shape[0] < 1:
                    features[key] = np.zeros([1, 256])
            elif key == 'rec_features':
                if features[key].shape[0] < 1:
                    features[key] = np.zeros([1, 256])
            elif key == 'boxes':
                if features[key].shape[0] < 1:
                    features[key] = np.zeros([1, 4])
            elif key == 'texts':
                if len(features[key]) == 0:
                    features[key] = ['no_token']
            elif key == 'scores':
                if len(features[key]) == 0:
                    features[key] = [0]

            if isinstance(feature, np.ndarray):
                features[key] = torch.tensor(feature, dtype=torch.float32)

        ocr_fasttext_features = self.load_fasttext_features(image_id)
        ocr_nums = ocr_fasttext_features.shape[0]
        keys = ["det_features", "rec_features", "texts", "boxes"]
        for key in keys:
            features[key] = features[key][:ocr_nums]
        return {
            "ocr_det_features": features["det_features"],
            "ocr_rec_features": features["rec_features"],
            "ocr_texts": features["texts"],
            "ocr_boxes": features["boxes"],
            "ocr_fasttext_features": ocr_fasttext_features,  # Thêm đặc trưng FastText
        }

    def load_features(self, image_id: int) -> Dict[str, Any]:
        image_features = self.load_image_features(image_id)
        scene_text_features = self.load_scene_text_features(image_id)
        features = {
            **image_features,
            **scene_text_features
        }

        return features

    def __getitem__(self, idx: int):
        features = self.load_features(self.annotations[idx]["image_id"])

        item = self.annotations[idx]
        question = item["question"]
        question_tokens = self.vocab.encode_question(question)
        answer = item["answer"]

        ocr_tokens = [text if text.strip() != "" else self.vocab.padding_token 
                      for text in features["ocr_texts"]]

        answer_tokens_ = self.tokenizer(" ".join(answer),
                                        padding="max_length",
                                        add_special_tokens=False,
                                        max_length=600,
                                        return_tensors="pt")

        answer_tokens = answer_tokens_.input_ids  # remove eos_token in answer
        answer_mask = answer_tokens_.attention_mask

        return Instance(
            **features,
            image_id=item["image_id"],
            filename=item["filename"],
            ocr_tokens=ocr_tokens,
            question=" ".join(question),
            question_tokens=question_tokens,
            answers=answer,
            answer_tokens=answer_tokens,
            answer_masks=answer_mask,
        )


@META_DATASET.register()
class ViTCDictionaryDataset(DictionaryDataset):
    def __init__(self,
                 json_path: str,
                 vocab,
                 config) -> None:
        super().__init__(json_path, vocab, config)
        self.fasttext_path = config.FEATURE_PATH.FASTTEXT

        # scene text features
        self.scene_text_features_path = config.FEATURE_PATH.SCENE_TEXT
        self.scene_text_threshold = config.SCENE_TEXT_THRESHOLD
        self.max_scene_text = config.MAX_SCENE_TEXT
        self.tokenizer = get_tokenizer(config.TOKENIZER.PRETRAINED_NAME)
    
    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    question = preprocess_sentence(ann["question"], None)
                    answers = [preprocess_sentence(answer, None) for answer in ann["answers"]]
                    answers = [" ".join(answer) for answer in answers]
                    annotation = {
                        "question_id": ann["id"],
                        "type": ann["QA-type"],
                        "question": question,
                        "answers": answers,
                        "image_id": ann["image_id"],
                        "filename": image["filename"]
                    }
                    break

            annotations.append(annotation)
        return annotations
        
    def load_image_features(self, image_id: int) -> Dict[str, Any]:
        feature_file = os.path.join(self.image_features_path, f"{image_id}.npy")
        features = np.load(feature_file, allow_pickle=True)[()]
        for key, feature in features.items():
            if isinstance(feature, np.ndarray):
                features[key] = torch.tensor(feature, dtype=torch.float32)

        return features

    def load_fasttext_features(self, image_id: int) -> Dict[str, Any]:
        feature_file = os.path.join(self.fasttext_path, f"{image_id}.npy")
        features = np.load(feature_file, allow_pickle=True)[()]
        
        if features.shape[0] < 1:
            features = np.zeros([1, 300])

        return torch.tensor(features, dtype=torch.float32)

    def load_scene_text_features(self, image_id: int) -> Dict[str, Any]:
        feature_file = os.path.join(self.scene_text_features_path, f"{image_id}.npy")
        features = np.load(feature_file, allow_pickle=True)[()]
        for key, feature in features.items():
            if key == 'det_features':
                if features[key].shape[0] < 1:
                    features[key] = np.zeros([1, 256])
            elif key == 'rec_features':
                if features[key].shape[0] < 1:
                    features[key] = np.zeros([1, 256])
            elif key == 'boxes':
                if features[key].shape[0] < 1:
                    features[key] = np.zeros([1, 4])
            elif key == 'texts':
                if len(features[key]) == 0:
                    features[key] = ['no_token']
            elif key == 'scores':
                if len(features[key]) == 0:
                    features[key] = [0]

            if isinstance(feature, np.ndarray):
                features[key] = torch.tensor(feature, dtype=torch.float32)
        
        ocr_fasttext_features = self.load_fasttext_features(image_id)
        ocr_nums = ocr_fasttext_features.shape[0]
        keys = ["det_features", "rec_features", "texts", "boxes"]
        for key in keys:
            features[key] = features[key][:ocr_nums]
        return {
            "ocr_det_features": features["det_features"],
            "ocr_rec_features": features["rec_features"],
            "ocr_texts": features["texts"],
            "ocr_boxes": features["boxes"],
            "ocr_fasttext_features": ocr_fasttext_features,  # Thêm đặc trưng FastText
        }
        

    def load_features(self, image_id: int) -> Dict[str, Any]:
        image_features = self.load_image_features(image_id)
        scene_text_features = self.load_scene_text_features(image_id)
        features = {
            **image_features,
            **scene_text_features
        }

        return features

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        image_id = item["image_id"]
        filename = item["filename"]
        features = self.load_features(image_id)
        answers = item["answers"]

        ocr_tokens = [text if text.strip() != "" else self.vocab.padding_token for text in features["ocr_texts"]]
        answer_tokens_ = self.tokenizer(" ".join(answers),
                                        padding="max_length",
                                        add_special_tokens=False,
                                        max_length=600,
                                        return_tensors="pt")

        answer_tokens = answer_tokens_.input_ids

        answer_mask = answer_tokens_.attention_mask
        return Instance(
            **features,
            question_id=item["question_id"],
            type=item["type"],
            image_id=image_id,
            filename=filename,
            ocr_tokens=ocr_tokens,
            answers=answers,
            answer_tokens=answer_tokens,
            answer_masks=answer_mask
        )
