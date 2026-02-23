import torch

from data_utils.datasets.feature_dataset import FeatureDataset
from data_utils.datasets.dictionary_dataset import DictionaryDataset
from utils.instance import Instance
from builders.dataset_builder import META_DATASET
from utils.logging_utils import setup_logger
import os
import numpy as np
from typing import Dict, Any

logger = setup_logger()

@META_DATASET.register()
class OcrFeatureDataset(FeatureDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__(json_path, vocab, config)
        self.fasttext_path = config.FEATURE_PATH.FASTTEXT

        # scene text features
        self.edge_features_path = False
        try:
            self.edge_features_path = config.FEATURE_PATH.EDGE
        except ValueError as e:
            logger.info(e)
            raise
            
        self.scene_text_features_path = config.FEATURE_PATH.SCENE_TEXT
        self.scene_text_threshold = config.SCENE_TEXT_THRESHOLD
        self.max_scene_text = config.MAX_SCENE_TEXT

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
                    features[key] = np.zeros([2, 256])
            elif key == 'rec_features':
                if features[key].shape[0] < 1:
                    features[key] = np.zeros([2, 256])
            elif key == 'boxes':
                if features[key].shape[0] < 1:
                    features[key] = np.zeros([2, 4])
            elif key == 'texts':
                if len(features[key]) == 0:
                    features[key] = ['no_token', 'no_token']
            elif key == 'scores':
                if len(features[key]) == 0:
                    features[key] = [0, 0]

            if isinstance(feature, np.ndarray):
                features[key] = torch.tensor(feature, dtype=torch.float32)
        
        ocr_fasttext_features = self.load_fasttext_features(image_id)
        ocr_nums = ocr_fasttext_features.shape[0]
        keys = ["det_features", "rec_features", "texts", "boxes"]
        for key in keys:
            features[key] = features[key][:ocr_nums]

        # OCR confidence scores from SwinTextSpotter
        ocr_scores = features.get("scores", [0.0] * ocr_nums)
        if isinstance(ocr_scores, torch.Tensor):
            ocr_scores = ocr_scores[:ocr_nums]
        else:
            ocr_scores = ocr_scores[:ocr_nums]
            ocr_scores = torch.tensor(ocr_scores, dtype=torch.float32)

        ocr_scores = ocr_scores.unsqueeze(-1)  # (num_ocr, 1) for collation

        return {
            "ocr_det_features": features["det_features"],
            "ocr_rec_features": features["rec_features"],
            "ocr_texts": features["texts"],
            "ocr_boxes": features["boxes"],
            "ocr_fasttext_features": ocr_fasttext_features,
            "ocr_scores": ocr_scores,
        }
        
    def load_edge_features(self, image_id: int) -> Dict[str, Any]:
        feature_file = os.path.join(self.edge_features_path, f"{image_id}_info.npy")
        features = np.load(feature_file, allow_pickle=True)[()]
        for key, feature in features.items():
            if isinstance(feature, np.ndarray):
                features[key] = torch.tensor(feature, dtype=torch.float32)
        return features
                
    def load_features(self, image_id: int) -> Dict[str, Any]:
        image_features = self.load_image_features(image_id)
        scene_text_features = self.load_scene_text_features(image_id)
        edge_features = None
        if self.edge_features_path:
            edge_features = self.load_edge_features(image_id)
            features = {
                **edge_features,
                **image_features,
                **scene_text_features
            }
        else:
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

        ocr_tokens = [text if text.strip() != "" else self.vocab.padding_token for text in features["ocr_texts"]]

        answer_tokens = self.vocab.encode_answer(answer, ocr_tokens)

        shifted_right_answer_tokens = torch.zeros_like(answer_tokens).fill_(self.vocab.padding_idx)
        shifted_right_answer_tokens[:-1] = answer_tokens[1:]
        answer_tokens = torch.where(answer_tokens == self.vocab.eos_idx, self.vocab.padding_idx, answer_tokens) # remove eos_token in answer

        answer_mask = torch.where(answer_tokens > 0, 1, 0)
        return Instance(
            **features,
            image_id=item["image_id"],
            filename=item["filename"],
            ocr_tokens=ocr_tokens,
            question=" ".join(question),
            question_tokens=question_tokens,
            answers=answer,
            answer_tokens=answer_tokens,
            answer_mask=answer_mask,
            shifted_right_answer_tokens=shifted_right_answer_tokens
        )

@META_DATASET.register()
class OcrDictionaryDataset(DictionaryDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__(json_path, vocab, config)
        self.fasttext_path = config.FEATURE_PATH.FASTTEXT

        # scene text features
        self.edge_features_path = False
        try:
            self.edge_features_path = config.FEATURE_PATH.EDGE
        except ValueError as e:
            logger.info(e)
            raise
            
        self.scene_text_features_path = config.FEATURE_PATH.SCENE_TEXT
        self.scene_text_threshold = config.SCENE_TEXT_THRESHOLD
        self.max_scene_text = config.MAX_SCENE_TEXT
    
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
                    features[key] = np.zeros([2, 256])
            elif key == 'rec_features':
                if features[key].shape[0] < 1:
                    features[key] = np.zeros([2, 256])
            elif key == 'boxes':
                if features[key].shape[0] < 1:
                    features[key] = np.zeros([2, 4])
            elif key == 'texts':
                if len(features[key]) == 0:
                    features[key] = ['no_token', 'no_token']
            elif key == 'scores':
                if len(features[key]) == 0:
                    features[key] = [0, 0]

            if isinstance(feature, np.ndarray):
                features[key] = torch.tensor(feature, dtype=torch.float32)
        
        ocr_fasttext_features = self.load_fasttext_features(image_id)
        ocr_nums = ocr_fasttext_features.shape[0]
        keys = ["det_features", "rec_features", "texts", "boxes"]
        for key in keys:
            features[key] = features[key][:ocr_nums]

        # OCR confidence scores from SwinTextSpotter
        ocr_scores = features.get("scores", [0.0] * ocr_nums)
        if isinstance(ocr_scores, torch.Tensor):
            ocr_scores = ocr_scores[:ocr_nums]
        else:
            ocr_scores = ocr_scores[:ocr_nums]
            ocr_scores = torch.tensor(ocr_scores, dtype=torch.float32)

        ocr_scores = ocr_scores.unsqueeze(-1)  # (num_ocr, 1) for collation

        return {
            "ocr_det_features": features["det_features"],
            "ocr_rec_features": features["rec_features"],
            "ocr_texts": features["texts"],
            "ocr_boxes": features["boxes"],
            "ocr_fasttext_features": ocr_fasttext_features,
            "ocr_scores": ocr_scores,
        }
        
    def load_edge_features(self, image_id: int) -> Dict[str, Any]:
        feature_file = os.path.join(self.edge_features_path, f"{image_id}_info.npy")
        features = np.load(feature_file, allow_pickle=True)[()]
        for key, feature in features.items():
            if isinstance(feature, np.ndarray):
                features[key] = torch.tensor(feature, dtype=torch.float32)
        return features

    def load_features(self, image_id: int) -> Dict[str, Any]:
        image_features = self.load_image_features(image_id)
        scene_text_features = self.load_scene_text_features(image_id)
        edge_features = None
        if self.edge_features_path:
            edge_features = self.load_edge_features(image_id)
            features = {
                **edge_features,
                **image_features,
                **scene_text_features
            }
        else:
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
        question = item["question"]
        question_tokens = self.vocab.encode_question(question)
        answers = item["answers"]

        ocr_tokens = [text if text.strip() != "" else self.vocab.padding_token for text in features["ocr_texts"]]
        answer_tokens = self.vocab.encode_answer(answers, ocr_tokens)

        answer_tokens = torch.where(answer_tokens == self.vocab.eos_idx, self.vocab.padding_idx, answer_tokens) # remove eos_token in answer
        answer_mask = torch.where(answer_tokens > 0, 1, 0)
        return Instance(
            **features,
            question_id=item["question_id"],
            type=item["type"],
            image_id=image_id,
            filename=filename,
            ocr_tokens=ocr_tokens,
            question=" ".join(question),
            question_tokens=question_tokens,
            answers=answers,
            answer_tokens=answer_tokens,
            answer_mask=answer_mask
        )
