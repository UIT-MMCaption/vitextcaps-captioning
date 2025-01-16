# from models.stacmr import VSRN
# from data_utils.datasets.vitextcap_dataset import ViTCFeatureDataset

# from configs.utils import get_config
# from builders.vocab_builder import build_vocab
# import torch
# from torch.utils.data import DataLoader
# from data_utils.utils import collate_fn

# config = get_config('configs/stacmr.yaml')
# vocab = build_vocab(config.DATASET.VOCAB)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model = VSRN(config.MODEL, vocab)
# model.to(device)
# # print(model)

# train_data = ViTCFeatureDataset(json_path='data/vitextcaps_dev.json',
#                                vocab=vocab,
#                                config=config.DATASET.FEATURE_DATASET)
# train_loader = DataLoader(train_data,
#                           batch_size=2,
#                           shuffle=True,
#                           collate_fn=collate_fn)
# sample = next(iter(train_loader))
# model.eval()
# with torch.no_grad():
#     output = model(sample, mode='inference')
# print(output['scores'].shape)
d_model = 768
warmup = 2000
initial_lr = (d_model ** -0.5) * (1 * warmup ** -1.5)
print(initial_lr)
# print(output['scores'].argmax(dim=-1))
# print(output['predicted_token'])
