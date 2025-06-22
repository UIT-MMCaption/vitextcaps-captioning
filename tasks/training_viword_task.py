import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import NLLLoss
from torch.utils.data import DataLoader
from utils.logging_utils import setup_logger
from tasks.open_ended_task import OpenEndedTask
from builders.task_builder import META_TASK
import evaluation
from data_utils.utils import collate_fn
import os
from tqdm import tqdm
import itertools
from shutil import copyfile
import json
from builders.task_builder import META_TASK
from torch.optim.lr_scheduler import LambdaLR
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
logger = setup_logger()

class CustomLoss(nn.Module):
    def __init__(self, ignore_index):
        super().__init__()

        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        losses = []
        print(logits.shape)
        B, S, _, V = logits.shape

        logit_flat = logits.view(B*S, _, V)
        target_flat = targets.view(B*S, _)

        for i in range(4):
            logits_i = logit_flat[:, i, :]
            target_i = target_flat[:, i]

            ignore_index_i = torch.tensor(self.ignore_index[i], device=targets.device)
            valid_mask = ~torch.isin(target_i, ignore_index_i)  # handle multiple index
            logits_valid = logits_i[valid_mask]
            target_valid = target_i[valid_mask]

            log_probs = logits_valid - logits_valid.logsumexp(dim=1, keepdim=True)
            loss_i = -log_probs[torch.arange(logits_valid.size(0)), target_valid]

            mean_loss_i = loss_i.mean() if loss_i.numel() > 0 else torch.tensor(0.0, device=logits.device)
            losses.append(mean_loss_i)

        total_loss = sum(losses)

        return total_loss

@META_TASK.register()
class TrainingViWord(OpenEndedTask):
    def __init__(self, config):
        super().__init__(config)
        self.scheduler = LambdaLR(self.optim, self.lambda_lr)
        self.loss_fn = CustomLoss(ignore_index=self.vocab.ignore_index).to('cuda')


    def create_dict_dataloaders(self, config):
        # creating dictionary iterable-dataset data loader
        self.train_dict_dataloader = DataLoader(
            dataset=self.train_dict_dataset,
            batch_size=config.DATASET.DICT_DATASET.BATCH_SIZE // config.TRAINING.TRAINING_BEAM_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )

        self.dev_dict_dataloader = DataLoader(
            dataset=self.dev_dict_dataset,
            batch_size=config.DATASET.DICT_DATASET.BATCH_SIZE // config.TRAINING.EVALUATING_BEAM_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )
        self.test_dict_dataloader = DataLoader(
            dataset=self.test_dict_dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=collate_fn
        )

    def evaluate_loss(self, dataloader):
        self.model.eval()
        self.model.train()
        running_loss = .0
        with tqdm(desc='Epoch %d - Validation' % self.epoch, unit='it', total=len(dataloader)) as pbar:
            for it, items in enumerate(dataloader):
                items = items.to(self.device)
                with torch.no_grad():
                    results = self.model(items)
                out = results["scores"]

                shifted_right_answer_tokens = items.shifted_right_answer_tokens

                shifted_right_answer_tokens = F.pad(shifted_right_answer_tokens, (0, 0, 0, self.model.max_iter - shifted_right_answer_tokens.shape[1]))
                loss = self.loss_fn(out, shifted_right_answer_tokens.type(torch.long))

                running_loss += loss.item()

                pbar.set_postfix(loss=running_loss / (it + 1), refresh=True)
                pbar.update()
        val_loss = running_loss / len(dataloader)
        self.model.eval()
        return val_loss

    def evaluate_metrics(self, dataloader):
        self.model.train()
        gens = {}
        gts = {}
        with tqdm(desc='Epoch %d - Evaluation' % self.epoch, unit='it', total=len(dataloader)) as pbar:
            for it, items in enumerate(dataloader):
                items = items.to(self.device)
                with torch.no_grad():
                    results = self.model(items)
                outs = results["scores"].argmax(dim=-1)

                answers_gt = items.answers
                answers_gen = self.vocab.decode_batch_caption(outs.contiguous(),
                                                                join_words=False)
                if not any(isinstance(i, list) for i in answers_gen):
                    answers_gen = [answers_gen]
                for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
                    gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                    gens['%d_%d' % (it, i)] = [gen_i, ]
                    gts['%d_%d' % (it, i)] = gts_i
                pbar.update()

        scores, _ = evaluation.compute_scores(gts, gens)

        return scores

    def train(self):
        self.model.train()
        running_loss = .0
        with tqdm(desc='Epoch %d - Training with cross-entropy loss' % self.epoch, unit='it', total=len(self.train_dataloader)) as pbar:
            for it, items in enumerate(self.train_dataloader):
                items = items.to(self.device)
                results = self.model(items)
                out = results["scores"]

                shifted_right_answer_tokens = items.shifted_right_answer_tokens
                self.optim.zero_grad()

                shifted_right_answer_tokens = F.pad(shifted_right_answer_tokens, (0, 0, 0, self.model.max_iter - shifted_right_answer_tokens.shape[1]))
                loss = self.loss_fn(out, shifted_right_answer_tokens.type(torch.long))

                loss.backward()
                self.optim.step()
                running_loss += loss.item()

                pbar.set_postfix(loss=running_loss / (it + 1), refresh=True)
                pbar.update()
                self.scheduler.step()

    # def start(self, epochs=10):
    #     if os.path.isfile(os.path.join(self.checkpoint_path, "last_model.pth")):
    #         checkpoint = self.load_checkpoint(os.path.join(self.checkpoint_path, "last_model.pth"))
    #         best_val_score = checkpoint["best_val_score"]
    #         patience = checkpoint["patience"]
    #         self.epoch = checkpoint["epoch"] + 1
    #         self.optim.load_state_dict(checkpoint['optimizer'])
    #         self.scheduler.load_state_dict(checkpoint['scheduler'])

    #     while self.epoch < epochs:
    #         self.train()

    #         # scores = self.evaluate_metrics(self.dev_dict_dataloader)
    #         # logger.info("Validation scores %s", scores)
    #         # val_score = scores[self.score]

    #         self.save_checkpoint({
    #                 'best_val_score': 0,
    #                 'patience': 0
    #             })

    def start(self):
        if os.path.isfile(os.path.join(self.checkpoint_path, "last_model.pth")):
            checkpoint = self.load_checkpoint(os.path.join(self.checkpoint_path, "last_model.pth"))
            best_val_score = checkpoint["best_val_score"]
            patience = checkpoint["patience"]
            self.epoch = checkpoint["epoch"] + 1
            self.optim.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            best_val_score = .0
            patience = 0

        while True:
            self.train()
            # self.evaluate_loss(self.dev_dataloader)

            # val scores
            scores = self.evaluate_metrics(self.dev_dict_dataloader)
            logger.info("Validation scores %s", scores)
            val_score = scores[self.score]

            # Prepare for next epoch
            best = False
            if val_score > best_val_score:
                best_val_score = val_score
                patience = 0
                best = True
            else:
                patience += 1

            exit_train = False

            if patience == self.patience:
                logger.info('patience reached.')
                exit_train = True

            self.save_checkpoint({
                'best_val_score': best_val_score,
                'patience': patience
            })

            if best:
                copyfile(os.path.join(self.checkpoint_path, "last_model.pth"),
                         os.path.join(self.checkpoint_path, "best_model.pth"))

            if exit_train:
                break

            self.epoch += 1


    def get_predictions(self):
        if not os.path.isfile(os.path.join(self.checkpoint_path, 'last_model.pth')):
            logger.error("Prediction require the model must be trained. There is no weights to load for model prediction!")
            raise FileNotFoundError("Make sure your checkpoint path is correct or the best_model.pth is available in your checkpoint path")

        self.load_checkpoint(os.path.join(self.checkpoint_path, "last_model.pth"))

        self.model.train()
        results = []
        overall_gens = {}
        overall_gts = {}
        with tqdm(desc='Getting predictions: ', unit='it', total=len(self.test_dict_dataloader)) as pbar:
            for it, items in enumerate(self.test_dict_dataloader):
                items = items.to(self.device)
                with torch.no_grad():
                    result = self.model(items)
                outs = result["scores"].argmax(dim=-1)

                answers_gt = items.answers
                answers_gen = self.vocab.decode_batch_caption(outs.contiguous(),
                                                              join_words=False)
                gts = {}
                gens = {}
                for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
                    gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                    gens['%d_%d' % (it, i)] = (gen_i)
                    gts['%d_%d' % (it, i)] = gts_i
                    overall_gens['%d_%d' % (it, i)] = [gen_i, ]
                    overall_gts['%d_%d' % (it, i)] = gts_i
                pbar.update()

                results.append({
                    "id": items.question_id,
                    "image_id": items.image_id,
                    "filename": items.filename,
                    "gens": gens,
                    "gts": gts
                })

                pbar.update()

        scores, _ = evaluation.compute_scores(overall_gts, overall_gens)
        logger.info("Evaluation scores on test: %s", scores)

        json.dump({
            "results": results,
            **scores,
        }, open(os.path.join(self.checkpoint_path, "test_results.json"), "w+"), ensure_ascii=False)