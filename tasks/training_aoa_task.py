import torch
from torch import nn
from torch.nn import functional as F

from utils.logging_utils import setup_logger
from tasks.open_ended_task import OpenEndedTask
from builders.task_builder import META_TASK
import evaluation
from pycocoevalcap.cider.cider import Cider
from data_utils.utils import get_tokenizer
import os
from tqdm import tqdm
import itertools
from shutil import copyfile
import json

from torch.autograd import Variable

logger = setup_logger()


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, logits, target, mask):
        """
        logits: shape of (N, seq_len, vocab_size)
        target: shape of (N, seq_len)
        mask: shape of (N, seq_len)
        """
        # truncate to the same size
        target = target[:, :logits.shape[1]]
        mask = mask[:, :logits.shape[1]]
        
        logits = logits.contiguous().view(-1, logits.shape[2])
        target = target.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        
        loss = self.loss_fn(logits, target)
        masked_loss = loss * mask
        output = masked_loss.sum() / mask.sum()  # Average over actual tokens
        return output


@META_TASK.register()
class TrainingAoA(OpenEndedTask):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.tokenizer = get_tokenizer(config.DATASET.FEATURE_DATASET.TOKENIZER.PRETRAINED_NAME)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def evaluate_loss(self, dataloader):
        self.model.eval()
        running_loss = .0
        with tqdm(desc='Epoch %d - Validation' % self.epoch, unit='it', total=len(dataloader)) as pbar:
            with torch.no_grad():
                for it, items in enumerate(dataloader):
                    items = items.to(self.device)
                    with torch.no_grad():
                        outs = self.model(items)
                    
                    shifted_right_answer_tokens = items.answer_tokens.squeeze()
                    # loss = self.loss_fn(out.view(-1, out.shape[-1]), shifted_right_answer_tokens.view(-1))
                    
                    answer_masks = items.answer_masks.squeeze()
                    
                    # caption_loss = self.loss_fn(outs, 
                    #                             shifted_right_answer_tokens, 
                    #                             answer_masks)
                    
                    caption_loss = self.loss_fn(outs, 
                                                shifted_right_answer_tokens)
                    
                    this_loss = caption_loss.item()
                    running_loss += this_loss

                    pbar.set_postfix(loss=running_loss / (it + 1))
                    pbar.update()

        val_loss = running_loss / len(dataloader)

        return val_loss

    def evaluate_metrics(self, dataloader):
        self.model.eval()
        gens = {}
        gts = {}
        with tqdm(desc='Epoch %d - Evaluation' % self.epoch, unit='it', total=len(dataloader)) as pbar:
            for it, items in enumerate(dataloader):
                items = items.to(self.device)
                with torch.no_grad():
                    outs = self.model(items)
                answers_gen_ids = outs.argmax(dim=-1)
                answers_gt = items.answers
                answers_gen = self.tokenizer.batch_decode(answers_gen_ids,
                                                          skip_special_tokens=True)
                for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
                    words = gen_i.split()
                    gen_i = ' '.join([k for k, g in itertools.groupby(words)])
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
                # self.adjust_learning_rate(self.optim, self.epoch)
                items = items.to(self.device)
                out = self.model(items)

                shifted_right_answer_tokens = items.answer_tokens.squeeze()
                    
                answer_masks = items.answer_masks.squeeze()
                self.optim.zero_grad()
                # loss = self.loss_fn(out.view(-1, out.shape[-1]), shifted_right_answer_tokens.view(-1))
                
                # caption_loss = self.loss_fn(out, 
                #                             shifted_right_answer_tokens, 
                #                             answer_masks)
                
                caption_loss = self.loss_fn(out, 
                                            shifted_right_answer_tokens)
                
                loss = caption_loss
                
                loss.backward()
                    
                self.optim.step()
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
                self.scheduler.step()

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
            self.evaluate_loss(self.dev_dataloader)

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

        self.model.eval()
        results = []
        overall_gens = {}
        overall_gts = {}
        with tqdm(desc='Getting predictions: ', unit='it', total=len(self.test_dict_dataloader)) as pbar:
            for it, items in enumerate(self.test_dict_dataloader):
                items = items.to(self.device)
                with torch.no_grad():
                    outs = self.model(items)
                    
                answers_gen_ids = outs.argmax(dim=-1)
                answers_gt = items.answers
                answers_gen = self.tokenizer.batch_decode(answers_gen_ids,
                                                          skip_special_tokens=True)
                gts = {}
                gens = {}
                for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
                    words = gen_i.split()
                    gen_i = ' '.join([k for k, g in itertools.groupby(words)])
                    gens['%d_%d' % (it, i)] = [gen_i, ]
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
