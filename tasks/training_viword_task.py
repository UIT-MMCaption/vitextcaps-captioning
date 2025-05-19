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
    def __init__(self, vocab, padding_idx=0):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained('NlpHUST/gpt2-vietnamese', force_download=True)
        
        if self.tokenizer.pad_token is None:
             self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.reward_model = GPT2LMHeadModel.from_pretrained('NlpHUST/gpt2-vietnamese', force_download=True)

        if len(self.tokenizer) > self.reward_model.get_input_embeddings().num_embeddings:
             self.reward_model.resize_token_embeddings(len(self.tokenizer))

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=padding_idx)
        self.vocab = vocab

        # Get the maximum sequence length from the reward model's configuration
        self.max_seq_length = self.reward_model.config.max_position_embeddings


        for param in self.reward_model.parameters():
            param.requires_grad = False


        self.reward_model.eval() # Set to evaluation mode (disables dropout, batchnorm updates, etc.)


    def forward(self, logits, answer_tokens, lambda_=0.1):
        
        padded_answer_tokens = F.pad(answer_tokens.type(torch.long), (0, 0, 0, 410 - answer_tokens.shape[1])).to(logits.device)
        loss = self.loss_fn(logits.view(-1, logits.shape[-1]), padded_answer_tokens.view(-1))

        # Decode the generated logits into a list of word strings (one for each item in the batch)
        word_strings = self.vocab.decode_batch_caption(logits.argmax(dim=-1),
                                            join_words=True)
        # Tokenize the generated word strings (passing a list of strings)
        # Add padding=True to explicitly handle padding for the batch
        tokenized_outputs = self.tokenizer(word_strings,
                                           return_tensors='pt',
                                           padding=True, # Explicitly enable padding
                                           truncation=True,
                                           max_length=self.max_seq_length)

        input_ids = tokenized_outputs['input_ids'].to(logits.device)
        attention_mask = tokenized_outputs['attention_mask'].to(logits.device)
        
        max_len_in_batch = input_ids.shape[1]
        position_ids = torch.arange(0, max_len_in_batch, dtype=torch.long, device=logits.device).unsqueeze(0).expand_as(input_ids)

        # Calculate reward loss using the reward model
        # The loss returned by the reward model is the average negative log-likelihood over the batch and sequence length
        outputs = self.reward_model(input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=input_ids)

        reward_loss_avg_nll_batch = outputs.loss

        # minimize `loss + lambda_ * reward_loss_avg_nll_batch`.
        total_loss = loss + lambda_ * reward_loss_avg_nll_batch

        return total_loss

@META_TASK.register()
class TrainingViWord(OpenEndedTask):
    def __init__(self, config):
        super().__init__(config)
        self.scheduler = LambdaLR(self.optim, self.lambda_lr)
        # self.loss_fn = BCEWithMaskLogitsLoss(ignore_index=self.vocab.padding_idx)
        # self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.vocab.padding_idx)
        # self.loss_fn = NLLLoss(ignore_index=self.vocab.padding_idx)
        self.loss_fn = CustomLoss(self.vocab).to('cuda')
        self.delta = config.TRAINING.AUX_LOSS_COEF

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