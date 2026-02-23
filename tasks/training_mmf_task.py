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
import csv
import time
from tqdm import tqdm
import itertools
from shutil import copyfile
import json
from builders.task_builder import META_TASK
from torch.optim.lr_scheduler import LambdaLR
logger = setup_logger()

class BCEWithMaskLogitsLoss(nn.Module):
    def __init__(self, ignore_index=0):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        loss_mask = (target == self.ignore_index)
        source = torch.ones_like(input)
        scattered_target = torch.zeros_like(input)
        scattered_target.scatter_(dim=-1, index=target.unsqueeze(-1), src=source)
        losses = F.binary_cross_entropy_with_logits(input, scattered_target, reduction="none")
        losses = losses.masked_fill(loss_mask.unsqueeze(-1), value=0)
        count = torch.max(torch.sum(loss_mask), torch.ones((1, )).to(loss_mask.device))
        loss = torch.sum(losses) / count
        return loss

@META_TASK.register()
class TrainingMMF(OpenEndedTask):
    def __init__(self, config):
        super().__init__(config)
        self.scheduler = LambdaLR(self.optim, self.lambda_lr)
        self.loss_fn = NLLLoss(ignore_index=self.vocab.padding_idx)

        # === Logging setup for paper reporting ===
        self.log_dir = os.path.join(self.checkpoint_path, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # CSV log for per-epoch metrics
        self.csv_path = os.path.join(self.log_dir, "training_log.csv")
        self.csv_columns = [
            "epoch", "train_loss", "val_loss", "lr",
            "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4",
            "METEOR", "ROUGE_L", "CIDEr", "SPICE",
            "Accuracy", "Precision", "Recall", "F1",
            "epoch_time_min"
        ]
        if not os.path.isfile(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_columns)
                writer.writeheader()

        # Save config snapshot
        config_snapshot_path = os.path.join(self.log_dir, "config.txt")
        if not os.path.isfile(config_snapshot_path):
            with open(config_snapshot_path, "w") as f:
                f.write(str(config))

        # Model summary
        model_info_path = os.path.join(self.log_dir, "model_info.txt")
        if not os.path.isfile(model_info_path):
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            with open(model_info_path, "w") as f:
                f.write(f"Total parameters: {total_params:,}\n")
                f.write(f"Trainable parameters: {trainable_params:,}\n")
                f.write(f"Vocab size: {len(self.vocab)}\n")
                f.write(f"\n{self.model}\n")
        logger.info("Logging setup complete. Logs will be saved to %s", self.log_dir)

    def _log_epoch(self, epoch, train_loss, val_loss, lr, scores, epoch_time):
        """Write one row to the CSV log and log to console."""
        bleu_scores = scores.get("BLEU", scores.get("Bleu", [0, 0, 0, 0]))
        if not isinstance(bleu_scores, (list, tuple)):
            bleu_scores = [bleu_scores, 0, 0, 0]

        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "lr": f"{lr:.8f}",
            "Bleu_1": f"{bleu_scores[0]:.4f}" if len(bleu_scores) > 0 else "0.0",
            "Bleu_2": f"{bleu_scores[1]:.4f}" if len(bleu_scores) > 1 else "0.0",
            "Bleu_3": f"{bleu_scores[2]:.4f}" if len(bleu_scores) > 2 else "0.0",
            "Bleu_4": f"{bleu_scores[3]:.4f}" if len(bleu_scores) > 3 else "0.0",
            "METEOR": f"{scores.get('METEOR', 0):.4f}",
            "ROUGE_L": f"{scores.get('ROUGE', scores.get('ROUGE_L', 0)):.4f}",
            "CIDEr": f"{scores.get('CIDEr', 0):.4f}",
            "SPICE": f"{scores.get('SPICE', 0):.4f}",
            "Accuracy": f"{scores.get('Accuracy', 0):.4f}",
            "Precision": f"{scores.get('Precision', 0):.4f}",
            "Recall": f"{scores.get('Recall', 0):.4f}",
            "F1": f"{scores.get('F1', 0):.4f}",
            "epoch_time_min": f"{epoch_time / 60:.2f}"
        }
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_columns)
            writer.writerow(row)

        logger.info(
            "Epoch %d | Train Loss: %.4f | Val Loss: %.4f | LR: %.2e | "
            "B@1: %s | B@4: %s | M: %s | R: %s | C: %s | S: %s | Time: %.1f min",
            epoch, train_loss, val_loss, lr,
            row["Bleu_1"], row["Bleu_4"], row["METEOR"],
            row["ROUGE_L"], row["CIDEr"], row["SPICE"],
            epoch_time / 60
        )

    def create_dict_dataloaders(self, config):
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
        self.model.train()  # teacher forcing for comparable val loss
        running_loss = .0
        with tqdm(desc='Epoch %d - Validation' % self.epoch, unit='it', total=len(dataloader)) as pbar:
            with torch.no_grad():
                for it, items in enumerate(dataloader):
                    items = items.to(self.device)
                    with torch.no_grad():
                        results = self.model(items)
                    out = results["scores"].contiguous()
                    out = F.log_softmax(out, dim=-1)
                    shifted_right_answer_tokens = items.shifted_right_answer_tokens
                    loss = self.loss_fn(out.view(-1, out.shape[-1]), shifted_right_answer_tokens.view(-1))
                    this_loss = loss.item()
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
                    results = self.model(items)
                outs = results["scores"].argmax(dim=-1)
                answers_gt = items.answers
                answers_gen = self.vocab.decode_answer(outs.contiguous(),
                                                       items.ocr_tokens,
                                                       join_words=False)
                for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
                    gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                    gens['%d_%d' % (it, i)] = [gen_i, ]
                    # gts_i is a list of words; join into a sentence string for metrics
                    gts['%d_%d' % (it, i)] = [' '.join(gts_i)]
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
                out = results["scores"].contiguous()
                out = F.log_softmax(out, dim=-1)
                shifted_right_answer_tokens = items.shifted_right_answer_tokens
                self.optim.zero_grad()
                loss = self.loss_fn(out.view(-1, out.shape[-1]), shifted_right_answer_tokens.view(-1))
                loss.backward()
                self.optim.step()
                this_loss = loss.item()
                running_loss += this_loss
                pbar.set_postfix(loss=running_loss / (it + 1), refresh=True)
                pbar.update()
                self.scheduler.step()
        return running_loss / len(self.train_dataloader)

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
            epoch_start = time.time()

            train_loss = self.train()
            val_loss = self.evaluate_loss(self.dev_dataloader)

            # val scores
            scores = self.evaluate_metrics(self.dev_dict_dataloader)
            logger.info("Validation scores %s", scores)
            val_score = scores[self.score]

            epoch_time = time.time() - epoch_start
            current_lr = self.scheduler.get_last_lr()[0]
            self._log_epoch(self.epoch, train_loss, val_loss, current_lr, scores, epoch_time)

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

        self.load_checkpoint(os.path.join(self.checkpoint_path, "best_model.pth"))

        self.model.eval()
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
                answers_gen, in_fixed_vocab = self.vocab.decode_answer_with_determination(outs.contiguous().view(-1, self.vocab.max_answer_length),
                                                        items.ocr_tokens, join_words=False)
                gts = {}
                gens = {}
                for i, (gts_i, gen_i, in_fixed_vocab_i) in enumerate(zip(answers_gt, answers_gen, in_fixed_vocab)):
                    gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                    gens['%d_%d' % (it, i)] = (gen_i, in_fixed_vocab_i)
                    gts['%d_%d' % (it, i)] = [' '.join(gts_i)]
                    overall_gens['%d_%d' % (it, i)] = [gen_i, ]
                    overall_gts['%d_%d' % (it, i)] = [' '.join(gts_i)]
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
