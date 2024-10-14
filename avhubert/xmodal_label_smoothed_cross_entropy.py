# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class XmodalLabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    flow_coef: float = field(
        default=0.1,
        metadata={"help": "temporal flow loss scaling coefficient"},
    )
    direction_coef: float = field(
        default=0.05,
        metadata={"help": "temporal direction loss scaling coefficient"},
    )
    speed_coef: float = field(
        default=0.05,
        metadata={"help": "temporal speed loss scaling coefficient"},
    )
    reg_coef: float = field(
        default=0.1,
        metadata={"help": "regression loss scaling coefficient"},
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "xmodal_label_smoothed_cross_entropy", dataclass=XmodalLabelSmoothedCrossEntropyCriterionConfig
)
class XmodalLabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        flow_coef,
        direction_coef,
        speed_coef,
        reg_coef,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.flow_coef = flow_coef
        self.direction_coef = direction_coef
        self.speed_coef = speed_coef
        self.reg_coef = reg_coef
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.embed_dim = 1024
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.mse_loss = nn.MSELoss(reduction="none")
        self.bce_loss = nn.BCEWithLogitsLoss(weight=None, reduction="none")

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output, layer_results = model(**sample["net_input"])
        loss, nll_loss, aux_loss = self.compute_loss(model, net_output, layer_results, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        logging_output["temporal_flow_loss_video"] = aux_loss.get("temporal_flow_loss_video", 0)
        logging_output["temporal_flow_loss_audio"] = aux_loss.get("temporal_flow_loss_audio", 0)
        logging_output["temporal_va_flow_loss_video"] = aux_loss.get("temporal_va_flow_loss_video", 0)
        logging_output["temporal_va_flow_loss_audio"] = aux_loss.get("temporal_va_flow_loss_audio", 0)
        logging_output["temporal_direction_loss_video"] = aux_loss.get("temporal_direction_loss_video", 0)
        logging_output["temporal_direction_loss_audio"] = aux_loss.get("temporal_direction_loss_audio", 0)
        logging_output["temporal_speed_loss_video"] = aux_loss.get("temporal_speed_loss_video", 0)
        logging_output["temporal_speed_loss_audio"] = aux_loss.get("temporal_speed_loss_audio", 0)
        logging_output["regression_loss_video"] = aux_loss.get("regression_loss_video", 0)
        logging_output["regression_loss_audio"] = aux_loss.get("regression_loss_audio", 0)
        
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def get_temporal_flow_loss(self, temporal_output):
        # out (temporal_output): 2 * B * (T-1) * 1
        temporal_output = temporal_output.squeeze(-1)
        forward_targets = torch.ones_like(temporal_output[0]).to(dtype=temporal_output.dtype)
        backward_targets = torch.zeros_like(temporal_output[1]).to(dtype=temporal_output.dtype)
        targets = torch.stack([forward_targets, backward_targets], 0)
        bce_loss = self.bce_loss(temporal_output, targets).sum(-1).mean()
        return bce_loss

    def get_temporal_direction_loss(self, temporal_output):
        # out (temporal_output): 2 * B * (T-1) * 1
        temporal_output = temporal_output.squeeze(-1)
        forward_targets = torch.ones_like(temporal_output[0]).to(dtype=temporal_output.dtype)
        backward_targets = torch.zeros_like(temporal_output[1]).to(dtype=temporal_output.dtype)
        targets = torch.stack([forward_targets, backward_targets], 0)
        bce_loss = self.bce_loss(temporal_output, targets).sum(-1).mean()
        return bce_loss
    
    def get_temporal_speed_loss(self, temporal_output):
        # out (temporal_output): 2 * B * (T-1) * 1
        temporal_output = temporal_output.squeeze(-1)
        forward_targets = torch.ones_like(temporal_output[0]).to(dtype=temporal_output.dtype)
        backward_targets = torch.zeros_like(temporal_output[1]).to(dtype=temporal_output.dtype)
        targets = torch.stack([forward_targets, backward_targets], 0)
        bce_loss = self.bce_loss(temporal_output, targets).sum(-1).mean()
        return bce_loss

    def compute_loss(self, model, net_output, layer_results, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )

        layer_results_video, layer_results_audio = layer_results
        aux_loss = dict()

        if "temp_flow" in layer_results_video:
            temporal_flow_loss_video = self.get_temporal_flow_loss(layer_results_video["temp_flow"])
            loss += self.flow_coef * temporal_flow_loss_video
            aux_loss["temporal_flow_loss_video"] = (self.flow_coef * temporal_flow_loss_video).data
        if "temp_flow" in layer_results_audio:
            temporal_flow_loss_audio = self.get_temporal_flow_loss(layer_results_audio["temp_flow"])
            loss += self.flow_coef * temporal_flow_loss_audio
            aux_loss["temporal_flow_loss_audio"] = (self.flow_coef * temporal_flow_loss_audio).data

        if "temp_va_flow" in layer_results_video:
            va_temporal_flow_loss_video = self.get_temporal_flow_loss(layer_results_video["temp_va_flow"])
            loss += self.flow_coef * va_temporal_flow_loss_video
            aux_loss["temporal_va_flow_loss_video"] = (self.flow_coef * va_temporal_flow_loss_video).data
        if "temp_va_flow" in layer_results_audio:
            va_temporal_flow_loss_audio = self.get_temporal_flow_loss(layer_results_audio["temp_va_flow"])
            loss += self.flow_coef * va_temporal_flow_loss_audio
            aux_loss["temporal_va_flow_loss_audio"] = (self.flow_coef * va_temporal_flow_loss_audio).data

        if "temp_direction" in layer_results_video:
            temporal_direction_loss_video = self.get_temporal_direction_loss(layer_results_video["temp_direction"])
            loss += self.direction_coef * temporal_direction_loss_video
            aux_loss["temporal_direction_loss_video"] = (self.direction_coef * temporal_direction_loss_video).data
        if "temp_direction" in layer_results_audio:
            temporal_direction_loss_audio = self.get_temporal_direction_loss(layer_results_audio["temp_direction"])
            loss += self.direction_coef * temporal_direction_loss_audio
            aux_loss["temporal_direction_loss_audio"] = (self.direction_coef * temporal_direction_loss_audio).data

        if "temp_speed" in layer_results_video:
            temporal_speed_loss_video = self.get_temporal_speed_loss(layer_results_video["temp_speed"])
            loss += self.speed_coef * temporal_speed_loss_video
            aux_loss["temporal_speed_loss_video"] = (self.speed_coef * temporal_speed_loss_video).data
        if "temp_speed" in layer_results_audio:
            temporal_speed_loss_audio = self.get_temporal_speed_loss(layer_results_audio["temp_speed"])
            loss += self.speed_coef * temporal_speed_loss_audio
            aux_loss["temporal_speed_loss_audio"] = (self.speed_coef * temporal_speed_loss_audio).data

        if "xattn_feat" in layer_results_video:
            regression_loss_video = (self.mse_loss(layer_results_video["xattn_feat"], layer_results_video["feat"]) / self.embed_dim).sum()
            loss += self.reg_coef * regression_loss_video
            aux_loss["regression_loss_video"] = (self.reg_coef * regression_loss_video).data
        if "xattn_feat" in layer_results_audio:
            regression_loss_audio = (self.mse_loss(layer_results_audio["xattn_feat"], layer_results_audio["feat"]) / self.embed_dim).sum()
            loss += self.reg_coef * regression_loss_audio
            aux_loss["regression_loss_audio"] = (self.reg_coef * regression_loss_audio).data

        return loss, nll_loss, aux_loss
    
    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        temporal_flow_loss_video_sum = sum(log.get("temporal_flow_loss_video", 0) for log in logging_outputs)
        temporal_flow_loss_audio_sum = sum(log.get("temporal_flow_loss_audio", 0) for log in logging_outputs)
        temporal_va_flow_loss_video_sum = sum(log.get("temporal_va_flow_loss_video", 0) for log in logging_outputs)
        temporal_va_flow_loss_audio_sum = sum(log.get("temporal_va_flow_loss_audio", 0) for log in logging_outputs)
        temporal_direction_loss_video_sum = sum(log.get("temporal_direction_loss_video", 0) for log in logging_outputs)
        temporal_direction_loss_audio_sum = sum(log.get("temporal_direction_loss_audio", 0) for log in logging_outputs)
        temporal_speed_loss_video_sum = sum(log.get("temporal_speed_loss_video", 0) for log in logging_outputs)
        temporal_speed_loss_audio_sum = sum(log.get("temporal_speed_loss_audio", 0) for log in logging_outputs)
        regression_loss_video_sum = sum(log.get("regression_loss_video", 0) for log in logging_outputs)
        regression_loss_audio_sum = sum(log.get("regression_loss_audio", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "temporal_flow_loss_video", temporal_flow_loss_video_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "temporal_flow_loss_audio", temporal_flow_loss_audio_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "temporal_va_flow_loss_video", temporal_va_flow_loss_video_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "temporal_va_flow_loss_audio", temporal_va_flow_loss_audio_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "temporal_direction_loss_video", temporal_direction_loss_video_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "temporal_direction_loss_audio", temporal_direction_loss_audio_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "temporal_speed_loss_video", temporal_speed_loss_video_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "temporal_speed_loss_audio", temporal_speed_loss_audio_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "regression_loss_video", regression_loss_video_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "regression_loss_audio", regression_loss_audio_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
