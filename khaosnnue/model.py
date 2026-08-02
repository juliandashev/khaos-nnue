"""The trainable model (torch). Topology matches include/nnue.h:

    768 -> HIDDEN (feature transformer) x2 -> concat -> L2 -> 1

Clipped ReLU on [0, 1] (quantized: [0, QA]). The float forward mirrors the
integer one so a checkpoint and its export agree up to rounding;
scripts/check_engine.py verifies that against the engine.
"""

import torch
import torch.nn as nn

from . import quant


class KhaosNet(nn.Module):
    def __init__(self, hidden=quant.HIDDEN, l2=quant.L2):
        super().__init__()
        self.hidden = hidden

        # Feature transformer (exporter transposes to feature-major), then one
        # hidden layer over the concatenated perspectives, then the output.
        self.feature_transformer = nn.Linear(quant.INPUTS, hidden)
        self.hidden2 = nn.Linear(2 * hidden, l2)
        self.output = nn.Linear(l2, 1)

        self._init_weights()

    def _init_weights(self):
        # Small feature init keeps the accumulator inside int16 range.
        nn.init.uniform_(self.feature_transformer.weight, -0.01, 0.01)
        nn.init.zeros_(self.feature_transformer.bias)
        nn.init.uniform_(self.hidden2.weight, -0.1, 0.1)
        nn.init.zeros_(self.hidden2.bias)
        nn.init.uniform_(self.output.weight, -0.1, 0.1)
        nn.init.zeros_(self.output.bias)

    def forward(self, own_features, their_features):
        """Evaluate a batch.

        own_features / their_features are sparse (batch, INPUTS) float tensors
        -- or anything matmul-compatible -- already arranged so that
        `own` is the side-to-move's perspective.

        Returns the raw output; sigmoid(raw) is the predicted win probability
        and raw * EVAL_SCALE is the evaluation in engine units.
        """
        own_acc = self.feature_transformer(own_features)
        their_acc = self.feature_transformer(their_features)

        x = torch.cat(
            [own_acc.clamp(0.0, 1.0), their_acc.clamp(0.0, 1.0)], dim=1
        )
        h = self.hidden2(x).clamp(0.0, 1.0)

        return self.output(h).squeeze(1)

    @torch.no_grad()
    def clamp_weights(self):
        """Keep feature weights inside the range the int16 accumulator allows.

        Called after every optimizer step. Without it, training is free to grow
        weights until 32 pieces' worth of accumulation overflows int16 at
        inference -- which would not show up in the training loss at all, only
        as a net that plays bizarrely once quantized.
        """
        self.feature_transformer.weight.clamp_(
            -quant.FEATURE_WEIGHT_CLAMP, quant.FEATURE_WEIGHT_CLAMP
        )
        self.feature_transformer.bias.clamp_(
            -quant.FEATURE_WEIGHT_CLAMP, quant.FEATURE_WEIGHT_CLAMP
        )
        # Post-accumulator weights must fit int16 after * QB.
        lim = quant.LAYER_WEIGHT_CLAMP
        self.hidden2.weight.clamp_(-lim, lim)
        self.output.weight.clamp_(-lim, lim)


def quantized_tensors(model):
    """Convert a trained model to the integer arrays write_net() wants:
    (feature_weights, feature_bias, l2_weights, l2_bias, output_weights,
    output_bias) as flat Python lists of ints."""
    with torch.no_grad():
        # (hidden, INPUTS) -> (INPUTS, hidden): feature-major for the accumulator.
        fw = model.feature_transformer.weight.t().contiguous()
        feature_weights = torch.round(fw * quant.QA).to(torch.int32)
        feature_bias = torch.round(
            model.feature_transformer.bias * quant.QA
        ).to(torch.int32)

        # nn.Linear weight is already [out, in] = output-major; no transpose.
        l2_weights = torch.round(model.hidden2.weight * quant.QB).to(torch.int32)
        l2_bias = torch.round(
            model.hidden2.bias * quant.QA * quant.QB
        ).to(torch.int32)

        output_weights = torch.round(
            model.output.weight.squeeze(0) * quant.QB
        ).to(torch.int32)
        output_bias = int(
            round(float(model.output.bias.item()) * quant.QA * quant.QB)
        )

    return (
        feature_weights.flatten().tolist(),
        feature_bias.tolist(),
        l2_weights.flatten().tolist(),
        l2_bias.tolist(),
        output_weights.tolist(),
        output_bias,
    )
