"""The trainable model. Requires torch.

Topology, matching include/nnue.h:

    768 -> HIDDEN  (feature transformer, shared by both perspectives)
    concat[own, their] -> 1

Activation is a clipped ReLU on [0, 1]; at inference the same clamp happens on
[0, QA] because the quantized accumulator is QA times the float one.

The float forward pass here is written to mirror the integer one exactly (same
operation order, same clamp), so a trained checkpoint and its exported net
should agree to within quantization rounding. scripts/check_engine.py is what
verifies the exported net against the engine.
"""

import torch
import torch.nn as nn

from . import quant


class KhaosNet(nn.Module):
    def __init__(self, hidden=quant.HIDDEN):
        super().__init__()
        self.hidden = hidden

        # Feature transformer. Stored as (hidden, INPUTS) like any nn.Linear;
        # the exporter transposes to the feature-major layout the engine wants.
        self.feature_transformer = nn.Linear(quant.INPUTS, hidden)

        # Output layer over [own half, their half].
        self.output = nn.Linear(2 * hidden, 1)

        self._init_weights()

    def _init_weights(self):
        # Small init keeps the early accumulator well inside int16 range and
        # avoids saturating the clipped ReLU before anything is learned.
        nn.init.uniform_(self.feature_transformer.weight, -0.01, 0.01)
        nn.init.zeros_(self.feature_transformer.bias)
        nn.init.uniform_(self.output.weight, -0.05, 0.05)
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

        activated = torch.cat(
            [own_acc.clamp(0.0, 1.0), their_acc.clamp(0.0, 1.0)], dim=1
        )

        return self.output(activated).squeeze(1)

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


def quantized_tensors(model):
    """Convert a trained model to the integer arrays the .nnue format wants.

    Returns (feature_weights, feature_bias, output_weights, output_bias) as
    flat Python lists of ints, ready for khaosnnue.format.write_net().
    """
    with torch.no_grad():
        # (hidden, INPUTS) -> (INPUTS, hidden) so one feature's weights are
        # contiguous, which is the layout the accumulator update reads.
        fw = model.feature_transformer.weight.t().contiguous()
        feature_weights = torch.round(fw * quant.QA).to(torch.int32)
        feature_bias = torch.round(
            model.feature_transformer.bias * quant.QA
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
        output_weights.tolist(),
        output_bias,
    )
