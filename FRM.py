import torch.nn as nn
import numpy as np

    
class AutoEncoder(nn.Module):
    def __init__(self, input_size = 768, num_encoder_layers=3, num_decoder_layers=3, bottleneck_dim=96, dropout_prob=0.5):

        super(AutoEncoder, self).__init__()
        input_dim = input_size
        output_dim = input_size

        encoder_dims = self._interpolate_dims(input_dim, bottleneck_dim, num_encoder_layers)
        decoder_dims = self._interpolate_dims(bottleneck_dim, output_dim, num_decoder_layers)

        def linear_block(in_features, out_features, apply_dropout=True):
            return nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
                nn.Dropout(dropout_prob if apply_dropout else 0)
            )

        self.encoder = nn.Sequential(*[
            linear_block(encoder_dims[i], encoder_dims[i+1], apply_dropout=(i != len(encoder_dims) - 2))
            for i in range(len(encoder_dims) - 1)
        ])

        self.decoder = nn.Sequential(*[
            linear_block(decoder_dims[i], decoder_dims[i+1], apply_dropout=(i != len(decoder_dims) - 2))
            if i != len(decoder_dims) - 2 else nn.Linear(decoder_dims[i], decoder_dims[i+1])
            for i in range(len(decoder_dims) - 1)
        ])

    def _interpolate_dims(self, start_dim, end_dim, steps):

        return [int(x) for x in np.linspace(start_dim, end_dim, steps + 1)]

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

