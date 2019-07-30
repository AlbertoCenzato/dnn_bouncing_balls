import torch
import torch.nn as nn

from .lstm_cell_stack import LSTMCellStack
from .lstm_stack import LSTMStack


class Seq2SeqLSTM(nn.Module):
    """
    Implementation of the model described in 'Unsupervised Learning of Video 
    Representations using LSTMs', N. Srivastava, E. Mansimov, R. Salakhutdinov
    https://arxiv.org/pdf/1502.04681.pdf

    It is composed by an LSTM which acts as an encoder for a video sequence 
    and one or multiple decoders (LSTM but possibly other models too) that, 
    given the same input representations, execute various tasks.
    """
    
    def __init__(self, input_size, hidden_size, batch_first, decoding_steps=-1):
        """
        :param input_size: int
        :param hidden_size: List[int]
        :param batch_first: bool
        :param decoding_steps: int
        """
        super(Seq2SeqLSTM, self).__init__()
        self.batch_first = batch_first
        self.input_size  = input_size
        self.decoding_steps = -1

        self.encoder = LSTMStack(
            input_size=self.input_size, 
            hidden_size=hidden_size, 
            batch_first=False
        )

        sizes = [self.input_size, *hidden_size]
        decoding_sizes = list(reversed(sizes))
        self.input_reconstruction = LSTMCellStack(
                                        input_size=self.input_size, 
                                        hidden_size=decoding_sizes
                                    )

        self.future_prediction = LSTMCellStack(
                                     input_size=self.input_size,
                                     hidden_size=decoding_sizes
                                 )

    def forward(self, input_sequence):
        """
        :param input_sequence: torch.Tensor
        :return: Tuple[torch.Tensor, torch.Tensor]
        """
        sequence = input_sequence.transpose(0,1) if self.batch_first else input_sequence  # always work in sequence-first mode
        sequence_len = sequence.size(0)

        # encode
        _, hidden_state = self.encoder(sequence)  # discard output, we are interested only in hidden state to initialize the decoders
        # LSTM state has shape (num_layers * num_directions, batch, hidden_size) =
        # (1, batch, hidden_size) but LSTMCell expects h and c to have shape 
        # (batch, hidden_size), so we have to remove the first dimension
        h_n, c_n = hidden_state
        h_n_last, c_n_last = h_n[-1], c_n[-1]  # take the last layer's hidden state ...
        representation = (h_n_last.squeeze(dim=0), c_n_last.squeeze(dim=0))  # ... and use it as compressed representation of what the model has seen so far

        steps = self.decoding_steps if self.decoding_steps != -1 else sequence_len
        
        # decode for input reconstruction
        output_seq_recon = self._decode(self.input_reconstruction, sequence, # last_frame, 
                                        representation, steps)

        # decode for future prediction
        output_seq_pred = self._decode(self.future_prediction, sequence,
                                       representation, steps)

        if self.batch_first:  # if input was batch_first restore dimension order
            reconstruction = output_seq_recon.transpose(0,1)
            prediction     = output_seq_pred .transpose(0,1)
        else:
            reconstruction = output_seq_recon
            prediction     = output_seq_pred

        return reconstruction, prediction

    def _decode(self, decoder, input_sequence, representation, steps):
        """
        :param decoder: LSTMCellStack
        :param input_sequence: torch.Tensor
        :param representation: Tuple[torch.Tensor, torch.Tensor]
        :param steps: int
        :return: torch.Tensor
        """
        output_seq = []
        sequence_reversed = input_sequence.flip(0)

        h_0, c_0 = decoder.init_hidden(input_sequence.size(1))
        # use encoder's last layer hidden state to initalize decoders hidden state
        h_0[0], c_0[0] = representation[0], representation[1]

        state = (h_0, c_0)
        for t in range(steps):
            output, state = decoder(sequence_reversed[t,:], state)
            output_seq.append(output)

        return torch.stack(output_seq, dim=0)  # dim 0 because we are working with batch_first=False


class ImgLSTMAutoencoder(nn.Module):

    def __init__(self, image_size, hidden_size, batch_first, decoding_steps=-1):
        """
        :param image_size: Tuple[int, int, int]
        :param hidden_size: List[int]
        :param batch_first: bool
        :param decoding_steps: int
        """
        super(ImgLSTMAutoencoder, self).__init__()
        self.image_size  = image_size
        self.input_size  = image_size[0] * image_size[1] * image_size[2]
        self.batch_first = batch_first

        self.lstm_autoencoder = Seq2SeqLSTM(self.input_size, hidden_size, False, decoding_steps)

    def forward(self, input):
        """
        :param input: torch.Tensor
        :return: Tuple[torch.Tensor, torch.Tensor]
        """
        sequence = input.transpose(0,1) if self.batch_first else input  # always work in sequence-first mode
        sequence_len = sequence.size(0)
        batch_size   = sequence.size(1)

        flattened_sequence = input.view((sequence_len, batch_size, -1))

        reconstruction, prediction = self.lstm_autoencoder(flattened_sequence)

        sequence_shape = (self.decoding_steps, batch_size,) + self.image_size
        reconstruction_img = reconstruction.view(sequence_shape)
        prediction_img     = prediction    .view(sequence_shape)

        recon_out = reconstruction_img.transpose(0,1) if self.batch_first else reconstruction_img
        pred_out  = prediction_img    .transpose(0,1) if self.batch_first else prediction_img

        return recon_out, pred_out

    @property
    def decoding_steps(self):
        return self.lstm_autoencoder.decoding_steps

    @decoding_steps.setter
    def decoding_steps(self, steps):
        self.lstm_autoencoder.decoding_steps = steps