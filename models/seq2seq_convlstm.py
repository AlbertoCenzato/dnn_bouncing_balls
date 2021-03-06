import torch
from torch import nn

from .convlstm import ConvLSTM


class Seq2SeqConvLSTM(nn.Module):
    """
    This model is an implementation of the sequence-to-sequence convolutional LSTM
    model proposed in 'Convolutional LSTM Network: A Machine Learning Approach 
    for Precipitation Nowcasting', Shi et al., 2015, http://arxiv.org/abs/1506.04214
    Instead of one decoding network, as proposed in the paper, this model has two
    decoding networks as in 'Unsupervised Learning of Video Representations using LSTMs',
    Srivastava et al., 2016.

    The encoding network receives a sequence of images and outputs its hidden state that
    should represent a compressed representation of the sequence. Its hidden state is then
    used as initial hidden state for the two decoding networks that use the information
    contained in it to respectively reconstruct the input sequence and to predict future 
    frames.
    """
    
    def __init__(self, input_size, input_ch, hidden_ch, kernel_size, batch_first=True,
                 bias=True, decoding_steps=-1):
        """
        :param input_size: Tuple[int, int]
        :param input_ch: int
        :param hidden_ch: List[int]
        :param kernel_size: List[Tuple[int, int]]
        :param batch_first: bool
        :param bias: bool
        :param decoding_steps: int
        """
        super(Seq2SeqConvLSTM, self).__init__()
        self.decoding_steps = decoding_steps
        self.input_size  = input_size
        self.input_ch   = input_ch
        self.img_size = input_size[0] * input_size[1]
        self.hidden_ch  = hidden_ch
        self.kernel_size = kernel_size
        self.batch_first = batch_first
        self.num_enc_layers = len(hidden_ch)

        self.encoder = ConvLSTM(
            input_size=input_size, 
            input_dim=input_ch, 
            hidden_dim=hidden_ch, 
            kernel_size=kernel_size, 
            num_layers=self.num_enc_layers, 
            batch_first=False, 
            bias=bias, 
            mode=ConvLSTM.SEQUENCE
        )

        # reverse the order of hidden dimensions and kernels
        decoding_hidden_dim  = hidden_ch  
        decoding_kernel_size = kernel_size

        self.future_prediction = ConvLSTM(
                                    input_size=input_size,
                                    input_dim=input_ch,
                                    hidden_dim=decoding_hidden_dim,
                                    kernel_size=decoding_kernel_size,
                                    num_layers=self.num_enc_layers,
                                    batch_first=False,
                                    bias=bias,
                                    mode=ConvLSTM.STEP_BY_STEP
                                 )
        
    def forward(self, input_sequence):
        """
        :param input_sequence: torch.Tensor
        :return: List[torch.Tensor]
        """
        sequence = input_sequence.transpose(0,1) if self.batch_first else input_sequence  # always work in sequence-first mode
        sequence_len = sequence.size(0)

        # ------------------ encoding ------------------------
        _, hidden_state = self.encoder(sequence)

        last_frame = sequence[-1, :]
        h_n, c_n = hidden_state

        representation = (h_n, c_n)

        # ----------------- decoder --------------------
        steps = self.decoding_steps if self.decoding_steps != -1 else sequence_len

        output_seq_pred = self._decode(self.future_prediction, last_frame,
                                       representation, steps)

        prediction = output_seq_pred.transpose(0,1) if self.batch_first else output_seq_pred

        return prediction

    def _decode(self, decoder, last_frame, representation, steps):
        """
        :param decoder: ConvLSTM
        :param last_frame: torch.Tensor
        :param representation: HiddenState
        :param steps: int
        :return: torch.Tensor
        """
        decoded_sequence = []

        h_n, c_n = representation
        h_0, c_0 = decoder.init_hidden(last_frame.size(0))

        h_0[:len(h_n)] = h_n
        c_0[:len(c_n)] = c_n

        state = (h_0, c_0)
        output = last_frame
        for _ in range(steps):
            output, state = decoder(output, state)
            decoded_sequence.append(output)

        return torch.stack(decoded_sequence, dim=0)
