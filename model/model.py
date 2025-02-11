import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from moment_utils import get_moment_model
from xlstm_utils import get_xlstm_model


#############################################################################################################
class CLEF(nn.Module):
  def __init__(self,
               seq_encoder, nfeat, nconcepts, ntoken, nhead, nlayers,         ### For sequence encoder
               condition, condition_sz = 1,                                   ### For condition encoder
               apply_ct_proj = None,                                          ### For concept encoder/decoder
               concept_ones = False,                                          ### For baseline
               conv_sz = 0, qkv_proj_sz = 0,                                  ### For xLSTM-only
               dropout = 0.5
              ):
    super(CLEF, self).__init__()

    self.seq_encoder = seq_encoder
    self.nfeat = nfeat
    self.nconcepts = nconcepts
    self.ntoken = ntoken
    self.nhead = nhead
    self.nlayers = nlayers
    self.conv_sz = conv_sz
    self.qkv_proj_sz = qkv_proj_sz
    self.condition = condition
    self.condition_sz = condition_sz

    # For baselines
    self.concept_ones = concept_ones # Boolean; all concept values are ones
    self.apply_ct_proj = apply_ct_proj # Apply lab decoder on the concept vector * next date embedding (boolean)

    if self.nconcepts == 0: # Baseline: Without concepts
      
      # Sequence encoder
      self.init_seq_encoder(self.nfeat)
      self.seq_decoder = nn.Linear(self.nfeat, self.nfeat)
      
      # Condition encoder
      if self.condition: self.condition_encoder = nn.Linear(self.condition_sz, self.nfeat)
      self.condition_decoder = nn.Linear(self.nfeat, self.nfeat)
    
    else: # Full model: With concepts

      # Sequence encoder
      self.init_seq_encoder(self.nconcepts)
      self.concept_proj = nn.Linear(self.nconcepts, self.nfeat) # Applicable only when self.apply_ct_proj == True
      self.concept_act = nn.GELU()
      
      # Condition encoder
      if self.condition: self.condition_encoder = nn.Linear(self.condition_sz, self.nconcepts)
      self.condition_decoder = nn.Linear(self.nconcepts, self.nconcepts)

  def init_seq_encoder(self, outdim):
    if self.seq_encoder == "transformer":
        self.encoder = TransformerEncoder(self.ntoken, self.nfeat, self.nhead, self.nfeat, self.nlayers, outdim)
    elif self.seq_encoder == "moment":
      self.encoder = get_moment_model("embedding")
      self.encoder_proj = nn.Linear(1024, outdim) # Pretrained embeddings are 1024 dim
      self.time_encoder = DateTimeEmbedding(self.ntoken, self.nfeat)
    elif self.seq_encoder == "xlstm":
      self.encoder = get_xlstm_model(outdim, self.nlayers, self.conv_sz, self.qkv_proj_sz, self.nhead)
      self.encoder_proj = nn.Linear(outdim, outdim)
      self.time_encoder = DateTimeEmbedding(self.ntoken, self.nfeat)
    else:
      raise NotImplementedError

  def extract_pad_mask(self, x_mask):
    return (x_mask.sum(-1) / x_mask.shape[-1]) == 1

  def extract_date_by_index(self, dates, index):
    return {k: v[:, index].unsqueeze(-1) for k, v in dates.items()}

  def calc_time_delta(self, dates, next_date, enc):
    prev_date = self.extract_date_by_index(dates, -1)
    prev_date_emb = enc(prev_date)
    next_date_emb = enc(next_date)
    return next_date_emb - prev_date_emb

  def encode_inputs(self, x, x_mask, dates, next_date, next_condition, src_mask, device):
    if self.nconcepts != 0 and self.concept_ones: # Baseline
      c = torch.ones((x.shape[0], self.nconcepts)).to(device)

    else:
      
      if self.seq_encoder == "transformer":
        c = self.encoder(x, self.extract_pad_mask(x_mask), dates, src_mask, device)
        
        # Time delta
        time_diff = self.calc_time_delta(dates, next_date, self.encoder.pos_encoder)
        next_cond_emb = self.condition_decoder(time_diff).squeeze(1)

      elif self.seq_encoder == "moment":
        x_padded = torch.permute(x, (0, 2, 1))
        if x_padded.shape[2] < 8: x_padded = F.pad(x_padded, (0, 8 - x_padded.shape[2]))
        c = self.encoder(x_enc = x_padded).embeddings
        c = self.encoder_proj(c)
        
        # Time delta
        time_diff = self.calc_time_delta(dates, next_date, self.time_encoder)
        next_cond_emb = self.condition_decoder(time_diff).squeeze(1)

      elif self.seq_encoder == "xlstm":
        c = self.encoder(x)
        c = self.encoder_proj(c.mean(axis = 1))

        # Time delta
        time_diff = self.calc_time_delta(dates, next_date, self.time_encoder)
        next_cond_emb = self.condition_decoder(time_diff).squeeze(1)

      else:
        raise NotImplementedError

      # Apply condition/action
      if self.condition:
        next_cond_emb = self.condition_decoder(next_cond_emb + self.condition_encoder(next_condition)).squeeze(1)

      # Apply time embedding to concept embedding
      c = c * next_cond_emb

    return c

  def forward(self, x, x_mask, dates, next_date, next_condition, src_mask, device = None):
    
    # Encode input sequence data and condition token
    c = self.encode_inputs(x, x_mask, dates, next_date, next_condition, src_mask, device)

    # Apply concept vector
    if self.nconcepts != 0:
      if x.dim() == 3:
        x[x_mask] = float('nan')
        x = torch.nan_to_num(torch.nanmean(x, 1), 1e-10) # Epsilon to be less susceptible to missing data

      # Apply lab decoder on c * next_cond_emb
      if self.apply_ct_proj: c = self.concept_proj(c)

      # Learn rates of change (values >= 0)
      if not self.concept_ones: c = self.concept_act(c)
      
      # Concept-based lab decoder
      assert x.shape == c.shape
      x = c * x
      return x, c

    else:
      c = self.seq_decoder(c)
      return c

  def inference_edit(self, edit, x, x_mask, dates, next_date, next_condition, src_mask, device = None):
    assert edit.shape[1] == self.nconcepts # Batch x Features
    assert self.nconcepts != 0 and not self.concept_ones

    # Encode input sequence data and condition token
    c = self.encode_inputs(x, x_mask, dates, next_date, next_condition, src_mask, device)

    # Apply concept vector
    if x.dim() == 3:
      x[x_mask] = float('nan')
      x = torch.nan_to_num(torch.nanmean(x, 1), 1e-10) # Epsilon to be less susceptible to missing data

    # Apply lab decoder on c * next_cond_emb
    if self.apply_ct_proj: c = self.concept_proj(c)

    # Learn rates of change (values >= 0)
    if not self.concept_ones: c = self.concept_act(c)

    # Apply edit to concepts
    c = torch.where(torch.isnan(edit), c, edit)

    # Concept-based lab decoder
    assert x.shape == c.shape
    x = c * x
    return x, c


#############################################################################################################
# Sinusoidal positional embedding adapted from https://github.com/whaleloops/TransformEHR/blob/a3031c1cd926b39077560999c5047c586027feb9/icdmodelbart.py#L1219
#
class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions, embedding_dim, padding_idx=None):
        super(SinusoidalPositionalEmbedding, self).__init__(num_positions, embedding_dim)

        if embedding_dim % 2 != 0:
            raise NotImplementedError(f"odd embedding_dim {embedding_dim} not supported")
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """Identical to the XLM create_sinusoidal_embeddings except features are not interleaved.
            The cos features are in the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )

        out.detach_()
        out.requires_grad = False
        out[:, 0 : dim // 2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))  # This line breaks for odd n_pos
        out[:, dim // 2 :] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        return out

    @torch.no_grad()
    def forward(self, input_ids, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        if use_cache:
            positions = input_ids.data.new(1, 1).fill_(seq_len - 1)  # called before slicing
        else:
            # starts at 0, ends at 1-seq_len
            positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions)


#############################################################################################################
# Time embedding adapted from https://github.com/whaleloops/TransformEHR/blob/a3031c1cd926b39077560999c5047c586027feb9/icdmodelbart.py#L1205
#
class DateTimeEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings(year) and learned positional embedding(month day hour)"""
    def __init__(self, num_positions, embedding_dim, padding_idx=None):
        super(DateTimeEmbedding, self).__init__(num_embeddings=num_positions, embedding_dim=embedding_dim, padding_idx=padding_idx)

        self.embed_year = SinusoidalPositionalEmbedding(num_positions, embedding_dim, padding_idx=padding_idx)
        self.embed_month = nn.Embedding(13, embedding_dim)
        self.embed_day = nn.Embedding(32, embedding_dim)
        self.embed_time = nn.Embedding(25, embedding_dim) # Time = hour

    def forward(self, input, use_cache=False):
        year = self.embed_year(input["year"])
        month = self.embed_month(input["month"])
        day = self.embed_day(input["day"])
        time = self.embed_time(input["time"])
        return year + month + day + time


#############################################################################################################
# Transformer encoder adapted from https://github.com/pytorch/examples/blob/main/word_language_model/model.py
#
class TransformerEncoder(nn.Transformer):
  def __init__(self, ntoken, ninp, nhead, nhid, nlayers, nout, dropout=0.5, position="time"):
    super(TransformerEncoder, self).__init__(d_model = ninp, nhead = nhead, dim_feedforward = nhid, num_encoder_layers = nlayers, num_decoder_layers = 1, batch_first = True)

    self.src_mask = None
    self.ntoken = ntoken  # max_timepoints
    self.ninp = ninp      # nfeat

    self.position = position
    if self.position == "time":
      self.pos_encoder = DateTimeEmbedding(self.ntoken, ninp) # Replace position encoding with time encoding
    else:
      self.pos_encoder = PositionalEncoding(ninp, dropout)

    self.layer_norm = nn.LayerNorm(ninp)
    self.dropout = nn.Dropout(dropout)

    self.decoder = nn.Linear(ninp, nout) #ntoken

    self.init_weights()

  def _generate_square_subsequent_mask(self, sz):
      return torch.log(torch.tril(torch.ones(sz,sz)))

  def init_weights(self):
      initrange = 0.1
      nn.init.zeros_(self.decoder.bias)
      nn.init.uniform_(self.decoder.weight, -initrange, initrange)

  def forward(self, src, src_key_padding_mask, dates=None, has_mask=True, device = None):
      if has_mask:
          if self.src_mask is None or self.src_mask.size(0) != src.shape[1]:
              mask = self._generate_square_subsequent_mask(src.shape[1]).to(device)
              self.src_mask = mask
          else:
            self.src_mask = self.src_mask.to(device)
      else:
          self.src_mask = None

      if self.position == "time": time_emb = self.pos_encoder(dates)
      else: time_emb = self.pos_encoder(src)

      src = self.dropout(src + time_emb)
      src = self.layer_norm(src)
      if src_key_padding_mask is not None: src_key_padding_mask = src_key_padding_mask.float() # Fixes the mismatch deprecated warning
      output = self.encoder(src, src_key_padding_mask = src_key_padding_mask, mask = self.src_mask)

      # Extract last token
      output = output[:, -1, :]

      output = self.decoder(output)
      return output


#############################################################################################################
# Positional encoding adapted from https://github.com/pytorch/examples/blob/main/word_language_model/model.py
#
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
