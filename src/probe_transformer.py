"""Run the ethics probe set through the saved basic-Transformer checkpoint."""
import sentencepiece as spm
import torch

from common import CKPT, ethics_probe
from train_transformer import Seq2SeqTransformer, translate

sp = spm.SentencePieceProcessor(model_file=str(CKPT / "spm_joint.model"))
model = Seq2SeqTransformer(sp.get_piece_size())
model.load_state_dict(torch.load(CKPT / "transformer_best.pt"))
ethics_probe("transformer", lambda s: translate(model, sp, s))
