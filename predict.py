from SmilesVisualizer import SmilesVisualizer
import torch
from torch._C import _propagate_and_assign_input_shapes
import torch.nn as nn
import os
import sys
from PyQt5.QtWidgets import QApplication
from torchtext.vocab import build_vocab_from_iterator
from models.Transformer import Transformer
import argparse

tokenize = lambda x : list(x)

def predict(model, input_sequence, max_length=150, PAD_token=1, SOS_token=2, EOS_token=3):
    model.eval()
    
    y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)

    num_tokens = len(input_sequence[0])

    for _ in range(max_length):
        # Get source mask
        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)
        
        pred = model(input_sequence, y_input, tgt_mask)
        
        next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
        next_item = torch.tensor([[next_item]], device=device)

        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1)

        # Stop if model predicts end of sentence
        if next_item.view(-1).item() == EOS_token or next_item.view(-1).item() == PAD_token:
            break

    return y_input.view(-1).tolist()

def protein_to_numbers(protein, protein_vocab):
    return [protein_vocab[token] for token in tokenize(protein)]

def smiles_to_string(smiles, smiles_vocab):
    return ''.join([smiles_vocab.get_itos()[word] for word in smiles])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='none')
    parser.add_argument('--vis', type=bool, default=True)
    parser.add_argument('--max', type=int, default=150)
    parser.add_argument('--pad', type=int, default=1)
    parser.add_argument('--sos', type=int, default=2)
    parser.add_argument('--eos', type=int, default=3)
    args = parser.parse_args()

    if args.input == 'none':
        print("Please provide an --input argument.")
        exit()

    root = os.path.dirname(__file__)
    protein_vocab = torch.load(os.path.join(root, 'utils/vocab/protein-vocab.pt'))
    smiles_vocab = torch.load(os.path.join(root, 'utils/vocab/smiles-vocab.pt'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer(
        src_tokens=len(protein_vocab),
        trg_tokens=len(smiles_vocab), 
        dim_model=256, 
        num_heads=8, 
        num_encoder_layers=6, 
        num_decoder_layers=6, 
        dropout_p=0.1
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(root, 'checkpoints/checkpoint.pth'), map_location=torch.device('cpu')))
    
    input = torch.tensor([protein_to_numbers(args.input, protein_vocab)], dtype=torch.long, device=device)
    result = predict(model, input, args.max, args.pad, args.sos, args.eos)
    result = smiles_to_string(result[1:-1], smiles_vocab)
    print(f"Predicted SMILES: {result}")
    
    if args.vis:
        app = QApplication(sys.argv)
        window = SmilesVisualizer(result)
        window.show()
        sys.exit(app.exec_())