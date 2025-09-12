import torch

def build_vec_len_lookup(tokenizer, device=None, dtype=torch.float32):
    vocab_size = len(tokenizer)
    vec_to_angle = torch.full((vocab_size,), fill_value=-1, dtype=torch.long, device=device)
    len_to_length = torch.full((vocab_size,), fill_value=float("nan"), dtype=dtype, device=device)

    for tok_id in range(vocab_size):
        tok = tokenizer.decode([tok_id])
        if tok.startswith("VEC_"):
            try:
                angle_idx = int(tok.replace("VEC_", "").strip())
                vec_to_angle[tok_id] = angle_idx
            except ValueError:
                pass  # malformed token, skip
        elif tok.startswith("LEN_"):
            try:
                length_val = float(tok.replace("LEN_", "").strip())
                len_to_length[tok_id] = length_val
            except ValueError:
                pass

    return vec_to_angle, len_to_length

