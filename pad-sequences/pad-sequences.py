import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    if max_len is None:
        max_len = max(len(seq) for seq in seqs) if seqs else 0

    padded = []
    for seq in seqs:
        seq = np.array(seq)
        if len(seq) < max_len:
            pad = np.full(max_len - len(seq), pad_value)
            seq = np.concatenate([seq, pad])
        else:
            seq = seq[:max_len]
        padded.append(seq)

    return np.array(padded)