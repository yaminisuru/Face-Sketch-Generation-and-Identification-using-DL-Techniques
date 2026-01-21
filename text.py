import numpy as np
emb = np.load("photo_embeddings.npy", allow_pickle=True).item()
print(list(emb.keys())[:5])  # prints first 5 photo filenames
