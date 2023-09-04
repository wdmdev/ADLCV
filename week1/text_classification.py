import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
from torchtext import data, datasets, vocab
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformer import TransformerClassifier, to_device

NUM_CLS = 2
VOCAB_SIZE = 50_000
SAMPLED_RATIO = 0.2
MAX_SEQ_LEN = 512

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def prepare_data_iter(sampled_ratio=0.2, batch_size=16):
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False)
    tdata, _ = datasets.IMDB.splits(TEXT, LABEL)
    # Reduce dataset size
    reduced_tdata, _ = tdata.split(split_ratio=sampled_ratio)
    # Create train and test splits
    train, test = reduced_tdata.split(split_ratio=0.8)
    print('training: ', len(train), 'test: ', len(test))
    TEXT.build_vocab(train, max_size= VOCAB_SIZE - 2)
    LABEL.build_vocab(train)
    train_iter, test_iter = data.BucketIterator.splits((train, test), 
                                                       batch_size=batch_size, 
                                                       device=to_device()
    )

    return train_iter, test_iter


def main(train_iter, test_iter, embed_dim=128, num_heads=4, num_layers=4, num_epochs=20,
         pos_enc='fixed', pool='max', dropout=0.0, fc_dim=None,
         batch_size=16, lr=1e-4, warmup_steps=625, 
         weight_decay=1e-4, gradient_clipping=1
    ):

    
    
    loss_function = nn.CrossEntropyLoss()


    model = TransformerClassifier(embed_dim=embed_dim, 
                                  num_heads=num_heads, 
                                  num_layers=num_layers,
                                  pos_enc=pos_enc,
                                  pool=pool,  
                                  dropout=dropout,
                                  fc_dim=fc_dim,
                                  max_seq_len=MAX_SEQ_LEN, 
                                  num_tokens=VOCAB_SIZE, 
                                  num_classes=NUM_CLS,
                                  )

    
    if torch.cuda.is_available():
        model = model.to('cuda')

    opt = torch.optim.AdamW(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / warmup_steps, 1.0))

    # training loop
    accuracies = []
    for e in range(num_epochs):
        # print(f'\n epoch {e}')
        model.train()
        # for batch in tqdm.tqdm(train_iter):
        for batch in train_iter:
            opt.zero_grad()
            input_seq = batch.text[0]
            batch_size, seq_len = input_seq.size()
            label = batch.label - 1
            if seq_len > MAX_SEQ_LEN:
                input_seq = input_seq[:, :MAX_SEQ_LEN]
            out = model(input_seq)
            loss = loss_function(out, label)
            loss.backward()
            # if the total gradient vector has a length > 1, we clip it back down to 1.
            if gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            opt.step()
            sch.step()

        with torch.no_grad():
            model.eval()
            tot, cor= 0.0, 0.0
            for batch in test_iter:
                input_seq = batch.text[0]
                batch_size, seq_len = input_seq.size()
                label = batch.label - 1
                if seq_len > MAX_SEQ_LEN:
                    input_seq = input_seq[:, :MAX_SEQ_LEN]
                out = model(input_seq).argmax(dim=1)
                tot += float(input_seq.size(0))
                cor += float((label == out).sum().item())
            acc = cor / tot
            accuracies.append(acc)
            # print(f'-- {"validation"} accuracy {acc:.3}')
    
    return accuracies, model


def plot_positional_encoding(name, network):
    plt.figure(figsize=(10, 6))

    # positional encoding for a embed_dim-dimensional vector 
    # pe = PositionalEncoding(embed_dim=embed_dim, max_seq_len=max_seq_len)
    pe = network.positional_encoding

    # token input: batch_size x sequence_length x embed_dim
    token = torch.zeros(1, network.max_seq_len, network.embed_dim).to('cuda')
    positions = pe(token).cpu().detach().numpy()

    # Each row in the plot corresponds to the vector we are adding 
    # to our embedding vector when the word is at that position in the sentence
    sns.heatmap(positions.squeeze(0), cmap=sns.color_palette("viridis", as_cmap=True))
    plt.xlabel('Dimension')
    plt.ylabel('Position in the sequence')
    plt.gca().invert_yaxis()
    plt.savefig(f'{name}.png')



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]= str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    set_seed(seed=1)

    epochs = 20
    batch_size = 16
    train_iter, test_iter = prepare_data_iter(sampled_ratio=SAMPLED_RATIO, 
                                            batch_size=batch_size
    )


    embed_dim_range = [64, 128, 256, 512]
    num_heads_range = [2, 4, 8, 16]
    num_layers_range = [2, 4, 6, 8]
    pos_enc_range = ['learnable', 'fixed']
    pool_range = ['mean', 'max']

    #Performance evaluations

    results = pd.DataFrame(columns=['embed_dim_64', 'embed_dim_128', 'embed_dim_256', 'embed_dim_512', 
                                    'num_heads_2', 'num_heads_4', 'num_heads_8', 'num_heads_16', 
                                    'num_layers_2', 'num_layers_4', 'num_layers_6', 'num_layers_8', 
                                    'pos_enc_fixed', 'pos_enc_learnable', 'pool_mean', 'pool_max', 'epoch'])
    
    results['epoch'] = list(range(1, epochs+1))

    # # 1. Embedding dimension
    # print('embedding dimension')
    # for embed_dim in embed_dim_range: 
    #     accuracies, model = main(train_iter=train_iter, test_iter=test_iter, embed_dim=embed_dim)
    #     name = f'embed_dim_{embed_dim}'
    #     results[name] = accuracies
    #     plot_positional_encoding(f'heat_maps/{name}', model)
    
    # # 2. Number of heads
    # print('number of heads')
    # for num_heads in num_heads_range:
    #     accuracies, model = main(train_i        self.positional_embedding = nn.Embedding(max_seq_len, embed_dim)ter=train_iter, test_iter=test_iter, num_heads=num_heads)
    #     name = f'num_heads_{num_heads}'
    #     results[name] = accuracies
    #     plot_positional_encoding(f'heat_maps/{name}', model)
    
    # # 3. Number of layers
    # print('number of layers')
    # for num_layers in num_layers_range:
    #     accuracies, model = main(train_iter=train_iter, test_iter=test_iter, num_layers=num_layers)
    #     name = f'num_layers_{num_layers}'
    #     results[name] = accuracies
    #     plot_positional_encoding(f'heat_maps/{name}', model)
    
    # 4. Positional encoding
    print('positional encoding')
    for pos_enc in pos_enc_range:
        accuracies, model = main(train_iter=train_iter, test_iter=test_iter, 
                                    pos_enc=pos_enc, num_epochs=epochs, embed_dim=1024)
        name = f'pos_enc_{pos_enc}'
        results[name] = accuracies
        plot_positional_encoding(f'heat_maps/pos_enc_embed_dim_1024_{name}', model)
    
    # # 5. Pooling
    # print('pooling')
    # for pool in pool_range:
    #     accuracies, model = main(train_iter=train_iter, test_iter=test_iter, pool=pool)
    #     name = f'pool_{pool}'
    #     results[name] = accuracies
    #     plot_positional_encoding(f'heat_maps/{name}', model)
    
    #Save results
    results.to_csv('pos_enc_embed_dim_1024_batch_results.csv', index=False)
