import torch, torchtext


TEXT = torchtext.data.Field(lower=True, fix_length=20, batch_first=True)
LABEL = torchtext.data.Field(sequential=False)

train, test = torchtext.datasets.IMDB.splits(TEXT, LABEL)

torchtext.data.BucketIterator.splits((train, test), batch_size=128)