# Self-Attention and Scaled Dot-Product Attention

Self-attention is the core operation of the Transformer. It lets each position in a
sequence build a new representation of itself by mixing in information from every
other position, weighted by how relevant those positions are. The word "self"
means the queries, keys, and values all come from the same sequence, unlike
encoder-decoder attention where they come from different sequences.

## Queries, keys, and values

For each input token the model computes three vectors: a query, a key, and a
value. These are produced by multiplying the token's representation by three
learned weight matrices, W_Q, W_K, and W_V. Intuitively the query represents what
the token is looking for, the key represents what the token offers to others, and
the value is the content that will actually be passed on if the token is attended
to. The three matrices are learned during training and are different in every
layer and every head.

## Computing the attention weights

Attention scores are computed as the dot product between one token's query and
every token's key. A large dot product means the query and key point in similar
directions, so that position is treated as relevant. For a query at position i,
this produces one raw score for every position j in the sequence.

The scores for a given query are divided by the square root of the key dimension
d_k, then passed through a softmax, which turns them into a set of non-negative
weights that sum to one. The output for position i is the weighted sum of all
value vectors using these softmax weights. Positions with high weight contribute
most of their value into the result.

The complete operation is written compactly as Attention(Q, K, V) equals
softmax(Q times K transposed, divided by the square root of d_k) times V. Because
it is expressed as two matrix multiplications with a softmax in between, attention
for all positions is computed at once on a GPU, with no loop over the sequence.
The time and memory cost is proportional to the sequence length squared times the
model dimension, which becomes the main bottleneck for very long inputs.

## Why the scaling factor

The division by the square root of d_k is called scaling. If the components of the
query and key are roughly independent with zero mean and unit variance, then their
dot product over d_k dimensions has variance d_k. As d_k grows, the dot products
grow large in magnitude, which pushes the softmax into a region where one weight is
near one and the rest are near zero. In that region the softmax gradient is tiny,
so learning stalls. Dividing by the square root of d_k rescales the scores back to
unit variance and keeps training well conditioned. In the base model d_k is 64, so
the scaling factor is 8.

## Masked self-attention in the decoder

In the decoder, self-attention is masked so that a position can only attend to
earlier positions and to itself. This preserves the autoregressive property: when
predicting token t the model must not see tokens t plus 1 and beyond, because at
inference time those do not exist yet. The mask is applied by setting the
disallowed scores to negative infinity before the softmax, so their weights become
exactly zero. The encoder uses no such mask, since it is allowed to see the whole
input at once.

## Additive versus dot-product attention

An earlier alternative, additive attention, computes the score with a small
feed-forward network instead of a dot product. It performs similarly for small
d_k, but dot-product attention is much faster in practice because it maps directly
onto highly optimized matrix multiplication routines. The scaled dot-product form
is what makes the Transformer efficient enough to train at scale.
