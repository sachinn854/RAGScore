# Feed-Forward Networks, Residuals, and Layer Normalization

Every Transformer layer pairs its attention sublayer with a position-wise
feed-forward network, and wraps both sublayers in residual connections and layer
normalization. These components are less discussed than attention but are
essential to making deep Transformers trainable.

## The position-wise feed-forward network

The feed-forward network is applied to each position separately and identically. It
consists of two linear transformations with a nonlinear activation between them.
In the base model the inner layer has dimension 2048 while the input and output
have dimension 512, so the network expands the representation by a factor of four
and then projects it back down. The "big" model uses an inner dimension of 4096.
The original paper uses a ReLU activation; many later models use GELU, and some
use gated variants such as SwiGLU, which change the exact formula but keep the
expand-then-contract shape.

The feed-forward network holds a large share of the model's parameters, roughly
two thirds in a standard block. Its role is complementary to attention: attention
moves information sideways between positions, and the feed-forward network then
transforms each position's mixed representation on its own, adding nonlinear
capacity. Recent interpretability work also describes the feed-forward layers as
key-value memories that store factual associations learned during training.

## Residual connections

Around every sublayer, both attention and feed-forward, the Transformer adds a
residual connection: the output is the sublayer's result added to its input. If
the sublayer is written as a function F, the block computes x plus F of x rather
than just F of x. Residual connections give gradients a direct, unobstructed path
back through the network, which is what makes it possible to stack dozens or
hundreds of layers without the gradient vanishing or exploding. They also let a
layer default to doing nothing, by driving F toward zero, which stabilizes early
training.

## Layer normalization

After the residual addition in the original design, layer normalization is
applied. Layer normalization computes the mean and variance across the feature
dimension for each position independently, normalizes that vector to zero mean and
unit variance, and then rescales and shifts it with two learned parameter vectors,
gamma and beta. Unlike batch normalization, it does not use statistics from other
examples in the batch, so it behaves identically during training and inference and
works naturally with variable-length sequences and small batches.

## Pre-norm versus post-norm

The original paper places normalization after the residual addition, a design
called post-norm. Post-norm models are sensitive to initialization and usually
need a learning-rate warmup to train stably. Most modern implementations move the
normalization to before each sublayer, called pre-norm, so the residual path
carries an un-normalized signal straight through the stack. Pre-norm makes very
deep models much easier to optimize and often removes the need for warmup, at a
small cost in final quality that is usually recovered by other means.

## Dropout

Dropout is applied to the output of each sublayer before it is added back to the
residual, and also to the sum of the token embeddings and positional encodings.
The base model uses a dropout rate of 0.1, and the "big" model uses 0.3 for
English-to-German. Together with label smoothing, dropout is the main
regularizer in the original training recipe.
