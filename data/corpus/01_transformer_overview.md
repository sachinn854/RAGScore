# The Transformer Architecture

The Transformer is a neural network architecture introduced in the 2017 paper
"Attention Is All You Need" by Vaswani and colleagues at Google Brain and Google
Research. It was designed for sequence-to-sequence tasks such as machine
translation, and it replaced the recurrent and convolutional layers that dominated
earlier models with a mechanism based entirely on attention. The title of the
paper is a literal claim: no recurrence and no convolution are needed to reach
state-of-the-art results on translation.

## What came before

Before the Transformer, the leading sequence models were recurrent neural networks
such as the LSTM and the GRU, usually combined with an attention mechanism between
the encoder and decoder. These models process a sequence one token at a time,
carrying a hidden state forward. This sequential dependency has two costs. First,
it makes training slow because the computation for step t cannot begin until step
t minus 1 is finished, so the work cannot be spread across the time dimension on a
GPU. Second, long-range dependencies are hard to learn, because information from an
early token must survive many recurrent updates without being overwritten before
it can influence a later token.

Convolutional sequence models such as ByteNet and ConvS2S removed the sequential
bottleneck but still needed many stacked layers to connect distant positions,
because each convolution only looks at a small local window. The number of
operations needed to relate two positions grew with the distance between them.

## The core idea

The Transformer removes recurrence entirely. Every position in the sequence is
processed at the same time, and self-attention lets any position look directly at
any other position in a single step. This makes training highly parallel and
reduces the path length that information must travel between any two tokens to a
constant, independent of how far apart they are. Shorter paths make it easier for
gradients to flow and for the model to learn long-range structure.

## Encoder and decoder

The original architecture has two halves. The encoder reads the entire input
sequence and produces a sequence of continuous representations, one vector per
input token, each enriched with context from the whole sentence. The decoder
consumes those representations and generates the output sequence one token at a
time, feeding its own previously generated tokens back in as additional input.
This last property is called autoregressive generation.

The base model in the paper, referred to as the "base" configuration, used 6
stacked layers in the encoder and 6 in the decoder, a model dimension d_model of
512, a feed-forward inner dimension of 2048, and 8 attention heads, for about 65
million parameters. The larger "big" configuration used d_model of 1024, a
feed-forward dimension of 4096, 16 heads, and dropout 0.3, for about 213 million
parameters.

Each encoder layer contains two sublayers: a multi-head self-attention block and a
position-wise feed-forward network. Each decoder layer contains three sublayers: a
masked multi-head self-attention block, a multi-head encoder-decoder attention
block that lets the decoder look at the encoder output, and a position-wise
feed-forward network. Every sublayer is wrapped with a residual connection
followed by layer normalization, and the input embeddings are combined with
positional encodings before entering the first layer.

## Results and impact

On the WMT 2014 English-to-German translation benchmark, the big Transformer
reached a BLEU score of 28.4, more than 2 points above the best previously
published models, including ensembles. On English-to-French it reached 41.8 BLEU,
a new single-model state of the art, after training for 3.5 days on 8 GPUs, a
small fraction of the cost of the best earlier models.

The architecture generalized far beyond translation. Encoder-only descendants such
as BERT, decoder-only descendants such as the GPT family, and encoder-decoder
descendants such as T5 all keep the same attention-plus-feed-forward block and
mostly change the training objective, the depth, and the width. The Transformer is
now the standard backbone for large language models, and also for models in
vision, speech, and protein structure prediction.
