# Positional Encoding

Attention has no built-in sense of order. If the input tokens were shuffled, the
set of query-key dot products would be exactly the same, just permuted, and the
output would be permuted the same way. A plain attention model therefore treats a
sentence as a bag of words. Word order clearly carries meaning in language, so the
Transformer must inject position information in some other way. This is the job of
positional encoding.

## Adding position to the embedding

The original Transformer adds a positional encoding vector to each token embedding
before the first layer. The encoding has the same dimension as the embedding,
d_model, so the two vectors can simply be summed. After the addition, a token's
vector carries both what the word is and where it sits in the sequence, and every
later layer can use that combined signal.

## The sinusoidal scheme

The paper uses a fixed, non-learned sinusoidal encoding. For a given position and a
given dimension of the vector, the value is a sine or cosine wave whose frequency
depends on the dimension index. Even dimensions use sine, odd dimensions use
cosine, and the wavelengths form a geometric progression from about 2 pi at the
lowest dimension up to about 10000 times 2 pi at the highest. Low dimensions
oscillate quickly and encode fine position differences; high dimensions oscillate
slowly and encode coarse position.

A useful mathematical property is that the encoding for position p plus k can be
written as a fixed linear function of the encoding for position p. This means the
model can learn to attend by relative offset, for example "three tokens back,"
using a simple linear map that does not depend on the absolute position.

## Learned alternative

The authors also tested learned positional embeddings, where each position index
has its own trainable vector, exactly like a word embedding table but indexed by
position. On their translation task the two approaches gave nearly identical BLEU.
They kept the sinusoidal version mainly because it is defined by a formula for
every position, so in principle it can extrapolate to sequences longer than any
seen during training, whereas a learned table has no entry for positions beyond
its trained range.

## Later approaches

Subsequent architectures use other schemes for the same purpose. BERT and GPT-2
use learned absolute position embeddings. Transformer-XL and T5 add a relative
position bias directly to the attention scores, so the model reasons about the
distance between two tokens rather than their absolute indices. Rotary position
embedding, or RoPE, rotates the query and key vectors by an angle proportional to
their position, which injects relative position into the dot product itself and
has become common in recent large language models. ALiBi instead adds a linear
penalty to attention scores that grows with distance. All of these methods exist
because attention on its own cannot tell where a token is.
