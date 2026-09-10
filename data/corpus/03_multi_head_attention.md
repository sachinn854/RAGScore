# Multi-Head Attention

Multi-head attention runs several attention operations in parallel instead of one,
and then combines their results. Each parallel copy is called a head. It is the
form of attention actually used everywhere in the Transformer.

## Why more than one head

A single attention function forces the model to average many kinds of
relationships into one set of weights. One head might want to track subject-verb
agreement, another might want to link a pronoun to the noun it refers to, a third
might follow simple positional proximity, and a fourth might attend to the end of
the sentence. Collapsing all of these into one softmax distribution loses
information, because a weighted average cannot express "attend strongly to two
unrelated places for two different reasons." Multiple heads let the model attend to
different positions in different representation subspaces at the same time, and
then merge what each head found.

## How it works

In multi-head attention the query, key, and value vectors are each linearly
projected into a lower-dimensional space, once per head, using separate learned
matrices. Scaled dot-product attention is applied independently inside each head,
producing one output vector per position per head. The per-head outputs for a
position are concatenated back into a single vector and passed through one final
learned linear projection, W_O, which returns to the model dimension and lets the
heads' contributions interact.

The original base model uses 8 heads. Each head operates in a space of size
d_model divided by the number of heads, that is 512 divided by 8, or 64 dimensions
per head, which also makes d_k and d_v equal to 64. Because the per-head dimension
shrinks as the number of heads grows, the total computation and parameter cost of
multi-head attention is close to that of a single full-dimension attention head.
The benefit of multiple views comes almost for free.

## The three places attention appears

Multi-head attention is used in three distinct roles in the Transformer. In the
encoder it is self-attention over the input sequence, with every position free to
attend to every other. In the decoder it is masked self-attention over the tokens
generated so far, so a position sees only itself and earlier positions. Between
these is encoder-decoder attention, also called cross-attention, where the queries
come from the current decoder layer and the keys and values come from the output
of the final encoder layer. Cross-attention is the channel through which the
decoder reads the source sentence while producing the translation.

## What heads learn in practice

Probing studies on trained Transformers show that individual heads often
specialize. Some heads learn clear syntactic functions, such as attending from a
verb to its direct object or from a determiner to its noun. Some heads mostly
attend to the immediately previous or next token, acting like a learned
convolution. Some heads track coreference, linking pronouns to their antecedents.
And a substantial fraction of heads become nearly redundant: they can be removed
after training with little or no loss in accuracy, which has motivated later work
on head pruning and on models that use fewer key-value heads than query heads,
such as multi-query and grouped-query attention.
