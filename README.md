# TripletEmbedding Criterion

This aims to reproduce the loss function used in Google's [FaceNet paper](http://arxiv.org/abs/1503.03832v1).

```lua
criterion = nn.TripletEmbeddingCriterion([alpha])
```

The cost function can be expressed as follow

```lua
loss({a, p, n}) = 1/N \sum max(0, ||a_i - p_i||^2 + alpha - ||a_i - n_i||^2)
```

where `a`, `p` and `n` are batches of the embedding of *ancore*, *positive* and *negative* samples respectively.

If the margin `alpha` is not specified, it is set to `0.2` by default.

## Test

In order to test the criterion, someone can run the [`test`](test.lua) script as

```lua
th test.lua
```

which shows how to use the criterion and checks the correctness of the gradient.
