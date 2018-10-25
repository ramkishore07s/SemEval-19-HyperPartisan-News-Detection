# HyperPartisan News Detection

**News articles in this Dataset cite various sources throughout. We can leverage this information when constructing representation for sentences.**

Let representation of a given sentence `s` be `h(s)` and assume that `s` had a hyperlink to some link `k`, meaning the information `s` was obtained from source `k`. 

Then we can modify the representation as:
`SentenceRepresentation(s) = tanh(W_{k}(h(s)) + bias_{k})`

If a sentence does not have any citations, then a default Matrix is used.

An alterate approach could be to have a weightage vector for each source.
