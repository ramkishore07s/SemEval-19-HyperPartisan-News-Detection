# HyperPartisan News Detection

**News articles in this Dataset cite various sources throughout. We can leverage this information when constructing representation for sentences.**

Let representation of a given sentence `s` be `h(s)` and assume that `s` had a hyperlink to some link `k`, meaning the information `s` was obtained from source `k`. 

Then we can modify the representation as:
`SentenceRepresentation(s) = tanh(W_{k}(h(s)) + bias_{k})`

If a sentence does not have any citations, then a default Matrix is used.

An alterate approach could be to have a weightage vector for each source.

## Model:
* `U_{i}` -> Urls cited in the corresponding sentence `S_{i}`
<img src='https://raw.githubusercontent.com/ramkishore07s/SemEval-19-HyperPartisan-News-Detection/master/CredVectorModel/model.jpg?token=AUR6a5Bx8WGc9V7PvW7YPwbhA0CVRTdBks5b2rBjwA%3D%3D'/>
