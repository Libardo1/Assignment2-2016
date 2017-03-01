# Assigment 2 (2016)

This repository contains all my solutions for the [Assigment 2](http://cs224d.stanford.edu/assignment2/index.html) of the course CS224d: Deep Learning for Natural Language Processing (in the year 2016). All the code can be found in the source folder. A report for the written assignment is in the folder report.

### Requirements
* Numpy
* Matplotlib
* Tensorflow
* Pandas


## Example

```
$ python q3_RNNLM.py

929589.0 total words with 10000 uniques
929589.0 total words with 10000 uniques
Epoch 0
Training perplexity: 444.84475708
Validation perplexity: 288.029968262
Total time: 271.806652069
Epoch 1
Training perplexity: 235.130401611
Validation perplexity: 224.262451172
Total time: 295.947225094
Epoch 2
Training perplexity: 184.838226318
Validation perplexity: 197.621871948
Total time: 267.479329824
Epoch 3
Training perplexity: 157.324035645
Validation perplexity: 183.068054199
Total time: 286.485746861
Epoch 4
Training perplexity: 139.520019531
Validation perplexity: 174.463821411
Total time: 284.408274174
Epoch 5
Training perplexity: 127.013710022
Validation perplexity: 168.909576416
Total time: 300.423648119
Epoch 6
Training perplexity: 117.584762573
Validation perplexity: 165.419036865
Total time: 289.815206051
Epoch 7
Training perplexity: 110.105171204
Validation perplexity: 163.875228882
Total time: 278.592165947
Epoch 8
Training perplexity: 103.75844574
Validation perplexity: 163.170974731
Total time: 278.172949791
Epoch 9
Training perplexity: 98.6021957397
Validation perplexity: 163.926651001
Total time: 279.617377996
Epoch 10
Training perplexity: 94.2514877319
Validation perplexity: 166.024658203
Total time: 271.736068964
Epoch 11
Training perplexity: 90.4200973511
Validation perplexity: 166.484313965
=-==-==-==-==-=152.166793823
Test perplexity: 151.285949707
=-==-==-==-==-=
 
=-==-==-==-==-=
Sentence generator
Type '*end*' to break the loop
=-==-==-==-==-=
in palo alto <unk> and <unk> democracy in his family in delaware area inc. there <eos>
> *end*
The best validation perplexity is 163.170974731 and the whole
      training takes 3729.75997996(s) 
```
