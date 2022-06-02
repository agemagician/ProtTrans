Important Notes:
----------------
0. The notebooks are only for illustrative purposes; mind that ProtT5 outperformed ProtBert in all our benchmarks [use-cases](https://github.com/agemagician/ProtTrans#-use-cases)
1. These are examples for fine-tuning the whole pretrained models, and not feature extraction.
2. If you are intersted on using feature extraction as mentioned on the paper, please refer to the [embeding section](https://github.com/agemagician/ProtTrans/tree/master/Embedding) .
3. You must not freeze the pretrained models, otherwise, you will get worse results.
4. Applying hyperparameter search (learning rate, seed, number of epochs, etc.) should give you better results than the mentioned results.

Update:
----------------
The notebook `protBERT-BFD-lightning-multitasks.ipynb` was kindly provided by @ratthachat in this [issue](https://github.com/agemagician/ProtTrans/issues/74#issuecomment-1120174837). It contains updates/fixes to our previous fine-tuning examples . Specifically, it shows how to fine-tune ProtBERT-BFD on the prediction of subcellular localization as well as differentiation between membrane-bound and soluble proteins (multi-task fine-tuning).
