<br/>
<h1 align="center">ProtTrans</h1>
<br/>

<br/>

[ProtTrans](https://github.com/agemagician/ProtTrans/) is providing **state of the art pre-trained models for proteins**. ProtTrans was trained on **thousands of GPUs from Summit** and **hundreds of Google TPUs** using various **Transformer models**.

Have a look at our paper [ProtTrans: cracking the language of life’s code through self-supervised deep learning and high performance computing](https://doi.org/10.1109/TPAMI.2021.3095381) for more information about our work. 

<br/>
<p align="center">
    <img width="70%" src="https://github.com/agemagician/ProtTrans/raw/master/images/transformers_attention.png" alt="ProtTrans Attention Visualization">
</p>
<br/>


This repository will be updated regulary with **new pre-trained models for proteins** as part of supporting **bioinformatics** community in general, and **Covid-19 research** specifically through our [Accelerate SARS-CoV-2 research with transfer learning using pre-trained language modeling models](https://covid19-hpc-consortium.org/projects/5ed56e51a21132007ebf57bf) project.

Table of Contents
=================
* [ ⌛️&nbsp; News](#news)
* [ 🚀&nbsp; Quick Start](#quick)
* [ ⌛️&nbsp; Models Availability](#models)
* [ ⌛️&nbsp; Dataset Availability](#datasets)
* [ 🚀&nbsp; Usage ](#usage)
  * [ 🧬&nbsp; Feature Extraction (FE)](#feature-extraction)
  * [ 🚀&nbsp; Logits extraction](#logits-extraction)
  * [ 💥&nbsp; Fine Tuning (FT)](#fine-tuning)
  * [ 🧠&nbsp; Prediction](#prediction)
  * [ ⚗️&nbsp; Protein Sequences Generation ](#protein-generation)
  * [ 🧐&nbsp; Visualization ](#visualization)
  * [ 📈&nbsp; Benchmark ](#benchmark)
* [ 📊&nbsp; Original downstream Predictions  ](#results)
* [ 📊&nbsp; Followup use-cases  ](#inaction)
* [ 📊&nbsp; Comparisons to other tools ](#comparison)
* [ ❤️&nbsp; Community and Contributions ](#community)
* [ 📫&nbsp; Have a question? ](#question)
* [ 🤝&nbsp; Found a bug? ](#bug)
* [ ✅&nbsp; Requirements ](#requirements)
* [ 🤵&nbsp; Team ](#team)
* [ 💰&nbsp; Sponsors ](#sponsors)
* [ 📘&nbsp; License ](#license)
* [ ✏️&nbsp; Citation ](#citation)


<a name="news"></a>
## ⌛️&nbsp; News
* 2022/11/18: Availability: [LambdaPP](https://embed.predictprotein.org/) offers a simple web-service to access ProtT5-based predictions and UniProt now offers to download [pre-computed ProtT5 embeddings](https://www.uniprot.org/help/embeddings) for a subset of selected organisms. 

<a name="quick"></a>
## 🚀&nbsp; Quick Start
Example for how to derive embeddings from our best-performing protein language model, ProtT5-XL-U50 (aka ProtT5); also available as [colab](https://colab.research.google.com/drive/1h7F5v5xkE_ly-1bTQSu-1xaLtTP2TnLF?usp=sharing):
```python
from transformers import T5Tokenizer, T5EncoderModel
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False).to(device)

# Load the model
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

# only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
model.full() if device=='cpu' else model.half()

# prepare your protein sequences as a list
sequence_examples = ["PRTEINO", "SEQWENCE"]

# replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

# tokenize sequences and pad up to the longest sequence in the batch
ids = tokenizer.batch_encode_plus(sequences_example, add_special_tokens=True, padding="longest")

input_ids = torch.tensor(ids['input_ids']).to(device)
attention_mask = torch.tensor(ids['attention_mask']).to(device)

# generate embeddings
with torch.no_grad():
    embedding_rpr = model(input_ids=input_ids,attention_mask=attention_mask)

# extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens ([0,:7]) 
emb_0 = embedding_repr.last_hidden_state[0,:7] # shape (7 x 1024)
# same for the second ([1,:]) sequence but taking into account different sequence lengths ([1,:8])
emb_1 = embedding_repr.last_hidden_state[1,:8] # shape (8 x 1024)

# if you want to derive a single representation (per-protein embedding) for the whole protein
emb_0_per_protein = emb_0.mean(dim=0) # shape (1024)
```


We also have a [script](https://github.com/agemagician/ProtTrans/blob/master/Embedding/prott5_embedder.py) which simplifies deriving per-residue and per-protein embeddings from ProtT5 for a given FASTA file:
```
python prott5_embedder.py --input sequences/some.fasta --output embeddings/residue_embeddings.h5
python prott5_embedder.py --input sequences/some.fasta --output embeddings/protein_embeddings.h5 --per_protein 1
```

<a name="models"></a>
## ⌛️&nbsp; Models Availability

|          Model                |                              Hugging Face                                  |                         Zenodo                | Colab |
| ----------------------------- | :------------------------------------------------------------------------: |:---------------------------------------------:|---------------------------------------------:|
| ProtT5-XL-UniRef50 (also **ProtT5-XL-U50**)            |  [Download](https://huggingface.co/Rostlab/prot_t5_xl_uniref50/tree/main)  | [Download](https://zenodo.org/record/4644188) | [**Colab**](https://colab.research.google.com/drive/1TUj-ayG3WO52n5N50S7KH9vtt6zRkdmj?usp=sharing)|
| ProtT5-XL-BFD                 |  [Download](https://huggingface.co/Rostlab/prot_t5_xl_bfd/tree/main)       | [Download](https://zenodo.org/record/4633924) |
| ProtT5-XXL-UniRef50           |  [Download](https://huggingface.co/Rostlab/prot_t5_xxl_uniref50/tree/main) | [Download](https://zenodo.org/record/4652717) |
| ProtT5-XXL-BFD                |  [Download](https://huggingface.co/Rostlab/prot_t5_xxl_bfd/tree/main)      | [Download](https://zenodo.org/record/4635302) |
| ProtBert-BFD                  |  [Download](https://huggingface.co/Rostlab/prot_bert_bfd/tree/main)        | [Download](https://zenodo.org/record/4633647) |
| ProtBert                      |  [Download](https://huggingface.co/Rostlab/prot_bert/tree/main)            | [Download](https://zenodo.org/record/4633691) |
| ProtAlbert                    |  [Download](https://huggingface.co/Rostlab/prot_albert/tree/main)          | [Download](https://zenodo.org/record/4633687) |
| ProtXLNet                     |  [Download](https://huggingface.co/Rostlab/prot_xlnet/tree/main)           | [Download](https://zenodo.org/record/4633987) |
| ProtElectra-Generator-BFD     |  [Download](https://huggingface.co/Rostlab/prot_electra_generator_bfd/tree/main)           | [Download](https://zenodo.org/record/4633813) |
| ProtElectra-Discriminator-BFD |  [Download](https://huggingface.co/Rostlab/prot_electra_discriminator_bfd/tree/main)           | [Download](https://zenodo.org/record/4633717) |


<a name="datasets"></a>
## ⌛️&nbsp; Datasets Availability
|          Dataset              |                                    Dropbox                                    |  
| ----------------------------- | :---------------------------------------------------------------------------: |
|	NEW364			|      [Download](https://www.dropbox.com/s/g49lb352ij4cnt7/NEW364.csv?dl=1)    |
|	Netsurfp2       	| [Download](https://www.dropbox.com/s/98hovta9qjmmiby/Train_HHblits.csv?dl=1)  |
|	CASP12			| [Download](https://www.dropbox.com/s/te0vn0t7ocdkra7/CASP12_HHblits.csv?dl=1) |
|	CB513			| [Download](https://www.dropbox.com/s/9mat2fqqkcvdr67/CB513_HHblits.csv?dl=1) |
|	TS115			| [Download](https://www.dropbox.com/s/68pknljl9la8ax3/TS115_HHblits.csv?dl=1) |
|	DeepLoc Train		| [Download](https://www.dropbox.com/s/vgdqcl4vzqm9as0/deeploc_per_protein_train.csv?dl=1) |
|	DeepLoc Test		| [Download](https://www.dropbox.com/s/jfzuokrym7nflkp/deeploc_per_protein_test.csv?dl=1) |

<a name="usage"></a>
## 🚀&nbsp; Usage  

How to use ProtTrans:

<a name="feature-extraction"></a>
 * <b>🧬&nbsp; Feature Extraction (FE):</b><br/>
 Please check:
 [Embedding Section](https://github.com/agemagician/ProtTrans/tree/master/Embedding). [Colab](https://colab.research.google.com/drive/1TUj-ayG3WO52n5N50S7KH9vtt6zRkdmj?usp=sharing) example for feature extraction via ProtT5-XL-U50 

<a name="logits-extraction"></a>
 * <b>🚀&nbsp; Logits Extraction:</b><br/>
 For ProtT5-logits extraction, please check:
 [VESPA logits script](https://github.com/Rostlab/VESPA#step-3-log-odds-ratio-of-masked-marginal-probabilities). 

<a name="fine-tuning"></a>
 * <b>💥&nbsp; Fine Tuning (FT):</b><br/>
 Please check:
 [Fine Tuning Section](https://github.com/agemagician/ProtTrans/tree/master/Fine-Tuning). More information coming soon.

<a name="prediction"></a>
 * <b>🧠&nbsp; Prediction:</b><br/>
 Please check:
 [Prediction Section](https://github.com/agemagician/ProtTrans/tree/master/Prediction). [Colab](https://colab.research.google.com/drive/1TUj-ayG3WO52n5N50S7KH9vtt6zRkdmj?usp=sharing) example for secondary structure prediction via ProtT5-XL-U50 and [Colab](https://colab.research.google.com/drive/1W5fI20eKLtHpaeeGDcKuXsgeiwujeczX?usp=sharing) example for subcellular localization prediction as well as differentiation between membrane-bound and water-soluble proteins via ProtT5-XL-U50.
  
<a name="protein-generation"></a>
 * <b>⚗️&nbsp; Protein Sequences Generation:</b><br/>
 Please check:
 [Generate Section](https://github.com/agemagician/ProtTrans/tree/master/Generate). More information coming soon.
 
<a name="visualization"></a>
* <b>🧐&nbsp; Visualization:</b><br/> 
Please check:
 [Visualization Section](https://github.com/agemagician/ProtTrans/tree/master/Visualization). More information coming soon.
 
<a name="benchmark"></a>
* <b>📈&nbsp; Benchmark:</b><br/> 
Please check:
 [Benchmark Section](https://github.com/agemagician/ProtTrans/tree/master/Benchmark). More information coming soon.

<a name="results"></a>
## 📊&nbsp; Original downstream Predictions 

<a name="q3"></a>
 * <b>🧬&nbsp; Secondary Structure Prediction (Q3):</b><br/>
 
|          Model             |       CASP12       |       TS115      |       CB513      |
| -------------------------- | :----------------: | :-------------:  | :-------------:  |
| ProtT5-XL-UniRef50         |         81         |        87        |        86        |
| ProtT5-XL-BFD              |         77         |        85        |        84        |
| ProtT5-XXL-UniRef50        |         79         |        86        |        85        |
| ProtT5-XXL-BFD             |         78         |        85        |        83        |
| ProtBert-BFD               |         76         |        84        |        83        |
| ProtBert                   |         75         |        83        |        81        |
| ProtAlbert                 |         74         |        82        |        79        |
| ProtXLNet                  |         73         |        81        |        78        |
| ProtElectra-Generator      |         73         |        78        |        76        |
| ProtElectra-Discriminator  |         74         |        81        |        79        |
| ProtTXL                    |         71         |        76        |        74        |
| ProtTXL-BFD                |         72         |        75        |        77        |

🆕 Predict your sequence live on [predictprotein.org](https://predictprotein.org).

<a name="q8"></a>
 * <b>🧬&nbsp; Secondary Structure Prediction (Q8):</b><br/>
 
|          Model             |       CASP12       |       TS115      |       CB513      |
| -------------------------- | :----------------: | :-------------:  | :-------------:  |
| ProtT5-XL-UniRef50         |         70         |        77        |        74        |
| ProtT5-XL-BFD              |         66         |        74        |        71        |
| ProtT5-XXL-UniRef50        |         68         |        75        |        72        |
| ProtT5-XXL-BFD             |         66         |        73        |        70        |
| ProtBert-BFD               |         65         |        73        |        70        |
| ProtBert                   |         63         |        72        |        66        |
| ProtAlbert                 |         62         |        70        |        65        |
| ProtXLNet                  |         62         |        69        |        63        |
| ProtElectra-Generator      |         60         |        66        |        61        |
| ProtElectra-Discriminator  |         62         |        69        |        65        |
| ProtTXL                    |         59         |        64        |        59        |
| ProtTXL-BFD                |         60         |        65        |        60        |

🆕 Predict your sequence live on [predictprotein.org](https://predictprotein.org).

<a name="q2"></a>
 * <b>🧬&nbsp; Membrane-bound vs Water-soluble (Q2):</b><br/>
 
|          Model             |    DeepLoc         |
| -------------------------- | :----------------: |
| ProtT5-XL-UniRef50         |         91         |
| ProtT5-XL-BFD              |         91         |
| ProtT5-XXL-UniRef50        |         89         |
| ProtT5-XXL-BFD             |         90         |
| ProtBert-BFD               |         89         |
| ProtBert                   |         89         |
| ProtAlbert                 |         88         |
| ProtXLNet                  |         87         |
| ProtElectra-Generator      |         85         |
| ProtElectra-Discriminator  |         86         |
| ProtTXL                    |         85         |
| ProtTXL-BFD                |         86         |


<a name="q10"></a>
 * <b>🧬&nbsp; Subcellular Localization (Q10):</b><br/>
 
|          Model             |    DeepLoc         |
| -------------------------- | :----------------: |
| ProtT5-XL-UniRef50         |         81         |
| ProtT5-XL-BFD              |         77         |
| ProtT5-XXL-UniRef50        |         79         |
| ProtT5-XXL-BFD             |         77         |
| ProtBert-BFD               |         74         |
| ProtBert                   |         74         |
| ProtAlbert                 |         74         |
| ProtXLNet                  |         68         |
| ProtElectra-Generator      |         59         |
| ProtElectra-Discriminator  |         70         |
| ProtTXL                    |         66         |
| ProtTXL-BFD                |         65         |


<a name="inaction"></a>
## 📊&nbsp; Use-cases 
| Level | Type  | Tool |  Task | Manuscript | Webserver |
| ----- |  ---- | -- | -- | -- | -- |
| Protein | Function | Light Attention | Subcellular localization | [Light attention predicts protein location from the language of life](https://doi.org/10.1093/bioadv/vbab035) | ([Web-server](https://embed.protein.properties/)) |
| Residue | Function | bindEmbed21 | Binding Residues | [Protein embeddings and deep learning predict binding residues for various ligand classes](https://www.nature.com/articles/s41598-021-03431-4) | (Coming soon)  |
| Residue | Function | VESPA           | Conservation & effect of Single Amino Acid Variants (SAVs) | [Embeddings from protein language models predict conservation and variant effects](https://rdcu.be/cD7q5) | (coming soon) |
| Protein | Structure | ProtTucker      | Protein 3D structure similarity prediction                 | [Contrastive learning on protein embeddings enlightens midnight zone at lightning speed](https://www.biorxiv.org/content/10.1101/2021.11.14.468528v2) |  |
| Residue | Structure | ProtT5dst       | Protein 3D structure prediction                            | [Protein language model embeddings for fast, accurate, alignment-free protein structure prediction](https://www.biorxiv.org/content/10.1101/2021.07.31.454572v1.abstract) |  |

<a name="comparison"></a>
## 📊&nbsp; Comparison to other protein language models (pLMs)
While developing the [use-cases](#inaction), we compared ProtTrans models to other protein language models, for instance the [ESM](https://github.com/facebookresearch/esm) models. To focus on the effect of changing input representaitons, the following comparisons use the same architectures on top on different embedding inputs.

|          Task/Model             |  ProtBERT-BFD      | ProtT5-XL-U50    |       ESM-1b    |       ESM-1v      | Metric | Reference |
| -------------------------- | :--------------:   | :--------------: | :-----------:   | :-----------:  | :-----------: | :-----------: |
| Subcell. loc. (setDeepLoc) |  80    | <b>86</b>    |   83        |    -         | Accuracy |  [Light-attention](https://academic.oup.com/view-large/figure/321379865/vbab035f2.tif) |
| Subcell. loc. (setHard)    |  58    | <b>65</b>    |   62        |    -         | Accuracy |  [Light-attention](https://academic.oup.com/view-large/figure/321379865/vbab035f2.tif) |
| Conservation (ConSurf-DB)  |  0.540 | <b>0.596</b> |   0.563     |    -         | MCC      | [ConsEmb](https://rdcu.be/cD7q5) | 
| Variant effect (DMS-data)  |  -     | <b>0.53</b>  |   -         |    0.49      | Spearman (Mean) | [VESPA](https://rdcu.be/cD7q5) |
| Variant effect (DMS-data)  |  -     | <b>0.53</b>  |   -         | <b>0.53</b>  | Spearman (Median) | [VESPA](https://rdcu.be/cD7q5) |
| CATH superfamily (unsup.)  |  18    | <b>64</b>    |   57        |    -         | Accuracy | [ProtTucker](https://www.biorxiv.org/content/10.1101/2021.11.14.468528v1) |
| CATH superfamily (sup.)    |  39    | <b>76</b>    |   70        |    -         | Accuracy | [ProtTucker](https://www.biorxiv.org/content/10.1101/2021.11.14.468528v1) |
| Binding residues           |  -     | <b>39</b>    |   32        |    -        | F1 | [bindEmbed21](https://www.nature.com/articles/s41598-021-03431-4) |

Important note on ProtT5-XL-UniRef50 (dubbed ProtT5-XL-U50): all performances were measured using only embeddings extracted from the encoder-side of the underlying T5 model as described [here](https://github.com/agemagician/ProtTrans/blob/master/Embedding/PyTorch/Advanced/ProtT5-XL-UniRef50.ipynb). Also, experiments were ran in half-precision mode (model.half()), to speed-up embedding generation. No performance degradation could be observed in any of the experiments when running in half-precision.

<a name="community"></a>
## ❤️&nbsp; Community and Contributions

The ProtTrans project is a **open source project** supported by various partner companies and research institutions. We are committed to **share all our pre-trained models and knowledge**. We are more than happy if you could help us on sharing new ptrained models, fixing bugs, proposing new feature, improving our documentation, spreading the word, or support our project.

<a name="question"></a>
## 📫&nbsp; Have a question?

We are happy to hear your question in our issues page [ProtTrans](https://github.com/agemagician/ProtTrans/issues)! Obviously if you have a private question or want to cooperate with us, you can always **reach out to us directly** via our [RostLab email](mailto:assistant@rostlab.org?subject=[GitHub]ProtTrans) 

<a name="bug"></a>
## 🤝&nbsp; Found a bug?

Feel free to **file a new issue** with a respective title and description on the the [ProtTrans](https://github.com/agemagician/ProtTrans/issues) repository. If you already found a solution to your problem, **we would love to review your pull request**!.

<a name="requirements"></a>
## ✅&nbsp; Requirements

For protein feature extraction or fine-tuninng our pre-trained models, [Pytorch](https://github.com/pytorch/pytorch) and [Transformers](https://github.com/huggingface/transformers) library from huggingface is needed. For model visualization, you need to install [BertViz](https://github.com/jessevig/bertviz) library.

<a name="team"></a>
## 🤵&nbsp; Team

 * <b>Technical University of Munich:</b><br/>
 
| Ahmed Elnaggar       |      Michael Heinzinger  |  Christian Dallago | Ghalia Rehawi | Burkhard Rost |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
| <img width=120/ src="https://github.com/agemagician/ProtTrans/blob/master/images/ElnaggarAhmend.jpg?raw=true"> | <img width=120/ src="https://github.com/agemagician/ProtTrans/blob/master/images/MichaelHeinzinger-2.jpg?raw=true"> | <img width=120/ src="https://github.com/agemagician/ProtTrans/blob/master/images/christiandallago.png?raw=true"> | <img width=120/ src="https://github.com/agemagician/ProtTrans/blob/master/images/female.png?raw=true"> | <img width=120/ src="https://github.com/agemagician/ProtTrans/blob/master/images/B.Rost.jpg?raw=true"> |

 * <b>Med AI Technology:</b><br/>

| Yu Wang       |
|:-------------------------:|
| <img width=120/ src="https://github.com/agemagician/ProtTrans/blob/master/images/yu-wang.jpeg?raw=true"> |

* <b>Google:</b><br/>

| Llion Jones       |
|:-------------------------:|
| <img width=120/ src="https://github.com/agemagician/ProtTrans/blob/master/images/Llion-Jones.jpg?raw=true"> |

* <b>Nvidia:</b><br/>

| Tom Gibbs       | Tamas Feher | Christoph Angerer |
|:-------------------------:|:-------------------------:|:-------------------------:|
| <img width=120/ src="https://github.com/agemagician/ProtTrans/blob/master/images/Tom-Gibbs.png?raw=true"> | <img width=120/ src="https://github.com/agemagician/ProtTrans/blob/master/images/Tamas-Feher.jpeg?raw=true"> | <img width=120/ src="https://github.com/agemagician/ProtTrans/blob/master/images/Christoph-Angerer.jpg?raw=true"> |

* <b>Seoul National University:</b><br/>

| Martin Steinegger       |
|:-------------------------:|
| <img width=120/ src="https://github.com/agemagician/ProtTrans/raw/master/images/Martin-Steinegger.png"> |


* <b>ORNL:</b><br/>

| Debsindhu Bhowmik       |
|:-------------------------:|
| <img width=120/ src="https://github.com/agemagician/ProtTrans/blob/master/images/Debsindhu-Bhowmik.jpg?raw=true"> |

<a name="sponsors"></a>
## 💰&nbsp; Sponsors

<!--
<div id="banner" style="overflow: hidden;justify-content:space-around;display:table-cell; vertical-align:middle; text-align:center">
  <div class="" style="max-width: 20%;max-height: 20%;display: inline-block;">
      <img width="14%" src="https://github.com/agemagician/ProtTrans/blob/master/images/1200px-Nvidia_image_logo.svg.png?raw=true" alt="nvidia logo">
  </div>

  <div class="" style="max-width: 20%;max-height: 20%;display: inline-block;">
      <img width="22%" src="https://github.com/agemagician/ProtTrans/blob/master/images/Google-Logo.jpg?raw=true" alt="google cloud logo">
  </div>

  <div class="" style="max-width: 20%;max-height: 20%;display: inline-block;">
      <img width="20%" src="https://github.com/agemagician/ProtTrans/blob/master/images/Oak_Ridge_National_Laboratory_logo.svg.png?raw=true" alt="ornl logo">
  </div>
  
  <div class="" style="max-width: 20%;max-height: 20%;display: inline-block;">
      <img width="12%" src="https://github.com/agemagician/ProtTrans/blob/master/images/SOFTWARE_CAMPUS_logo_cmyk.jpg?raw=true" alt="software campus logo">
  </div>
  
</div>
-->

Nvidia       |      Google  |      Google  | ORNL | Software Campus
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/agemagician/ProtTrans/blob/master/images/1200px-Nvidia_image_logo.svg.png?raw=true) | ![](https://github.com/agemagician/ProtTrans/blob/master/images/google-cloud-logo.jpg?raw=true) | ![](https://github.com/agemagician/ProtTrans/blob/master/images/tfrc.png?raw=true) | ![](https://github.com/agemagician/ProtTrans/blob/master/images/Oak_Ridge_National_Laboratory_logo.svg.png?raw=true) | ![](https://github.com/agemagician/ProtTrans/blob/master/images/SOFTWARE_CAMPUS_logo_cmyk.jpg?raw=true)

<a name="license"></a>
## 📘&nbsp; License
The ProtTrans pretrained models are released under the under terms of the [Academic Free License v3.0 License](https://choosealicense.com/licenses/afl-3.0/).

<a name="citation"></a>
## ✏️&nbsp; Citation
If you use this code or our pretrained models for your publication, please cite the original paper:
```
@ARTICLE
{9477085,
author={Elnaggar, Ahmed and Heinzinger, Michael and Dallago, Christian and Rehawi, Ghalia and Yu, Wang and Jones, Llion and Gibbs, Tom and Feher, Tamas and Angerer, Christoph and Steinegger, Martin and Bhowmik, Debsindhu and Rost, Burkhard},
journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
title={ProtTrans: Towards Cracking the Language of Lifes Code Through Self-Supervised Deep Learning and High Performance Computing},
year={2021},
volume={},
number={},
pages={1-1},
doi={10.1109/TPAMI.2021.3095381}}
```
