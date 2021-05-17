<br/>
<h1 align="center">ProtTrans</h1>
<br/>

<br/>

[ProtTrans](https://github.com/agemagician/ProtTrans/) is providing **state of the art pre-trained models for proteins**. ProtTrans was trained on **thousands of GPUs from Summit** and **hundreds of Google TPUs** using various **Transformers Models**.

Have a look at our paper [ProtTrans: cracking the language of life’s code through self-supervised deep learning and high performance computing](https://www.biorxiv.org/content/10.1101/2020.07.12.199554v2) for more information about our work. 

<br/>
<p align="center">
    <img width="70%" src="https://github.com/agemagician/ProtTrans/raw/master/images/transformers_attention.png" alt="ProtTrans Attention Visualization">
</p>
<br/>


This repository will be updated regulary with **new pre-trained models for proteins** as part of supporting **bioinformatics** community in general, and **Covid-19 research** specifically through our [Accelerate SARS-CoV-2 research with transfer learning using pre-trained language modeling models](https://covid19-hpc-consortium.org/projects/5ed56e51a21132007ebf57bf) project.

Table of Contents
=================
* [ ⌛️&nbsp; Models Availability](#models)
* [ ⌛️&nbsp; Dataset Availability](#datasets)
* [ 🚀&nbsp; Usage ](#usage)
  * [ 🧬&nbsp; Feature Extraction (FE)](#feature-extraction)
  * [ 💥&nbsp; Fine Tuning (FT)](#fine-tuning)
  * [ 🧠&nbsp; Prediction](#prediction)
  * [ ⚗️&nbsp; Protein Sequences Generation ](#protein-generation)
  * [ 🧐&nbsp; Visualization ](#visualization)
  * [ 📈&nbsp; Benchmark ](#benchmark)
* [ 📊&nbsp; Expected Results  ](#results)
  * [ 🧬&nbsp; Secondary Structure Prediction (Q3) ](#q3)
  * [ 🧬&nbsp; Secondary Structure Prediction (Q8) ](#q8)
  * [ 🧬&nbsp; Membrane-bound vs Water-soluble (Q2) ](#q2)
  * [ 🧬&nbsp; Subcellular Localization (Q10) ](#q10)
* [ ❤️&nbsp; Community and Contributions ](#community)
* [ 📫&nbsp; Have a question? ](#question)
* [ 🤝&nbsp; Found a bug? ](#bug)
* [ ✅&nbsp; Requirements ](#requirements)
* [ 🤵&nbsp; Team ](#team)
* [ 💰&nbsp; Sponsors ](#sponsors)
* [ 📘&nbsp; License ](#license)
* [ ✏️&nbsp; Citation ](#citation)

<a name="models"></a>
## ⌛️&nbsp; Models Availability

|          Model                |                              Hugging Face                                  |                         Zenodo                |
| ----------------------------- | :------------------------------------------------------------------------: |:---------------------------------------------:|
| ProtT5-XL-UniRef50            |  [Download](https://huggingface.co/Rostlab/prot_t5_xl_uniref50/tree/main)  | [Download](https://zenodo.org/record/4644188) |
| ProtT5-XL-BFD                 |  [Download](https://huggingface.co/Rostlab/prot_t5_xl_bfd/tree/main)       | [Download](https://zenodo.org/record/4633924) |
| ProtT5-XXL-UniRef50           |  [Download](https://huggingface.co/Rostlab/prot_t5_xxl_uniref50/tree/main) | [Download](https://zenodo.org/record/4652717) |
| ProtT5-XXL-BFD                |  [Download](https://huggingface.co/Rostlab/prot_t5_xxl_bfd/tree/main)      | [Download](https://zenodo.org/record/4635302) |
| ProtBert-BFD                  |  [Download](https://huggingface.co/Rostlab/prot_bert_bfd/tree/main)        | [Download](https://zenodo.org/record/4633647) |
| ProtBert                      |  [Download](https://huggingface.co/Rostlab/prot_bert/tree/main)            | [Download](https://zenodo.org/record/4633691) |
| ProtAlbert                    |  [Download](https://huggingface.co/Rostlab/prot_albert/tree/main)          | [Download](https://zenodo.org/record/4633687) |
| ProtXLNet                     |  [Download](https://huggingface.co/Rostlab/prot_xlnet/tree/main)           | [Download](https://zenodo.org/record/4633987) |
| ProtElectra-Generator-BFD     |  [Download](https://huggingface.co/Rostlab/prot_electra_generator_bfd/tree/main)           | [Download](https://zenodo.org/record/4633813) |
| ProtElectra-Discriminator-BFD |  [Download](https://huggingface.co/Rostlab/prot_electra_discriminator_bfd/tree/main)           | [Download](https://zenodo.org/record/4633717) |
| ProtElectra-Generator         |     coming soon    |
| ProtElectra-Discriminator     |     coming soon    |
| ProtTXL                       |     coming soon    |
| ProtTXL-BFD                   |     coming soon    |

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
 [Embedding Section](https://github.com/agemagician/ProtTrans/tree/master/Embedding). More information coming soon.

<a name="fine-tuning"></a>
 * <b>💥&nbsp; Fine Tuning (FT):</b><br/>
 Please check:
 [Fine Tuning Section](https://github.com/agemagician/ProtTrans/tree/master/Fine-Tuning). More information coming soon.

<a name="prediction"></a>
 * <b>🧠&nbsp; Prediction:</b><br/>
 Please check:
 [Prediction Section](https://github.com/agemagician/ProtTrans/tree/master/Prediction). More information coming soon.
  
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
## 📊&nbsp; Expected Results 

<a name="q3"></a>
 * <b>🧬&nbsp; Secondary Structure Prediction (Q3):</b><br/>
 
|          Model             |       CASP12       |       TS115      |       CB513      |
| -------------------------- | :----------------: | :-------------:  | :-------------:  |
| ProtT5-XL-UniRef50         |         81         |        87        |        86        |
| ProtT5-XL-BFD              |         77         |        85        |        84        |
| ProtBert-BFD               |         76         |        84        |        83        |
| ProtBert                   |         75         |        83        |        81        |
| ProtAlbert                 |         74         |        82        |        79        |
| ProtXLNet                  |         73         |        81        |        78        |
| ProtElectra-Generator      |         73         |        78        |        76        |
| ProtElectra-Discriminator  |         74         |        81        |        79        |
| ProtTXL                    |         71         |        76        |        74        |
| ProtTXL-BFD                |         72         |        75        |        77        |

<a name="q8"></a>
 * <b>🧬&nbsp; Secondary Structure Prediction (Q8):</b><br/>
 
|          Model             |       CASP12       |       TS115      |       CB513      |
| -------------------------- | :----------------: | :-------------:  | :-------------:  |
| ProtT5-XL-UniRef50         |         70         |        77        |        74        |
| ProtT5-XL-BFD              |         66         |        74        |        71        |
| ProtBert-BFD               |         65         |        73        |        70        |
| ProtBert                   |         63         |        72        |        66        |
| ProtAlbert                 |         62         |        70        |        65        |
| ProtXLNet                  |         62         |        69        |        63        |
| ProtElectra-Generator      |         60         |        66        |        61        |
| ProtElectra-Discriminator  |         62         |        69        |        65        |
| ProtTXL                    |         59         |        64        |        59        |
| ProtTXL-BFD                |         60         |        65        |        60        |

<a name="q2"></a>
 * <b>🧬&nbsp; Membrane-bound vs Water-soluble (Q2):</b><br/>
 
|          Model             |    DeepLoc         |
| -------------------------- | :----------------: |
| ProtT5-XL-UniRef50         |         91         |
| ProtT5-XL-BFD              |         91         |
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
| ProtBert-BFD               |         74         |
| ProtBert                   |         74         |
| ProtAlbert                 |         74         |
| ProtXLNet                  |         68         |
| ProtElectra-Generator      |         59         |
| ProtElectra-Discriminator  |         70         |
| ProtTXL                    |         66         |
| ProtTXL-BFD                |         65         |

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
@article {Elnaggar2020.07.12.199554,
	author = {Elnaggar, Ahmed and Heinzinger, Michael and Dallago, Christian and Rehawi, Ghalia and Wang, Yu and Jones, Llion and Gibbs, Tom and Feher, Tamas and Angerer, Christoph and Steinegger, Martin and BHOWMIK, DEBSINDHU and Rost, Burkhard},
	title = {ProtTrans: Towards Cracking the Language of Life{\textquoteright}s Code Through Self-Supervised Deep Learning and High Performance Computing},
	elocation-id = {2020.07.12.199554},
	year = {2021},
	doi = {10.1101/2020.07.12.199554},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Computational biology and bioinformatics provide vast data gold-mines from protein sequences, ideal for Language Models taken from NLP. These LMs reach for new prediction frontiers at low inference costs. Here, we trained two auto-regressive models (Transformer-XL, XLNet) and four auto-encoder models (BERT, Albert, Electra, T5) on data from UniRef and BFD containing up to 393 billion amino acids. The LMs were trained on the Summit supercomputer using 5616 GPUs and TPU Pod up-to 1024 cores. Dimensionality reduction revealed that the raw protein LM-embeddings from unlabeled data captured some biophysical features of protein sequences. We validated the advantage of using the embeddings as exclusive input for several subsequent tasks. The first was a per-residue prediction of protein secondary structure (3-state accuracy Q3=81\%-87\%); the second were per-protein predictions of protein sub-cellular localization (ten-state accuracy: Q10=81\%) and membrane vs. water-soluble (2-state accuracy Q2=91\%). For the per-residue predictions the transfer of the most informative embeddings (ProtT5) for the first time outperformed the state-of-the-art without using evolutionary information thereby bypassing expensive database searches. Taken together, the results implied that protein LMs learned some of the grammar of the language of life. To facilitate future work, we released our models at \&lt;a href="https://github.com/agemagician/ProtTrans"\&gt;https://github.com/agemagician/ProtTrans\&lt;/a\&gt;.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2021/05/04/2020.07.12.199554},
	eprint = {https://www.biorxiv.org/content/early/2021/05/04/2020.07.12.199554.full.pdf},
	journal = {bioRxiv}
}
```
