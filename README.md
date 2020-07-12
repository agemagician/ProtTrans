<br/>
<h1 align="center">ProtTrans</h1>
<br/>

<br/>

[ProtTrans](https://github.com/agemagician/ProtTrans/) is providing **state of the art pre-trained models for proteins**. ProtTrans was trained on **thousands of GPUs from Summit** and **hundreds of Google TPUs** using various **Transformers Models**.

Have a look at our paper [ProtTrans: cracking the language of life’s code through self-supervised deep learning and high performance computing](https://arxiv.com/) for more information about our work. 

<br/>
<p align="center">
    <img width="70%" src="https://github.com/agemagician/ProtTrans/raw/master/images/transformers_attention.png" alt="ProtTrans Attention Visualization">
</p>
<br/>


This repository will be updated regulary with **new pre-trained models for proteins** as part of supporting **bioinformatics** community in general, and **Covid-19 research** specifically through our [Accelerate SARS-CoV-2 research with transfer learning using pre-trained language modeling models](https://covid19-hpc-consortium.org/projects/5ed56e51a21132007ebf57bf) project.

Table of Contents
=================
* [ 🚀&nbsp; Usage ](#usage)
  * [ 🧬&nbsp; Feature Extraction ](#feature-extraction)
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

<a name="usage"></a>
## 🚀&nbsp; Usage  

How to use ProtTrans:

<a name="feature-extraction"></a>
 * <b>🧬&nbsp; Feature Extraction:</b><br/>
 Please check:
 [Embedding Section](https://github.com/agemagician/ProtTrans/tree/master/Embedding). More information coming soon.
 
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
| ProtBert                   |         75         |        83        |        81        |
| ProtAlbert                 |         74         |        82        |        79        |
| ProtXLNet                  |         73         |        81        |        78        |
| ProtElectra-Generator      |         73         |        78        |        76        |
| ProtElectra-Discriminator  |         74         |        81        |        79        |
| ProtTXL                    |         71         |        76        |        74        |
| ProtTXL-BFD                |         72         |        75        |        77        |
| ProtT5                     |     coming soon    |   coming soon    |   coming soon    |
| ProtBert-BFD               |     coming soon    |   coming soon    |   coming soon    |

<a name="q8"></a>
 * <b>🧬&nbsp; Secondary Structure Prediction (Q8):</b><br/>
 
|          Model             |       CASP12       |       TS115      |       CB513      |
| -------------------------- | :----------------: | :-------------:  | :-------------:  |
| ProtBert                   |         63         |        72        |        66        |
| ProtAlbert                 |         62         |        70        |        65        |
| ProtXLNet                  |         62         |        69        |        63        |
| ProtElectra-Generator      |     coming soon    |   coming soon    |   coming soon    |
| ProtElectra-Discriminator  |     coming soon    |   coming soon    |   coming soon    |
| ProtTXL                    |         59         |        64        |        59        |
| ProtTXL-BFD                |         60         |        65        |        60        |
| ProtT5                     |     coming soon    |   coming soon    |   coming soon    |
| ProtBert-BFD               |     coming soon    |   coming soon    |   coming soon    |

<a name="q2"></a>
 * <b>🧬&nbsp; Membrane-bound vs Water-soluble (Q2):</b><br/>
 
|          Model             |       DeepLoc      |
| -------------------------- | :----------------: |
| ProtBert                   |         89         |
| ProtAlbert                 |         88         |
| ProtXLNet                  |         87         |
| ProtElectra-Generator      |     coming soon    |
| ProtElectra-Discriminator  |     coming soon    |
| ProtTXL                    |         85         |
| ProtTXL-BFD                |         86         |
| ProtT5                     |     coming soon    |
| ProtBert-BFD               |     coming soon    |

<a name="q10"></a>
 * <b>🧬&nbsp; Subcellular Localization (Q10):</b><br/>
 
|          Model             |      DeepLoc      |
| -------------------------- | :----------------: |
| ProtBert                   |         74         |
| ProtAlbert                 |         74         |
| ProtXLNet                  |         68         |
| ProtElectra-Generator      |     coming soon    |
| ProtElectra-Discriminator  |     coming soon    |
| ProtTXL                    |         66         |
| ProtTXL-BFD                |         65         |
| ProtT5                     |     coming soon    |
| ProtBert-BFD               |     coming soon    |

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
 
| Ahmed Elnaggar       |      Michael Heinzinger  |  Christian Dallago | Ghalia Rihawi | Burkhard Rost |
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
| <img width=120/ src="https://cdn.shopify.com/s/files/1/1061/1924/products/Unknown_Man_Emoji_large.png?v=1571606038"> | <img width=120/ src="https://cdn.shopify.com/s/files/1/1061/1924/products/Unknown_Man_Emoji_large.png?v=1571606038"> | <img width=120/ src="https://cdn.shopify.com/s/files/1/1061/1924/products/Unknown_Man_Emoji_large.png?v=1571606038"> |

* <b>ORNL:</b><br/>

| Debsindhu Bhowmik       |
|:-------------------------:|
| <img width=120/ src="https://cdn.shopify.com/s/files/1/1061/1924/products/Unknown_Man_Emoji_large.png?v=1571606038"> |

<a name="sponsors"></a>
## 💰&nbsp; Sponsors

<!--
<div id="banner" style="overflow: hidden;justify-content:space-around;display:table-cell; vertical-align:middle; text-align:center">
  <div class="" style="max-width: 20%;max-height: 20%;display: inline-block;">
      <img width="14%" src="https://github.com/agemagician/ProtTrans/blob/master/images/1200px-Nvidia_image_logo.svg.png?raw=true" alt="nvidia logo">
  </div>

  <div class="" style="max-width: 20%;max-height: 20%;display: inline-block;">
      <img width="22%" src="https://github.com/agemagician/ProtTrans/blob/master/images/google-cloud-logo.jpg?raw=true" alt="google cloud logo">
  </div>

  <div class="" style="max-width: 20%;max-height: 20%;display: inline-block;">
      <img width="20%" src="https://github.com/agemagician/ProtTrans/blob/master/images/Oak_Ridge_National_Laboratory_logo.svg.png?raw=true" alt="ornl logo">
  </div>
  
  <div class="" style="max-width: 20%;max-height: 20%;display: inline-block;">
      <img width="12%" src="https://github.com/agemagician/ProtTrans/blob/master/images/SOFTWARE_CAMPUS_logo_cmyk.jpg?raw=true" alt="software campus logo">
  </div>
  
</div>
-->

Nvidia       |      Google  |  ORNL | Software Campus
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/agemagician/ProtTrans/blob/master/images/1200px-Nvidia_image_logo.svg.png?raw=true)  |  ![](https://github.com/agemagician/ProtTrans/blob/master/images/google-cloud-logo.jpg?raw=true)  |  ![](https://github.com/agemagician/ProtTrans/blob/master/images/Oak_Ridge_National_Laboratory_logo.svg.png?raw=true)  |  ![](https://github.com/agemagician/ProtTrans/blob/master/images/SOFTWARE_CAMPUS_logo_cmyk.jpg?raw=true)



<a name="license"></a>
## 📘&nbsp; License
The ProtTrans pretrained models are released under the under terms of the [MIT License](LICENSE).

<a name="citation"></a>
## ✏️&nbsp; Citation
If you use this code or our pretrained models for your publication, please cite the original paper:
```
@inproceedings{elnnaggar2020prottrans,
  title = {{ProtTrans}: towards cracking the language of life’s code through self-supervised deep learning and high performance computing},
  author = {Ahmed Elnaggar, Michael, .....},
  booktitle = {Arxiv},
  year = {2020},
  url = {https://arxiv/.....}
}
```
