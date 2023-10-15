# NSFW Content Detection

__Based on the repository [TextClassifiers](https://github.com/datnnt1997/textclassifiers) by [datnnt1997](https://github.com/datnnt1997).__
Textclassifiers: Collection of Text Classification/Document Classification/Sentence Classification/Sentiment Analysis models for PyTorch

## Prerequisites
Install dependencies:  
`pip3 install -r requirements.txt`

## Training
`python3 run.py --mode train --config configs/fasttext_config.yaml`

## Models
- <b>FastText</b> released with the paper [Bag of tricks for efficient text classification](https://arxiv.org/abs/1607.01759) by Joulin, Armand, et al.
- <b>TextRNN</b>
- <b>TextCNN</b> released with the paper [Convolutional neural networks for sentence classification](https://arxiv.org/abs/1408.5882) by Kim, Yoon.
- <b>RCNN</b> released with the paper [Recurrent convolutional neural networks for text classification](http://zhengyima.com/my/pdfs/Textrcnn.pdf) by Lai, Siwei, et al.
- <b>LSTM + Attention</b> released with the paper [Text classification research with attention-based recurrent neural networks](https://pdfs.semanticscholar.org/7ac1/e870f767b7d51978e5096c98699f764932ca.pdf) by Du, Changshun, and Lei Huang.
- <b>Transformer</b> released with the paper [Attention is all you need](https://user.phil.hhu.de/~cwurm/wp-content/uploads/2020/01/7181-attention-is-all-you-need.pdf) by Vaswani, Ashish, et al.
## Results

### Initial results on the test set

<table>
  <tr>
    <th rowspan=2>S. No.</th>
    <th rowspan=2>Model</th>
    <th colspan=2>Training</th>
    <th colspan=4>Test</th>
  </tr>
  <tr>
    <th>Loss</td>
    <th>F1</td>
    <th>Loss</td>
    <th>Precision</td>
    <th>Recall</td>
    <th>F1</td>
  </tr>
  <tr>
    <td>1</td>
    <td>TextRNN</td>
    <td>0.097278</td>
    <td>0.4952</td>
    <td>0.098633</td>
    <td>0.28</td>
    <td>0.01</td>
    <td>0.01</td>
  </tr>
  <tr>
    <td>2</td>
    <td>TextCNN</td>
    <td>0.082481</td>
    <td>0.5407</td>
    <td>0.092145</td>
    <td>0.32</td>
    <td>0.06</td>
    <td>0.1</td>
  </tr>
  <tr>
    <td>3</td>
    <td>LSTM Attention</td>
    <td>0.084443</td>
    <td>0.498</td>
    <td>0.088827</td>
    <td>0.31</td>
    <td>0.02</td>
    <td>0.04</td>
  </tr>
  <tr>
    <td>4</td>
    <td>RCNN</td>
    <td>0.071199</td>
    <td>0.5568</td>
    <td>0.086385</td>
    <td>0.43</td>
    <td>0.02</td>
    <td>0.04</td>
  </tr>
  <tr>
    <td>5</td>
    <td>FastText</td>
    <td>0.096211</td>
    <td>0.4972</td>
    <td>0.099686</td>
    <td>0.33</td>
    <td>0.01</td>
    <td>0.02</td>
  </tr>
  <tr>
    <td>6</td>
    <td>Transformer</td>
    <td>0.09527</td>
    <td>0.501</td>
    <td>0.097861</td>
    <td>0.33</td>
    <td>0.01</td>
    <td>0.03</td>
  </tr>
</table>

### Results on balanced dataset (under sampling)

<table>
  <tr>
    <th rowspan=2>S. No.</th>
    <th rowspan=2>Model</th>
    <th colspan=2>Training</th>
    <th colspan=4>Test</th>
  </tr>
  <tr>
    <th>Loss</th>
    <th>F1</th>
    <th>Loss</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1</th>
  </tr>
  <tr>
    <td>1</td>
    <td>TextRNN</td>
    <td>0.536789</td>
    <td>0.7388</td>
    <td>0.572004</td>
    <td>0.71</td>
    <td>0.71</td>
    <td>0.71</td>
  </tr>
  <tr>
    <td>2</td>
    <td>TextCNN</td>
    <td>0.176651</td>
    <td>0.9328</td>
    <td>0.86567</td>
    <td>0.64</td>
    <td>0.78</td>
    <td>0.71</td>
  </tr>
  <tr>
    <td>3</td>
    <td>LSTM Attention</td>
    <td>0.537625</td>
    <td>0.7379</td>
    <td>0.590331</td>
    <td>0.7</td>
    <td>0.7</td>
    <td>0.7</td>
  </tr>
  <tr>
    <td>4</td>
    <td>RCNN</td>
    <td>0.442955</td>
    <td>0.7994</td>
    <td>0.560319</td>
    <td>0.75</td>
    <td>0.67</td>
    <td>0.71</td>
  </tr>
  <tr>
    <td>5</td>
    <td>FastText</td>
    <td>0.659113</td>
    <td>0.6</td>
    <td>0.647117</td>
    <td>0.67</td>
    <td>0.6</td>
    <td>0.63</td>
  </tr>
  <tr>
    <td>6</td>
    <td>Transformer</td>
    <td>0.479358</td>
    <td>0.773</td>
    <td>0.612778</td>
    <td>0.71</td>
    <td>0.68</td>
    <td>0.7</td>
  </tr>
</table>


### Results on balanced dataset (over sampling)

<table>
  <tr>
    <th rowspan=2>S. No.</th>
    <th rowspan=2>Model</th>
    <th colspan=2>Training</th>
    <th colspan=4>Test</th>
  </tr>
  <tr>
    <th>Loss</th>
    <th>F1</th>
    <th>Loss</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1</th>
  </tr>
  <tr>
    <td>1</td>
    <td>TextRNN</td>
    <td>0.519968</td>
    <td>0.7486</td>
    <td>0.600619</td>
    <td>0.7</td>
    <td>0.68</td>
    <td>0.69</td>
  </tr>
  <tr>
    <td>2</td>
    <td>TextCNN</td>
    <td>0.375668</td>
    <td>0.8312</td>
    <td>0.887177</td>
    <td>0.79</td>
    <td>0.46</td>
    <td>0.58</td>
  </tr>
  <tr>
    <td>3</td>
    <td>LSTM Attention</td>
    <td>0.574681</td>
    <td>0.7066</td>
    <td>0.594618</td>
    <td>0.71</td>
    <td>0.65</td>
    <td>0.68</td>
  </tr>
  <tr>
    <td>4</td>
    <td>RCNN</td>
    <td>0.536011</td>
    <td>0.734</td>
    <td>0.559934</td>
    <td>0.74</td>
    <td>0.68</td>
    <td>0.71</td>
  </tr>
  <tr>
    <td>5</td>
    <td>FastText</td>
    <td>0.73637</td>
    <td>0.5292</td>
    <td>0.683402</td>
    <td>0.65</td>
    <td>0.43</td>
    <td>0.52</td>
  </tr>
  <tr>
    <td>6</td>
    <td>Transformer</td>
    <td>0.530028</td>
    <td>0.7354</td>
    <td>0.60349</td>
    <td>0.71</td>
    <td>0.69</td>
    <td>0.7</td>
  </tr>
</table>


## References
[1] Joulin, Armand, Edouard Grave, and Piotr Bojanowski Tomas Mikolov. "Bag of Tricks for Efficient Text Classification." EACL 2017 (2017): 427.

[2] Kim, Yoon. "Convolutional Neural Networks for Sentence Classification." Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2014.

[3] Lai, Siwei, et al. "Recurrent convolutional neural networks for text classification." In Proc. Conference of the Association for the Advancement of Artificial Intelligence (AAAI). 2015.

[4] Du, Changshun, and Lei Huang. "Text classification research with attention-based recurrent neural networks." International Journal of Computers Communications & Control 13.1 (2018): 50-61.

[5] Vaswani, Ashish, et al. "Attention is all you need." Proceedings of the 31st International Conference on Neural Information Processing Systems. Curran Associates Inc., 2017.
