### Project: Classify Query Intention

### Highlights:
  - This is a **multi-class text classification (sentence classification)** problem.
  - The goal of this project is to **classify Kaggle San Francisco Crime Description into 39 classes**.
  - This model was built with **CNN, RNN (LSTM and GRU) and Word Embeddings** on **Tensorflow**.

### Data: [Kaggle San Francisco Crime](https://www.kaggle.com/c/sf-crime/data)
  - Input: **Query**
  - Output: **Category**
  - Examples:

    Descript   | Category
    -----------|-----------
    GRAND THEFT FROM LOCKED AUTO|LARCENY/THEFT
    POSSESSION OF NARCOTICS PARAPHERNALIA|DRUG/NARCOTIC
    AIDED CASE, MENTAL DISTURBED|NON-CRIMINAL
    AGGRAVATED ASSAULT WITH BODILY FORCE|ASSAULT
    ATTEMPTED ROBBERY ON THE STREET WITH A GUN|ROBBERY
    
### Train:
  - Command: ```python train.py ./data/train.csv ./training_config.json
  ```

### Predict:
  - Command: ```python3 util.py```
  
### Reference:
 - [Implement a cnn for text classification in tensorflow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)

