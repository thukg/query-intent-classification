### Project:  Classify Query Intention

### Highlights:
  - The goal of this project is to **classify user's query into four categories of intention**.
  - This model was built with **CNN, RNN (LSTM and GRU) and Word Embeddings** on **Tensorflow**.

### Task:
  - To identify the searching intention according to the query sentence typed by users in academic search engine.
  - The intention has four main categories: **topic**, **paper**, **expert** and **venue**.
  
### Data:
  - Input: **Query**
  - Output: **Category**
  - Examples:

    Query   | Category
    -----------|-----------
    what's the most popular research area in data mining? | topic
    the papers about RL published in AAAI | paper
    researchers who're experted at deep learning | expert
    what's the top conference about databases? | venue
    
### Prepare Data:
  - We generate training data using mannually defined templates using the following commands.
  - ```cd templates```
  - ```python3 scholar_simulator.py ../data/train.csv```
  
### Train:
  - Command: ```python train.py ./data/train.csv```

### Predict:
  - Command: ```python predict.py```
  
### Reference:
 - [Implement a cnn for text classification in tensorflow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)

