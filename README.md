# small_lm


## Dataset 

* There are two train Train datasets one contains 10M words and other contain 100M words. 

* Validation datasets and train datasets contains approzimately 10M words 

* There are 6 types of text files in each dataset 
    * bnc_spoken
    * childes
    * gutenberg
    * open_subtitles
    * simple_wiki
    * switchboard

* all these files can be cleaned by file clean.py which contain all the functions to clean these file

--- 

## Building Tokenizers 

* Using BPE algorithms to train tokenizers From scratch with a vocabulary size of 32000 
* Separate Tokenizers is build for train_10M and train_100M datasets 
* To build the dataset run all the cells from build_tokenizers.ipynb
    * This will first create a cleaned dataset 
    * Then using the BPE algorithms creates a tokenizer for both these datasets