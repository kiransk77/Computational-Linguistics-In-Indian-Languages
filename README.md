# Computational-Linguistics-In-Indian-Languages
Information Retrieval

Constraints:
------------
1. Install all the dependencies mentioned in the requirements along with the routine ones.
2. Input data files are not provided in the zip as they are very large. 
3. Please check the path and add corresponding files in the locations for processing.

Output formats:
---------------
Q1.  Q1_<EMBEDDING><DIMENSION>_similarity_<THRESHOLD>.csv

	Example: For EMBEDDING = Cbow, DIMENSION=50 and THRESHOLD = 4
		Output file is: Q1_Cbow50_similarity_4.csv
Q2: F1-scores are printed in the notebook itself.

Q3: a. Top-100 most frequent unigrams,bigrams,trigram and quadgrams for characters,words and syllables are
	stored in three separate csv files each for characters,words and syllables.
	b. Naming convention: top100_ngram_<type>.csv where type= char/word/syl.
		Example: top100_ngram_char for characters
	c. Columns in each file: unigram, unigramfreq, bigram, bigramfreq, trigram, trigramfreq, quadgram, quadgramfreq
	
	*NOTE*: 1. quadgrams are asked for only characters.
		     2. bigrams and trigrams for words are not computed due to limitations in laptop hardware.
		     3. code for the above computationfor words is commented in the ipynb notebook.
	d. Zifian distribution graphs are printed in the notebook itself.

References:
-----------

Q2. To solve this question I have refered to various online resources which are listed below:

	https://huggingface.co/ai4bharat/indic-bert

	https://skimai.com/how-to-fine-tune-bert-for-named-entity-recognition-ner/
	
	https://www.freecodecamp.org/news/getting-started-with-ner-models-using-huggingface/
	
	https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT_only_first_wordpiece.ipynb
	
	
