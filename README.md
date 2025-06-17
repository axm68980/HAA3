🌟 University of Central Missouri
🎓 Assignment Overview
This assignment explores Natural Language Processing and deep learning concepts from Chapters 7 and 9. The key topics include:

Recurrent Neural Networks (RNN) for text generation
NLP preprocessing: tokenization, stopword removal, and stemming
Named Entity Recognition using SpaCy
Scaled dot-product attention mechanism
Sentiment analysis with HuggingFace Transformers
All tasks were implemented using Python and Google Colab notebooks. Code is fully commented, outputs are demonstrated, and short theoretical answers are included

🧑‍💻 Tasks Breakdown
✍️ Q1: Text Generation using LSTM
Task: Train an LSTM model to predict the next character in a text.
Dataset: Custom small literary text.
Steps:
Convert text to character-level sequences.
One-hot encode inputs.
Build LSTM model using tf.keras.layers.LSTM().
Sample output text using temperature scaling.
Temperature: Controls randomness (low = deterministic, high = creative).
Output: Model-generated sample text shown in video.
🔢 Q2: NLP Preprocessing Pipeline
Sentence: "NLP techniques are used in virtual assistants like Alexa and Siri."
Steps:
Tokenize the sentence.
Remove stopwords.
Apply stemming using NLTK.
Results:
List of all tokens
Tokens after stopword removal
Final stemmed tokens
💡 Q3: Named Entity Recognition with SpaCy
Input Sentence: "Barack Obama served as the 44th President..."
Tool: SpaCy's pre-trained NLP pipeline
Output:
Entity Text
Entity Label (e.g., PERSON, DATE)
Start and End Character Positions
🔄 Q4: Scaled Dot-Product Attention
Input: Q, K, V matrices
Steps:
Compute dot product QKᵀ
Scale using √d
Apply softmax to get attention weights
Multiply weights with V
Output:
Attention weights matrix
Final result matrix
🏃 Q5: Sentiment Analysis with HuggingFace
Input: "Despite the high price, the performance of the new MacBook is outstanding."
Library: transformers from HuggingFace
Model: Pre-trained pipeline
Output:
Sentiment Label: POSITIVE
Confidence Score: e.g., 0.9987
🚀 Technologies Used
Python
TensorFlow / Keras
NLTK
SpaCy
HuggingFace Transformers
NumPy
Google Colab
