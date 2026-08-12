# What I Learned Building a Spam Classifier

This project (FastAPI + XGBoost spam detection) taught me more than how to wire up a model — it taught me how machine learning actually *thinks*. Here's what I took away.

## 1. ML is math, and models learn from numbers
The biggest lesson: **a model doesn't understand words or meaning it only understands numbers.** Every machine learning model, no matter what it does, needs its inputs turned into numbers before it can learn anything. This project finally made that click for me: "ML is maths." The classifier isn't "reading" your message; it's doing arithmetic on the numbers that represent it.

## 2. Every model has its own preprocessing
Depending on what a model works with (text, images, tables), it has its own way of getting ready. For **NLP**, that pipeline looked like:

- **Lowercasing** : so "WIN", "Win", and "win" count as the same word.
- **Removing punctuation** : a period or exclamation mark isn't a useful signal.
- **Tokenizing** : splitting a message into individual words (tokens).
- **Removing stopwords** : words like "the", "and", "is" carry almost no meaning. I learned that **not all words are useful information**; a lot of them are just noise that makes the problem harder.
- **Stemming → lemmatization** : reducing words to their base form. I started with `PorterStemmer` and upgraded to POS-aware `WordNetLemmatizer`, which tags each word (noun/verb/adjective) first so it reduces them correctly (e.g. "running" → "run").
- **Vectorizing** : converting text to numbers with `CountVectorizer` / `TfidfVectorizer`, which builds a vocabulary and scores how important each word is.

## 3. Labels are numbers too
Spam/ham aren't "text" to the model : I used `LabelEncoder` to turn "spam" and "ham" into 0 and 1. Same idea: everything becomes numbers.

## 4. Training and evaluating honestly
You don't train on your whole dataset : you split it (`train_test_split`) so you can test on data the model has *never seen*. Then you measure with real metrics: accuracy, precision, recall, and F1. Precision/recall matter more than accuracy on an imbalanced problem like spam, because "almost everything is ham" would already score high accuracy.

## 5. Models are limited by their training data
My proudest *and* most humbling moment: the model nailed the classic SMS spam ("Congratulations, you've won a FREE iPhone") but said "not spam" to a phishing-style email like "Your account has been compromised." Why? Because the model was trained on the UCI SMS dataset, which never contains email-phishing phrasings. **A model can only catch what its training data looks like.** Out-of-distribution text — new words, new styles — is exactly where it fails. That's not a bug; it's a fundamental truth about ML, and knowing it makes you a better engineer.

## 6. Deployment is part of the job
A notebook model isn't a product. I learned how to wrap it in a FastAPI app, build a simple web UI, and deploy it on Render — including real-world gotchas:

- Python version matters (3.14 vs 3.11 broke the build because no prebuilt wheels existed).
- The web server must bind to the port Render provides (`--host 0.0.0.0 --port $PORT`).
- Runtime dependencies should be trimmed, not `pip freeze`'d blindly.

## Takeaway
Machine learning feels magical until you realize it's just math on numbers — and your whole job as the developer is deciding *which* numbers matter and feeding the model the right ones. This project turned that abstract idea into something I built, broke, and understood.

##NEED FINETUNING 

