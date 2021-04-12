# lyrics-artist-predictor

A web-based lyrics artist predictor, predicts the artist who sings the lyrics which the user had typed.

Three artists where chosen: Frank Sinatra , Ed Sheeran , Taylor Swift , but you can also choose your favourite artists.

to train the model we collect songs for each artist from Genius.com with LyricsGenius API.

to clean the songs lyrics, we used two options:
- clean with Natural-Language tool-kit(NLTK).
- clean with spaCy.

spaCy provides the best way to do it.It provides the fastest and most accurate syntactic analysis of any NLP library released to date.

Then the model was trained to predict the artist based on two models: Random Forest classifier, and Multinomial Naive baise, with a Grid Search to find the optimal parameters.

### Clean text with cpaCy:
we used spaCy which is an open-source software library for advanced natural language processing to clean the songs lyrics through:
- Lemmatization.
- remove noise (stopwords, digit, puntuaction and single characters).
- remove empty tokens and one letter tokens.
- remove non alphabetic characters.
- remove non-Unicode characters.
<img src="images/spacy.png">

### Usage:
- clone the repo.
- install the requirements.
- you can use the terminal for prediction by running main.py with some lyrics:

  python main.py 'lyrics goes here'
- or you can use the web interface by running: python application.py then go to your browser and type: localhost/5000

<img src="images/image1.png">

<img src="images/imag2.png">



## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**








