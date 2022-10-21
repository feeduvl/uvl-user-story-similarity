# uvl-propose-acceptance-criteria-vsm

This is a microservice which takes user stories (linked with acceptance criteria) as input, computes pairwise similarity between those user stories and returns those which are similar to each other.

For the computation three different NLP techniques based on three different language models (vector space model, Wordnet and Word2vec) can be chosen from.

## REST API

TODO: OpenAPI doc

## Parameter

- `selected_technique`: Which NLP technique to use for the similarity measurement.
- `threshold`: A similarity score for two user stories above this threshold treats them to be similar and they will be contained in the output. 
- `focused_document_ids` (optional): If not set, all pairwise similarities between user stories will be calculated. If set, only similarites between user stories pairs are considered that include at least one of the IDs given in this parameter.

## Build and run docker container

- `docker build -t <name> .`
- `docker run -p 9697:9697 <name>`

## Run locally

1. If you want to use the Word2vec approache:
  - Download the Word2vec model from [OneDrive](https://drive.google.com/file/d/1hlGZrC5IMJjiYPvTxgHnG85R0XK0wA_i/view?usp=sharing) (alternatively see how it is downloaded in the Dockerfile)
2. If you do not want to use the Word2vec approach:
  - Comment out the two lines calling the `before_start_up()` function in app.py
3. Unpack the file and store it in the __data__ folder
4. Start the app 
  - via: `python3 app.py` (for debugging mode change `debug=` to `True` in the bottom of app.py)
  - alternatively via `flask --debug run`

## Run tests

- `coverage run --source=src -m pytest`
- `coverage report`