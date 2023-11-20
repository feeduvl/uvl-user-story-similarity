FROM python:3.9-slim-buster

WORKDIR /app
COPY . .

RUN pip3 install --no-cache-dir --upgrade pip gdown 
RUN cd data && gdown 1hlGZrC5IMJjiYPvTxgHnG85R0XK0wA_i
RUN cd data && gzip -d GoogleNews-vectors-negative300-SLIM.bin.gz

RUN pip3 install --no-cache-dir --upgrade pip -r requirements.txt
RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader wordnet
RUN python3 -m nltk.downloader omw-1.4
RUN python3 -m nltk.downloader averaged_perceptron_tagger

EXPOSE 9697
CMD [ "python3", "./app.py" ]