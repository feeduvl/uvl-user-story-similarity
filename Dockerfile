FROM python:3.10-slim-buster

WORKDIR /app
COPY . .

RUN pip3 install --no-cache-dir --upgrade pip -r requirements.txt
RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader wordnet
RUN python3 -m nltk.downloader omw-1.4
RUN python3 -m nltk.downloader averaged_perceptron_tagger

EXPOSE 9697
CMD [ "python3", "./app.py" ]