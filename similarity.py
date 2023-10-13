import json
import requests
from sklearn.metrics.pairwise import cosine_similarity

API_SBERT = "http://211.109.9.73:7008/api/v1/slx-sbert/encode"
API_MUSE = "http://211.109.9.73:5006/analyze"


def callMuse(sentence):
    data = {
        "userutterance": "",
        "intentsamples": [sentence]
    }
    resp = requests.post(url=API_MUSE, json=data)
    result = json.loads(resp.text)
    return result['vectors'][0]


def callSBert(sentence):
    data = {
        'id': '449535cb-44b0-40a8-b91e-300193e0171b',
        'texts': [sentence]
    }
    resp = requests.post(url=API_SBERT, json=data)
    result = json.loads(resp.text)
    return result['result'][0]

if __name__ == '__main__':

    sentence_1 = 'Dirígete al canal de noticias, por favor.'
    sentence_2 = 'Muévete al canal de noticias, por favor.'

    vector_1 = callMuse(sentence_1)
    vector_2 = callMuse(sentence_2)

    # vector_1 = callSBert(sentence_1)
    # vector_2 = callSBert(sentence_2)

    matrix = cosine_similarity([vector_1], [vector_2])
    score = matrix[0][0]

    print(score)
