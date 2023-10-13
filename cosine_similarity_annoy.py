import json

import numpy as np
import pandas
import requests
import re
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import annoy

tagSentences = []
sentences = []
vectors = []
similarities = []
PATCH_SIZE = 500

API_SBERT = "http://211.109.9.73:7008/api/v1/slx-sbert/encode"
API_MUSE = "http://211.109.9.73:5006/analyze"
VERTOR_SIZE = 512


def readExcel():
    excel_data_df = pandas.read_excel('./data/test_15000.xls', sheet_name='Sheet1', usecols=['Sentence'])

    tagSentences = excel_data_df['Sentence'].tolist()
    samples = []
    for sample in tagSentences:
        samples.append(re.sub('<.*?>', '', sample))

    return tagSentences, samples


def callMuse(sentences):
    data = {
        "userutterance": "",
        "intentsamples": sentences
    }
    resp = requests.post(url=API_MUSE, json=data)
    return resp


def callSBert(sentence):
    data = {
        'id': '449535cb-44b0-40a8-b91e-300193e0171b',
        'texts': [sentence]
    }
    resp = requests.post(url=API_SBERT, json=data)
    result = json.loads(resp.text)
    return result['result'][0]


def callMusePatch(sentences):
    n_patch = len(sentences) / PATCH_SIZE
    if n_patch < 1:
        n_patch = 1

    sentences_patch = np.array_split(sentences, n_patch)

    vectors_tmp = []
    for samples in sentences_patch:
        data = {
            "userutterance": "",
            "intentsamples": list(samples)
        }
        resp = requests.post(url=API_MUSE, json=data)
        result = json.loads(resp.text)
        vectors_tmp += result['vectors']

    return vectors_tmp


def callSBertPatch(sentences):
    n_patch = len(sentences) / PATCH_SIZE
    if n_patch < 1:
        n_patch = 1

    vectors_tmp = []
    sentences_patch = np.array_split(sentences, n_patch)
    for samples in sentences_patch:
        data = {
            'id': '449535cb-44b0-40a8-b91e-300193e0171b',
            'texts': list(samples)
        }
        resp = requests.post(url=API_SBERT, json=data)
        result = json.loads(resp.text)
        # vectors.append(result['result'])
        vectors_tmp += result['result']

    return vectors_tmp


if __name__ == '__main__':

    now = datetime.now()
    print("Start read data: ", now.strftime("%H:%M:%S"))
    tagSentences, sentences = readExcel()
    # r = callSBert(sentences[0])
    # callMuse(sentences)

    now = datetime.now()
    print("Start call sBert: ", now.strftime("%H:%M:%S"))
    # vectors = callSBertPatch(sentences)
    vectors = callMusePatch(sentences)

    now = datetime.now()
    print("Start add annoy: ", now.strftime("%H:%M:%S"))
    annoy_instance = annoy.AnnoyIndex(VERTOR_SIZE, 'angular')
    for annoy_idx, vector in enumerate(vectors):
        annoy_instance.add_item(annoy_idx, vector)

    annoy_instance.build(n_trees=30,  n_jobs=-1)

    indexTuples = []
    indexStr = ''

    now = datetime.now()
    print("Start filter score: ", now.strftime("%H:%M:%S"))

    for i, vector in enumerate(vectors):
        ids = annoy_instance.get_nns_by_vector(vector, 20)
        for idx in range(len(ids)):
            if i != ids[idx]:
                corresponding_vector = annoy_instance.get_item_vector(ids[idx])
                # one times call cosin_similarity function consumes a lot of  time
                matrix = cosine_similarity([vector], [corresponding_vector])
                similarity = float(matrix[0][0])

                if similarity > 0.97:
                    if i < ids[idx]:
                        indexStr = str(i) + str(ids[idx])
                    else:
                        indexStr = str(ids[idx]) + str(i)

                    if indexStr not in indexTuples:
                        indexTuples.append(indexStr)
                        similarities.append([i, ids[idx], similarity, 2])


    now = datetime.now()
    print("Start write file: ", now.strftime("%H:%M:%S"))

    originTextForm = "[Dòng {i}] {origin}"
    duplicateTextForm = "\n\t[Dòng {j}] [{score:.2f}] {dupSentence}"
    nTuples = len(similarities)
    index = 0

    # check duplicate times
    while index < nTuples:
        originIndex = index
        count = similarities[index][3]
        while ((index+1) < nTuples) and (similarities[index][0] == similarities[index+1][0]):
            count += 1
            index += 1

        similarities[originIndex][3] = count
        index += 1

    index = 0
    with open(r'./result/result_muse097.txt', 'w') as fp:
        while index < nTuples:
            if (similarities[index][3] > 3) :
                text = "\n-----------------\n" + originTextForm.format(i=similarities[index][0] + 1, origin=tagSentences[similarities[index][0]]) \
                      + duplicateTextForm.format(j=similarities[index][1] + 1, score=similarities[index][2], dupSentence=tagSentences[similarities[index][1]])

                while ((index+1) < nTuples) and (similarities[index][0] == similarities[index+1][0]):
                    text += duplicateTextForm.format(j=similarities[index+1][1] + 1, score=similarities[index+1][2], dupSentence=tagSentences[similarities[index+1][1]])
                    index += 1

                # print(text)
                fp.write(text)

            index += 1

    now = datetime.now()
    print("End write file: ", now.strftime("%H:%M:%S"))

