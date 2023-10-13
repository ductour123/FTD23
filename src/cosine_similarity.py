import json

import numpy as np
import pandas
import requests
import re
from sklearn.metrics.pairwise import cosine_similarity
# from datetime import datetime

class SimilarityFilter:
    def __init__(self, api_sBert: str, api_muse: str, patch_size: int = 500, threshold: float = 0.97):
        self.API_SBERT = api_sBert
        self.API_MUSE = api_muse
        self.PATCH_SIZE = patch_size
        self.threshold = threshold


    def readExcel(self, data_file_path: str):
        try:
            excel_data_df = pandas.read_excel(data_file_path, engine='openpyxl', sheet_name='Sheet1', usecols=['Sentence'])
        except:
            print("-----------" + data_file_path)

        tagSentences = excel_data_df['Sentence'].tolist()
        samples = []
        for sample in tagSentences:
            if str(sample) != 'nan':
                samples.append(re.sub('<.*?>', '', sample))

        return tagSentences, samples


    def callMusePatch(self, sentences: [str], patch_size: int):
        n_patch = len(sentences) / patch_size
        if n_patch < 1:
            n_patch = 1

        sentences_patch = np.array_split(sentences, n_patch)

        vectors_tmp = []
        for samples in sentences_patch:
            data = {
                "userutterance": "",
                "intentsamples": list(samples)
            }
            resp = requests.post(url=self.API_MUSE, json=data)
            result = json.loads(resp.text)
            vectors_tmp += result['vectors']

        return vectors_tmp


    def callSBertPatch(self, sentences: [str], patch_size: int):
        n_patch = len(sentences) / patch_size
        if n_patch < 1:
            n_patch = 1

        vectors_tmp = []
        sentences_patch = np.array_split(sentences, n_patch)
        for samples in sentences_patch:
            data = {
                'id': '449535cb-44b0-40a8-b91e-300193e0171b',
                'texts': list(samples)
            }
            resp = requests.post(url=self.API_SBERT, json=data)
            result = json.loads(resp.text)
            # vectors.append(result['result'])
            vectors_tmp += result['result']

        return vectors_tmp

    def evaluate_similarity(self, sentences: [str], patch_size: int, target_api: str):
        similarities = []
        if target_api == 'muse':
            vectors = self.callMusePatch(sentences, patch_size)
        else:
            vectors = self.callSBertPatch(sentences, patch_size)

        matrix = cosine_similarity(vectors, vectors)
        index_tuples = []
        index_str = ''

        for i, matrixY in enumerate(matrix):
            for j, score in enumerate(matrixY):
                if (i != j) and (score > self.threshold):
                    if i <= j:
                        index_str = str(i + 1) + str(j + 1)
                    else:
                        index_str = str(j + 1) + str(i + 1)

                    if index_str not in index_tuples:
                        index_tuples.append(index_str)
                        similarities.append([i, j, score, 2])

        return similarities


    def filt_similarity(self, data_file_path: str, result_file_path: str):

        tagSentences, sentences = self.readExcel(data_file_path)

        similarities = self.evaluate_similarity(sentences, self.PATCH_SIZE, 'muse')
        similarities += self.evaluate_similarity(sentences, self.PATCH_SIZE, 'sBert')

        originTextForm = "[Dòng {i}] {origin}"
        duplicateTextForm = "\n\t[Dòng {j}] [{score:.2f}] {dupSentence}"
        nTuples = len(similarities)
        index = 0

        # check duplicate times
        while index < nTuples:
            originIndex = index
            count = similarities[index][3]
            while ((index + 1) < nTuples) and (similarities[index][0] == similarities[index + 1][0]):
                count += 1
                index += 1

            similarities[originIndex][3] = count
            index += 1

        found_sen_quantity = 0
        index = 0
        with open(result_file_path, 'w') as fp:
            while index < nTuples:
                if (similarities[index][3] > 3):
                    text = "\n-----------------\n" + originTextForm.format(i=similarities[index][0] + 1,
                                                                           origin=tagSentences[similarities[index][0]]) \
                           + duplicateTextForm.format(j=similarities[index][1] + 1, score=similarities[index][2],
                                                      dupSentence=tagSentences[similarities[index][1]])

                    while ((index + 1) < nTuples) and (similarities[index][0] == similarities[index + 1][0]):
                        text += duplicateTextForm.format(j=similarities[index + 1][1] + 1,
                                                         score=similarities[index + 1][2],
                                                         dupSentence=tagSentences[similarities[index + 1][1]])
                        index += 1

                    # print(text)
                    fp.write(text)
                    found_sen_quantity += 1

                index += 1

        # rate = round((found_sen_quantity/len(sentences)) * 100)
        return found_sen_quantity, len(sentences)




