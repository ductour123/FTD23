import json
import os.path

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
            # excel_data_df = pandas.read_excel(data_file_path, engine='openpyxl', sheet_name=0, usecols=['Sentence'])
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

        # sort list similarities after combine the results of Muse and sBert
        # sorted(similarities, key=lambda x: x[0])

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
                    text = "\n-----------------\n" + originTextForm.format(i=similarities[index][0] + 2,
                                                                           origin=tagSentences[similarities[index][0]]) \
                           + duplicateTextForm.format(j=similarities[index][1] + 2, score=similarities[index][2],
                                                      dupSentence=tagSentences[similarities[index][1]])

                    while ((index + 1) < nTuples) and (similarities[index][0] == similarities[index + 1][0]):
                        text += duplicateTextForm.format(j=similarities[index + 1][1] + 2,
                                                         score=similarities[index + 1][2],
                                                         dupSentence=tagSentences[similarities[index + 1][1]])
                        index += 1

                    fp.write(text)
                    found_sen_quantity += 1

                index += 1

        return found_sen_quantity, len(sentences)


    def filt_similarity_excel(self, data_file_path: str, result_file_path: str):
        """
        data_file_path: file .xlsx
        result_file_path: file .xlsx
        """

        tagSentences, sentences = self.readExcel(data_file_path)

        similarities = self.evaluate_similarity(sentences, self.PATCH_SIZE, 'muse')
        similarities += self.evaluate_similarity(sentences, self.PATCH_SIZE, 'sBert')

        # sort list similarities after combine the results of Muse and sBert
        # sorted(similarities, key=lambda x: x[0])

        originTextForm = "[Row {i}] {origin} \n"
        duplicateTextForm = "[Row {j}] {dupSentence} \n"
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

        sentence_lst = []
        fuzzies_lst = []

        while index < nTuples:
            if (similarities[index][3] > 3):
                origin = originTextForm.format(i=similarities[index][0] + 2, origin=tagSentences[similarities[index][0]])
                sentence_lst.append(origin)

                fuzzies = duplicateTextForm.format(j=similarities[index][1] + 2, dupSentence=tagSentences[similarities[index][1]])
                fuzzies_lst.append(fuzzies)

                while ((index + 1) < nTuples) and (similarities[index][0] == similarities[index + 1][0]):
                    fuzzies = duplicateTextForm.format(j=similarities[index + 1][1] + 2, dupSentence=tagSentences[similarities[index + 1][1]])
                    fuzzies_lst.append(fuzzies)
                    sentence_lst.append("")

                    index += 1

                # fuzzies_lst.append(fuzzies)
                found_sen_quantity += 1

            index += 1

        # write fuzzies file excel
        data = {
            'Sentence': sentence_lst,
            'Fuzzies': fuzzies_lst
        }

        df = pandas.DataFrame(data, dtype=str)

        if os.path.exists(result_file_path):
            os.remove(result_file_path)

        # df.to_excel(result_file_path, sheet_name='High Fuzzy Matches', index=False, engine='openpyxl')
        writer = pandas.ExcelWriter(result_file_path, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='High Fuzzy Matches', index=False)
        workbook = writer.book
        worksheet = writer.sheets['High Fuzzy Matches']
        # worksheet.set_column('A:B', 80)
        cell_format_0 = workbook.add_format({'text_wrap': True})
        worksheet.set_column(0, 0, 80, cell_format_0)
        cell_format_1 = workbook.add_format({'text_wrap': True, 'font_color': 'red'})
        worksheet.set_column(1, 1, 80, cell_format_1)

        writer.save()

        return found_sen_quantity, len(sentences)


    def filt_similarity_excel_2(self, data_file_path: str, result_file_path: str):
        """
        data_file_path: file .xlsx
        result_file_path: file .xlsx
        """

        tagSentences, sentences = self.readExcel(data_file_path)

        similarities = self.evaluate_similarity(sentences, self.PATCH_SIZE, 'muse')
        similarities += self.evaluate_similarity(sentences, self.PATCH_SIZE, 'sBert')

        # sort list similarities after combine the results of Muse and sBert
        # sorted(similarities, key=lambda x: x[0])

        originTextForm = "[Row {i}] {origin} \n"
        duplicateTextForm = " -[Row {j}] {dupSentence} \n"
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

        sentence_lst = []
        fuzzies_lst = []

        while index < nTuples:
            if (similarities[index][3] > 3):
                origin = originTextForm.format(i=similarities[index][0] + 2, origin=tagSentences[similarities[index][0]])
                sentence_lst.append(origin)

                fuzzies = duplicateTextForm.format(j=similarities[index][1] + 2, dupSentence=tagSentences[similarities[index][1]])

                while ((index + 1) < nTuples) and (similarities[index][0] == similarities[index + 1][0]):
                    fuzzies += duplicateTextForm.format(j=similarities[index + 1][1] + 2, dupSentence=tagSentences[similarities[index + 1][1]])
                    index += 1

                fuzzies_lst.append(fuzzies)
                found_sen_quantity += 1

            index += 1

        # write fuzzies file excel
        data = {
            'Sentence': sentence_lst,
            'Fuzzies': fuzzies_lst
        }

        df = pandas.DataFrame(data, dtype=str)

        if os.path.exists(result_file_path):
            os.remove(result_file_path)

        writer = pandas.ExcelWriter(result_file_path, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='High Fuzzy Matches', index=False)
        workbook = writer.book
        worksheet = writer.sheets['High Fuzzy Matches']
        cell_format_0 = workbook.add_format({'text_wrap': True})
        worksheet.set_column(0, 0, 80, cell_format_0)
        cell_format_1 = workbook.add_format({'text_wrap': True, 'font_color': 'red'})
        worksheet.set_column(1, 1, 80, cell_format_1)

        writer.save()

        return found_sen_quantity, len(sentences)


    def filt_similarity_excel_3(self, data_file_path: str, result_file_path: str):
        """
        data_file_path: file .xlsx
        result_file_path: file .xlsx
        """

        tagSentences, sentences = self.readExcel(data_file_path)

        similarities = self.evaluate_similarity(sentences, self.PATCH_SIZE, 'muse')
        similarities += self.evaluate_similarity(sentences, self.PATCH_SIZE, 'sBert')

        # sort list similarities after combine the results of Muse and sBert
        # sorted(similarities, key=lambda x: x[0])

        duplicateTextForm = " -[Row {j}] {dupSentence} \n"
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

        # read origin data file
        data_df = pandas.read_excel(data_file_path, engine='openpyxl', sheet_name='Sheet1')
        fuzzies_lst = [''] * len(data_df.index)

        found_sen_quantity = 0
        index = 0

        while index < nTuples:
            if (similarities[index][3] > 3):

                fuzzies = duplicateTextForm.format(j=similarities[index][1] + 2, dupSentence=tagSentences[similarities[index][1]])

                while ((index + 1) < nTuples) and (similarities[index][0] == similarities[index + 1][0]):
                    fuzzies += duplicateTextForm.format(j=similarities[index + 1][1] + 2, dupSentence=tagSentences[similarities[index + 1][1]])
                    index += 1

                fuzzies_lst[similarities[index][0]] = fuzzies
                found_sen_quantity += 1

            index += 1

        # write data file
        origin_column_n = len(data_df.columns)
        data_df.insert(loc=origin_column_n, column='Fuzzies', value=fuzzies_lst)

        if os.path.exists(result_file_path):
            os.remove(result_file_path)

        writer = pandas.ExcelWriter(result_file_path, engine='xlsxwriter')
        data_df.to_excel(writer, sheet_name='High Fuzzy Matches', index=False)
        workbook = writer.book
        worksheet = writer.sheets['High Fuzzy Matches']

        cell_format = workbook.add_format({'text_wrap': True})
        function_idx = data_df.columns.get_loc("Function")
        sentence_idx = data_df.columns.get_loc("Sentence")
        worksheet.set_column(function_idx, function_idx, 20)
        worksheet.set_column(sentence_idx, sentence_idx, 80, cell_format)

        fuzzy_cell_format = workbook.add_format({'text_wrap': True, 'bold': True, 'font_color': 'red'})
        worksheet.set_column(origin_column_n, origin_column_n, 80, fuzzy_cell_format)
        # # data Trung
        # worksheet.set_column(origin_column_n - 1, origin_column_n - 1, 50, fuzzy_cell_format)

        writer.save()

        return found_sen_quantity, len(sentences)
