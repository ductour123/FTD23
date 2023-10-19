import json
import os
from cosine_similarity import SimilarityFilter

data_dir_path = "../data/ftd23"
result_dir_path = "../result/ftd23"

def get_batchs():

    folders = os.listdir(data_dir_path)
    # print(folders)
    batchs = {}
    for folder in folders:
        ESUS_file_lst = []
        FRCA_file_lst = []
        files = os.listdir(data_dir_path + '/' + folder)
        for file in files:
            if 'ESUS' in file:
                ESUS_file_lst.append(file)
            else:
                FRCA_file_lst.append(file)

        batchs[folder] = {
            "ESUS": ESUS_file_lst,
            "FRCA": FRCA_file_lst
        }

    return batchs


if __name__ == '__main__':
    batchs = get_batchs()

    API_SBERT = "http://211.109.9.73:7008/api/v1/slx-sbert/encode"
    API_MUSE = "http://211.109.9.73:5006/analyze"
    similarity_filter = SimilarityFilter(API_SBERT, API_MUSE, 500, 0.97)

    batchs_rate = {}
    for batch_name in batchs:
        batch_rate = {}
        for language in batchs[batch_name]:
            if len(batchs[batch_name][language]) > 0:
                evaluate = [0, 0, 0]
                for file_name in batchs[batch_name][language]:
                    # check similarity in file
                    # found_sen_quantity, sen_total = similarity_filter.filt_similarity(data_dir_path + '/' + batch_name + '/' + file_name,
                    #                                   result_dir_path + '/' + batch_name + '/' + file_name[0:(len(file_name) - 5)] + '.txt')

                    # found_sen_quantity, sen_total = similarity_filter.filt_similarity_excel_2(
                    #     data_dir_path + '/' + batch_name + '/' + file_name,
                    #     result_dir_path + '/' + batch_name + '/' + file_name[0:(len(file_name) - 5)] + '_fuzzies.xlsx')

                    found_sen_quantity, sen_total = similarity_filter.filt_similarity_excel_3(
                        data_dir_path + '/' + batch_name + '/' + file_name,
                        result_dir_path + '/' + batch_name + '/' + file_name)

                    evaluate[0] += found_sen_quantity
                    evaluate[1] += sen_total

                evaluate[2] = round((evaluate[0]/evaluate[1]) * 100, 2)
                batch_rate[language] = '{:,}'.format(evaluate[0]) + "/" + '{:,}'.format(evaluate[1]) + " ~ " + '{:,}'.format(evaluate[2]) + "%"

        batchs_rate[batch_name] = batch_rate

    batchs_rate_json_str = json.dumps(batchs_rate, indent=2)

    with open(result_dir_path + '/report.txt', 'w') as fp:
        fp.write(batchs_rate_json_str)

    print('Completed')
