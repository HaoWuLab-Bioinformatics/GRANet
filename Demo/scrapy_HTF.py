import requests
import  pandas as pd
import numpy as np
import json
import jsonpath
import urllib.request, urllib.parse

def Judge_regulatory(TF, Target):
    try:
        url = "http://guolab.wchscu.cn/hTFtarget/api/quick_search?"
        data = {
            'kw' : Target
        }
        headers = {
        'referer':'https://guolab.wchscu.cn/hTFtarget/',
        "cookie":"acw_tc=1a0c380817363145762755583e0102fd845cbce510e5e6b020ed5567a73cd1",
        "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0"
        }

        response = requests.get(url=url, params=data, headers=headers)
        content = response.text

        with open('HTF.json', mode='w', encoding='utf-8') as f:
            f.write(content)

        obj = json.load(open('HTF.json', mode='r', encoding='utf-8'))
        Query_gene = jsonpath.jsonpath(obj, '$..target')
        Query_gene_information = Query_gene[0][0]['id']

        second_url = "http://guolab.wchscu.cn/hTFtarget/api/chipseq/targets/target?"
        data_second = {
            'target':Query_gene_information
        }
        headers_second = {
            'cookie':'acw_tc=ac11000117363190596772269e0093cc073adfa25d69761ebb28c2ecc7a9e9',
            'user-agent':"Mozilla/5.0(WindowsNT10.0;Win64;x64)AppleWebKit/537.36(KHTML, likeGecko) Chrome/131.0.0.0Safari/537.36Edg/131.0.0.0"
        }

        data_second = urllib.parse.urlencode(data_second)
        second_url = second_url + data_second
        Request = urllib.request.Request(url=second_url, headers=headers_second)
        response = urllib.request.urlopen(Request)
        content = response.read().decode('utf-8')

        with open('HTF_list.json', mode='w', encoding='utf-8') as f:
            f.write(content)

        obj = json.load(open('HTF_list.json', mode='r', encoding='utf-8'))
        TF_list = jsonpath.jsonpath(obj, '$..tf_id')
        result = TF in TF_list
    except IndexError as e:
        return -1
    else:
        return result

if __name__ == '__main__':
    cell_type = 'mHSC-GM'
    TF = 'EGR1'
    data = pd.read_csv(f'./Regulatory_relationship/{cell_type}_{TF}.csv', index_col=0)
    TF = data.iloc[0, 0]
    Targets = data.iloc[:, 1]

    result = []
    for target in Targets:
        relationship = Judge_regulatory(TF, target)
        result.append(relationship)
        print(f"{target} 受 {TF} 的调控" if  relationship else f"{target} 不受 {TF} 的调控")

    data['relation'] = np.array(result)
    data.to_csv(f"./Regulatory_relationship/{cell_type}_{TF}_relationship.csv")




