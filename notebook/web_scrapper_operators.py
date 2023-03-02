# %%
import os
import sys
import time
import warnings
import numpy as np
import progressbar
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
warnings.filterwarnings('ignore')
from bs4 import BeautifulSoup
sys.path.append('../')
from thop.vision.onnx_counter import onnx_operators

url = "https://github.com/onnx/onnx"

try:
    response = requests.get(f'{url}/blob/main/docs/Operators.md', verify=False)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table')
    records = []
    columns = []
    for tr in table.findAll("tr"):
        ths = tr.findAll("th")
        if ths != []:
            for each in ths:
                columns.append(each.text)
        else:
            trs = tr.findAll("td")
            record = []
            for each in trs:
                try:
                    link = each.find('a')['href']
                    text = each.text
                    record.append((link, text))
                except:
                    text = each.text
                    record.append(('', text))
            records.append(record)
    df = pd.DataFrame(data=records, columns=columns)
except Exception as e:
    print(e)

# %%
l_onnxOps = list(map(lambda x: x[1], df['Operator']))

# %%
l_cur_onnxOps = list(filter(lambda x: x != None, onnx_operators.keys()))

# %%
na_onnxOps = list(set(l_onnxOps) - set(l_cur_onnxOps))

# %%
df_ops = pd.DataFrame.from_dict({'Onnx OPS': sorted(l_onnxOps), 'Current Supported Onnx Ops': sorted(l_cur_onnxOps), 'Not Supported Onnx Ops': sorted(na_onnxOps)}, orient='index')

# %%
writer = pd.ExcelWriter('Operations.xlsx', engine='xlsxwriter')
df_ops = df_ops.transpose()
df_ops.to_excel(writer, sheet_name='Ops')
writer.save()


