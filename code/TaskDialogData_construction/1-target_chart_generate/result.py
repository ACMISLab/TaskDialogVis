
import json
from tqdm import tqdm
import pandas as pd
import runpy

with open("../dataset/message.json", 'r') as file:
    data_list = json.load(file)
from Transform import GetNewColumnType

chart_list = []

task_rank = {
    'change_over_time': 3,
    'characterize_distribution': 1,
    'comparison': 1,
    'compute_derived_value': 1,
    'correlate': 1,
    'determine_range': 2,
    'find_anomalies': 3,
    'find_extremum': 1,
    'retrieve_value': 1
}
tasks = list(task_rank.keys())
namespace = runpy.run_module(mod_name='task_chart_generate')
for task in tqdm(tasks):
    for num in tqdm(range(len(data_list))):
        print(data_list[num]['id'])
        df = pd.read_csv(f"../dataset/data_files/{data_list[num]['id']}.csv")
        columnType = data_list[num]['columns']
        types = []
        for i in columnType:
            types.append({"field": i, "type": columnType[i]})
        # 调用函数
        ans = namespace[f'{task}_chart'](df, types)
        for chart in ans:
            chart['task'] = task
            chart['dataset'] = f"{data_list[num]['id']}.csv"
        if len(ans) == 0:
            continue
        chart_df = pd.DataFrame(ans)
        # print(chart_df)
        chart_df = chart_df[['dataset', 'task', 'chart_type', 'mark', 'vega-lite']]
        chart_df.to_csv(f'{task}.csv', mode='a', index=False, header=False, encoding='utf-8')
