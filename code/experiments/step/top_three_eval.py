import json

import numpy as np
import sacrebleu
import copy
from similarity import vegalite_to_string
from tqdm import tqdm

def calculate_lcs_length(s1: str, s2: str) -> int:

    m = len(s1)
    n = len(s2)

    dp = np.zeros((m + 1, n + 1), dtype=int)

    # 填充 DP 表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                # 如果字符匹配，LCS 长度加 1
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                # 如果字符不匹配，取上方或左方单元格的最大值
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # DP 表右下角的值即为 s1 和 s2 的 LCS 长度
    return dp[m][n]


def rouge_l_similarity(s1: str, s2: str) -> float:

    m = len(s1)
    n = len(s2)

    # 处理空字符串的边界情况
    if m == 0 or n == 0:
        return 0.0

    # 计算 LCS 长度
    lcs_len = calculate_lcs_length(s1, s2)

    # 如果 LCS 长度为 0，则相似度为 0
    if lcs_len == 0:
        return 0.0

    recall_lcs = lcs_len / m


    precision_lcs = lcs_len / n


    f1_lcs = 2 * (recall_lcs * precision_lcs) / (recall_lcs + precision_lcs)

    return f1_lcs


def bleu_eval(candidates, references):
    references_sacre_formatted = [references]

    bleu_sacre = sacrebleu.sentence_bleu(candidates, references_sacre_formatted)

    return bleu_sacre.score / 100


def transfer_chart(chart):
    chart_new = {"mark": chart["mark"], "encoding": {}}
    if "x" in chart["encoding"]:
        chart_new["encoding"]["x"] = {}
        if "field" in chart["encoding"]["x"]:
            chart_new["encoding"]["x"]["field"] = chart["encoding"]["x"]["field"]
        if "sort" in chart["encoding"]["x"]:
            chart_new["encoding"]["x"]["sort"] = chart["encoding"]["x"]["sort"]
        if "bin" in chart["encoding"]["x"]:
            chart_new["encoding"]["x"]["bin"] = chart["encoding"]["x"]["bin"]
        if "aggregate" in chart["encoding"]["x"]:
            chart_new["encoding"]["x"]["aggregate"] = chart["encoding"]["x"]["aggregate"]
    if "y" in chart["encoding"]:
        chart_new["encoding"]["y"] = {}
        if "field" in chart["encoding"]["y"]:
            chart_new["encoding"]["y"]["field"] = chart["encoding"]["y"]["field"]
        if "sort" in chart["encoding"]["y"]:
            chart_new["encoding"]["y"]["sort"] = chart["encoding"]["y"]["sort"]
        if "bin" in chart["encoding"]["y"]:
            chart_new["encoding"]["y"]["bin"] = chart["encoding"]["y"]["bin"]
        if "aggregate" in chart["encoding"]["y"]:
            chart_new["encoding"]["y"]["aggregate"] = chart["encoding"]["y"]["aggregate"]
    if "color" in chart["encoding"]:
        chart_new["encoding"]["color"] = {}
        if "field" in chart["encoding"]["color"]:
            chart_new["encoding"]["color"]["field"] = chart["encoding"]["color"]["field"]
        if "aggregate" in chart["encoding"]["color"]:
            chart_new["encoding"]["color"]["aggregate"] = chart["encoding"]["color"]["aggregate"]
    if "theta" in chart["encoding"]:
        chart_new["encoding"]["theta"] = {}
        if "field" in chart["encoding"]["theta"]:
            chart_new["encoding"]["theta"]["field"] = chart["encoding"]["theta"]["field"]
        if "aggregate" in chart["encoding"]["theta"]:
            chart_new["encoding"]["theta"]["aggregate"] = chart["encoding"]["theta"]["aggregate"]
    return chart_new


def accuracy_eval(candidates_chart, references_chart):
    global candidates_filter_ast, references_filter_ast, candidates_chart_ast, references_chart_ast
    from filter_ast import filter_to_ast
    candidates_filter_str = ""
    references_filter_str = ""
    if "transform" in candidates_chart:
        for transform in candidates_chart['transform']:
            if "filter" in transform:
                candidates_filter_str = transform["filter"].replace("datum.", "").replace("===", "==")
                break
    if "transform" in references_chart:
        for transform in references_chart['transform']:
            if "filter" in transform:
                references_filter_str = transform["filter"].replace("datum.", "").replace("===", "==")
                break

    if (candidates_filter_str == "" and references_filter_str != "") or (
            references_filter_str != "" and candidates_filter_str == ""):
        return False
    candidates_filter_ast = ""
    references_filter_ast = ""

    if candidates_filter_str != "":
        candidates_filter_ast = filter_to_ast(candidates_filter_str)
    if references_filter_str != "":
        references_filter_ast = filter_to_ast(references_filter_str)
    else:
        references_chart_ast = ""

    if candidates_filter_ast != references_filter_ast and candidates_filter_str != references_filter_str:
        return False

    if candidates_chart["mark"] != references_chart["mark"]:
        return False

    if len(candidates_chart["encoding"]) != len(references_chart["encoding"]):
        return False

    if "x" in candidates_chart["encoding"] and "y" in candidates_chart["encoding"] and "x" in references_chart[
        "encoding"] and "y" in references_chart["encoding"]:
        if candidates_chart["encoding"]["x"] == references_chart["encoding"]["y"] and candidates_chart["encoding"][
            "y"] == references_chart["encoding"]["x"]:
            return True

    if "x" in candidates_chart["encoding"] and "x" in references_chart["encoding"]:
        if "field" in candidates_chart["encoding"]["x"] and "field" in references_chart["encoding"]["x"]:
            if candidates_chart["encoding"]["x"]["field"] != references_chart["encoding"]["x"]["field"]:
                return False
        elif ("field" in candidates_chart["encoding"]["x"] and "field" not in references_chart["encoding"][
            "x"]) or (
                "field" not in candidates_chart["encoding"]["x"] and "field" in references_chart["encoding"][
            "x"]):
            return False

        if "aggregate" in candidates_chart["encoding"]["x"] and "aggregate" in references_chart["encoding"]["x"]:
            if candidates_chart["encoding"]["x"]["aggregate"] != references_chart["encoding"]["x"]["aggregate"]:
                return False
        elif ("aggregate" in candidates_chart["encoding"]["x"] and "aggregate" not in references_chart["encoding"][
            "x"]) or (
                "aggregate" not in candidates_chart["encoding"]["x"] and "aggregate" in references_chart["encoding"][
            "x"]):
            return False

        if "bin" in candidates_chart["encoding"]["x"] and "bin" in references_chart["encoding"]["x"]:
            if candidates_chart["encoding"]["x"]["bin"] != references_chart["encoding"]["x"]["bin"]:
                return False
        elif ("bin" in candidates_chart["encoding"]["x"] and "bin" not in references_chart["encoding"]["x"]) or (
                "bin" not in candidates_chart["encoding"]["x"] and "bin" in references_chart["encoding"]["x"]):
            return False

        if "sort" in candidates_chart["encoding"]["x"] and "sort" in references_chart["encoding"]["x"]:
            if candidates_chart["encoding"]["x"]["sort"] != references_chart["encoding"]["x"]["sort"]:
                return False
        elif ("sort" in candidates_chart["encoding"]["x"] and "sort" not in references_chart["encoding"]["x"]) or (
                "sort" not in candidates_chart["encoding"]["x"] and "sort" in references_chart["encoding"]["x"]):
            return False
    elif ("x" in candidates_chart["encoding"] and "x" not in references_chart["encoding"]) or (
            "x" not in candidates_chart["encoding"] and "x" in references_chart["encoding"]):
        return False

    if "y" in candidates_chart["encoding"] and "y" in references_chart["encoding"]:
        if "field" in candidates_chart["encoding"]["y"] and "field" in references_chart["encoding"]["y"]:
            if candidates_chart["encoding"]["y"]["field"] != references_chart["encoding"]["y"]["field"]:
                return False
        elif ("field" in candidates_chart["encoding"]["y"] and "field" not in references_chart["encoding"][
            "y"]) or (
                "field" not in candidates_chart["encoding"]["y"] and "field" in references_chart["encoding"][
            "y"]):
            return False

        if "aggregate" in candidates_chart["encoding"]["y"] and "aggregate" in references_chart["encoding"]["y"]:
            if candidates_chart["encoding"]["y"]["aggregate"] != references_chart["encoding"]["y"]["aggregate"]:
                return False
        elif ("aggregate" in candidates_chart["encoding"]["y"] and "aggregate" not in references_chart["encoding"][
            "y"]) or (
                "aggregate" not in candidates_chart["encoding"]["y"] and "aggregate" in references_chart["encoding"][
            "y"]):
            return False

        if "bin" in candidates_chart["encoding"]["y"] and "bin" in references_chart["encoding"]["y"]:
            if candidates_chart["encoding"]["y"]["bin"] != references_chart["encoding"]["y"]["bin"]:
                return False
        elif ("bin" in candidates_chart["encoding"]["y"] and "bin" not in references_chart["encoding"]["y"]) or (
                "bin" not in candidates_chart["encoding"]["y"] and "bin" in references_chart["encoding"]["y"]):
            return False

        if "sort" in candidates_chart["encoding"]["y"] and "sort" in references_chart["encoding"]["y"]:
            if candidates_chart["encoding"]["y"]["sort"] != references_chart["encoding"]["y"]["sort"]:
                return False
        elif ("sort" in candidates_chart["encoding"]["y"] and "sort" not in references_chart["encoding"]["y"]) or (
                "sort" not in candidates_chart["encoding"]["y"] and "sort" in references_chart["encoding"]["y"]):
            return False
    elif ("y" in candidates_chart["encoding"] and "y" not in references_chart["encoding"]) or (
            "y" not in candidates_chart["encoding"] and "y" in references_chart["encoding"]):
        return False

    if "color" in candidates_chart["encoding"] and "color" in references_chart["encoding"]:
        if "field" in candidates_chart["encoding"]["color"] and "field" in references_chart["encoding"][
            "color"]:
            if candidates_chart["encoding"]["color"]["field"] != references_chart["encoding"]["color"]["field"]:
                return False
        elif ("field" in candidates_chart["encoding"]["color"] and "field" not in references_chart["encoding"][
            "color"]) or (
                "field" not in candidates_chart["encoding"]["color"] and "field" in
                references_chart["encoding"][
                    "color"]):
            return False

        if "aggregate" in candidates_chart["encoding"]["color"] and "aggregate" in references_chart["encoding"][
            "color"]:
            if candidates_chart["encoding"]["color"]["aggregate"] != references_chart["encoding"]["color"]["aggregate"]:
                return False
        elif ("aggregate" in candidates_chart["encoding"]["color"] and "aggregate" not in references_chart["encoding"][
            "color"]) or (
                "aggregate" not in candidates_chart["encoding"]["color"] and "aggregate" in
                references_chart["encoding"][
                    "color"]):
            return False

        if "bin" in candidates_chart["encoding"]["color"] and "bin" in references_chart["encoding"]["color"]:
            if candidates_chart["encoding"]["color"]["bin"] != references_chart["encoding"]["color"]["bin"]:
                return False
        elif ("bin" in candidates_chart["encoding"]["color"] and "bin" not in references_chart["encoding"][
            "color"]) or (
                "bin" not in candidates_chart["encoding"]["color"] and "bin" in references_chart["encoding"]["color"]):
            return False

        if "sort" in candidates_chart["encoding"]["color"] and "sort" in references_chart["encoding"]["color"]:
            if candidates_chart["encoding"]["color"]["sort"] != references_chart["encoding"]["color"]["sort"]:
                return False
        elif ("sort" in candidates_chart["encoding"]["color"] and "sort" not in references_chart["encoding"][
            "color"]) or (
                "sort" not in candidates_chart["encoding"]["color"] and "sort" in references_chart["encoding"][
            "color"]):
            return False
    elif ("color" in candidates_chart["encoding"] and "color" not in references_chart["encoding"]) or (
            "color" not in candidates_chart["encoding"] and "color" in references_chart["encoding"]):
        return False

    if "theta" in candidates_chart["encoding"] and "theta" in references_chart["encoding"]:
        if "field" in candidates_chart["encoding"]["theta"] and "field" in references_chart["encoding"][
            "theta"]:
            if candidates_chart["encoding"]["theta"]["field"] != references_chart["encoding"]["theta"]["field"]:
                return False
        elif ("field" in candidates_chart["encoding"]["theta"] and "field" not in references_chart["encoding"][
            "theta"]) or (
                "field" not in candidates_chart["encoding"]["theta"] and "field" in
                references_chart["encoding"][
                    "theta"]):
            return False

        if "aggregate" in candidates_chart["encoding"]["theta"] and "aggregate" in references_chart["encoding"][
            "theta"]:
            if candidates_chart["encoding"]["theta"]["aggregate"] != references_chart["encoding"]["theta"]["aggregate"]:
                return False
        elif ("aggregate" in candidates_chart["encoding"]["theta"] and "aggregate" not in references_chart["encoding"][
            "theta"]) or (
                "aggregate" not in candidates_chart["encoding"]["theta"] and "aggregate" in
                references_chart["encoding"][
                    "theta"]):
            return False

        if "bin" in candidates_chart["encoding"]["theta"] and "bin" in references_chart["encoding"]["theta"]:
            if candidates_chart["encoding"]["theta"]["bin"] != references_chart["encoding"]["theta"]["bin"]:
                return False
        elif ("bin" in candidates_chart["encoding"]["theta"] and "bin" not in references_chart["encoding"][
            "theta"]) or (
                "bin" not in candidates_chart["encoding"]["theta"] and "bin" in references_chart["encoding"]["theta"]):
            return False

        if "sort" in candidates_chart["encoding"]["theta"] and "sort" in references_chart["encoding"]["theta"]:
            if candidates_chart["encoding"]["theta"]["sort"] != references_chart["encoding"]["theta"]["sort"]:
                return False
        elif ("sort" in candidates_chart["encoding"]["theta"] and "sort" not in references_chart["encoding"][
            "theta"]) or (
                "sort" not in candidates_chart["encoding"]["theta"] and "sort" in references_chart["encoding"][
            "theta"]):
            return False
    elif ("theta" in candidates_chart["encoding"] and "theta" not in references_chart["encoding"]) or (
            "theta" not in candidates_chart["encoding"] and "theta" in references_chart["encoding"]):
        return False
    return True

with open(r"output/deepseek_r1_check.json", "r", encoding="utf-8") as file:
    output = file.read()

with open(r"../test1.json", "r", encoding="utf-8") as file:
    ture_data = file.read()

output = json.loads(output)
ture_data = json.loads(ture_data)
rouge_l_score = 0
bleu_score = 0
consistency = 0
task_num = 0
all_of_result = []
percentage_of_dialogues_correct = 0

for index, dio in enumerate(tqdm(output)):
    diolage_result = []
    a = True
    print(dio)
    if dio["dialogues"] == []:
        continue
    dio_accuracy_num = 0
    for i, chart in enumerate(tqdm(dio["dialogues"])):
        print(consistency)
        true_task = False
        ture_chart = False
        max_rouge = 0
        max_bleu = 0
        for n in chart["charts"]:
            try:
                chart_list = [n["chart"], ture_data[index]["dialogues"][i]["chart"]]
                task_list = [n["analytic task"], ture_data[index]["dialogues"][i]["analytic task"]]
                candidate_chart_string = vegalite_to_string(n["chart"])
                reference_chart_string = vegalite_to_string(ture_data[index]["dialogues"][i]["chart"])
                rouge_l = rouge_l_similarity(candidate_chart_string, reference_chart_string)
                bleu = bleu_eval(candidate_chart_string, reference_chart_string)
                if rouge_l > max_rouge:
                    max_rouge = rouge_l
                if bleu > max_bleu:
                    max_bleu = bleu
                if n["analytic task"] == ture_data[index]["dialogues"][i]["analytic task"]:
                    true_task = True
                if accuracy_eval(n["chart"], ture_data[index]["dialogues"][i]["chart"]):
                    ture_chart = True
            except:
                continue
        if ture_chart:
            consistency += 1
        rouge_l_score += max_rouge
        bleu_score += max_bleu
        if true_task:
            task_num+=1


rouge_l_score = rouge_l_score / 554
bleu_score = bleu_score / 554
consistency = consistency / 554
task_accuracy = task_num / 554

print(rouge_l_score)
print(bleu_score)
print(consistency)
print(task_accuracy)