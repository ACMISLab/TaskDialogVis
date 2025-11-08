import json

import numpy as np
import sacrebleu
from similarity import vegalite_to_string
from tqdm import tqdm


def calculate_lcs_length(s1: str, s2: str) -> int:
    """
    使用动态规划计算两个字符串的最长公共子序列 (LCS) 的长度。

    Args:
        s1: 第一个字符串。
        s2: 第二个字符串。

    Returns:
        两个字符串的 LCS 长度。
    """
    m = len(s1)
    n = len(s2)

    # 创建一个 DP 表来存储子问题的结果
    # dp[i][j] 将存储 s1 的前 i 个字符和 s2 的前 j 个字符的 LCS 长度
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
    """
    基于最长公共子序列 (LCS) 长度计算两个字符串的 ROUGE-L 相似度 (F1 分数)。

    Args:
        s1: 第一个字符串 (通常视为参考字符串/reference)。
        s2: 第二个字符串 (通常视为候选字符串/candidate)。

    Returns:
        ROUGE-L F1 分数，范围在 0.0 到 1.0 之间。
    """
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

    # 计算召回率 (Recall)
    recall_lcs = lcs_len / m

    # 计算精确率 (Precision)
    precision_lcs = lcs_len / n

    # 计算 F1 分数
    f1_lcs = 2 * (recall_lcs * precision_lcs) / (recall_lcs + precision_lcs)

    return f1_lcs


def bleu_eval(candidates, references):
    references_sacre_formatted = [references]
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


with open(r"output_transform.json", "r", encoding="utf-8") as file:
    output = file.read()

with open(r"../test1.json", "r", encoding="utf-8") as file:
    ture_data = file.read()

output = json.loads(output)
ture_data = json.loads(ture_data)
rouge_l_score = 0
bleu_score = 0
consistency = 0
diolage = 0
task_num = 0
all_of_result = []
percentage_of_dialogues_correct = 0
task_count = {}
task_right_count = {}
for task_name in ["Comparison", "Modify Chart", "Compute Derived Value", "Correlate", "Find Extremum", "Retrieve Value",
                  "Characterize Distribution", "Change Over Time", "Find Anomalies"]:
    task_right_count[task_name] = 0

for index, dio in enumerate(tqdm(output)):
    diolage_result = []
    a = True
    if dio["dialogues"] == []:
        continue
    dio_accuracy_num = 0
    for i, chart in enumerate(tqdm(dio["dialogues"])):
        right = False
        print(chart["chart"])
        print(ture_data[index]["dialogues"])
        chart_list = [chart["chart"], ture_data[index]["dialogues"][i]["chart"]]
        # task_list = [chart["analyzing tasks"], ture_data[index]["dialogues"][i]["analyzing tasks"]]
        # if ture_data[index]["dialogues"][i]["analyzing tasks"] not in task_count:
        #     task_count[ture_data[index]["dialogues"][i]["analyzing tasks"]] = 1
        # else:
        #     task_count[ture_data[index]["dialogues"][i]["analyzing tasks"]] += 1
        candidate_chart_string = vegalite_to_string(chart["chart"])
        reference_chart_string = vegalite_to_string(ture_data[index]["dialogues"][i]["chart"])
        rouge_l_score += rouge_l_similarity(candidate_chart_string, reference_chart_string)
        bleu_score += bleu_eval(candidate_chart_string, reference_chart_string)
        # print(chart["analyzing tasks"], ture_data[index]["dialogues"][i]["analyzing tasks"])

        # if chart["analyzing tasks"] == ture_data[index]["dialogues"][i]["analyzing tasks"]:
        #     task_num += 1

        if accuracy_eval(chart["chart"], ture_data[index]["dialogues"][i]["chart"]):
            consistency += 1
            right = True
            dio_accuracy_num += 1
            # if ture_data[index]["dialogues"][i]["analyzing tasks"] not in task_right_count:
            #     task_right_count[ture_data[index]["dialogues"][i]["analyzing tasks"]] = 1
            # else:
            #     task_right_count[ture_data[index]["dialogues"][i]["analyzing tasks"]] += 1
        else:
            a = False

        diolage_result.append(
            {"right": right, "utterance": ture_data[index]["dialogues"][i]["utterance"], "charts": chart_list})
        print(consistency)
        print(diolage)
    dio_accuracy_num = dio_accuracy_num / len(dio["dialogues"])
    percentage_of_dialogues_correct += dio_accuracy_num
    if a:
        diolage += 1
    all_of_result.append({"index": index, "diolage": diolage_result})
    with open(f"right_output.json", 'w', encoding='utf-8') as f:
        # 确保默认的ensure_ascii为False，这样非ASCII字符才能被正确写入
        json.dump(all_of_result, f, ensure_ascii=False, indent=4)

print(consistency)
rouge_l_score = rouge_l_score / 554
bleu_score = bleu_score / 554
consistency = consistency / 554
# task = task_num / 554

print(rouge_l_score)
print(bleu_score)
print(consistency)
