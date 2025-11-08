from nl4dv import NL4DV
import json
from tqdm import tqdm


def more_zero_nl4dv(nl4dv_instance, num, query):
    try:
        response = nl4dv_instance.analyze_query(query=query, dialog=True, dialog_id="0", query_id=str(num - 1))
    except:
        response = False
    return response


def nl4dv_dialog_output(dialog, data_url):
    global chart_list
    nl4dv_instance = NL4DV(data_url=data_url)
    dependency_parser_config = {"name": "spacy", "model": "en_core_web_sm", "parser": None}
    nl4dv_instance.set_dependency_parser(config=dependency_parser_config)
    # ans = []
    chart_ans = []

    response1 = nl4dv_instance.analyze_query(dialog[0]["utterance"])
    print(dialog[0]["utterance"])
    for res in response1["visList"]:
        chart_list = []
        chart_list.append({"tasks": res["tasks"], "chart": res["vlSpec"]})
    print(response1["visList"])
    chart_ans.append({"turn": 0, "utterance": dialog[0]["utterance"], "charts": chart_list})

    response2 = more_zero_nl4dv(nl4dv_instance, 1, dialog[1]["utterance"])
    if not response2:
        chart_ans.append({"turn": 1, "utterance": dialog[1]["utterance"], "charts": None})
    else:
        for res in response2["visList"]:
            chart_list = []
            chart_list.append({"tasks": res["tasks"], "chart": res["vlSpec"]})
        chart_ans.append({"turn": 1, "utterance": dialog[1]["utterance"], "charts": chart_list})

    response3 = more_zero_nl4dv(nl4dv_instance, 2, dialog[2]["utterance"])
    if not response3:
        chart_ans.append({"turn": 2, "utterance": dialog[2]["utterance"], "charts": None})
    else:
        for res in response3["visList"]:
            chart_list = []
            chart_list.append({"tasks": res["tasks"], "chart": res["vlSpec"]})
        chart_ans.append({"turn": 2, "utterance": dialog[2]["utterance"], "charts": chart_list})

    response4 = more_zero_nl4dv(nl4dv_instance, 3, dialog[3]["utterance"])
    if not response4:
        chart_ans.append({"turn": 3, "utterance": dialog[3]["utterance"], "charts": None})
    else:
        for res in response4["visList"]:
            chart_list = []
            chart_list.append({"tasks": res["tasks"], "chart": res["vlSpec"]})
        chart_ans.append({"turn": 3, "utterance": dialog[3]["utterance"], "charts": chart_list})

    if len(dialog) == 5:

        chart_list = []
        response5 = more_zero_nl4dv(nl4dv_instance, 4, dialog[4]["utterance"])
        if not response5:
            chart_ans.append({"turn": 4, "utterance": dialog[3]["utterance"], "charts": None})
        else:
            for res in response5["visList"]:
                chart_list.append({"tasks": res["tasks"], "chart": res["vlSpec"]})
            chart_ans.append({"turn": 4, "utterance": dialog[4]["utterance"], "charts": chart_list})

    return chart_ans


def filter_dialog_data(dialog):
    for i, chart in enumerate(dialog):
        if chart["chart"]["mark"] not in ["bar", "point", "rect", "line"]:
            return False
        field_type = []
        if "x" in chart["chart"]["encoding"] and "type" in chart["chart"]["encoding"]["x"]:
            if chart["chart"]["encoding"]["x"]["type"] not in field_type:
                field_type.append(chart["chart"]["encoding"]["x"]["type"])

        if "y" in chart["chart"]["encoding"] and "type" in chart["chart"]["encoding"]["y"]:
            if chart["chart"]["encoding"]["y"]["type"] not in field_type:
                field_type.append(chart["chart"]["encoding"]["y"]["type"])

        if "color" in chart["chart"]["encoding"] and "type" in chart["chart"]["encoding"]["color"]:
            if chart["chart"]["encoding"]["color"]["type"] not in field_type:
                field_type.append(chart["chart"]["encoding"]["color"]["type"])

        if "theta" in chart["chart"]["encoding"] and "type" in chart["chart"]["encoding"]["theta"]:
            if chart["chart"]["encoding"]["theta"]["type"] not in field_type:
                field_type.append(chart["chart"]["encoding"]["theta"]["type"])
        if "quantitative" not in field_type:
            return False
    return True
