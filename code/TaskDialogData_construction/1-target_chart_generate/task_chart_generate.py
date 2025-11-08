
from output import TaskVisAPIs



def get_field_list_with_layer(vega):
    field_list = []
    for encode in vega['layer'][0]['encoding']:
        if 'field' in vega['layer'][0]['encoding'][encode]:
            field_list.append(vega["layer"][0]['encoding'][encode]['field'])
    return field_list


def get_field_list(vega):
    field_list = []
    for encode in vega["encoding"]:
        if 'field' in vega["encoding"][encode]:
            field_list.append(vega["encoding"][encode]['field'])
    return field_list


def judgement_field(field_dict, field_list, num):
    for field in field_list:
        if field not in field_dict:
            field_dict[field] = 1
        else:
            if field_dict[field] == num:
                return False
            field_dict[field] = field_dict[field] + 1
    return True


def find_anomalies_chart(df, types):
    recos, _ = TaskVisAPIs(df, types, task="find_anomalies", mode=1)

    # Define maximum counts for each type of visualization
    max_counts = {
        'boxplot_1': 1,
        'boxplot_2': 1,
        'point': 1,
        'point_color': 1
    }

    # Initialize the list to store the results
    ans = []
    if recos is None:
        return []
    boxplot = {}
    point = {}
    # Iterate over
    for reco in recos:
        vega = reco.props
        mark = vega["mark"]["type"]
        encoding_length = len(vega["encoding"])
        # Handle boxplot visualizations
        field_list = get_field_list(vega)
        if mark == "boxplot" and judgement_field(boxplot, field_list, 2):
            if encoding_length == 1 and max_counts['boxplot_1'] > 0:
                max_counts['boxplot_1'] -= 1
                ans.append({"chart_type": "Box Plot", "mark": "boxplot", "vega-lite": vega})
            elif encoding_length == 2 and max_counts['boxplot_2'] > 0:
                max_counts['boxplot_2'] -= 1
                ans.append({"chart_type": "Box Plot", "mark": "boxplot", "vega-lite": vega})

        # Handle point visualizations
        elif mark == "point" and judgement_field(point, field_list, 5):
            if encoding_length == 2 and max_counts['point'] > 0:
                max_counts['point'] -= 1
                ans.append({"chart_type": "Scatter", "mark": "point", "vega-lite": vega})
            elif encoding_length == 3:
                if "color" in vega["encoding"] and max_counts['point_color'] > 0:
                    max_counts['point_color'] -= 1
                    ans.append({"chart_type": "Grouping Scatter", "mark": "point", "vega-lite": vega})
    print(max_counts)
    return ans


def find_extremum_chart(df, types):
    recos, _ = TaskVisAPIs(df, types, task="find_extremum", mode=1)
    max_counts = {
        'bar': 1,
        'stack_bar': 1,
        'point': 1,
        'point_color': 1
    }
    ans = []
    if recos is None:
        return []
    bar = {}
    point = {}
    for reco in recos:
        vega = reco.props
        mark = vega["mark"]["type"]
        encoding_length = len(vega["encoding"])
        field_list = get_field_list(vega)
        if mark == "bar":
            if encoding_length == 2 and max_counts['bar'] > 0:
                max_counts['bar'] -= 1
                ans.append({"chart_type": "Bar", "mark": "bar", "vega-lite": vega})
            elif encoding_length == 3 and max_counts['stack_bar'] > 0:
                max_counts['stack_bar'] -= 1
                ans.append({"chart_type": "Stacked Bar", "mark": "bar", "vega-lite": vega})

        elif mark == "point":
            if encoding_length == 2 and max_counts['point'] > 0:
                max_counts['point'] -= 1
                ans.append({"chart_type": "Scatter", "mark": "point", "vega-lite": vega})
            elif encoding_length == 3:
                if "color" in vega["encoding"] and max_counts['point_color'] > 0:
                    max_counts['point_color'] -= 1
                    ans.append({"chart_type": "Grouping Scatter", "mark": "point", "vega-lite": vega})
    print(max_counts)
    return ans


def part_to_whole_chart(df, types):
    recos, _ = TaskVisAPIs(df, types, task="part_to_whole", mode=1)
    max_counts = {
        'pie_no_field': 2,
        'pie_field': 2
    }
    if recos is None:
        return []
    ans = []
    pie = {}
    for reco in recos:
        vega = reco.props
        field_list = get_field_list(vega)
        if 'field' in vega["encoding"]['theta'] and max_counts['pie_field'] > 0:
            if judgement_field(pie, field_list, 5):
                max_counts['pie_field'] -= 1
                ans.append({"chart_type": "Pie", "mark": "arc", "vega-lite": vega})
        elif 'field' not in vega["encoding"]['theta'] and max_counts['pie_no_field'] > 0:
            if judgement_field(pie, field_list, 5):
                max_counts['pie_no_field'] -= 1
                ans.append({"chart_type": "Pie", "mark": "arc", "vega-lite": vega})
    print(max_counts)
    return ans


def change_over_time_chart(df, types):
    recos, _ = TaskVisAPIs(df, types, task="change_over_time", mode=1)
    max_counts = {
        'area': 1,
        'stack_area': 2,
        'line': 1,
        'stack_line': 2
    }
    if recos is None:
        return []
    ans = []
    area = {}
    line = {}
    for reco in recos:
        vega = reco.props
        mark = vega["mark"]["type"]
        encoding_length = len(vega["encoding"])
        field_list = get_field_list(vega)
        if mark == "area" and judgement_field(area, field_list, 10):
            if encoding_length == 2 and max_counts['area'] > 0:
                max_counts['area'] -= 1
                ans.append({"chart_type": "Area", "mark": "area", "vega-lite": vega})
            elif encoding_length == 3 and max_counts['stack_area'] > 0:
                max_counts['stack_area'] -= 1
                ans.append({"chart_type": "Stacked Area", "mark": "area", "vega-lite": vega})

        elif mark == "line" and judgement_field(line, field_list, 10):
            if encoding_length == 2 and max_counts['line'] > 0:
                max_counts['line'] -= 1
                ans.append({"chart_type": "Line", "mark": "line", "vega-lite": vega})
            elif encoding_length == 3 and max_counts['stack_line'] > 0:
                max_counts['stack_line'] -= 1
                ans.append({"chart_type": "Grouping Line", "mark": "line", "vega-lite": vega})
    print(max_counts)
    return ans


def retrieve_value_chart(df, types):
    recos, _ = TaskVisAPIs(df, types, task="retrieve_value", mode=1)
    max_counts = {
        'rect': 1,
        'rect_field': 1
    }
    ans = []
    if recos is None:
        return []
    rect = {}
    for reco in recos:
        vega = reco.props
        field_list = get_field_list_with_layer(vega)
        if 'field' in vega['layer'][0]['encoding']['color'] and max_counts['rect_field'] > 0:
            if judgement_field(rect, field_list, 5):
                max_counts['rect_field'] -= 1
                ans.append({"chart_type": "Heatmap", "mark": "rect", "vega-lite": vega})
        elif 'field' not in vega['layer'][0]['encoding']['color'] and max_counts['rect'] > 0:
            if judgement_field(rect, field_list, 5):
                max_counts['rect'] -= 1
                ans.append({"chart_type": "Heatmap", "mark": "rect", "vega-lite": vega})
    print(max_counts)
    return ans


def trend_chart(df, types):
    recos, _ = TaskVisAPIs(df, types, task="trend", mode=1)
    ans = []
    max_counts = {
        'point': 2
    }
    if recos is None:
        return []
    point = {}
    for reco in recos:
        vega = reco.props
        field_list = get_field_list_with_layer(vega)
        if max_counts['point'] > 0:
            if judgement_field(point, field_list, 3):
                max_counts['point'] -= 1
                ans.append({"chart_type": "Scatter", "mark": "point", "vega-lite": vega})
    print(max_counts)
    return ans


def characterize_distribution_chart(df, types):
    recos, _ = TaskVisAPIs(df, types, task="characterize_distribution", mode=1)
    max_counts = {
        'boxplot_1': 1,
        'boxplot_2': 1,
        'histogram': 2,
        'point': 1,
        'point_color': 1,
    }
    ans = []
    boxplot = {}
    point = {}
    histogram = {}
    if recos is None:
        return []
    for reco in recos:
        vega = reco.props
        mark = vega["mark"]["type"]
        encoding_length = len(vega["encoding"])
        field_list = get_field_list(vega)
        if mark == "boxplot" and judgement_field(boxplot, field_list, 2):
            if encoding_length == 1 and max_counts['boxplot_1'] > 0:
                max_counts['boxplot_1'] -= 1
                ans.append({"chart_type": "Box Plot", "mark": "boxplot", "vega-lite": vega})
            elif encoding_length == 2 and max_counts['boxplot_2'] > 0:
                max_counts['boxplot_2'] -= 1
                ans.append({"chart_type": "Box Plot", "mark": "boxplot", "vega-lite": vega})
        elif mark == "point" and judgement_field(point, field_list, 5):
            if encoding_length == 2 and max_counts['point'] > 0:
                max_counts['point'] -= 1
                ans.append({"chart_type": "Scatter", "mark": "point", "vega-lite": vega})
            elif encoding_length == 3:
                if "size" in vega["encoding"] and max_counts['point_size'] > 0:
                    max_counts['point_size'] -= 1
                    ans.append({"chart_type": "Bubble Scatter", "mark": "point", "vega-lite": vega})
                elif "color" in vega["encoding"] and max_counts['point_color'] > 0:
                    max_counts['point_color'] -= 1
                    ans.append({"chart_type": "Grouping Scatter", "mark": "point", "vega-lite": vega})
                elif "shape" in vega["encoding"] and max_counts['point_shape'] > 0:
                    max_counts['point_shape'] -= 1
                    ans.append({"chart_type": "Grouping Scatter", "mark": "point", "vega-lite": vega})
        elif mark == 'bar' and max_counts['histogram'] > 0 and judgement_field(histogram, field_list, 3):
            max_counts['histogram'] -= 1
            ans.append({"chart_type": "Histogram", "mark": "bar", "vega-lite": vega})
    print(max_counts)
    return ans


def comparison_chart(df, types):
    recos, _ = TaskVisAPIs(df, types, task="comparison", mode=1)
    max_counts = {
        'bar': 1,
        'line_color':1,
        'point_color': 1
    }
    ans = []
    if recos is None:
        return []
    bar = {}
    point = {}
    line = {}
    for reco in recos:
        vega = reco.props
        mark = vega["mark"]["type"]
        encoding_length = len(vega["encoding"])
        field_list = get_field_list(vega)
        if mark == "bar" and max_counts['bar'] > 0 and judgement_field(bar, field_list, 3):
            max_counts['bar'] -= 1
            ans.append({"chart_type": "Bar", "mark": "bar", "vega-lite": vega})
        elif mark == "point":
            if "color" in vega["encoding"] and max_counts['point_color'] > 0:
                max_counts['point_color'] -= 1
                ans.append({"chart_type": "Grouping Scatter", "mark": "point", "vega-lite": vega})
        elif mark == 'line' and max_counts['line_color'] > 0 and judgement_field(line, field_list, 3):
            max_counts['line_color'] -= 1
            ans.append({"chart_type": "Grouping Line", "mark": "line", "vega-lite": vega})
    print(max_counts)
    return ans


def compute_derived_value_chart(df, types):
    recos, _ = TaskVisAPIs(df, types, task="compute_derived_value", mode=1)
    max_counts = {
        'bar': 1,
        'stack_bar': 1,
        'rect': 1,
        'arc': 1,
        'arc_no_field': 1
    }
    ans = []
    if recos is None:
        return []
    rect = {}
    bar = {}
    pie = {}
    for reco in recos:
        vega = reco.props
        if 'layer' in vega:
            field_list = get_field_list_with_layer(vega)
        else:
            field_list = get_field_list(vega)
        if 'layer' in vega and max_counts['rect'] > 0 and judgement_field(rect, field_list, 3):
            max_counts['rect'] -= 1
            ans.append({"chart_type": "Heatmap", "mark": "rect", "vega-lite": vega})
        elif 'layer' not in vega and vega["mark"]["type"] == "bar":
            if len(vega["encoding"]) == 2 and max_counts['bar'] > 0:
                max_counts['bar'] -= 1
                ans.append({"chart_type": "Bar", "mark": "bar", "vega-lite": vega})
            elif len(vega["encoding"]) == 3 and max_counts['stack_bar'] > 0:
                max_counts['stack_bar'] -= 1
                ans.append({"chart_type": "Stacked Bar", "mark": "bar", "vega-lite": vega})
        elif 'layer' not in vega and vega["mark"]["type"] == "arc" and judgement_field(pie, field_list, 3):
            if 'field' not in vega["encoding"]['theta'] and max_counts['arc'] > 0:
                max_counts['arc'] -= 1
                ans.append({"chart_type": "Pie", "mark": "arc", "vega-lite": vega})
            elif 'field' in vega["encoding"]['theta'] and max_counts['arc_no_field'] > 0:
                max_counts['arc_no_field'] -= 1
                ans.append({"chart_type": "Pie", "mark": "arc", "vega-lite": vega})
    print(max_counts)
    return ans


def correlate_chart(df, types):
    recos, _ = TaskVisAPIs(df, types, task="correlate", mode=1)
    max_counts = {
        'point': 1,
        'point_three': 1
    }
    ans = []
    if recos is None:
        return []
    point = {}
    for reco in recos:
        vega = reco.props
        mark = vega["mark"]["type"]
        encoding_length = len(vega["encoding"])
        field_list = get_field_list(vega)
        if len(vega['encoding']) == 2 and max_counts['point'] > 0 and judgement_field(point, field_list, 6):
            max_counts['point'] -= 1
            ans.append({"chart_type": "Scatter", "mark": "point", "vega-lite": vega})
        elif len(vega['encoding']) == 3 and max_counts['point_three'] > 0 and judgement_field(point, field_list, 6):
            max_counts['point_three'] -= 1
            ans.append({"chart_type": "Grouping Scatter", "mark": "point", "vega-lite": vega})
    print(max_counts)
    return ans


def determine_range_chart(df, types):
    recos, _ = TaskVisAPIs(df, types, task="determine_range", mode=1)

    # Define maximum counts for each type of visualization
    max_counts = {
        'boxplot_1': 2,
        'boxplot_2': 2
    }

    # Initialize the list to store the results
    ans = []
    if recos is None:
        return []
    # Iterate over recommendations
    boxplot = {}
    for reco in recos:
        vega = reco.props
        mark = vega["mark"]["type"]
        encoding_length = len(vega["encoding"])
        field_list = get_field_list(vega)
        # Handle boxplot visualizations
        if mark == "boxplot" and judgement_field(boxplot, field_list, 5):
            if encoding_length == 1 and max_counts['boxplot_1'] > 0:
                max_counts['boxplot_1'] -= 1
                ans.append({"chart_type": "Box Plot", "mark": "boxplot", "vega-lite": vega})
            elif encoding_length == 2 and max_counts['boxplot_2'] > 0:
                max_counts['boxplot_2'] -= 1
                ans.append({"chart_type": "Box Plot", "mark": "boxplot", "vega-lite": vega})
    print(max_counts)
    return ans


def deviation_chart(df, types):
    recos, _ = TaskVisAPIs(df, types, task="deviation", mode=1)

    # Define maximum counts for each type of visualization
    max_counts = {
        'bar': 2,
        'point': 2
    }

    # Initialize the list to store the results
    ans = []
    if recos is None:
        return []
    bar = {}
    point = {}
    # Iterate over recommendations
    for reco in recos:
        vega = reco.props
        mark = vega['layer'][0]["mark"]["type"]
        field_list = get_field_list_with_layer(vega)
        # Handle boxplot visualizations
        if mark == "bar" and max_counts['bar'] > 0 and judgement_field(bar, field_list, 3):
            max_counts['bar'] -= 1
            ans.append({"chart_type": "bar", "mark": "bar", "vega-lite": vega})
        elif mark == "point" and max_counts['point'] > 0 and judgement_field(point, field_list, 3):
            max_counts['point'] -= 1
            ans.append({"chart_type": "Scatter", "mark": "point", "vega-lite": vega})
    print(max_counts)
    return ans
