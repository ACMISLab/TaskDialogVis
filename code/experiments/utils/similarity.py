import json



def vegalite_to_string(chart):
    mark = chart["mark"]
    encodings = "" + str(mark)
    if "x" in chart["encoding"]:
        if "field" in chart["encoding"]["x"]:
            encodings += " " + chart["encoding"]["x"]["field"]
        else:
            encodings += " none"

        if "aggregate" in chart["encoding"]["x"]:
            encodings += " " + str(chart["encoding"]["x"]["aggregate"])
        elif "bin" in chart["encoding"]["x"]:
            encodings += " bin"
        else:
            encodings += " none"
    else:
        encodings += " none none"

    if "y" in chart["encoding"]:
        if "field" in chart["encoding"]["y"]:
            encodings += " " + chart["encoding"]["y"]["field"]
        else:
            encodings += " none"

        if "aggregate" in chart["encoding"]["y"]:
            encodings += " " + str(chart["encoding"]["y"]["aggregate"])
        elif "bin" in chart["encoding"]["y"]:
            encodings += " bin"
        else:
            encodings += " none"
    else:
        encodings += " none none"

    if "color" in chart["encoding"]:
        if "field" in chart["encoding"]["color"]:
            encodings += " " + chart["encoding"]["color"]["field"]
        else:
            encodings += " none"
        if "aggregate" in chart["encoding"]["color"]:
            encodings += " " + chart["encoding"]["color"]["aggregate"]
        else:
            encodings += " none"
    else:
        encodings += " none none"

    if "theta" in chart["encoding"]:
        if "field" in chart["encoding"]["theta"]:
            encodings += " " + chart["encoding"]["theta"]["field"]
        else:
            encodings += " none"
        if "aggregate" in chart["encoding"]["theta"]:
            encodings += " " + chart["encoding"]["theta"]["aggregate"]
        else:
            encodings += " none"
    else:
        encodings += " none none"

    f_ture = False
    if "transform" in chart:
        for f in chart["transform"]:
            if "filter" in f:
                f_ture = True
                filters = f["filter"].replace("datum.", "").replace(" ", "")
                encodings += f" {filters}"
                break
        if not f_ture:
            encodings += " none none"
    else:
        encodings += " none none"

    s_ture = False
    if "x" in chart["encoding"]:
        if "sort" in chart["encoding"]["x"]:
            s_ture = True
            if chart["encoding"]["x"]["sort"] == "-y":
                encodings += f" y desc"
            elif chart["encoding"]["x"]["sort"] == "y":
                encodings += f" y asc"
    if "y" in chart["encoding"]:
        if "sort" in chart["encoding"]["y"]:
            s_ture = True
            if chart["encoding"]["y"]["sort"] == "-x":
                encodings += f" x desc"
            elif chart["encoding"]["y"]["sort"] == "x":
                encodings += f" x asc"
    if not s_ture:
        encodings += " none none"
    return encodings