import dash
import dash_bootstrap_components as dbc
from dash import html

default_title = "What .net version to choose?"

default_text = """\
I am writing public .net class library version for our online rest service and i can't decide which version of .net to choose. 
I would like to use .net 4.0 version but such compiled class library can't be used in .net 2.0 version? Maybe there is statistic on how many developers use .net 2.0 version ?
"""


def get_app() -> dash.Dash:
    app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = dbc.Container(
        [
            html.Div(
                [
                    dbc.Label(
                        children="Enter a title",
                        style={"margin": "10px"},
                    ),
                    dbc.Input(
                        value=default_title,
                        id="title",
                        style={"width": "800px"},
                    ),
                ],
                style={"margin": "10px"},
            ),
            html.Div(
                [
                    dbc.Label(
                        children="Enter a text",
                        style={"margin": "10px"},
                    ),
                    dbc.Textarea(
                        value=default_text,
                        id="text",
                        style={"height": "240px", "width": "800px"},
                    ),
                ],
                style={"margin": "10px"},
            ),
            dbc.Button(children="Predict", id="predict"),
            html.Div(
                [
                    dbc.Label(
                        "Predicted tags :",
                        style={"margin": "10px"},
                    ),
                    dbc.Container([], id="container"),
                ]
            ),
        ],
    )

    return app
