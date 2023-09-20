import urllib.parse as url_parse

import dash
import dash_bootstrap_components as dbc
import requests
from dash.dependencies import Input, Output
from ui import get_app

app = get_app()


@app.callback(
    Output("container", "children"),
    [Input("predict", "n_clicks")],
    [Input("title", "value"), Input("text", "value")],
    prevent_initial_call=True,
)
def update_title(n_clicks, title, text):
    ctx = dash.ctx
    if ctx.triggered and ctx.triggered[0]["prop_id"] == "predict.n_clicks":
        title = url_parse.quote_plus(title)
        text = url_parse.quote_plus(text)

        req = f"http://localhost:8000/predict?{title=}&{text=}"
        print("[INFO] Requesting at", req)

        res = requests.get(req)

        preds = res.json()

        print("[INFO] res = ", preds)

        return [dbc.Badge(children=tag, style={"margin": "10px"}) for tag in preds]


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0")
