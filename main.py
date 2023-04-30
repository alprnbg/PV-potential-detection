import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go
import requests
import ast
import pandas as pd
from PIL import Image
import dash_bootstrap_components as dbc
import plotly.express as px
import os


import numpy as np
import math
from pyproj import Transformer

# Information extracted from the dataset header
XLLCORNER = 3280500
YLLCORNER = 5237500
NROWS = 866
CELLSIZE = 1000
NODATA_VALUE = -999

# Load data as 2d array
data = np.loadtxt("data/images/grids_germany_annual_radiation_global_2022.asc", skiprows=28)
data[data == -999] = np.nan

# Define coordinate systems
from_crs = "EPSG:4326"  # WGS 84
to_crs = "EPSG:31467"  # Gauss Krüger Zone 3

# Create transformer object
transformer = Transformer.from_crs(from_crs, to_crs)



# Define the bounding box of the region in Munich (latitude, longitude)
bbox = (48.139394, 11.564956, 48.149394, 11.574956)

traces = []
df = pd.read_csv('data/images/roof_houses_all_output.csv')
#print(len(df))
df = df[np.logical_or(df["orientation"]=="True", df["orientation"]=="False")]
df = df[df["orientation"].notnull()]
df = df.sample(frac=1).iloc[:1000]
#print(df["orientation"].value_counts())
#df = df[df["orientation"].notnull()]
for i, row in df.iterrows():
    center = ast.literal_eval(row['center'])
    trace = go.Scattermapbox(
        lon=[center[0]],
        lat=[center[1]],
        mode="markers",
        marker=dict(size=10),
        hoverinfo="text",
        text="selectedBuilding",
        customdata=[[row["area"], row["orientation"], row["id"], row["potential_output"]]],
        name="buildingName",
        visible=True,
    )
    traces.append(trace)

# Create a layout with a map centered on the region in Munich
layout = go.Layout(
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=(bbox[0]+bbox[2])/2, lon=(bbox[1]+bbox[3])/2),
        zoom=4,
    ),
)

# Create a Figure with the building traces and layout
fig = go.Figure(data=traces, layout=layout)
fig.update_layout(showlegend=False,
title={
        'text': "All Buildings to Install Solar Panels On",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}
)
fig.update_layout(shapes=[go.layout.Shape(
    type='rect',
    xref='paper',
    yref='paper',
    x0=0,
    y0=0,
    x1=1.0,
    y1=1.0,
    line={'width': 1, 'color': 'black'}
)])

fig.update_layout(height=800)



traces3 = []
df3 = pd.read_csv('data/images/top_100.csv')
df3 = df3[df3["orientation"].notnull()]
for i, row in df3.iterrows():
    center = ast.literal_eval(row['center'])
    trace = go.Scattermapbox(
        lon=[center[0]],
        lat=[center[1]],
        mode="markers",
        marker=dict(size=10),
        hoverinfo="text",
        text="selectedBuilding",
        customdata=[[row["area"], row["orientation"], row["id"], row["potential_output"]]],
        name="buildingName",
        visible=True,
    )
    traces3.append(trace)

# Create a layout with a map centered on the region in Munich
layout3 = go.Layout(
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=(bbox[0]+bbox[2])/2, lon=(bbox[1]+bbox[3])/2),
        zoom=4,
    ),
)

# Create a Figure with the building traces and layout
fig3 = go.Figure(data=traces3, layout=layout3)
fig3.update_layout(showlegend=False,
title={
        'text': "The Most Efficient Buildings to Install Solar Panels On",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}
)
fig3.update_layout(shapes=[go.layout.Shape(
    type='rect',
    xref='paper',
    yref='paper',
    x0=0,
    y0=0,
    x1=1.0,
    y1=1.0,
    line={'width': 1, 'color': 'black'}
)])

fig3.update_layout(height=800)



import json
with open("4_niedrig.geo.json", encoding="utf-8") as response:
    counties = json.load(response)

features_df = pd.read_csv("features_df.csv")

fig2 = px.choropleth_mapbox(features_df, geojson=counties, locations='id', color='Radiance',
                           color_continuous_scale="Turbo",
                           mapbox_style="carto-positron",
                           zoom=4, center = {"lat": 50.896636, "lon": 10.3456715},
                           opacity=0.5,
                           labels={'Radiance':'Radiance (kWh/m2)', 'NAME_0': 'Country', "NAME_1": "State", "NAME_2": "District", "NAME_3": "County"},
                           hover_data=["NAME_0", "NAME_1", "NAME_2", "NAME_3"]
                          )
fig2.update_layout(margin={"r":0,"t":0,"l":0,"b":0})


# Create the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


nav_bar = dbc.Navbar(
    [
        html.A(
            dbc.Row([dbc.Col(dbc.NavbarBrand("photongraphy | genistat"))],
                    align="navbar-center"),
            className="navbar-nav mx-auto"),
        dbc.NavbarToggler(id="navbar-toggler"),
    ],
    color="dark",
    dark=True,
    className="mb-5")


# Define the app layout
app.layout = html.Div([
    nav_bar,
    dcc.Tabs(id="tabs-styled-with-props", value='tab-1', children=[
    dcc.Tab(label='Germany Map', value='tab-1',
    children=html.Div([
    dcc.Graph(id='map', figure=fig),
    html.Div(
        id='output-table',
        style={
            'margin': 'auto',
            'width': '1000px',
            'border': '1px solid #ddd',
            'border-radius': '5px',
            'padding': '10px',
            'text-align': 'center'
        }
    ),
    html.Div(id='output-image', style={'width': '25%', 'display': 'inline-block', 'padding-left': '30px'}),
    html.Div(id='output-image2', style={'width': '25%', 'display': 'inline-block', 'padding-left': '30px'}),
    html.Div(id='output-image3', style={'width': '25%', 'display': 'inline-block', 'padding-left': '30px'}),
    html.Div(id='output-image4', style={'width': '25%', 'display': 'inline-block', 'padding-left': '30px'}),
    html.Br(),
    html.Br()
    ])
),

dcc.Tab(label='Germany Map Top 100', value='tab-2',
children=html.Div([
dcc.Graph(id='map3', figure=fig3),
html.Div(
    id='output-table3',
    style={
        'margin': 'auto',
        'width': '1000px',
        'border': '1px solid #ddd',
        'border-radius': '5px',
        'padding': '10px',
        'text-align': 'center'
    }
),
html.Div(id='output-imagee', style={'width': '25%', 'display': 'inline-block', 'padding-left': '20px'}),
html.Div(id='output-imagee2', style={'width': '25%', 'display': 'inline-block'}),
html.Div(id='output-imagee3', style={'width': '25%', 'display': 'inline-block'}),
html.Div(id='output-imagee4', style={'width': '25%', 'display': 'inline-block'}),
html.Br(),
html.Br()
])
)
,

dcc.Tab(label='Statistics', value='tab-3',
children=html.Div([
html.Div([
    html.H1('Overall Statistics'),
    html.Div([
        html.P('German primary yearly energy consumption coming from fossil sources amounted to 9144.75 Petajoule, which is equal to 2540411550000 kWh'),
        html.P("Average roof area of the houses in Germany is 129.92 m2"),
        html.P("Average solar panel could produce 16156 kWh per year per household"),
        html.P("So, we need to equip 157242606 different houses on average to eliminate fossil fuel dependency completely")
    ])
])


])
),


dcc.Tab(label='Germany Radiation Map', value='tab-4',
children=html.Div([
dcc.Graph(id='map2', figure=fig2)


])
),dcc.Tab(label='Manual Energy Calculation', value='tab-5',
children=html.Div([
html.Header("Calculate Energy Production of a Solar Panel"),
        html.Br(),
        html.I("Area of the panel (m2)"),
        html.I("Latitude", style={'padding-left': '35px'}),
        html.I("Longitude", style={'padding-left': '135px'}),
        html.I("Azimuth", style={'padding-left': '120px'}),
        html.I("Solar panel yield", style={'padding-left': '130px'}),
        html.I("Performance ratio", style={'padding-left': '75px'}),
        html.Br(),
        dcc.Input(id="input1-energy", type="number", value=0),
        dcc.Input(id="input2-energy", type="number", value=48.137827),
        dcc.Input(id="input3-energy", type="number", value=11.574949),
        dcc.Input(id="input4-energy", type="number", value=0),
        dcc.Input(id="input5-energy", type="number", value=0.15),
        dcc.Input(id="input6-energy", type="number", value=0.75),
        html.Br(),
        html.Br(),
        html.Div(id="output-energy"),
        html.Br(),
        html.Div(id="output-energy2"),
        html.Br(),
        html.Div(id="output-energy3")

])
)



]

)


])


@app.callback(
    dash.dependencies.Output("output-energy", "children"),
    dash.dependencies.Output("output-energy2", "children"),
    dash.dependencies.Output("output-energy3", "children"),
    dash.dependencies.Input("input1-energy", "value"),
    dash.dependencies.Input("input2-energy", "value"),
    dash.dependencies.Input("input3-energy", "value"),
    dash.dependencies.Input("input4-energy", "value"),
    dash.dependencies.Input("input5-energy", "value"),
    dash.dependencies.Input("input6-energy", "value"),
)
def update_output(input1, input2, input3, input4, input5, input6):

    # Coordinates of TU Munich
    latitude, longitude = input2, input3


    try:
        # Convert latitude and longitude to Gauss Krüger coordinates
        h, r = transformer.transform(latitude, longitude)

        y, x = math.floor((r - XLLCORNER) / CELLSIZE), NROWS - math.ceil((h - YLLCORNER) / CELLSIZE)
        radiance = data[x, y]
    except:
        radiance = 0


    potential_output = input1 * radiance * input5 * input6
    if (input4>45 and input4<135) or (input4>225 and input4<315):
        potential_output *= 0.85
    if radiance!=0:
        return (f"{potential_output} kWh", f"Yearly total sun radiation in your region: {radiance} kWh", f"You could save up to {potential_output*0.4} euros yearly if you build a solar panel in your house.")
    if radiance==0:
        return (f"0 kWh", f"Your region is out of Germany.", f"You could save up to 0 euros yearly if you build a solar panel in your house.")


# Define the callback to update the table
@app.callback([
    dash.dependencies.Output('output-table', 'children'),
    dash.dependencies.Output('output-image', 'children'),
    dash.dependencies.Output('output-image2', 'children'),
    dash.dependencies.Output('output-image3', 'children'),
    dash.dependencies.Output('output-image4', 'children'),
    dash.dependencies.Input('map', 'clickData')])
def update_table(clickData):
    if clickData:
        padding = '50px'
        lat = round(clickData['points'][0]['lat'], 3)
        lon = round(clickData['points'][0]['lon'], 3)
        customdata = round(clickData['points'][0]['customdata'][0], 3)
        orientation = clickData['points'][0]['customdata'][1]

        if orientation=="True":
            orientation = "South-North"
        elif orientation=="False":
            orientation = "East-West"
        else:
            orientation = "Unknown"
        base_img_path = clickData['points'][0]['customdata'][2]
        base_img_path += ".jpg"

        potential_kwh = clickData['points'][0]['customdata'][3]


        img_src1 = os.path.join("data", "images", "raw", base_img_path)

        try:
            pil_img1 = Image.open(img_src1).resize((240, 240))
        except:
            pil_img1 = ""


        img_src2 = os.path.join("data", "images", "roof_images", base_img_path)
        try:
            pil_img2 = Image.open(img_src2).resize((240, 240))
        except:
            pil_img2 = ""


        img_src3 = os.path.join("data", "images", "segments", base_img_path)
        try:
            pil_img3 = Image.open(img_src3).resize((240, 240))
        except:
            pil_img3 = ""

        img_src4 = os.path.join("data", "images", "orientation", base_img_path)
        try:
            pil_img4 = Image.open(img_src4).resize((240, 240))
        except:
            pil_img4 = ""



        return (html.Table(
            style={'border': '1px solid black', 'border-collapse': 'separate', 'margin': '0px'},
            children=[
                html.Thead(
                    html.Tr(
                        [html.Th('LATITUDE', style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Th('LONGITUDE', style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Th('AREA', style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Th('ORIENTATION', style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Th('KWH POTENTIAL', style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'})],
                        style={'background-color': 'lightblue', 'text-align': 'center'}
                    )
                ),
                html.Tbody(
                    html.Tr(
                        [html.Td(lat, style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Td(lon, style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Td(str(customdata)+" m2", style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Td(orientation, style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Td(potential_kwh, style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'})],
                        style={'text-align': 'center'}
                    )
                )
            ]),
            html.Img(src=pil_img1),
            html.Img(src=pil_img2),
            html.Img(src=pil_img3),
            html.Img(src=pil_img4)
        )
    else:
        padding = '50px'
        img_src = 'image_1.png'
        pil_img = Image.open(img_src)
        return (html.Table(
            style={'border': '1px solid black', 'border-collapse': 'separate', 'margin': '0px'},
            children=[
                html.Thead(
                    html.Tr(
                        [html.Th('LATITUDE', style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Th('LONGITUDE', style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Th('AREA', style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Th('ORIENTATION', style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Th('KWH POTENTIAL', style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'})],
                        style={'background-color': 'lightblue', 'text-align': 'center'}
                    )
                ),
                html.Tbody(
                    html.Tr(
                        [html.Td('-', style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Td('-', style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Td('-', style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Td("-", style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Td("-", style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'})],
                        style={'text-align': 'center'}
                    )
                )
            ]
        ),
        html.Img(src=""),
        html.Img(src=""),
        html.Img(src=""),
        html.Img(src="")
        )



# Define the callback to update the table
@app.callback([
    dash.dependencies.Output('output-table3', 'children'),
    dash.dependencies.Output('output-imagee', 'children'),
    dash.dependencies.Output('output-imagee2', 'children'),
    dash.dependencies.Output('output-imagee3', 'children'),
    dash.dependencies.Output('output-imagee4', 'children'),
    dash.dependencies.Input('map3', 'clickData')])
def update_tablee(clickData):
    if clickData:
        padding = '50px'
        lat = round(clickData['points'][0]['lat'], 3)
        lon = round(clickData['points'][0]['lon'], 3)
        customdata = round(clickData['points'][0]['customdata'][0], 3)
        orientation = clickData['points'][0]['customdata'][1]
        if orientation=="True":
            orientation = "South-North"
        elif orientation=="False":
            orientation = "East-West"
        else:
            orientation = "Unknown"
        base_img_path = clickData['points'][0]['customdata'][2]
        base_img_path += ".jpg"

        potential_kwh = clickData['points'][0]['customdata'][3]


        img_src1 = os.path.join("data", "images", "raw", base_img_path)

        try:
            pil_img1 = Image.open(img_src1).resize((240, 240))
        except:
            pil_img1 = ""


        img_src2 = os.path.join("data", "images", "roof_images", base_img_path)
        try:
            pil_img2 = Image.open(img_src2).resize((240, 240))
        except:
            pil_img2 = ""


        img_src3 = os.path.join("data", "images", "segments", base_img_path)
        try:
            pil_img3 = Image.open(img_src3).resize((240, 240))
        except:
            pil_img3 = ""

        img_src4 = os.path.join("data", "images", "orientation", base_img_path)
        try:
            pil_img4 = Image.open(img_src4).resize((240, 240))
        except:
            pil_img4 = ""



        return (html.Table(
            style={'border': '1px solid black', 'border-collapse': 'separate', 'margin': '0px'},
            children=[
                html.Thead(
                    html.Tr(
                        [html.Th('LATITUDE', style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Th('LONGITUDE', style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Th('AREA', style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Th('ORIENTATION', style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Th('KWH POTENTIAL', style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'})],
                        style={'background-color': 'lightblue', 'text-align': 'center'}
                    )
                ),
                html.Tbody(
                    html.Tr(
                        [html.Td(lat, style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Td(lon, style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Td(str(customdata)+" m2", style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Td(orientation, style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Td(potential_kwh, style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'})],
                        style={'text-align': 'center'}
                    )
                )
            ]),
            html.Img(src=pil_img1),
            html.Img(src=pil_img2),
            html.Img(src=pil_img3),
            html.Img(src=pil_img4)
        )
    else:
        padding = '50px'
        img_src = 'image_1.png'
        pil_img = Image.open(img_src)
        return (html.Table(
            style={'border': '1px solid black', 'border-collapse': 'separate', 'margin': '0px'},
            children=[
                html.Thead(
                    html.Tr(
                        [html.Th('LATITUDE', style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Th('LONGITUDE', style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Th('AREA', style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Th('ORIENTATION', style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Th('KWH POTENTIAL', style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'})],
                        style={'background-color': 'lightblue', 'text-align': 'center'}
                    )
                ),
                html.Tbody(
                    html.Tr(
                        [html.Td('-', style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Td('-', style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Td('-', style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Td("-", style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'}), html.Td("-", style={'padding-left': padding, 'padding-right': padding, 'text-align': 'center'})],
                        style={'text-align': 'center'}
                    )
                )
            ]
        ),
        html.Img(src=""),
        html.Img(src=""),
        html.Img(src=""),
        html.Img(src="")
        )


# Run the application
if __name__ == "__main__":
    app.run_server(debug=True)
