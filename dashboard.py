import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Load and preprocess data
try:
    df = pd.read_csv(r"C:\Users\adars\OneDrive\Desktop\jupyter projects\earthquake_data.csv")
    print("Dataset loaded successfully. Shape:", df.shape)
    print("Sample of latitude column:", df['latitude'].head().tolist())
    if df.empty or df['latitude'].isna().all():
        print("Error: DataFrame is empty or 'latitude' column contains only NaN values.")
        exit(1)
except FileNotFoundError:
    print("Error: Dataset not found at specified path. Please check the file path.")
    exit(1)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Preprocessing
df['Station Count'] = df['Station Count'].fillna(0)
df['Azimuth Gap'] = df['Azimuth Gap'].fillna(0)
df['Distance'] = df['Distance'].fillna(0)
df['RMS'] = df['RMS'].fillna(0)
df['horizontalError'] = df['horizontalError'].fillna(0)
df['magError'] = df['magError'].fillna(0)
df['magNst'] = df['magNst'].fillna(0)
df['time'] = pd.to_datetime(df['time'], format='ISO8601', utc=True, errors='coerce')
if df['time'].isna().any():
    print("Warning: Some 'time' values could not be parsed. Rows with invalid 'time' will be dropped.")
    df = df.dropna(subset=['time'])
print("Time column parsed. Shape after time drop:", df.shape)
df['region'] = df['place'].str.extract(r',\s*(.*)').fillna('Unknown')
df['Magnitude_Category'] = df['mag'].apply(
    lambda x: 'Below 2.5' if x < 2.5 else ('2.5 - 4.5' if x <= 4.5 else 'Above 4.5')
)

# Check and create bins only if latitude has valid data
if not df['latitude'].empty and not df['latitude'].isna().all():
    df['lat_bin'] = pd.cut(df['latitude'], bins=20, labels=False)
    df['lon_bin'] = pd.cut(df['longitude'], bins=20, labels=False)
else:
    print("Error: 'latitude' or 'longitude' column is empty or contains only NaN values. Binning skipped.")
    exit(1)

# Initialize Dash app
app = Dash(__name__)

# Gradient background style
app.layout = html.Div([
    # Header with logo and title moved to left
    html.Div([
        html.Img(src="/assets/planet-earth.png", 
                 style={'verticalAlign': 'middle', 'marginRight': '15px', 'width': '60px', 'height': '60px'}),
        html.H1("Earthquake Analytics Dashboard", style={
            'color': '#FF6F61', 'fontSize': '48px', 'padding': '15px 0', 'textShadow': '3px 3px 6px #000',
            'fontWeight': 'bold', 'display': 'inline-block', 'letterSpacing': '1px'
        })
    ], style={
        'background': 'linear-gradient(135deg, #1a1a1a, #2a2a2a)', 'borderBottom': '4px solid #FF6F61',
        'display': 'flex', 'alignItems': 'center', 'padding': '10px 20px', 'justifyContent': 'flex-start'
    }),

    # Main layout with sidebar and content
    html.Div([
        # Sidebar
        html.Div([
            html.H3("Controls", style={'color': '#FF8E53', 'fontSize': '26px', 'marginBottom': '25px', 'textShadow': '1px 1px 3px #000'}),
            html.Label("Select Tab:", style={'color': '#FFCC70', 'fontSize': '16px'}),
            dcc.RadioItems(
                id='tab-selector',
                options=[
                    {'label': 'Overview', 'value': 'overview'},
                    {'label': 'Depth & Magnitude', 'value': 'depth-mag'},
                    {'label': 'Regional Analysis', 'value': 'regional'},
                    {'label': 'Magnitude Breakdown', 'value': 'mag-breakdown'}
                ],
                value='overview',
                style={'color': '#FFCC70', 'marginBottom': '30px', 'fontSize': '14px'}
            ),
            html.Label("Time Range:", style={'color': '#FFCC70', 'fontSize': '16px'}),
            dcc.DatePickerRange(
                id='date-picker',
                min_date_allowed=df['time'].min().date(),
                max_date_allowed=df['time'].max().date(),
                start_date=df['time'].min().date(),
                end_date=df['time'].max().date(),
                style={'marginBottom': '30px', 'width': '100%', 'fontSize': '14px'}
            ),
            html.Label("Magnitude Range:", style={'color': '#FFCC70', 'fontSize': '16px'}),
            dcc.RangeSlider(
                id='mag-slider',
                min=df['mag'].min(),
                max=df['mag'].max(),
                step=0.1,
                value=[df['mag'].min(), df['mag'].max()],
                marks={i: str(i) for i in range(int(df['mag'].min()), int(df['mag'].max()) + 1)},
                tooltip={"placement": "bottom", "always_visible": True},
                updatemode='drag'
            ),
            html.Label("Region:", style={'color': '#FFCC70', 'fontSize': '16px'}),
            dcc.Dropdown(
                id='region-dropdown',
                options=[{'label': 'All', 'value': 'All'}] + [
                    {'label': region, 'value': region} for region in sorted(df['region'].unique())
                ],
                value='All',
                multi=True,
                style={'marginBottom': '30px', 'fontSize': '14px', 'color': '#000000', 'backgroundColor': '#2a2a2a', 'border': '1px solid #FF6F61', 'borderRadius': '5px'}
            ),
            html.Label("Depth Range:", style={'color': '#FFCC70', 'fontSize': '16px'}),
            dcc.RangeSlider(
                id='depth-slider',
                min=df['depth'].min(),
                max=df['depth'].max(),
                step=1,
                value=[df['depth'].min(), df['depth'].max()],
                marks={i: str(i) for i in range(int(df['depth'].min()), int(df['depth'].max()) + 1, 100)},
                tooltip={"placement": "bottom", "always_visible": True},
                updatemode='drag'
            ),
            html.Button("Reset Filters", id="btn-reset", style={
                'background': 'linear-gradient(45deg, #FF6F61, #FF8E53)', 'color': '#000000', 'border': 'none', 'padding': '15px 30px',
                'cursor': 'pointer', 'marginTop': '30px', 'width': '100%', 'fontSize': '18px', 'borderRadius': '10px', 'boxShadow': '2px 2px 5px #000'
            }),
            html.Button("Download Data", id="btn-download", style={
                'background': 'linear-gradient(45deg, #FF6F61, #FF8E53)', 'color': '#000000', 'border': 'none', 'padding': '15px 30px',
                'cursor': 'pointer', 'marginTop': '20px', 'width': '100%', 'fontSize': '18px', 'borderRadius': '10px', 'boxShadow': '2px 2px 5px #000'
            }),
            dcc.Download(id="download-data"),
            html.Div([
                html.P("Developed by: Adarsh Kumar", style={'color': '#FFCC70', 'fontSize': '16px', 'marginTop': '30px', 'textAlign': 'center', 'fontWeight': 'bold'}),
                html.P("Data Science Student | Aspiring Data Analyst", style={'color': '#FFCC70', 'fontSize': '14px', 'textAlign': 'center'}),
                html.P("Email: adarshsingh6534@gmail.com", style={'color': '#FFCC70', 'fontSize': '14px', 'textAlign': 'center'})
            ], style={'padding': '20px', 'borderTop': '1px solid #FF6F61', 'marginTop': '40px', 'textAlign': 'center'})
        ], style={
            'width': '25%', 'background': 'linear-gradient(135deg, #2a2a2a, #3a3a3a)', 'padding': '30px', 'height': '100vh',
            'color': 'white', 'overflowY': 'auto', 'borderRight': '3px solid #FF6F61', 'boxShadow': '5px 0 10px rgba(0,0,0,0.5)'
        }),

        # Main content with dynamic layout and graph selector
        html.Div([
            dcc.Dropdown(
                id='graph-selector-dropdown',
                options=[
                    {'label': 'None', 'value': 'none'},
                    {'label': '3D Depth Visualization', 'value': '3d', 'disabled': True},
                    {'label': 'High-Risk Zones', 'value': 'risk', 'disabled': True}
                ],
                value='none',
                style={'width': '200px', 'marginBottom': '20px', 'fontSize': '16px', 'color': '#000000', 'background': 'linear-gradient(45deg, #FF6F61, #FF8E53)', 'border': '1px solid #FF6F61', 'borderRadius': '5px', 'display': 'none'}
            ),
            html.Div(id='tab-content', style={'padding': '30px'})
        ], style={'width': '75%', 'background': 'linear-gradient(135deg, #121212, #1e1e1e)', 'minHeight': '100vh'})
    ], style={'display': 'flex', 'flexDirection': 'row', 'animation': 'fadeIn 1s'})
], style={'background': 'linear-gradient(135deg, #1a1a1a, #2a2a2a, #3a3a3a)', 'minHeight': '100vh'})
