import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf  
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats
from scipy.stats import kurtosis, shapiro, jarque_bera
import numpy as np
from dash.dash_table import DataTable
from dash.exceptions import PreventUpdate



# Leer y concatenar archivos de texto
files = [
    "/Users/juansebastianquintanacontreras/Desktop/Dash_avan#2_PF/traiding 01-02-2023_part1.txt",
    "/Users/juansebastianquintanacontreras/Desktop/Dash_avan#2_PF/traiding 01-02-2023_part2.txt",
    "/Users/juansebastianquintanacontreras/Desktop/Dash_avan#2_PF/traiding 01-02-2023_part3.txt"
]

dfs = [pd.read_csv(file, sep=",", header=0) for file in files]
dftra = pd.concat(dfs, ignore_index=True)

# Leer el archivo CSV con los resultados de los modelos
resultados_ml = pd.read_csv('/Users/juansebastianquintanacontreras/Desktop/Dash_avan#2_PF/results_project_models.csv')

# Eliminar espacios en los nombres de las columnas
dftra.columns = dftra.columns.str.strip()

# Seleccionar solo las columnas deseadas
columnas_deseadas = ['fecha', 'hora_exacta', 'precio_bid', 'volumen_bid', 'precio_ask', 'volumen_ask']
dftra = dftra[columnas_deseadas]

# Obtener estadísticas descriptivas solo para las columnas deseadas y añadir etiquetas de estadísticos
stats = dftra.describe().transpose()
stats.index.name = 'Estadísticas'
stats = stats.rename(columns={
    'count': 'Cuenta',
    'mean': 'Media',
    'std': 'Desviación estándar',
    'min': 'Mínimo',
    '25%': 'Percentil 25',
    '50%': 'Mediana',
    '75%': 'Percentil 75',
    'max': 'Máximo'
})

# Filtrar columnas necesarias
columnasf = ['fecha', 'hora_exacta', 'precio_bid', 'volumen_bid', 'precio_ask']
dftra = dftra[columnasf]

# Convertir 'fecha' a formato datetime y manejar errores, especificando el formato
# Cambia '%Y-%m-%d %H:%M:%S' por el formato correcto si los datos tienen una estructura diferente
dftra['fecha'] = pd.to_datetime(dftra['fecha'], format='mixed')

# Elimina filas donde 'fecha' o 'precio_bid' sean nulos
dftra.dropna(subset=['fecha', 'precio_bid'], inplace=True)

# Si 'fecha' aún está en las columnas, crea 'fechac' y establece 'fecha' como índice
if 'fecha' in dftra.columns:
    dftra['fechac'] = dftra['fecha']  # Copia de 'fecha' en 'fechac'
    dftra.set_index('fecha', inplace=True)  # Establece 'fecha' como índice

# Ahora que 'fecha' es el índice, utiliza 'fechac' para cualquier operación en columna
dftra['hora'] = dftra.index.hour  # Extrae la hora del índice 'fecha'
dftra['INTERVALO_HORAS'] = (dftra['hora'] // 2) * 2  # Crea intervalos de 2 en 2 horas
dftra['INTERVALO_HORAS'] = dftra['INTERVALO_HORAS'].apply(lambda x: f"{x:02d}:00 - {x+2:02d}:00")

# Verifica la estructura del DataFrame
# Verifica la estructura del DataFrame después de la limpieza
print("Columnas finales:", dftra.columns)
print(dftra[['fechac', 'hora', 'INTERVALO_HORAS']].head())


# Histograma para 'precio_bid' en Dash con un diseño más atractivo
def histograma_precio_bid():
    histograma = go.Figure()

    # Agregar la traza del histograma
    histograma.add_trace(go.Histogram(
        x=dftra['precio_bid'],
        nbinsx=50,
        marker=dict(color='lightblue', line=dict(color='black', width=1)),
        opacity=0.75
    ))

    # Configurar el layout del gráfico
    histograma.update_layout(
        title='Distribución de Precio Bid',
        xaxis_title='Precio Bid',
        yaxis_title='Frecuencia',
        bargap=0.2,
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgrey'),
        yaxis=dict(gridcolor='lightgrey'),
        title_font=dict(size=24, color='#264653'),
        xaxis_title_font=dict(size=16, color='#264653'),
        yaxis_title_font=dict(size=16, color='#264653'),
        font=dict(size=14, color='#264653')
    )

    return dcc.Graph(
        id='histograma-precio-bid',
        figure=histograma
    )

# Histograma para 'precio_ask' en Dash con un diseño más atractivo
def histograma_precio_ask():
    histograma = go.Figure()
    
    # Agregar la traza del histograma
    histograma.add_trace(go.Histogram(
        x=dftra['precio_ask'],
        nbinsx=50,
        marker=dict(color='lightcoral', line=dict(color='black', width=1)),
        opacity=0.75
    ))

    # Configurar el layout del gráfico
    histograma.update_layout(
        title='Distribución de Precio Ask',
        xaxis_title='Precio Ask',
        yaxis_title='Frecuencia',
        bargap=0.2,
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgrey'),
        yaxis=dict(gridcolor='lightgrey'),
        title_font=dict(size=24, color='#264653'),
        xaxis_title_font=dict(size=16, color='#264653'),
        yaxis_title_font=dict(size=16, color='#264653'),
        font=dict(size=14, color='#264653')
    )

    return dcc.Graph(
        id='histograma-precio-ask',
        figure=histograma
    )

# Calcular el total de valores NA en el DataFrame
total_na = dftra.isna().sum().sum()

def mapa_calor_correlaciones_plotly(df):
    # Seleccionar las columnas numéricas
    df_numeric = df[['hora_exacta', 'precio_bid', 'volumen_bid', 'precio_ask']]

    # Calcular la matriz de correlación
    correlation_matrix = df_numeric.corr()

    # Crear el mapa de calor
    heatmap = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='Viridis',  # Escala de colores
        colorbar=dict(title='Correlación'),
        text=correlation_matrix.values,  # Valores de correlación
        texttemplate='%{text:.2f}',  # Formato del texto
        hoverinfo='text'  # Mostrar el texto en el hover
    ))

    # Actualizar el layout
    heatmap.update_layout(
        title='Mapa de Calor de Correlaciones',
        xaxis=dict(title='Variables'),
        yaxis=dict(title='Variables'),
        plot_bgcolor='white'
    )

    return dcc.Graph(
        id='mapa-calor-correlaciones',
        figure=heatmap
    )

# Ejemplo de cómo crear fig_bid y fig_ask
fig_bid = go.Figure()
fig_bid.add_trace(go.Scatter(x=dftra.index, y=dftra['precio_bid'], mode='lines', name='Precio Bid'))

fig_ask = go.Figure()
fig_ask.add_trace(go.Scatter(x=dftra.index, y=dftra['precio_ask'], mode='lines', name='Precio Ask'))

# Función para generar gráficos ACF y PACF usando plotly
def create_acf_pacf_figures(data, title_prefix, acf_color, pacf_color, lag_count):
    acf_values = acf(data.dropna(), nlags=lag_count)
    pacf_values = pacf(data.dropna(), nlags=lag_count)

    # Crear figura de ACF
    fig_acf = go.Figure()
    fig_acf.add_trace(go.Bar(x=list(range(len(acf_values))), y=acf_values, marker_color=acf_color))
    fig_acf.update_layout(title=f'Autocorrelación - {title_prefix}', xaxis_title='Lags', yaxis_title='ACF')

    # Crear figura de PACF
    fig_pacf = go.Figure()
    fig_pacf.add_trace(go.Bar(x=list(range(len(pacf_values))), y=pacf_values, marker_color=pacf_color))
    fig_pacf.update_layout(title=f'Autocorrelación Parcial - {title_prefix}', xaxis_title='Lags', yaxis_title='PACF')

    return fig_acf, fig_pacf

# Convertir columnas específicas a numéricas, evitando 'INTERVALO_HORAS' y otras que no deberían ser numéricas
numeric_cols = ['hora_exacta', 'precio_bid', 'volumen_bid', 'precio_ask']
for col in numeric_cols:
    dftra[col] = pd.to_numeric(dftra[col], errors='coerce')

# Identificar columnas no numéricas y convertirlas a numéricas donde sea posible
non_numeric_cols = dftra.select_dtypes(exclude=['number', 'datetime']).columns
for col in non_numeric_cols:
    dftra[col] = pd.to_numeric(dftra[col], errors='coerce')

# Crear un DataFrame sin columnas no numéricas para resamplear
dftra_numeric = dftra.select_dtypes(include=['number', 'datetime'])

# Realizar el resampleo solo en las columnas numéricas
df5min = dftra_numeric.resample('5T').mean()
df30min = dftra_numeric.resample('30T').mean()
dfhora = dftra_numeric.resample('H').mean()

# Preparar los datos para el boxplot
def prepare_boxplot_data(df, period):
    df = df.reset_index()
    df['period'] = period
    return df

# Preparar datos para boxplots
dftra_boxplot = prepare_boxplot_data(dftra, 'General')
df5min_boxplot = prepare_boxplot_data(df5min, 'Cada 5 Minutos')
df30min_boxplot = prepare_boxplot_data(df30min, 'Cada 30 Minutos')
dfhora_boxplot = prepare_boxplot_data(dfhora, 'Cada Hora')

# Combinar todos los DataFrames para graficar
combined_df = pd.concat([dftra_boxplot, df5min_boxplot, df30min_boxplot, dfhora_boxplot])

# Asegúrate de que 'fechac' esté en formato datetime
dftra['fechac'] = pd.to_datetime(dftra['fechac'])

def resample_data(df, interval):
    # Crea una copia temporal del DataFrame con 'fechac' como índice
    df_temp = df.set_index('fechac')
    # Resamplea el DataFrame y calcula el promedio para 'precio_bid'
    df_resampled = df_temp['precio_bid'].resample(interval).mean().reset_index()
    
    return df_resampled

# Función para graficar la serie original y la diferenciada
def graficar_diferencias():
    if dftra['precio_bid'].isnull().any():
        raise ValueError("No se pueden calcular diferencias, hay valores nulos en 'precio_bid'.")

    dftra['precio_bid_diff'] = dftra['precio_bid'].diff()

    fig = go.Figure()

    # Gráfico de la serie original
    fig.add_trace(go.Scatter(
        x=dftra.index,
        y=dftra['precio_bid'],
        mode='lines',
        name='Serie Original',
        line=dict(color='blue')
    ))

    # Gráfico de la serie diferenciada
    fig.add_trace(go.Scatter(
        x=dftra.index,
        y=dftra['precio_bid_diff'],
        mode='lines',
        name='Serie Diferenciada',
        line=dict(color='red')
    ))

    fig.update_layout(
        title='Serie Original y Serie Diferenciada de Precio Bid',
        xaxis_title='Fecha',
        yaxis_title='Valor',
        plot_bgcolor='white'
    )

    return dcc.Graph(
        id='grafico-diferencias',
        figure=fig
    )

def histograma_residuos(residuos):
    histograma = go.Figure()
    histograma.add_trace(go.Histogram(
        x=residuos,
        nbinsx=50,
        marker=dict(color='lightblue', line=dict(color='black', width=1)),
        opacity=0.75
    ))

    histograma.update_layout(
        title='Histograma de Residuos',
        xaxis_title='Residuos',
        yaxis_title='Frecuencia',
        bargap=0.2,
        plot_bgcolor='white'
    )

    return histograma  # Retorna la figura sin el dcc.Graph

def grafico_residuos(residuos):
    fig_residuos = go.Figure()
    fig_residuos.add_trace(go.Scatter(
        x=list(range(len(residuos))),
        y=residuos,
        mode='markers',
        name='Residuos',
        marker=dict(color='red', size=6)
    ))

    fig_residuos.update_layout(
        title='Gráfico de Residuos',
        xaxis_title='Observaciones',
        yaxis_title='Residuos',
        plot_bgcolor='white'
    )

    return fig_residuos  # Retorna la figura sin el dcc.Graph

def pacf_residuos(residuos):
    fig = go.Figure()
    pacf_values = pacf(residuos, nlags=30)  # Usa pacf directamente
    fig.add_trace(go.Bar(x=list(range(len(pacf_values))), y=pacf_values, marker_color='blue'))
    fig.update_layout(title='PACF de Residuos', xaxis_title='Lags', yaxis_title='PACF', plot_bgcolor='white')
    return fig  # Retorna la figura sin el dcc.Graph



def calcular_metricas(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calcular RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Calcular MAPE
    lst = []
    for i in range(len(y_true)):
        if y_true[i] != 0:
            num = (y_true[i] - y_pred[i]) / y_true[i]
            lst.append(num)
    mape = np.mean(lst) * 100
    
    # Calcular R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Pruebas de Ljung-Box y Jarque-Bera
    lb_test = acorr_ljungbox(y_true - y_pred, lags=[10], return_df=True)
    ljung_box_p_value = lb_test['lb_pvalue'].iloc[0]
    jb_test = jarque_bera(y_true - y_pred)
    jarque_bera_p_value = jb_test[1]

    # Calcular Shapiro-Wilk y Kurtosis
    shapiro_stat, shapiro_p_value = shapiro(y_true - y_pred)
    kurt = kurtosis(y_true - y_pred)

    return {
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'Ljung-Box p-value': ljung_box_p_value,
        'Jarque-Bera p-value': jarque_bera_p_value,
        'Shapiro-Wilk p-value': shapiro_p_value,
        'Curtosis': kurt
    }

# Inicializa la aplicación Dash
app = dash.Dash(__name__,  suppress_callback_exceptions=True)
server = app.server

app.title = "Aplicación de Análisis Financiero"

# Estilos CSS para darle un diseño más elegante
button_style = {
    'backgroundColor': '#2a9d8f', 'color': 'white', 'border': 'none', 'padding': '12px 24px',
    'margin': '5px', 'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '16px'
}

header_style = {
    'textAlign': 'center', 'backgroundColor': '#264653', 'padding': '20px', 'color': 'white',
    'borderRadius': '8px', 'marginBottom': '20px'
}

table_style = {
    'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '8px', 'boxShadow': '0px 4px 8px rgba(0,0,0,0.2)',
    'overflowX': 'auto'
}

# Agregar botones para las nuevas páginas en el layout
app.layout = html.Div(style={'backgroundColor': '#f4f4f9', 'padding': '40px'}, children=[
    dcc.Location(id='url', refresh=False),
    html.Div([
        html.H1("Proyecto Final de Machine Learning", style={'marginBottom': '0'}),
        html.H4("Panel de Visualización datos de traiding", style={'color': '#a8dadc'})
    ], style=header_style),
    html.Div([
        html.Button("EDA", id='btn-home', n_clicks=0, style=button_style),
        html.Button("Series de Tiempo", id='btn-page1', n_clicks=0, style=button_style),
        html.Button("Resampleo Precio bid", id='btn-page2', n_clicks=0, style=button_style),
        html.Button("Análisis de Diferencias", id='btn-page3', n_clicks=0, style=button_style),
        html.Button("Modelos ML", id='btn-page4', n_clicks=0, style=button_style)
    ], style={'textAlign': 'center', 'marginBottom': '30px'}),
    html.Div(id='page-content')
])

# Página principal con estadísticas descriptivas
def homepage_layout():
    # Página principal con estadísticas descriptivas
    return html.Div(style=table_style, children=[
    html.H2("Página Principal: Datos Generales", style={'textAlign': 'center', 'color': '#264653'}),

    # Mostrar el total de NA
    html.Div(style={'textAlign': 'center', 'color': '#264653', 'margin': '20px'}, children=[
        html.H4(f"Total NA en el DataFrame: {total_na}", style={'fontSize': '18px'})
    ]),
    
    # Div que contiene las dos tablas en un contenedor flex
    
    html.Div(style={'display': 'flex', 'justify-content': 'space-around', 'flex-wrap': 'wrap'}, children=[
        # Tabla que muestra el DataFrame con columnas específicas
        html.H4("Datos a usar", style={'textAlign': 'center', 'color': '#264653'}),
            dcc.Graph(
            id='general-data-table',
            figure={
                'data': [{
                    'type': 'table',
                    'header': {
                        'values': list(dftra.columns),
                        'fill': {'color': '#264653'},
                        'font': {'color': 'white', 'size': 12},
                        'align': 'center'
                    },
                    'cells': {
                        'values': [dftra[col].tolist() for col in dftra.columns],
                        'fill': {'color': ['#f1faee', '#e9ecef']},
                        'align': 'center'
                    }
                }]
            }
        ),

        # Tabla que muestra las estadísticas descriptivas
        html.Div(style={'width': '800px'}, children=[
            html.H4("Estadísticas Descriptivas", style={'textAlign': 'center', 'color': '#264653'}),
            dcc.Graph(
                id='stats-table',
                figure={
                    'data': [{
                        'type': 'table',
                        'header': {
                            'values': ['Estadísticas'] + stats.columns.tolist(),
                            'fill': {'color': '#264653'},
                            'font': {'color': 'white', 'size': 12},
                            'align': 'center'
                        },
                        'cells': {
                            'values': [stats.index.tolist()] + [stats[col].tolist() for col in stats.columns],
                            'fill': {'color': ['#f1faee', '#e9ecef']},
                            'align': 'center'
                        }
                    }]
                }
            ),
        ])
    ]),

    # Histograma de 'precio_bid' dentro de un Div
    html.Div([
        html.H4("Histograma de Precio Bid", style={'textAlign': 'center', 'color': '#264653'}),
        histograma_precio_bid()  # Llama a la función para agregar el histograma
    ]), 
     # Histograma de 'precio_ask' dentro de un Div
    html.Div([
        html.H4("Histograma de Precio Ask", style={'textAlign': 'center', 'color': '#264653'}),
        histograma_precio_ask()  # Llama a la función para agregar el histograma
    ]),
    # Mapa de calor de correlaciones
    html.Div([
        html.H4("Mapa de Calor de Correlaciones", style={'textAlign': 'center', 'color': '#264653'}),
        mapa_calor_correlaciones_plotly(dftra),
    ]) 
])

# Layout de la página de series de tiempo
def page1_layout():
    return html.Div(style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '8px', 'boxShadow': '0px 4px 8px rgba(0,0,0,0.2)', 'overflowX': 'auto'}, children=[
        html.H1('Visualización de Series de Tiempo'),
        dcc.Dropdown(
            id='serie-dropdown',
            options=[
                {'label': 'Precio Bid', 'value': 'precio_bid'},
                {'label': 'Precio Ask', 'value': 'precio_ask'}
            ],
            value='precio_bid',  # Valor por defecto
            placeholder="Seleccione una serie"
        ),
        html.Div(style={'display': 'flex', 'flexDirection': 'column'}, children=[
            dcc.Graph(id='serie-grafico', style={'flex': '1 1 50%'}),
            html.Div(id='acf-pacf-plots', style={'display': 'flex', 'flexDirection': 'row', 'width': '100%'})
        ])
    ])

# Layout de la página de resampleo con opciones para el boxplot
def page2_layout():
    return html.Div(style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '8px', 'boxShadow': '0px 4px 8px rgba(0,0,0,0.2)', 'overflowX': 'auto'}, children=[
        html.H1('Visualización de Resampleo de Precio Bid'),
        
        # Dropdown para seleccionar el intervalo de resampleo
        dcc.Dropdown(
            id='interval-dropdown',
            options=[
                {'label': 'Cada 5 Minutos', 'value': '5T'},
                {'label': 'Cada 30 Minutos', 'value': '30T'},
                {'label': 'Cada Hora', 'value': 'H'}
            ],
            placeholder="Seleccione un intervalo de resampleo"
        ),

        # Gráfico de la serie resampleada
        dcc.Graph(id='resampled-grafico'),

        html.Hr(),

        # Gráficos de ACF y PACF en paralelo
        html.Div([
            dcc.Graph(id='acf-graph', style={'display': 'inline-block', 'width': '49%'}),
            dcc.Graph(id='pacf-graph', style={'display': 'inline-block', 'width': '49%'})
        ]),

        html.Hr(),

        # Dropdown para seleccionar el período del boxplot
        dcc.Dropdown(
            id='period-dropdown',
            options=[
                {'label': 'General', 'value': 'General'},
                {'label': 'Cada 5 Minutos', 'value': 'Cada 5 Minutos'},
                {'label': 'Cada 30 Minutos', 'value': 'Cada 30 Minutos'},
                {'label': 'Cada Hora', 'value': 'Cada Hora'}
            ],
            value='General',
            clearable=False,
            style={'width': '50%', 'margin': '20px auto'}
        ),
        
        # Gráfico del boxplot
        dcc.Graph(id='boxplot-graph')
    ])

# Layout de la página de análisis de diferencias
def page3_layout():
    return html.Div(style={
        'padding': '20px', 
        'backgroundColor': 'white', 
        'borderRadius': '8px', 
        'boxShadow': '0px 4px 8px rgba(0,0,0,0.2)', 
        'overflowX': 'auto'
    }, children=[
        html.H1('Análisis de Diferencias'),
        
        # Llama a la función para graficar las diferencias
        graficar_diferencias(),  # Asegúrate de que esta función devuelva un dcc.Graph

        # Dropdown para seleccionar el intervalo de resampleo
        dcc.Dropdown(
            id='diferencias-resample-dropdown',
            options=[
                {'label': 'Cada 5 Minutos', 'value': '5T'},
                {'label': 'Cada 30 Minutos', 'value': '30T'},
                {'label': 'Cada Hora', 'value': 'H'}
            ],
            placeholder="Seleccione un intervalo de resampleo"
        ),

        # Gráfico para mostrar la serie diferenciada resampleada
        dcc.Graph(id='grafico-diferencias'),

        # Gráficos de ACF y PACF para la serie diferenciada
        html.Div([
            dcc.Graph(id='acf-graph-diferencias', style={'display': 'inline-block', 'width': '49%'}),
            dcc.Graph(id='pacf-graph-diferencias', style={'display': 'inline-block', 'width': '49%'})
        ])
    ])

def page4_layout():
    return html.Div(style={
        'padding': '20px', 
        'backgroundColor': 'white', 
        'borderRadius': '8px', 
        'boxShadow': '0px 4px 8px rgba(0,0,0,0.2)', 
        'overflowX': 'auto'
    }, children=[
        html.H1('Resultados de Modelos de Machine Learning'),

        # Dropdown para seleccionar el modelo
        dcc.Dropdown(
            id='modelo-dropdown',
            options=[
                {'label': 'Modelo Lineal', 'value': 'Linear'},
                {'label': 'KNN', 'value': 'Knn'},
                {'label': 'Ridge', 'value': 'Ridge'},
                {'label': 'Lasso', 'value': 'Lasso'}
            ],
            value='Linear',  # Modelo por defecto
            placeholder="Seleccione un modelo"
        ),

        # Gráficos en un layout de 2 columnas
        html.Div(style={
            'display': 'flex', 
            'flexWrap': 'wrap', 
            'justifyContent': 'space-between',  # Para el espacio entre columnas
            'alignItems': 'stretch'  # Asegúrate de que los gráficos se alineen
        }, children=[
            dcc.Graph(id='grafico-resultados-ml', style={'flex': '0 0 48%', 'margin': '1%'}),
            dcc.Graph(id='histograma-residuos', style={'flex': '0 0 48%', 'margin': '1%'}),
            dcc.Graph(id='grafico-residuos', style={'flex': '0 0 48%', 'margin': '1%'}),
            dcc.Graph(id='pacf-plot-residuos', style={'flex': '0 0 48%', 'margin': '1%'}),
        ]),

        # Div para mostrar métricas
        html.Div(id='metricas-resultado', style={'margin-top': '20px'}),
    ])


# Modificar la navegación para incluir las nuevas páginas
@app.callback(
    Output('url', 'pathname'),
    [Input('btn-home', 'n_clicks'), Input('btn-page1', 'n_clicks'), Input('btn-page2', 'n_clicks'), Input('btn-page3', 'n_clicks'), Input('btn-page4', 'n_clicks')]
)
def navigate(n_home, n_page1, n_page2, n_page3, n_page4):
    ctx = dash.callback_context
    if not ctx.triggered:
        return '/'
    elif ctx.triggered[0]['prop_id'] == 'btn-home.n_clicks':
        return '/'
    elif ctx.triggered[0]['prop_id'] == 'btn-page1.n_clicks':
        return '/page1'
    elif ctx.triggered[0]['prop_id'] == 'btn-page2.n_clicks':
        return '/page2'
    elif ctx.triggered[0]['prop_id'] == 'btn-page3.n_clicks':
        return '/page3'
    elif ctx.triggered[0]['prop_id'] == 'btn-page4.n_clicks':
        return '/page4'


# Callbacks para la navegación entre páginas
@app.callback(
    Output('page-content', 'children'),
    [Input('btn-home', 'n_clicks'), Input('btn-page1', 'n_clicks'), Input('btn-page2', 'n_clicks'), Input('btn-page3', 'n_clicks'), Input('btn-page4', 'n_clicks')]
)
def display_page(btn_home, btn_page1, btn_page2, btn_page3, btn_page4):
    ctx = dash.callback_context
    if not ctx.triggered:
        return homepage_layout()
    elif ctx.triggered[0]['prop_id'] == 'btn-home.n_clicks':
        return homepage_layout()
    elif ctx.triggered[0]['prop_id'] == 'btn-page1.n_clicks':
        return page1_layout()
    elif ctx.triggered[0]['prop_id'] == 'btn-page2.n_clicks':
        return page2_layout()
    elif ctx.triggered[0]['prop_id'] == 'btn-page3.n_clicks':
        return page3_layout()
    elif ctx.triggered[0]['prop_id'] == 'btn-page4.n_clicks':
        return page4_layout()

# Callback para actualizar la serie temporal y los gráficos de ACF y PACF
@app.callback(
    [Output('serie-grafico', 'figure'), Output('acf-pacf-plots', 'children')],
    Input('serie-dropdown', 'value')
)
def update_graph_and_acf_pacf(selected_value):
    if selected_value is None:  # Asegúrate de que hay un valor seleccionado
        raise PreventUpdate

    fig = go.Figure()
    acf_fig, pacf_fig = None, None
    lag_count = 30  # Define un valor de lags para ACF y PACF

    if selected_value == 'precio_bid':
        data = dftra['precio_bid']
        acf_color = 'rgba(0, 0, 255, 0.6)'  # Azul para ACF
        pacf_color = 'rgba(0, 255, 0, 0.6)'  # Verde para PACF
        fig.add_trace(go.Scatter(
            x=dftra.index,  # Usar 'fecha' como eje x
            y=data, 
            mode='lines', 
            name='Precio Bid', 
            line=dict(color=acf_color)
        ))
        fig.update_xaxes(range=[dftra.index.min(), dftra.index.max()], rangemode="normal")
        fig.update_yaxes(range=[data.min() - 5, data.max() + 5])

        # Generar gráficos ACF y PACF en azul
        acf_fig, pacf_fig = create_acf_pacf_figures(data, "precio_bid", acf_color, pacf_color, 30)

    elif selected_value == 'precio_ask':
        data = dftra['precio_ask']
        acf_color = 'rgba(255, 0, 0, 0.6)'  # Rojo para ACF
        pacf_color = 'rgba(255, 165, 0, 0.6)'  # Naranja para PACF
        fig.add_trace(go.Scatter(
            x=dftra.index,  # Usar 'fecha' como eje x
            y=data, 
            mode='lines', 
            name='Precio Ask', 
            line=dict(color=acf_color)
        ))
        fig.update_xaxes(range=[dftra.index.min(), dftra.index.max()], rangemode="normal")
        fig.update_yaxes(range=[data.min() - 5, data.max() + 5])

        # Generar gráficos ACF y PACF en rojo
        acf_fig, pacf_fig = create_acf_pacf_figures(data, "precio_ask", acf_color, pacf_color, 30)

    # Configuración de la gráfica de series de tiempo
    fig.update_layout(
        title=f'Serie Temporal - {selected_value.replace("_", " ").title()}',
        xaxis_title='Fecha',
        yaxis_title=selected_value.replace("_", " ").title(),
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgrey'),
        yaxis=dict(gridcolor='lightgrey'),
        title_font=dict(size=20, color='#264653'),
        font=dict(size=14, color='#264653')
    )
    
    # Componentes de gráficos de ACF y PACF
    acf_pacf_div = [
        dcc.Graph(figure=acf_fig, style={'height': '400px', 'flex': '1'}),
        dcc.Graph(figure=pacf_fig, style={'height': '400px', 'flex': '1'})
    ]
    
    return fig, acf_pacf_div

@app.callback(
    [Output('resampled-grafico', 'figure'), Output('acf-graph', 'figure'), Output('pacf-graph', 'figure')],
    Input('interval-dropdown', 'value')
)
def update_resampled_and_acf_pacf_graph(selected_interval):
    # Crear figura para la serie de tiempo resampleada
    resampled_fig = go.Figure()
    acf_fig = go.Figure()
    pacf_fig = go.Figure()
    
    # Inicializar la variable de conteo de lags
    if selected_interval == '5T':
        resample_lag_count = 60  # Número de lags para 5 minutos
        resampled_data = dftra.resample('5T').mean()['precio_bid'].dropna()
    elif selected_interval == '30T':
        resample_lag_count = 22  # Número de lags para 30 minutos
        resampled_data = dftra.resample('30T').mean()['precio_bid'].dropna()
    elif selected_interval == 'H':
        resample_lag_count = 11  # Número de lags para cada hora
        resampled_data = dftra.resample('H').mean()['precio_bid'].dropna()
    else:
        return resampled_fig, acf_fig, pacf_fig  # Si no hay selección, retorna gráficos vacíos

    # Verificar que los datos no estén vacíos
    if not resampled_data.empty:
        # Gráfico de serie de tiempo resampleada
        resampled_fig.add_trace(go.Scatter(
            x=resampled_data.index,
            y=resampled_data,
            mode='lines',
            name=f'Serie Resampleada - Intervalo: {selected_interval}',
            line=dict(color='blue')
        ))
        resampled_fig.update_layout(
            title=f'Serie Resampleada - Intervalo: {selected_interval}',
            xaxis_title='Fecha',
            yaxis_title='Precio Bid'
        )

        # Calcular ACF y PACF
        acf_values = acf(resampled_data, nlags=resample_lag_count)
        pacf_values = pacf(resampled_data, nlags=resample_lag_count)

        # Gráfico ACF
        acf_fig.add_trace(go.Bar(x=list(range(len(acf_values))), y=acf_values, marker_color='blue'))
        acf_fig.update_layout(title="Autocorrelación (ACF)", xaxis_title='Lags', yaxis_title='ACF')

        # Gráfico PACF
        pacf_fig.add_trace(go.Bar(x=list(range(len(pacf_values))), y=pacf_values, marker_color='blue'))
        pacf_fig.update_layout(title="Autocorrelación Parcial (PACF)", xaxis_title='Lags', yaxis_title='PACF')

    return resampled_fig, acf_fig, pacf_fig



# Callback para actualizar el boxplot en función del período seleccionado
@app.callback(
    Output('boxplot-graph', 'figure'),
    Input('period-dropdown', 'value')
)
def update_boxplot(period):
    # Seleccionar el DataFrame correspondiente al período
    if period == 'General':
        df = dftra_boxplot
    elif period == 'Cada 5 Minutos':
        df = df5min_boxplot
    elif period == 'Cada 30 Minutos':
        df = df30min_boxplot
    elif period == 'Cada Hora':
        df = dfhora_boxplot
    else:
        df = dftra_boxplot  # Default por si acaso
    
    # Crear el gráfico del boxplot usando Plotly Express
    fig = px.box(df, x='period', y='precio_bid', title=f'Boxplot - {period}')
    fig.update_layout(
        plot_bgcolor='white',
        xaxis_title='Período',
        yaxis_title='Precio Bid',
        title_font=dict(size=20, color='#264653'),
        font=dict(size=14, color='#264653')
    )
    
    return fig

# Callback para actualizar el gráfico de diferencias según el intervalo de resampleo seleccionado
@app.callback(
    Output('grafico-diferencias-resampleado', 'figure'),
    Input('diferencias-resample-dropdown', 'value')
)
def update_diferencias_resample(interval):
    # Realizar el resampleo según el intervalo seleccionado
    if interval:
        df_resampled = dftra.resample(interval).mean()
        df_resampled['precio_bid_diff'] = df_resampled['precio_bid'].diff()

        # Crear el gráfico
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_resampled.index,
            y=df_resampled['precio_bid'],
            mode='lines',
            name='Serie Original Resampleada',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=df_resampled.index,
            y=df_resampled['precio_bid_diff'],
            mode='lines',
            name='Serie Diferenciada',
            line=dict(color='red')
        ))

        fig.update_layout(
            title=f'Análisis de Diferencias - Resampleo cada {interval}',
            xaxis_title='Fecha',
            yaxis_title='Valor',
            plot_bgcolor='white'
        )
        return fig
    return go.Figure()  # Retornar un gráfico vacío si no hay selección

@app.callback(
    [Output('grafico-diferencias', 'figure'), 
     Output('acf-graph-diferencias', 'figure'),
     Output('pacf-graph-diferencias', 'figure')],
    Input('diferencias-resample-dropdown', 'value')
)
def update_differenced_graph(selected_interval):
    # Resamplear los datos según el intervalo seleccionado
    if selected_interval == '5T':
        resampled_data = dftra.resample('5T').mean()['precio_bid'].dropna()
        nlag = 70  # Lags para 5 minutos
    elif selected_interval == '30T':
        resampled_data = dftra.resample('30T').mean()['precio_bid'].dropna()
        nlag = 22  # Lags para 30 minutos
    elif selected_interval == 'H':
        resampled_data = dftra.resample('H').mean()['precio_bid'].dropna()
        nlag = 11  # Lags para cada hora
    else:
        return go.Figure(), go.Figure(), go.Figure()  # Retornar gráficos vacíos si no hay selección

    # Calcular la diferencia de la serie
    resampled_data_diff = resampled_data.diff().dropna()

    # Gráfico de la serie diferenciada
    diff_fig = go.Figure()
    diff_fig.add_trace(go.Scatter(
        x=resampled_data_diff.index,
        y=resampled_data_diff,
        mode='lines',
        name='Serie Diferenciada',
        line=dict(color='blue')
    ))
    diff_fig.update_layout(
        title=f'Serie Diferenciada - Intervalo: {selected_interval}',
        xaxis_title='Fecha',
        yaxis_title='Diferencia de Precio Bid'
    )

    # Calcular ACF y PACF
    acf_values = acf(resampled_data_diff, nlags=nlag)
    pacf_values = pacf(resampled_data_diff, nlags=nlag)

    # Gráfico ACF
    acf_fig = go.Figure()
    acf_fig.add_trace(go.Bar(x=list(range(len(acf_values))), y=acf_values, marker_color='blue'))
    acf_fig.update_layout(title="Autocorrelación (ACF)", xaxis_title='Lags', yaxis_title='ACF')

    # Gráfico PACF
    pacf_fig = go.Figure()
    pacf_fig.add_trace(go.Bar(x=list(range(len(pacf_values))), y=pacf_values, marker_color='blue'))
    pacf_fig.update_layout(title="Autocorrelación Parcial (PACF)", xaxis_title='Lags', yaxis_title='PACF')

    return diff_fig, acf_fig, pacf_fig

@app.callback(
    [Output('grafico-resultados-ml', 'figure'),
     Output('metricas-resultado', 'children'),
     Output('histograma-residuos', 'figure'),  # Gráfico de histograma
     Output('grafico-residuos', 'figure'),      # Gráfico de residuos
     Output('pacf-plot-residuos', 'figure')],   # Gráfico de PACF de residuos
    Input('modelo-dropdown', 'value')  # Actualiza el gráfico según el modelo seleccionado
)
def update_ml_results_graph(selected_model):
    # Crear figura para los resultados del modelo
    fig = go.Figure()

    # Agregar los valores reales
    fig.add_trace(go.Scatter(
        x=list(range(len(resultados_ml))),  # Índices para el eje x
        y=resultados_ml['reales'],
        mode='lines+markers',
        name='Reales',
        line=dict(color='black', width=2)
    ))

    # Agregar las predicciones del modelo seleccionado
    fig.add_trace(go.Scatter(
        x=list(range(len(resultados_ml))),  # Índices para el eje x
        y=resultados_ml[selected_model],
        mode='lines+markers',
        name=selected_model
    ))

    # Configurar el layout de la figura
    fig.update_layout(
        title='Comparación de Modelos de Machine Learning',
        xaxis_title='Observaciones',
        yaxis_title='Valor Predicho',
        plot_bgcolor='white',
        legend=dict(title='Modelos')
    )

    # Calcular los residuos
    resultados_ml['residuos'] = resultados_ml['reales'] - resultados_ml[selected_model]

    # Calcular las métricas
    metrics = calcular_metricas(resultados_ml['reales'], resultados_ml[selected_model])

    # Crear la tabla con las métricas
    metrics_table = DataTable(
        columns=[{"name": key, "id": key} for key in metrics.keys()],
        data=[{key: value for key, value in metrics.items()}],
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
        style_cell={'textAlign': 'left'}
    )

    # Crear gráficos de residuos
    histograma = histograma_residuos(resultados_ml['residuos'])  
    grafico = grafico_residuos(resultados_ml['residuos'])        
    pacf_fig = pacf_residuos(resultados_ml['residuos'])          

    # Combina todos los elementos a retornar
    return fig, metrics_table, histograma, grafico, pacf_fig




def display_page(pathname):
    if pathname == '/page1':
        return page1_layout()
    elif pathname == '/page2':
        return page2_layout()
    elif pathname == '/page3':  
        return page3_layout()
    elif pathname == '/page4':  
        return page4_layout()
    else:
        return homepage_layout()


if __name__ == '__main__':
    app.run_server(debug=True)
