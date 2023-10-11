#!/usr/bin/env python
# coding: utf-8
#Autores: Érika Espinosa, Mónica Quiroga y Karla Rodríguez

# In[1]:

import io
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc 
import plotly.express as px
import pandas as pd    
import numpy as np               
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact, Layout
import ipywidgets as widgets
import warnings
warnings.filterwarnings('ignore')
import re
import plotly.graph_objs as go
from wordcloud import WordCloud, ImageColorGenerator
from stop_words import get_stop_words
from io import BytesIO
import base64


import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup


# In[2]:


#Cargar data
df = pd.read_excel('https://raw.githubusercontent.com/moniiandrea/ProyectoMAD/main/DataMAD(1).xlsx')

# =============================================================================
# #Menú superior
# =============================================================================

sidebar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Resumen del proyecto", href="/resumen", active="exact")),
        dbc.NavItem(dbc.NavLink("Herramienta de predicción", href="/herramienta", active="exact")),
    ],
    brand="Análisis de sentimientos",
    brand_href="#",
    color="primary",
    dark=True,
)

content = html.Div(id="page-content")

app = dash.Dash(external_stylesheets=[dbc.themes.LUX],suppress_callback_exceptions=True)

app.layout = dbc.Container(html.Div([dcc.Location(id="url"), sidebar, content]))


# In[8]:

dropdownValues=df['Criterio'].unique().tolist()
dropdownValues.append('Todas las Universidades')
#Selección de Criterio de búsqueda
controls1 = dbc.Card(
    [dcc.Dropdown(id='dropdown',
                options=[{'label': i, 'value': i} for i in dropdownValues],
                value='Todas las Universidades',
                ),
    ], body=True,
)

#Mostrar Nube
controlsWC = dbc.Card(
    [
     html.Img(id='image_wc')
    ], body=True,
)


#Nube de palabras

def plot_wordcloud(dft):
    #Función para generar una nube de palabras a partir del dataframe que tiene la columna clean_text
    stop_words=get_stop_words("es")
    web_stopWords = ["q","vc","tipo","ta","pra","pq","ne","sobre","ser","cara","la",'mano',
                     'índice','flecha','ojo','hacia','derecha','_X000D_','corazón','cómo','piel','claro',
                     'UI','dorso','risa','ojos','sonriendo','así','hace','si','hoy','bandera','verde','rojo',
                     'de','RT','manos','aplaudiendo','levantadas','izquierda','tono','p','m','gt','abajo',
                     'azul','do','aquí']
    stop_words += web_stopWords
    text = " ".join(tweet for tweet in dft.clean_text)
    wc = WordCloud(stopwords=stop_words,background_color='white', width=600, height=500)
    wc.generate(text)
    return wc.to_image()

@app.callback(Output('image_wc', 'src'),
              [Input('image_wc', 'id'),
               Input('dropdown', 'value')])

def make_image(b,n):
    img = BytesIO()
    if n=='Todas las Universidades':
        plot_wordcloud(df).save(img, format='PNG')
        return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())
    else:
        plot_wordcloud(df[(df['Criterio']==n)]).save(img, format='PNG')
        return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())


@app.callback(
    Output('Cantidad_tweets', 'children'),
    [Input('dropdown','value')])

def actualiza_r(univ):
    if univ=='Todas las Universidades':
        return len(df)
    else:
        return len(df[(df['Criterio']==univ)])


@app.callback(
    Output('Porcentaje_positivos', 'children'),
    [Input('dropdown','value')])

def actualiza_r(univ):
    if univ=='Todas las Universidades':
        n_total=len(df)
        n_pos = len(df[(df['sentiment']=='Positivo')])
        n_pos_porc = (n_pos/n_total)*100
        return '{:.0f}%'.format(n_pos_porc)
    else:
        n_total=len(df[(df['Criterio']==univ)])
        n_pos = len(df[(df['Criterio']==univ)][(df[(df['Criterio']==univ)]['sentiment']=='Positivo')])
        n_pos_porc = (n_pos/n_total)*100
        return '{:.0f}%'.format(n_pos_porc)

@app.callback(
    Output('Porcentaje_neutros', 'children'),
    [Input('dropdown','value')])

def actualiza_r(univ):
    if univ=='Todas las Universidades':
        n_total=len(df)
        n_pos = len(df[(df['sentiment']=='Neutro')])
        n_pos_porc = (n_pos/n_total)*100
        return '{:.0f}%'.format(n_pos_porc)
    else:
        n_total=len(df[(df['Criterio']==univ)])
        n_pos = len(df[(df['Criterio']==univ)][(df[(df['Criterio']==univ)]['sentiment']=='Neutro')])
        n_pos_porc = (n_pos/n_total)*100
        return '{:.0f}%'.format(n_pos_porc)
    
@app.callback(
    Output('Porcentaje_negativos', 'children'),
    [Input('dropdown','value')])

def actualiza_r(univ):
    if univ=='Todas las Universidades':
        n_total=len(df)
        n_pos = len(df[(df['sentiment']=='Negativo')])
        n_pos_porc = (n_pos/n_total)*100
        return '{:.0f}%'.format(n_pos_porc)
    else:
        n_total=len(df[(df['Criterio']==univ)])
        n_pos = len(df[(df['Criterio']==univ)][(df[(df['Criterio']==univ)]['sentiment']=='Negativo')])
        n_pos_porc = (n_pos/n_total)*100
        return '{:.0f}%'.format(n_pos_porc)

@app.callback(
    Output('percepcion', 'children'),
    [Input('dropdown','value')])

def actualiza_r(univ):
    if univ=='Todas las Universidades':
        n_pos = len(df[(df['sentiment']=='Positivo')])
        n_neg = len(df[(df['sentiment']=='Negativo')])
        n_neu = len(df[(df['sentiment']=='Neutro')])
    else:
        n_pos = len(df[(df['Criterio']==univ)][(df[(df['Criterio']==univ)]['sentiment']=='Positivo')])
        n_neg = len(df[(df['Criterio']==univ)][(df[(df['Criterio']==univ)]['sentiment']=='Negativo')])
        n_neu = len(df[(df['Criterio']==univ)][(df[(df['Criterio']==univ)]['sentiment']=='Neutro')])
    
    if n_neg>n_neu and n_neg>n_pos:
       text='La percepción es principalmente negativa'
    elif n_pos>n_neu and n_pos>n_neg:
        text='La percepción es principalmente positiva'
    elif n_neu>n_neg and n_neu>n_pos:
        text='La percepción es principalmente neutra'
    else:
        text='No se define una percepción clara'
    return text


    
# In[11]:


cant_card=dbc.Card(
    [
        dbc.CardBody([
            html.P("Cantidad de Tweets", className="card-title",style={'textAlign':'center'}),
            html.H1(id='Cantidad_tweets',style={'textAlign':'center'}),
        ]),
    ], color="black",inverse=True, style={"width": "33rem"},
)
pos_card=dbc.Card(
    [
        dbc.CardBody([
            html.P("Porcentaje de Positivos", className="card-title",style={'textAlign':'center'}),
            html.H1(id='Porcentaje_positivos',style={'textAlign':'center'}),
        ]),
    ], color = "#d9e1ad",style={"width": "10rem"},
)
neu_card=dbc.Card(
    [
        dbc.CardBody([
            html.P("Porcentaje de Neutros", className="card-title",style={'textAlign':'center'}),
            html.H1(id='Porcentaje_neutros',style={'textAlign':'center'}),
        ]),
    ], style={"width": "10rem"},
)
neg_card=dbc.Card(
    [
        dbc.CardBody([
            html.P("Porcentaje de Negativos", className="card-title",style={'textAlign':'center'}),
            html.H1(id='Porcentaje_negativos',style={'textAlign':'center'}),
        ]),
    ],color = "#F5A9A9", style={"width": "10rem"},
)

percepcion=dbc.Card(
    [
        dbc.CardBody([
            html.H1(id='percepcion',style={'textAlign':'center'}),
        ]),
    ],color = "none", style={"width": "33rem"},
)
# =============================================================================
# ## Lógica para subir documento
# =============================================================================


upload = html.Div([
        dcc.Upload(
                id='upload-data',
                children=html.Div([
                html.A('Selecciona o arrastra tu archivo')
                ]),
                style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
                 },
                # Permite que múltiples archivos sean cargados
                multiple=True
        ),

        html.Div(id='output-data-upload'),
        ])
        
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
        # Si el archivo cargado es un CSV
            df_predictions = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            df_predictions[['label', 'Score Negativo', 'Score Neutro','Score Positivo']] = df_predictions["full_text"].apply(classifySentiment)
            df_predictions.to_csv('predictions.csv')
           
            df_predictions[df_predictions.label ==0].shape[0]
        elif 'xls' in filename:
        # Si el archivo cargado es un Excel
            df_predictions = pd.read_excel(io.BytesIO(decoded))
            
            df_predictions[['label', 'Score Negativo', 'Score Neutro','Score Positivo']] = df_predictions["full_text"].apply(classifySentiment)
            df_predictions.to_csv('predictions.csv')
       
            print(df_predictions[df_predictions.label ==0].shape[0])
            conteo_negativos = df_predictions[df_predictions.label ==0].shape[0]
            conteo_neutros = df_predictions[df_predictions.label ==1].shape[0]
            conteo_positivos = df_predictions[df_predictions.label ==2].shape[0]
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    def percepcion2(df_predictions):
        
        n_pos = df_predictions[df_predictions.label ==2].shape[0]
        n_neg = df_predictions[df_predictions.label ==0].shape[0]
        n_neu = df_predictions[df_predictions.label ==1].shape[0]
        
        if n_neg>n_neu and n_neg>n_pos:
           text='La percepción es principalmente negativa'
        elif n_pos>n_neu and n_pos>n_neg:
            text='La percepción es principalmente positiva'
        elif n_neu>n_neg and n_neu>n_pos:
            text='La percepción es principalmente neutra'
        else:
            text='No se define una percepción clara'
        return text

    percepcion2=dbc.Card(
        [
            dbc.CardBody([
                html.H3(percepcion2(df_predictions),style={'textAlign':'center'}),
            ]),
        ],color = "none", style={"width": "33rem"},
    )
    
    fig = go.Figure(layout = go.Layout(title='Frecuencia de sentimientos', yaxis_title='Cantidad', xaxis_title='Sentimiento'))
    bar2 = go.Bar(x=['Negativo','Neutro','Positivo'],
                  y=[conteo_negativos, conteo_neutros, conteo_positivos],
                  marker_color=['red','lightslategray','green'])
    
    fig.add_trace(bar2)
    image = BytesIO()
    #Eliminación de URLs
    df_clean = pd.DataFrame()
    df_clean['clean_text'] = df['full_text'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)

    #Eliminación de menciones
    df_clean['clean_text'] = df['clean_text'].replace(regex='(@\w+)|#|&|!',value='')


    import emoji
    def emoji2description(text):
      return emoji.replace_emoji(text, replace=lambda chars, data_dict: ' '+' '.join(data_dict['es'].split('_')).strip(':')+' ')

    df_clean['clean_text'] = df_clean['clean_text'].map(emoji2description)
    
    
    df_predictions=df_predictions.rename(columns={df_predictions.columns[0]: 'Texto'})
    plot_wordcloud(df_clean).save(image, format='PNG')
    imageWC = 'data:image/png;base64,{}'.format(base64.b64encode(image.getvalue()).decode())
    
    
    return html.Div([
        html.Hr(),
        dbc.Row(
            [
                html.H3('¡Ya están listos tus resultados!',style={'textAlign':'center'}),
                html.P('A continuación, encontrarás el resumen del análisis de sentimientos realizado a tu archivo. Puedes volver a cargar un nuevo documento si así lo deseas.'),
                downloadButton,
                html.Hr(),
                dbc.Col([html.Div(dbc.Card(
                    [
                     html.Img(src=imageWC)
                    ], body=True,
                )),]),
                dbc.Col([
                html.Div(percepcion2),    
                html.Div(dcc.Graph(figure=fig)),]),
                
                ]),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(html.Div([
                    html.P([
                html.Span("Aquí verás los tweets con el sentimiento "),
                html.Span("más positivo:", style={"color": "green"}),
            ]
                           ),
                    dbc.Table.from_dataframe(df_predictions[df_predictions.label ==2].sort_values(['Score Positivo'], ascending=[False])[['Texto','Score Positivo']].head(), striped=True, bordered=True, hover=True)
                    ])),
                dbc.Col(html.Div([
                    html.P([html.Span("Aquí verás los tweets con el sentimiento "),
                    html.Span("más negativo:", style={"color": "red"}),]),
                    dbc.Table.from_dataframe(df_predictions[df_predictions.label ==0].sort_values(['Score Negativo'], ascending=[False])[['Texto','Score Negativo']].head(), striped=True, bordered=True, hover=True)
                    ])),
            ]
        ),
       
    ])

def show_predictions(df_predictions):
    return html.Div([
        dbc.Table.from_dataframe(df_predictions.head(), striped=True, bordered=True, hover=True),
       
        downloadButton
    ])
    
@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
 
        
        return children
    

# =============================================================================
# Lógica para descargar documento
# =============================================================================

downloadButton = html.Div(
    [
        html.Button("Descargar resultados", id="btn_csv", style={'textAlign':'center'}),
        dcc.Download(id="download-dataframe-csv"),
    ]
)


@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    df_predictions=pd.read_csv('C:/Users/Erika/Dash_proyecto/predictions.csv') #Ruta de la carpeta donde se encuentran los resultados obtenidos por el modelo
    
    return dcc.send_data_frame(df_predictions.to_csv, "mydf.csv")
# =============================================================================
# Model
# =============================================================================

model_name = 'dccuchile/bert-base-spanish-wwm-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)

class BERTSentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(BERTSentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(model_name)
    self.drop = nn.Dropout(p=0.3)
    self.linear = nn.Linear(self.bert.config.hidden_size, n_classes) #revisar si mantienen o se quita y hidden_size
    self.gelu = nn.GELU() #función utilizada en bert en las capas internas, GELU combina y mejora los resultados por ser una distribución acumulativa, usada en NLP
    self.softmax = nn.Softmax(dim=1) # salida de probabilidades cuando se tienen más de dos categorias

  def forward(self, input_ids, attention_mask):
    _, cls_output = self.bert(
        input_ids = input_ids,
        attention_mask = attention_mask,
        return_dict=False
    )
    drop_output = self.drop(cls_output)
    output = self.linear(drop_output)
    output = self.gelu(output)
    return self.softmax(output)


model_path='C:/Users/Erika/Dash_proyecto/Modelo_Beto_Base' #Ruta dondese encuentra el modelo
model = torch.load(model_path, map_location=torch.device('cpu'))

MAX_LEN = 128

def classifySentiment(full_text_text):
  full_text_text = full_text_text.strip()
  encoding_full_text = tokenizer.encode_plus(
      full_text_text,
      max_length = MAX_LEN,
      truncation = True,
      add_special_tokens = True,
      return_token_type_ids = False,
      padding='max_length',
      return_attention_mask = True,
      return_tensors = 'pt'
      )

  input_ids = encoding_full_text['input_ids']
  attention_mask = encoding_full_text['attention_mask']
  output = model(input_ids, attention_mask)
  
  _, prediction = torch.max(output, dim=1)

  return pd.Series([prediction,output.detach().numpy()[0][0],output.detach().numpy()[0][1],output.detach().numpy()[0][2]])


# In[12]:


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/resumen":
        return html.Div([
            html.Hr(),
            html.H2("Análisis de Sentimientos con Enfoque en la Educación Superior en Colombia",style={'textAlign':'center'}),
            html.Hr(),html.Hr(),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div("Autores: Érika Espinosa, Mónica Quiroga, Karla Rodríguez",style={'textAlign':'center'}),
                        width={"size": 6, "offset": 3},
                    ),
                ]
            ),
            html.Br(),
            html.P('Esta herramienta permite analizar la polaridad de los sentimientos de textos, clasificándolos en positivo, neutro y negativo.'),
            html.P('Para ello se entrenaron modelos BERT multilenguaje y BETO, haciendo uso de 70.506 tweets relacionados con la Educación Superior en Colombia, tomando como referencia criterios de búsqueda dentro de esta red social basados en 100 universidades del país.'),
            html.P('Esta sección se divide en dos partes:'),
            html.P('''1. Descripción de los datos de entrenamiento que fueron etiquetados manualmente: 
                Se presentan algunos datos que pueden ser consultados por universidad, como cantidad de tweets y porcentajes de tweets positivos, neutros y negativos; 
                además se muestra una nube de palabras. 
            '''),
            html.P('''
            2. Métricas de entrenamiento de los modelos: 
                Se presenta a manera de informe los resultados de ambos modelos para el contexto evaluado y se presentan algunas
                métricas de interés.
                         '''),
            html.Hr(),
            html.H2('Descripción de los datos'),
            
            dbc.Row(controls1),
            html.P('** Recuerda que puedes buscar la universidad de interés para conocer más sobre los tweets utilizados en el entrenamiento de los modelos para dicha institución'),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col([
                        dbc.Row(dbc.Col(cant_card, width="auto")),
                        html.Br(),
                        dbc.Row([
                        dbc.Col(pos_card, width="auto"),
                        dbc.Col(neu_card, width="auto"),
                        dbc.Col(neg_card, width="auto"),
                        dbc.Row(dbc.Col(percepcion, width="auto"))
                        ])
                        
                        ]),
                    dbc.Col(controlsWC),
                ]
            ),
           
            html.Br(),
            html.Hr(),
            html.H2('Métricas de entrenamiento de los modelos'),html.Hr(),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.H3('Métricas BERT Multilenguaje'),
                    dbc.Row([dbc.Col(html.Div([dbc.Card([dbc.CardBody([
                                html.P("Exactitud", className="card-title",style={'textAlign':'center'}),
                                html.H1('77.8%',style={'textAlign':'center'}),
                            ],style={'textAlign':'center'}),
                        ], color="black",inverse=True, style={"width": "auto"},
                    ),]),style={'textAlign':'center'}),
                        dbc.Col([dbc.Card([dbc.CardBody([
                                html.P("Pérdida", className="card-title",style={'textAlign':'center'}),
                                html.H1('0.769',style={'textAlign':'center'}),
                            ]),
                        ], color="black",inverse=True, style={"width": "auto"},
                    ),],style={'textAlign':'center'})],style={'textAlign':'center'}),
                    
                    html.Hr(),html.Br(),
                    html.Div([html.Img(src='data:image/png;base64,{}'.format(base64.b64encode(open('C:/Users/Erika/Dash_proyecto/Matriz_BERTmul.png', 'rb').read()).decode()),alt='image',style={"width": "auto", "align":"middle"})],style={'textAlign': 'center'})
                    ]),
                dbc.Col([
                    html.H3('Métricas BETO'),
                    dbc.Row([dbc.Col([dbc.Card([dbc.CardBody([
                                html.P("Exactitud", className="card-title",style={'textAlign':'center'}),
                                html.H1('79.3%',style={'textAlign':'center'}),
                            ]),
                        ], color="black",inverse=True, style={"width": "auto"},
                    ),]),
                        dbc.Col([dbc.Card([dbc.CardBody([
                                html.P("Pérdida", className="card-title",style={'textAlign':'center'}),
                                html.H1('0.755',style={'textAlign':'center'}),
                            ]),
                        ], color="black",inverse=True, style={"width": "auto"},
                    ),])]),html.Hr(),html.Br(),
                    html.Div([html.Img(src='data:image/png;base64,{}'.format(base64.b64encode(open('C:/Users/Erika/Dash_proyecto/Matriz_BETO.png', 'rb').read()).decode()),alt='image',style={"width": "auto", "align":"middle"})],style={'textAlign': 'center'})
                    ]),
                
                ]),
            html.Br(),
            html.H4('De acuerdo con los resultados el mejor modelo corresponde a BETO',style={'textAlign':'center'}),
            
        
        ])
    elif pathname == "/herramienta":
        return dbc.Container(
        [
            html.Hr(),
            html.H1("Herramienta de cálculo de sentimiento",style={'textAlign':'center'}),
            html.Hr(),
            html.P('Calcula el sentimiento asociado a diferentes textos (Positivo, Neutro o Negativo), subiendo un archivo Excel (.xlsx) o separado por comas (.csv)'),
            html.P('Para un uso correcto, tu archivo debe contener una única columna con los textos que deseas analizar.'),
            html.B('Nota: Es posible que se tarde algunos minutos en obtener los resultados'),
            dbc.Row(upload),
            
            html.Br(),     
    
            
        ],
        fluid=True,
        )




# In[13]:


if __name__ == '__main__':
    app.run(debug=True, port=2888)

