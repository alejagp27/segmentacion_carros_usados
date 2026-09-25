# app.py — Segmentación de vehículos usados UK (M3: clustering K-means)
import pickle

import pandas as pd
import streamlit as st

st.set_page_config(page_title='Perfiles de vehículos usados', page_icon='🚗', layout='wide')


# ── Cargar modelo: [KMeans, columnas del entrenamiento, MinMaxScaler] ────
@st.cache_resource
def cargar_modelo():
    with open('modelo-cla.pkl', 'rb') as f:
        return pickle.load(f)


modelo, variables, min_max_scaler = cargar_modelo()
variables = list(variables)
NUMERICAS = ['year', 'mileage', 'tax', 'mpg', 'engineSize', 'price']

# Métricas de calidad del modelo final (evaluado con el 100% de los datos)
INERCIA_GLOBAL = 7607.7
SILUETA_GLOBAL = 0.644
META_SILUETA = 0.5

# ── Perfiles del modelo final (k=11, n_init=100, random_state=42) ────────
# p95: distancia al centroide que no supera el 95% de los vehículos del perfil
PERFILES = {
    0:  dict(nombre='Hyundai de gasolina, económico y urbano', segmento='Generalista',
             mediana=9491, rango='£6.100 – £17.500', silueta=0.676, p95=1.287,
             descripcion='El perfil más barato del inventario y el de motor más pequeño. Vehículos recientes '
                         'y con poco uso: su precio bajo se explica por la marca y el motor, no por el desgaste.'),
    1:  dict(nombre='BMW automático diésel, eficiente y con más uso', segmento='Premium automático accesible',
             mediana=18743, rango='£11.500 – £34.700', silueta=0.704, p95=1.334,
             descripcion='BMW automáticos con más años y kilometraje que los premium recientes. '
                         'Tienen el mejor rendimiento del inventario (67 mpg en promedio).'),
    2:  dict(nombre='Audi de gasolina manual con motor pequeño', segmento='Premium de entrada manual',
             mediana=16450, rango='£10.900 – £25.000', silueta=0.846, p95=0.257,
             descripcion='El Audi de entrada: manual, de gasolina, motor pequeño y uso moderado. '
                         'Es el premium manual más caro porque tiene menos millas que los diésel manuales.'),
    3:  dict(nombre='BMW diésel semiautomático casi nuevo', segmento='Premium alto',
             mediana=24490, rango='£15.600 – £42.000', silueta=0.799, p95=0.358,
             descripcion='BMW recientes, con poco uso y buen rendimiento. Es el perfil BMW más caro; '
                         'reúne desde versiones de entrada hasta modelos de gama alta.'),
    4:  dict(nombre='Audi de gasolina semiautomático reciente', segmento='Premium alto',
             mediana=23999, rango='£16.750 – £37.000', silueta=0.852, p95=0.317,
             descripcion='Audi casi nuevos y con pocas millas. Rinden poco y pagan un impuesto alto, '
                         'como los demás vehículos de gasolina recientes.'),
    5:  dict(nombre='Audi diésel manual con alto kilometraje', segmento='Premium de entrada manual',
             mediana=14995, rango='£9.000 – £22.000', silueta=0.817, p95=0.315,
             descripcion='Audi con más años y mucho uso, pero muy eficientes y con impuesto muy bajo. '
                         'Su precio refleja el kilometraje más que la marca.'),
    6:  dict(nombre='Audi diésel semiautomático de motor grande', segmento='Premium alto',
             mediana=24490, rango='£15.500 – £46.000', silueta=0.808, p95=0.342,
             descripcion='Audi de motor grande y uso moderado. Incluye modelos de gama alta: '
                         'el 10% más caro supera las £46.000.'),
    7:  dict(nombre='Audi automático, el más caro del inventario', segmento='Premium alto',
             mediana=26500, rango='£14.500 – £45.000', silueta=0.423, p95=0.927,
             descripcion='Audi automáticos recientes, diésel o gasolina. Tiene el mayor precio del inventario '
                         'e incluye modelos de gama alta.'),
    8:  dict(nombre='BMW diésel manual, el premium más económico', segmento='Premium de entrada manual',
             mediana=12450, rango='£8.500 – £22.500', silueta=0.821, p95=0.310,
             descripcion='El BMW más barato del inventario, con el mayor kilometraje y el menor impuesto. '
                         'Su precio se explica por el uso.'),
    9:  dict(nombre='BMW de gasolina con transmisión mixta', segmento='Premium alto',
             mediana=21995, rango='£12.800 – £38.000', silueta=0.305, p95=0.992,
             descripcion='Todos los BMW de gasolina, sin importar la transmisión. Recientes y con pocas millas, '
                         'pero con bajo rendimiento y el impuesto más alto del inventario.'),
    10: dict(nombre='Hyundai diésel e híbrido, con más uso', segmento='Generalista',
             mediana=14500, rango='£8.300 – £23.000', silueta=0.267, p95=1.508,
             descripcion='Hyundai diésel e híbridos con más kilometraje que el perfil 0. '
                         'Su precio es similar al de los premium manuales.'),
}

# Número de vehículos de cada perfil en el entrenamiento (total: 26.306)
N_VEHICULOS = {0: 2901, 1: 2694, 2: 2250, 3: 3009, 4: 1783, 5: 2118,
               6: 1808, 7: 2708, 8: 1630, 9: 3448, 10: 1957}
TOTAL = sum(N_VEHICULOS.values())

DESCRIPCION_SEGMENTOS = {
    'Premium alto': 'Audi y BMW recientes (2017–2019), con pocas millas y mayoritariamente automáticos '
                    'o semiautomáticos. Los de mayor precio del inventario.',
    'Premium automático accesible': 'BMW automáticos de 2016 con kilometraje medio y el mejor rendimiento '
                                    'del inventario. Precio intermedio.',
    'Premium de entrada manual': 'Audi y BMW 100% manuales, por debajo de £17.000 en promedio. '
                                 'Los diésel tienen el mayor kilometraje y el menor impuesto.',
    'Generalista': 'Todos los Hyundai: gasolina económico (perfil 0) y diésel e híbrido (perfil 10).',
}

SEGMENTOS = {
    'Premium alto': dict(
        compras='Captar vehículos de 2018–2019 con menos de 20.000 millas y usar la mediana del perfil como '
                'referencia de oferta, descontando cada año y cada tramo de millas adicional. En los perfiles 6 y 7, '
                'valorar la unidad por modelo; en el perfil 9, según la transmisión.',
        rotacion='Venta con certificación, garantía extendida y financiación con cuota final. Diésel semiautomáticos '
                 'hacia empresas y flotas. Vigilar los días en inventario: es el segmento de mayor capital inmovilizado.'),
    'Premium automático accesible': dict(
        compras='Exigir inspección mecánica por el kilometraje y valorar aparte los híbridos.',
        rotacion='Ofrecerlo como el paso del premium manual al automático, con el argumento de la eficiencia '
                 'y una garantía mecánica.'),
    'Premium de entrada manual': dict(
        compras='En los diésel con alto kilometraje, exigir historial de servicio y descontar el reacondicionamiento '
                'de la oferta. Los de gasolina con menos millas admiten una oferta más alta.',
        rotacion='Venta como “primer premium”, con financiación de cuota baja y en marketplaces. '
                 'Mostrar el historial para resolver la objeción del kilometraje.'),
    'Generalista': dict(
        compras='Compra por volumen a precio ajustado y con el mínimo reacondicionamiento. '
                'Valorar aparte los híbridos.',
        rotacion='Debe ser el segmento de rotación más rápida: precio competitivo, proceso ágil y '
                 'enfoque en primer auto y familias con presupuesto ajustado.'),
}

# ── Estilo visual por segmento: (emoji, color) ───────────────────────────
ESTILO_SEG = {
    'Premium alto': ('🏎️', '#1F4E79'),
    'Premium automático accesible': ('🚙', '#2E75B6'),
    'Premium de entrada manual': ('🚗', '#5B9BD5'),
    'Generalista': ('🚕', '#7F7F7F'),
}
fmt = lambda n: f'{n:,}'.replace(',', '.')
coma = lambda x, d=1: f'{x:.{d}f}'.replace('.', ',')


def rango_corto(rango):
    # '£14.500 – £45.000' -> '£14,5k–45k' (para que quepa en la tarjeta)
    a, b = [int(x.replace('.', '')) / 1000 for x in rango.replace('£', '').split(' – ')]
    k = lambda v: coma(v).replace(',0', '')
    return f'£{k(a)}k–{k(b)}k'

st.markdown("""
<style>
.hero {background: linear-gradient(135deg, #16365C 0%, #1F4E79 55%, #2E75B6 100%);
       color: #FFFFFF; border-radius: 16px; padding: 1.6rem 1.8rem 1.3rem; margin-bottom: 1.2rem;}
.hero h1 {color: #FFFFFF; font-size: 2rem; margin: 0 0 .35rem 0; padding: 0;}
.hero p {color: #DCE8F5; margin: 0; font-size: 1rem;}
.hero .autos {font-size: 1.6rem; letter-spacing: .35rem; margin-bottom: .4rem;}
.seg-card {border-left: 6px solid var(--c); background: rgba(128,128,128,.07);
           border-radius: 10px; padding: .75rem 1rem; margin-bottom: .8rem; min-height: 8.2rem;}
.seg-card .t {font-weight: 700; font-size: 1.02rem;}
.seg-card .n {font-size: .85rem; opacity: .75; margin: .15rem 0 .35rem 0;}
.seg-card .d {font-size: .9rem;}
.perf-card {border-top: 5px solid var(--c); background: rgba(128,128,128,.07); border-radius: 12px;
             padding: .8rem 1rem .9rem; margin-bottom: 1rem; min-height: 15.5rem;}
.perf-card .top {display: flex; justify-content: space-between; align-items: center;}
.perf-card .num {font-size: .8rem; font-weight: 700; letter-spacing: .06rem; text-transform: uppercase; opacity: .7;}
.perf-card .em {font-size: 1.6rem;}
.perf-card .nom {font-weight: 700; font-size: 1.05rem; margin: .25rem 0 .35rem 0; line-height: 1.3;}
.perf-card .tag {display: inline-block; background: var(--c); color: #FFFFFF; border-radius: 999px;
                 padding: .08rem .6rem; font-size: .75rem; margin-bottom: .5rem;}
.perf-card .d {font-size: .88rem; margin-bottom: .6rem;}
.perf-card .st {font-size: .82rem; border-top: 1px solid rgba(128,128,128,.25); padding-top: .45rem; line-height: 1.6;}
.perf-card .baja {color: #B45309; font-weight: 600;}
.res-card {border-left: 8px solid var(--c); background: rgba(128,128,128,.08);
           border-radius: 12px; padding: 1rem 1.2rem; margin: .6rem 0 1rem 0;}
.res-card .e {font-size: 2.2rem; line-height: 1;}
.res-card .p {font-size: 1.25rem; font-weight: 700; margin-top: .3rem;}
.res-card .s {display: inline-block; background: var(--c); color: #FFFFFF; border-radius: 999px;
              padding: .15rem .7rem; font-size: .85rem; margin-top: .45rem;}
.res-card .d {margin-top: .6rem;}
[data-testid="stMetric"] {background: rgba(128,128,128,.07); border-radius: 10px; padding: .6rem .9rem;}
button[kind="primary"], [data-testid="stBaseButton-primary"] {background-color: #1F4E79; border-color: #1F4E79;}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {background-color: #2E75B6; border-color: #2E75B6;}
.pie {text-align: center; opacity: .65; font-size: .85rem; margin-top: 1.5rem;}
</style>
""", unsafe_allow_html=True)

# ── Encabezado ───────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="autos">🚗 🚙 🏎️ 🚕 🚐</div>
  <h1>Perfiles de vehículos usados</h1>
  <p>Conoce los 11 perfiles de vehículos que componen el inventario y descubre a cuál pertenece un vehículo,
  con la estrategia de compra y venta que le corresponde.</p>
</div>
""", unsafe_allow_html=True)

# ── 1. Perfiles del inventario (foco principal) ──────────────────────────
st.markdown('### 🚗 Los 11 perfiles del inventario')
st.markdown('El modelo K-means agrupó los **26.306 vehículos** del inventario en **11 perfiles**. Cada tarjeta resume '
            'un perfil: cuántos vehículos tiene, su precio típico y qué tan bien definido está (silueta; la meta es 0,5).')

cols = st.columns(3)
for k, v in PERFILES.items():
    emoji, color = ESTILO_SEG[v['segmento']]
    n = N_VEHICULOS[k]
    sil = coma(v['silueta'], 3)
    sil_txt = (f'<span class="baja">⚠️ Silueta {sil} (bajo la meta)</span>' if v['silueta'] < META_SILUETA
               else f'Silueta {sil}')
    cols[k % 3].markdown(
        f'<div class="perf-card" style="--c:{color}">'
        f'<div class="top"><span class="num">Perfil {k}</span><span class="em">{emoji}</span></div>'
        f'<div class="nom">{v["nombre"]}</div>'
        f'<span class="tag">{v["segmento"]}</span>'
        f'<div class="d">{v["descripcion"]}</div>'
        f'<div class="st">🚗 {fmt(n)} vehículos ({coma(n / TOTAL * 100)}%)<br>'
        f'💰 Mediana £{fmt(v["mediana"])} · 80% entre {v["rango"]}<br>{sil_txt}</div></div>',
        unsafe_allow_html=True)

with st.expander('🧩 Cómo se agrupan los perfiles en segmentos de negocio'):
    st.markdown('Para la estrategia comercial, los perfiles se agrupan en **4 segmentos** según marca, '
                'transmisión y nivel de precio.')
    columnas = st.columns(2)
    for i_seg, (nombre, desc) in enumerate(DESCRIPCION_SEGMENTOS.items()):
        ids = [k for k, v in PERFILES.items() if v['segmento'] == nombre]
        n = sum(N_VEHICULOS[k] for k in ids)
        emoji, color = ESTILO_SEG[nombre]
        columnas[i_seg % 2].markdown(
            f'<div class="seg-card" style="--c:{color}">'
            f'<div class="t">{emoji} {nombre}</div>'
            f'<div class="n">Perfiles {", ".join(map(str, ids))} · {fmt(n)} vehículos ({coma(n / TOTAL * 100)}%)</div>'
            f'<div class="d">{desc}</div></div>', unsafe_allow_html=True)

st.divider()

# ── 2. Identificar el segmento de un vehículo ────────────────────────────
st.markdown('### 🔎 ¿A qué perfil pertenece un vehículo?')
with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox('Marca', ['Audi', 'BMW', 'Hyundai'])
        transmission = st.selectbox('Transmisión', ['Automatic', 'Manual', 'Semi-Auto'],
                                    format_func={'Automatic': 'Automática', 'Manual': 'Manual',
                                                 'Semi-Auto': 'Semiautomática'}.get)
        fuelType = st.selectbox('Combustible', ['Diesel', 'Hybrid', 'Petrol'],
                                format_func={'Diesel': 'Diésel', 'Hybrid': 'Híbrido', 'Petrol': 'Gasolina'}.get)
        year = st.slider('Año de matrícula', min_value=1998, max_value=2020, value=2017, step=1)
        price = st.number_input('Precio publicado (£)', min_value=1200, max_value=145000, value=20000, step=500)
    with col2:
        mileage = st.number_input('Millas recorridas', min_value=1, max_value=214000, value=20000, step=1000)
        tax = st.number_input('Impuesto anual (£)', min_value=0, max_value=580, value=145, step=5)
        mpg = st.number_input('Rendimiento (mpg)', min_value=5.5, max_value=470.8, value=50.0, step=0.5)
        engineSize = st.number_input('Tamaño del motor (L)', min_value=0.6, max_value=6.6, value=2.0, step=0.1)
    boton = st.button('🚗 Identificar perfil', type='primary', width='stretch')

if boton:
    # 1. DataFrame con las mismas variables del entrenamiento
    data = pd.DataFrame([[year, mileage, tax, mpg, engineSize, price, transmission, fuelType, brand]],
                        columns=NUMERICAS + ['transmission', 'fuelType', 'brand'])

    # 2. Normalizar numéricas con el escalador del entrenamiento (solo transform)
    data_preparada = data.copy()
    data_preparada[NUMERICAS] = min_max_scaler.transform(data_preparada[NUMERICAS])

    # 3. Dummies sin eliminar categorías y alineación con las columnas del entrenamiento
    data_preparada = pd.get_dummies(data_preparada, columns=['transmission', 'fuelType', 'brand'],
                                    drop_first=False, dtype=int)
    data_preparada = data_preparada.reindex(columns=variables, fill_value=0)

    # 4. Asignar el clúster (centroide más cercano) y su distancia
    cluster = int(modelo.predict(data_preparada)[0])
    distancia = float(modelo.transform(data_preparada)[0][cluster])
    perfil = PERFILES[cluster]
    segmento = SEGMENTOS[perfil['segmento']]
    emoji, color = ESTILO_SEG[perfil['segmento']]

    # 5. Resultado
    st.markdown(
        f'<div class="res-card" style="--c:{color}">'
        f'<div class="e">{emoji}</div>'
        f'<div class="p">Perfil {cluster}: {perfil["nombre"]}</div>'
        f'<div class="s">Segmento: {perfil["segmento"]}</div>'
        f'<div class="d">{perfil["descripcion"]}</div></div>', unsafe_allow_html=True)

    diferencia = (price - perfil['mediana']) / perfil['mediana'] * 100
    c1, c2, c3 = st.columns(3)
    c1.metric('Precio mediano del perfil', f"£{fmt(perfil['mediana'])}")
    c2.metric('Precio ingresado vs. mediana', f'{diferencia:+.0f}%')
    c3.metric('80% de los precios del perfil', rango_corto(perfil['rango']))

    if distancia > perfil['p95']:
        st.warning('⚠️ Este vehículo se aleja de lo habitual en su perfil (está más lejos del centroide que el 95% '
                   'de los vehículos del perfil). Valóralo de forma individual antes de hacer una oferta.')

    st.markdown('#### 🧭 Estrategia para este segmento')
    e1, e2 = st.columns(2)
    with e1.container(border=True):
        st.markdown(f"**🛒 Compra**\n\n{segmento['compras']}")
    with e2.container(border=True):
        st.markdown(f"**🔁 Rotación y venta**\n\n{segmento['rotacion']}")

    with st.expander('Datos ingresados'):
        st.dataframe(data, hide_index=True)

    # 6. Confiabilidad de la asignación para este perfil
    st.markdown('#### 🎯 Confiabilidad de este resultado')
    sil = perfil['silueta']
    st.metric(f'Silueta del perfil {cluster}', coma(sil, 3),
              delta=f'{coma(sil - META_SILUETA, 3) if sil < META_SILUETA else "+" + coma(sil - META_SILUETA, 3)} '
                    f'frente a la meta de 0,5')
    if sil < META_SILUETA:
        st.warning(f'La silueta de este perfil es {coma(sil, 3)}, por debajo de la meta de 0,5. '
                   'Es uno de los tres perfiles menos definidos del modelo (7, 9 y 10): sus vehículos se parecen '
                   'menos entre sí, así que la descripción, el precio mediano y la estrategia representan con menos '
                   'precisión a este vehículo. Valóralo de forma individual.')
    else:
        st.write('La silueta de este perfil supera la meta de 0,5: sus vehículos son parecidos entre sí y '
                 'están bien separados de los demás perfiles, así que la descripción es representativa.')

# ── 3. Calidad general del modelo (siempre visible) ──────────────────────
st.divider()
st.markdown('### 📈 Calidad del modelo')
m1, m2 = st.columns(2)
m1.metric('Silueta general', coma(SILUETA_GLOBAL, 3), delta='Cumple la meta de 0,5', delta_color='off')
m2.metric('Inercia general', f'{fmt(int(INERCIA_GLOBAL))},{str(INERCIA_GLOBAL).split(".")[1][0]}')
st.caption('**Silueta:** va de -1 a 1 y mide qué tan parecido es cada vehículo a los de su perfil frente a los del '
           'perfil más cercano. Valores cercanos a 1 indican perfiles bien definidos; la meta del proyecto es 0,5. '
           '**Inercia:** suma de las distancias al cuadrado de cada vehículo a su centroide, con los datos '
           'normalizados. Cuanto menor, más compactos son los perfiles; sirve para comparar modelos, no tiene unidades. '
           'Modelo K-means con 11 perfiles, entrenado con 26.306 vehículos. Los perfiles 7, 9 y 10 tienen silueta '
           'por debajo de la meta.')
st.markdown('<div class="pie">🚗 🚙 🏎️ 🚕 🚐<br>Tasación inteligente de vehículos usados mediante minería de datos</div>', unsafe_allow_html=True)
