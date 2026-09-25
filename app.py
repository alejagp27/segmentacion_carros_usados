# app.py — Segmentación de vehículos usados UK (M3: clustering K-means)
import pickle

import pandas as pd
import streamlit as st

st.set_page_config(page_title='Segmento del vehículo', page_icon='🚗', layout='centered')


# ── Cargar modelo: [KMeans, columnas del entrenamiento, MinMaxScaler] ────
@st.cache_resource
def cargar_modelo():
    with open('modelo-cla.pkl', 'rb') as f:
        return pickle.load(f)


modelo, variables, min_max_scaler = cargar_modelo()
variables = list(variables)
NUMERICAS = ['year', 'mileage', 'tax', 'mpg', 'engineSize', 'price']

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

# ── Interfaz ─────────────────────────────────────────────────────────────
st.title('🚗 Segmento del vehículo')
st.markdown('Ingresa las características del vehículo para saber a qué perfil y segmento del inventario pertenece, '
            'y qué estrategia de compra y venta aplica.')

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

if st.button('🔎 Identificar segmento', type='primary'):
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

    # 5. Resultado
    st.success(f"**Perfil {cluster}: {perfil['nombre']}**  \nSegmento: **{perfil['segmento']}**")
    st.write(perfil['descripcion'])

    diferencia = (price - perfil['mediana']) / perfil['mediana'] * 100
    c1, c2, c3 = st.columns(3)
    c1.metric('Precio mediano del perfil', f"£{perfil['mediana']:,.0f}".replace(',', '.'))
    c2.metric('Precio ingresado frente a la mediana', f'{diferencia:+.0f}%')
    c3.metric('80% de los precios del perfil', perfil['rango'])

    if distancia > perfil['p95']:
        st.warning('Este vehículo se aleja de lo habitual en su perfil (está más lejos del centroide que el 95% '
                   'de los vehículos del perfil). Valóralo de forma individual antes de hacer una oferta.')
    if perfil['silueta'] < 0.5:
        st.info('Este perfil es de los menos definidos del modelo (silueta por debajo de 0,5): '
                'su descripción representa con menos precisión a cada vehículo.')

    st.subheader('Estrategia para este segmento')
    st.markdown(f"**Compra:** {segmento['compras']}")
    st.markdown(f"**Rotación y venta:** {segmento['rotacion']}")

    st.caption('Datos ingresados')
    st.dataframe(data, hide_index=True)
