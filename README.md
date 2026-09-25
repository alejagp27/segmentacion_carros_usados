# 🚗 Segmentación de Vehículos Usados — Inventario UK

**Proyecto:** Tasación inteligente de vehículos usados mediante minería de datos — Dataset 26.000 UK Used Cars

Aplicación web que asigna un vehículo usado a uno de los perfiles del inventario, a partir de sus características, y muestra la estrategia de compra y venta del segmento al que pertenece. Desarrollada como parte del proyecto final del curso de Minería de Datos — Maestría en Ciencia de Datos, Universidad Pontificia Bolivariana.

---

## 📌 Descripción

La aplicación tiene tres secciones:

**1. Perfiles del inventario (sección principal).** Al ingresar, la aplicación muestra los 11 perfiles que resultaron del clustering, cada uno en una tarjeta con su descripción, segmento, número de vehículos, precio mediano, rango de precios y silueta. Debajo se puede consultar cómo se agrupan los perfiles en los 4 segmentos de negocio. No requiere ingresar ningún dato.

**2. Identificar el perfil de un vehículo.** El comprador ingresa las características del vehículo (marca, transmisión, combustible, año, millas, impuesto, rendimiento, tamaño del motor y precio publicado) y la aplicación devuelve:

- El **perfil** al que pertenece (uno de 11) y su descripción.
- El **segmento de negocio** (uno de 4).
- El precio mediano del perfil y la diferencia porcentual frente al precio ingresado.
- La **estrategia de compra y rotación** del segmento.
- La **silueta del perfil asignado** frente a la meta de 0,5, con un aviso si es uno de los perfiles menos definidos (7, 9 y 10).
- Una alerta cuando el vehículo se aleja de lo habitual en su perfil.

**3. Calidad del modelo.** Al final de la página se muestran siempre la silueta general (0,644) y la inercia general (7.607,7), con una explicación de qué significa cada una.

---

## 📊 Dataset

- **Fuente:** avisos de carros usados del Reino Unido (Audi, BMW y Hyundai)
- **Registros:** 26.306 (100% de los datos, tras la limpieza)
- **Variables:** año, millas, impuesto, rendimiento (mpg), tamaño del motor, precio, transmisión, combustible y marca

---

## 🤖 Modelo

**K-means** con `n_clusters=11`, `n_init=100`, `random_state=42`.

| Métrica | Valor | Meta |
|---|---|---|
| Inercia | 7.608 | Menor posible |
| Coeficiente de silueta | 0,644 | ≥ 0,5 ✅ |

Preparación: normalización Min-Max de las variables numéricas y variables dummy (sin eliminar categorías) para transmisión, combustible y marca. El número de clústeres se eligió con el método del codo, la silueta y la homogeneidad de marca: con k = 11 ningún clúster mezcla marcas.

### Segmentos

| Segmento | Perfiles | Vehículos |
|---|---|---|
| Premium alto | 3, 4, 6, 7 y 9 | 48,5% |
| Premium de entrada manual | 2, 5 y 8 | 22,8% |
| Generalista (Hyundai) | 0 y 10 | 18,5% |
| Premium automático accesible | 1 | 10,2% |

---

## 🗂️ Estructura del repositorio

```
📁 repo/
├── app.py                    # Aplicación Streamlit
├── modelo-cla.pkl            # Modelo K-means + columnas del entrenamiento + escalador Min-Max
├── requirements.txt          # Dependencias
├── .streamlit/config.toml    # Color del tema de la aplicación
└── README.md
```

---

## 🚀 Despliegue local

```bash
# 1. Clonar el repositorio
git clone https://github.com/alejagp27/segmentacion_carros_usados
cd segmentacion_carros_usados

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar la app
streamlit run app.py
```

---

## 🌐 Aplicación en línea

👉 [Ver aplicación en Streamlit](https://segmentacioncarrosusados-4cah8vvussugmmu4qs7s7c.streamlit.app)

---

## 🛠️ Tecnologías

- Python
- Scikit-learn
- Streamlit
- Pandas

---

## 📋 Metodología

El proyecto sigue las fases de **CRISP-DM**:

1. **Entendimiento del negocio** — Identificar qué segmentos de vehículo componen el inventario
2. **Entendimiento de los datos** — Perfilado, distribuciones y reglas de calidad
3. **Preparación de datos** — Limpieza, normalización Min-Max y variables dummy
4. **Modelamiento** — K-means, selección de k con codo, silueta y homogeneidad de marca
5. **Evaluación** — Inercia, silueta global y silueta por clúster
6. **Despliegue** — Aplicación interactiva en Streamlit

---

## ✍️ Autores

Cesar Maldonado, Diana Carolina López y Alejandra Galindo
