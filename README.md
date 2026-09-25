# 🚗 Segmentación de Vehículos Usados — Inventario UK

Aplicación web que asigna un vehículo usado a uno de los perfiles del inventario, a partir de sus características, y muestra la estrategia de compra y venta del segmento al que pertenece. Desarrollada como parte del proyecto final del curso de Minería de Datos — Maestría en Ciencia de Datos, Universidad Pontificia Bolivariana.

---

## 📌 Descripción

El comprador ingresa las características del vehículo (marca, transmisión, combustible, año, millas, impuesto, rendimiento, tamaño del motor y precio publicado) y la aplicación devuelve:

- El **perfil** al que pertenece (uno de 11) y su descripción.
- El **segmento de negocio** (uno de 4).
- El precio mediano del perfil y la diferencia porcentual frente al precio ingresado.
- La **estrategia de compra y rotación** del segmento.
- Una alerta cuando el vehículo se aleja de lo habitual en su perfil.

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
└── README.md
```

---

## 🚀 Despliegue local

```bash
# 1. Clonar el repositorio
git clone https://github.com/<usuario>/<repositorio>
cd <repositorio>

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar la app
streamlit run app.py
```

---

## 🌐 Aplicación en línea

👉 [Ver aplicación en Streamlit](<enlace de la app>)

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

<Nombres del equipo> — Maestría en Ciencia de Datos, UPB 2026
