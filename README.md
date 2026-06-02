# Spotify 2023 — Predicción de Éxito Musical con Machine Learning

Proyecto académico de análisis de datos y modelado predictivo sobre el dataset de Spotify 2023. El objetivo es identificar qué factores determinan que una canción alcance altos conteos de streams, comparando tres modelos de machine learning bajo un torneo de evaluación.

**Institución:** Benemérita Universidad Autónoma de Puebla — Facultad de Ciencias de la Computación  
**Autor:** Jonathan Ezequiel Quiñones Sánchez

---

## Dataset

**Fuente:** `spotify-2023.csv`  
**Registros:** 953 canciones  
**Variables:** 25 columnas que incluyen metadatos de la canción, métricas de plataforma y características de audio

| Categoría | Variables |
|---|---|
| Metadatos | nombre, artista(s), fecha de lanzamiento |
| Métricas de plataforma | Spotify playlists/charts, Apple Music, Deezer, Shazam |
| Características de audio | BPM, danceability, valence, energy, acousticness, instrumentalness, liveness, speechiness |
| Variable objetivo | `streams` (total de reproducciones) |

---

## Tecnologías

- **Lenguaje:** R
- **Manipulación de datos:** `dplyr`
- **Visualización:** `ggplot2`, `corrplot`
- **Modelos:** `rpart`, `rpart.plot`, `randomForest`
- **Evaluación:** `caret`, `Metrics`
- **Reporte:** R Markdown con tema CSS oscuro personalizado (Spotify branding)

---

## Instalación

```r
install.packages(c("dplyr", "ggplot2", "corrplot", "rpart", "rpart.plot",
                   "randomForest", "Metrics", "caret", "scales"))
```

---

## Ejecución

Script de análisis completo:

```r
source("proyectoSpotify.R")
```

Reporte HTML con documentación completa:

```r
rmarkdown::render("proyectoSpotify111.Rmd")
```

O desde RStudio: abrir el `.Rmd` y hacer clic en **Knit**.

---

## Pipeline de Análisis

```
Datos crudos
    │
    ▼
Limpieza (spotify_limpio)
  ├── Conversión de columnas Deezer/Shazam (string → numeric)
  ├── Eliminación de filas sin streams
  └── Imputación de ceros en métricas de plataforma
    │
    ▼
Análisis Exploratorio (EDA)
  ├── Distribución de streams
  ├── Scatter plots de características de audio vs. streams
  └── Matriz de correlación
    │
    ▼
Business Intelligence
  ├── Impacto de playlists en streams
  ├── Estacionalidad por mes de lanzamiento
  ├── "ADN musical" por artista (top 20)
  └── Impacto de colaboraciones (solistas vs. grupos)
    │
    ▼
Modelado (split 70/30 con caret::createDataPartition)
  ├── Regresión Lineal (modelo completo → reducido por significancia)
  ├── Árbol de Decisión CART (poda con grid de CP)
  └── Random Forest (tuning de mtry + análisis de error OOB)
    │
    ▼
Torneo de Evaluación
  ├── RMSE y R² sobre el mismo conjunto de prueba
  └── Accuracy por categoría: Bajo / Medio / Alto / Viral
      (cuartiles de streams → matrices de confusión)
```

---

## Modelos Comparados

| Modelo | Tipo | Parámetros clave |
|---|---|---|
| Regresión Lineal | Paramétrico | Selección por significancia estadística |
| Árbol de Decisión (CART) | No paramétrico | Poda por complexity parameter (CP) |
| Random Forest | Ensemble | Tuning de `mtry`, error OOB |

La evaluación final clasifica las predicciones en cuatro categorías de éxito definidas por cuartiles del target `streams`:

- **Bajo** — cuartil inferior
- **Medio** — rango intercuartil bajo
- **Alto** — rango intercuartil alto
- **Viral** — cuartil superior

---

## Estructura del Proyecto

```
spotify/
├── proyectoSpotify.R          # Script de análisis ejecutable (683 líneas)
├── proyectoSpotify111.Rmd     # Reporte R Markdown documentado (1,894 líneas)
├── spotify-2023.csv           # Dataset principal
├── CLAUDE.md                  # Guía para Claude Code
└── README.md                  # Este archivo
```

---

## Reporte

El archivo `.Rmd` genera un documento HTML con:

- Tema visual oscuro inspirado en Spotify (`#1DB954` / `#121212`)
- Tabla de contenidos interactiva
- Secciones narrativas con contexto teórico antes de cada bloque de código
- Todas las visualizaciones embebidas
