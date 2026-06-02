# PROYECTO SPOTIFY 2023 — PARTE 2

setwd("C:/Users/jhoni/Documents/ML/proyecto")
options(scipen = 999)

library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)
library(Metrics)
library(randomForest)
library(rpart)
library(rpart.plot)
library(glmnet)      
library(xgboost)     
library(e1071)        
library(scales)



# CARGA DE OBJETOS DE LA PARTE 1

spotify_limpio        <- readRDS("p1_spotify_limpio.rds")
datos_modelo_completo <- readRDS("p1_datos_modelo_completo.rds")
datos_torneo          <- readRDS("p1_datos_torneo.rds")
datos_train           <- readRDS("p1_datos_train.rds")
datos_test            <- readRDS("p1_datos_test.rds")
resultados_p1         <- readRDS("p1_resultados.rds")
cuartiles             <- readRDS("p1_cuartiles.rds")
load("p1_clasificar_exito.RData")

# Helper R²
r2_calc <- function(real, pred) {
  1 - sum((real - pred)^2) / sum((real - mean(real))^2)
}

cat("Datos Parte 1 cargados.\n")
cat("Train:", nrow(datos_train), "| Test:", nrow(datos_test), "\n\n")
cat("Referencia Parte 1:\n")
cat("  Reg. Lineal  → RMSE:", formatC(resultados_p1$lineal$rmse,  format="f", big.mark=",", digits=0),
    "| R²:", round(resultados_p1$lineal$r2  * 100, 2), "%\n")
cat("  Árbol        → RMSE:", formatC(resultados_p1$arbol$rmse,   format="f", big.mark=",", digits=0),
    "| R²:", round(resultados_p1$arbol$r2   * 100, 2), "%\n")
cat("  Random Forest→ RMSE:", formatC(resultados_p1$rf$rmse,      format="f", big.mark=",", digits=0),
    "| R²:", round(resultados_p1$rf$r2      * 100, 2), "%\n")


# FASE 1 — FEATURE ENGINEERING + MODELOS BASE RE-EVALUADOS

cat("\n\n========== FASE 1: FEATURE ENGINEERING ==========\n")

#Construcción de variables nuevas 
datos_fe <- datos_torneo %>%
  mutate(
    # Presencia total en todas las plataformas disponibles
    total_playlists        = in_spotify_playlists + in_apple_playlists,

    # Índice de viralidad: charts activos vs playlists pasivas
    ratio_charts_playlists = (in_spotify_charts + 1) / (in_spotify_playlists + 1),

    # Demanda externa: uso de Shazam como proxy de popularidad offline
    ratio_shazam_playlists = (in_shazam_charts + 1) / (in_spotify_playlists + 1),

    # Score ponderado: cada plataforma según su peso observado en P1
    score_popularidad      = (in_spotify_playlists * 2) +
                             (in_apple_playlists   * 1.5) +
                             (in_spotify_charts    * 3) +
                             (in_shazam_charts     * 1),

    # Antigüedad: tiempo de exposición acumulado hasta 2023
    antiguedad_anios       = 2023 - released_year
  )

cat("Variables originales:", ncol(datos_torneo) - 1, "\n")
cat("Variables con FE:    ", ncol(datos_fe) - 1, "\n")

# Correlación de las variables nuevas con streams
cor_fe <- cor(
  datos_fe %>% select(streams, total_playlists, ratio_charts_playlists,
                      ratio_shazam_playlists, score_popularidad, antiguedad_anios),
  use = "complete.obs"
)
cat("\nCorrelación variables nuevas con streams:\n")
print(round(cor_fe["streams", ], 3))

#Visualización: score_popularidad vs streams
ggplot(datos_fe, aes(x = score_popularidad, y = streams)) +
  geom_point(alpha = 0.4, color = "#1DB954", size = 2) +
  geom_smooth(method = "lm", color = "red", linetype = "dashed", se = FALSE) +
  scale_x_continuous(labels = scales::comma) +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title    = "Feature Engineering: Score de Popularidad vs Streams",
    subtitle = paste0("Correlación: ",
                      round(cor(datos_fe$score_popularidad,
                                datos_fe$streams, use = "complete.obs"), 3)),
    x = "Score de Popularidad Global", y = "Total de Streams"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

#Partición con FE 
set.seed(42)
idx_fe         <- createDataPartition(datos_fe$streams, p = 0.70, list = FALSE)
datos_train_fe <- datos_fe[ idx_fe, ]
datos_test_fe  <- datos_fe[-idx_fe, ]

cat("\nPartición FE — Train:", nrow(datos_train_fe),
    "| Test:", nrow(datos_test_fe), "\n")

#Regresión Lineal con FE
modelo_lm_fe <- lm(streams ~ ., data = datos_train_fe)

pred_lm_fe   <- predict(modelo_lm_fe, datos_test_fe)
rmse_lm_fe   <- rmse(datos_test_fe$streams, pred_lm_fe)
r2_lm_fe     <- r2_calc(datos_test_fe$streams, pred_lm_fe)

cat("\n--- Regresión Lineal + FE ---\n")
cat("RMSE:", formatC(rmse_lm_fe, format="f", big.mark=",", digits=0),
    "| R²:", round(r2_lm_fe * 100, 2), "%\n")
cat("Cambio vs Parte 1 — RMSE:",
    formatC(resultados_p1$lineal$rmse - rmse_lm_fe, format="f", big.mark=",", digits=0),
    "| R²: +", round((r2_lm_fe - resultados_p1$lineal$r2) * 100, 2), "pp\n")

mat_lm_fe <- confusionMatrix(
  factor(clasificar_exito(pred_lm_fe),         levels = c("Bajo","Medio","Alto","Viral")),
  factor(clasificar_exito(datos_test_fe$streams), levels = c("Bajo","Medio","Alto","Viral"))
)

#Árbol de Decisión con FE
modelo_arbol_fe <- rpart(
  streams ~ .,
  data    = datos_train_fe,
  method  = "anova",
  control = rpart.control(cp = 0.01, minsplit = 20, minbucket = 7)
)

pred_arbol_fe   <- predict(modelo_arbol_fe, datos_test_fe)
rmse_arbol_fe   <- rmse(datos_test_fe$streams, pred_arbol_fe)
r2_arbol_fe     <- r2_calc(datos_test_fe$streams, pred_arbol_fe)

cat("\n--- Árbol de Decisión + FE ---\n")
cat("RMSE:", formatC(rmse_arbol_fe, format="f", big.mark=",", digits=0),
    "| R²:", round(r2_arbol_fe * 100, 2), "%\n")
cat("Cambio vs Parte 1 — RMSE:",
    formatC(resultados_p1$arbol$rmse - rmse_arbol_fe, format="f", big.mark=",", digits=0),
    "| R²: +", round((r2_arbol_fe - resultados_p1$arbol$r2) * 100, 2), "pp\n")

mat_arbol_fe <- confusionMatrix(
  factor(clasificar_exito(pred_arbol_fe),          levels = c("Bajo","Medio","Alto","Viral")),
  factor(clasificar_exito(datos_test_fe$streams),  levels = c("Bajo","Medio","Alto","Viral"))
)

rpart.plot(modelo_arbol_fe,
           main = "Árbol de Decisión + Feature Engineering (cp=0.01)",
           type = 4, extra = 101, under = TRUE, faclen = 0,
           cex = 0.6, box.palette = "GnBu")

#Random Forest con FE
mtry_fe <- tuneRF(
  x        = datos_train_fe %>% select(-streams),
  y        = datos_train_fe$streams,
  ntreeTry = 300,
  stepFactor = 1.5,
  improve    = 0.01,
  trace      = FALSE,
  plot       = FALSE
)
mejor_mtry_fe <- mtry_fe[which.min(mtry_fe[, 2]), 1]
cat("\n--- Random Forest + FE ---\n")
cat("mtry seleccionado:", mejor_mtry_fe, "\n")

set.seed(123)
modelo_rf_fe <- randomForest(
  streams ~ .,
  data       = datos_train_fe,
  ntree      = 500,
  mtry       = mejor_mtry_fe,
  importance = TRUE
)

pred_rf_fe   <- predict(modelo_rf_fe, datos_test_fe)
rmse_rf_fe   <- rmse(datos_test_fe$streams, pred_rf_fe)
r2_rf_fe     <- r2_calc(datos_test_fe$streams, pred_rf_fe)

cat("RMSE:", formatC(rmse_rf_fe, format="f", big.mark=",", digits=0),
    "| R²:", round(r2_rf_fe * 100, 2), "%\n")
cat("Cambio vs Parte 1 — RMSE:",
    formatC(resultados_p1$rf$rmse - rmse_rf_fe, format="f", big.mark=",", digits=0),
    "| R²: +", round((r2_rf_fe - resultados_p1$rf$r2) * 100, 2), "pp\n")

mat_rf_fe <- confusionMatrix(
  factor(clasificar_exito(pred_rf_fe),             levels = c("Bajo","Medio","Alto","Viral")),
  factor(clasificar_exito(datos_test_fe$streams),  levels = c("Bajo","Medio","Alto","Viral"))
)

# Importancia de variables (¿las nuevas variables aportaron?)
imp_rf_fe          <- as.data.frame(importance(modelo_rf_fe))
imp_rf_fe$Variable <- rownames(imp_rf_fe)

ggplot(imp_rf_fe, aes(x = reorder(Variable, `%IncMSE`), y = `%IncMSE`)) +
  geom_bar(stat = "identity", fill = "#1DB954", alpha = 0.85, color = "black") +
  coord_flip() +
  labs(
    title    = "Importancia de Variables — RF + Feature Engineering",
    subtitle = "¿Las variables nuevas aparecen en el ranking?",
    x = "Variable", y = "% Incremento en MSE"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

#Tabla comparativa Fase 1 
tabla_fase1 <- data.frame(
  Modelo   = c("Reg. Lineal P1", "Árbol P1", "RF P1",
               "Reg. Lineal + FE", "Árbol + FE", "RF + FE"),
  RMSE     = c(resultados_p1$lineal$rmse, resultados_p1$arbol$rmse, resultados_p1$rf$rmse,
               rmse_lm_fe, rmse_arbol_fe, rmse_rf_fe),
  R2       = c(resultados_p1$lineal$r2, resultados_p1$arbol$r2, resultados_p1$rf$r2,
               r2_lm_fe, r2_arbol_fe, r2_rf_fe) * 100,
  Accuracy = c(resultados_p1$lineal$accuracy, resultados_p1$arbol$accuracy, resultados_p1$rf$accuracy,
               mat_lm_fe$overall["Accuracy"],
               mat_arbol_fe$overall["Accuracy"],
               mat_rf_fe$overall["Accuracy"]) * 100,
  Fase     = c("Parte 1","Parte 1","Parte 1","Fase 1","Fase 1","Fase 1")
)

cat("\n\nTabla comparativa Fase 1 (FE sin tuning):\n")
print(tabla_fase1 %>%
        mutate(RMSE     = formatC(RMSE, format="f", big.mark=",", digits=0),
               R2       = paste0(round(R2, 2), "%"),
               Accuracy = paste0(round(Accuracy, 2), "%")))

colores_fase1 <- c(
  "Reg. Lineal P1"  = "#ffccbc", "Árbol P1"  = "#c8e6c9", "RF P1"  = "#b3e5fc",
  "Reg. Lineal + FE"= "#e64a19", "Árbol + FE"= "#2e7d32", "RF + FE"= "#0277bd"
)

tabla_fase1$Modelo <- factor(
  tabla_fase1$Modelo,
  levels = tabla_fase1$Modelo[order(tabla_fase1$RMSE, decreasing = TRUE)]
)

ggplot(tabla_fase1, aes(x = Modelo, y = RMSE, fill = Modelo)) +
  geom_bar(stat = "identity", alpha = 0.9, color = "black", width = 0.65) +
  geom_text(aes(label = formatC(RMSE, format="f", big.mark=",", digits=0)),
            hjust = -0.1, fontface = "bold", size = 3.5) +
  coord_flip() +
  scale_fill_manual(values = colores_fase1) +
  scale_y_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.15))) +
  labs(
    title    = "Fase 1 — Impacto del Feature Engineering sobre modelos base",
    subtitle = "Colores claros = Parte 1 sin FE | Colores sólidos = con FE",
    x = NULL, y = "RMSE"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14), legend.position = "none")


# FASE 2 — TUNING DE HIPERPARÁMETROS

cat("\n\n========== FASE 2: TUNING DE HIPERPARÁMETROS ==========\n")

control_cv <- trainControl(
  method      = "cv",
  number      = 5,
  verboseIter = FALSE
)

# Regresión Linea
modelo_lm_step <- step(
  lm(streams ~ ., data = datos_train_fe),
  direction = "both",
  trace     = FALSE
)

pred_lm_step   <- predict(modelo_lm_step, datos_test_fe)
rmse_lm_step   <- rmse(datos_test_fe$streams, pred_lm_step)
r2_lm_step     <- r2_calc(datos_test_fe$streams, pred_lm_step)

cat("\n--- Regresión Lineal + FE + Stepwise ---\n")
cat("Variables seleccionadas:", length(coef(modelo_lm_step)) - 1, "\n")
cat("RMSE:", formatC(rmse_lm_step, format="f", big.mark=",", digits=0),
    "| R²:", round(r2_lm_step * 100, 2), "%\n")
cat("Mejora vs FE sin tuning — RMSE:",
    formatC(rmse_lm_fe - rmse_lm_step, format="f", big.mark=",", digits=0), "\n")

mat_lm_step <- confusionMatrix(
  factor(clasificar_exito(pred_lm_step),           levels = c("Bajo","Medio","Alto","Viral")),
  factor(clasificar_exito(datos_test_fe$streams),  levels = c("Bajo","Medio","Alto","Viral"))
)

#Árbol de Decisión — tuning de CP por CV 
grid_cp <- expand.grid(
  cp = c(0.0001, 0.0005, 0.001, 0.003, 0.005, 0.008, 0.01)
)

set.seed(42)
modelo_arbol_tuned <- train(
  streams ~ .,
  data      = datos_train_fe,
  method    = "rpart",
  tuneGrid  = grid_cp,
  trControl = control_cv,
  metric    = "RMSE"
)

cat("\n--- Árbol de Decisión + FE + CV ---\n")
cat("CP óptimo:", modelo_arbol_tuned$bestTune$cp, "\n")

ggplot(modelo_arbol_tuned) +
  labs(
    title    = "Tuning del Árbol — CP óptimo por CV",
    subtitle = paste0("CP seleccionado: ", round(modelo_arbol_tuned$bestTune$cp, 5)),
    x = "Complexity Parameter (CP)", y = "RMSE (Cross-Validation)"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

rpart.plot(modelo_arbol_tuned$finalModel,
           main = "Árbol Tuneado — CP Óptimo por CV",
           type = 4, extra = 101, under = TRUE, faclen = 0,
           cex = 0.6, box.palette = "GnBu")

pred_arbol_tuned   <- predict(modelo_arbol_tuned, datos_test_fe)
rmse_arbol_tuned   <- rmse(datos_test_fe$streams, pred_arbol_tuned)
r2_arbol_tuned     <- r2_calc(datos_test_fe$streams, pred_arbol_tuned)

cat("RMSE:", formatC(rmse_arbol_tuned, format="f", big.mark=",", digits=0),
    "| R²:", round(r2_arbol_tuned * 100, 2), "%\n")
cat("Mejora vs FE sin tuning — RMSE:",
    formatC(rmse_arbol_fe - rmse_arbol_tuned, format="f", big.mark=",", digits=0), "\n")

mat_arbol_tuned <- confusionMatrix(
  factor(clasificar_exito(pred_arbol_tuned),       levels = c("Bajo","Medio","Alto","Viral")),
  factor(clasificar_exito(datos_test_fe$streams),  levels = c("Bajo","Medio","Alto","Viral"))
)
print(mat_arbol_tuned)

ggplot(as.data.frame(mat_arbol_tuned$table),
       aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white", linewidth = 0.8) +
  geom_text(aes(label = Freq), size = 6, fontface = "bold", color = "black") +
  scale_fill_gradient(low = "#e8f5e9", high = "#1DB954") +
  labs(
    title    = "Matriz de Confusión — Árbol Tuneado",
    subtitle = paste0("Accuracy: ",
                      round(mat_arbol_tuned$overall["Accuracy"] * 100, 2), "%"),
    x = "Categoría Real", y = "Categoría Predicha", fill = "Frecuencia"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))
#Random Forest — tuning de mtry por CV 
grid_rf <- expand.grid(mtry = c(2, 3, 4, 5, 6, 7, 8, 9, 10))

set.seed(42)
modelo_rf_tuned <- train(
  streams ~ .,
  data       = datos_train_fe,
  method     = "rf",
  tuneGrid   = grid_rf,
  trControl  = control_cv,
  ntree      = 500,
  importance = TRUE
)

cat("\n--- Random Forest + FE + CV ---\n")
cat("mtry óptimo por CV:", modelo_rf_tuned$bestTune$mtry, "\n")

ggplot(modelo_rf_tuned) +
  labs(
    title    = "Random Forest — Tuning de mtry por CV",
    subtitle = paste0("mtry óptimo: ", modelo_rf_tuned$bestTune$mtry),
    x = "mtry (variables por split)", y = "RMSE (Cross-Validation)"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

pred_rf_tuned   <- predict(modelo_rf_tuned, datos_test_fe)
rmse_rf_tuned   <- rmse(datos_test_fe$streams, pred_rf_tuned)
r2_rf_tuned     <- r2_calc(datos_test_fe$streams, pred_rf_tuned)

cat("RMSE:", formatC(rmse_rf_tuned, format="f", big.mark=",", digits=0),
    "| R²:", round(r2_rf_tuned * 100, 2), "%\n")
cat("Mejora vs FE sin tuning — RMSE:",
    formatC(rmse_rf_fe - rmse_rf_tuned, format="f", big.mark=",", digits=0), "\n")

imp_rf_tuned          <- as.data.frame(importance(modelo_rf_tuned$finalModel))
imp_rf_tuned$Variable <- rownames(imp_rf_tuned)

ggplot(imp_rf_tuned, aes(x = reorder(Variable, `%IncMSE`), y = `%IncMSE`)) +
  geom_bar(stat = "identity", fill = "#0277bd", alpha = 0.85, color = "black") +
  coord_flip() +
  labs(
    title    = "Importancia de Variables — RF Tuneado",
    subtitle = paste0("mtry = ", modelo_rf_tuned$bestTune$mtry),
    x = "Variable", y = "% Incremento en MSE"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

mat_rf_tuned <- confusionMatrix(
  factor(clasificar_exito(pred_rf_tuned),          levels = c("Bajo","Medio","Alto","Viral")),
  factor(clasificar_exito(datos_test_fe$streams),  levels = c("Bajo","Medio","Alto","Viral"))
)
print(mat_rf_tuned)

ggplot(as.data.frame(mat_rf_tuned$table),
       aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white", linewidth = 0.8) +
  geom_text(aes(label = Freq), size = 6, fontface = "bold", color = "black") +
  scale_fill_gradient(low = "#e3f2fd", high = "#0277bd") +
  labs(
    title    = "Matriz de Confusión — Random Forest Tuneado",
    subtitle = paste0("Accuracy: ",
                      round(mat_rf_tuned$overall["Accuracy"] * 100, 2), "%"),
    x = "Categoría Real", y = "Categoría Predicha", fill = "Frecuencia"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

#Tabla comparativa Fase 2
tabla_fase2 <- data.frame(
  Modelo   = c("Reg. Lineal + FE", "Árbol + FE", "RF + FE",
               "Reg. Lineal Tuneada", "Árbol Tuneado", "RF Tuneado"),
  RMSE     = c(rmse_lm_fe, rmse_arbol_fe, rmse_rf_fe,
               rmse_lm_step, rmse_arbol_tuned, rmse_rf_tuned),
  R2       = c(r2_lm_fe, r2_arbol_fe, r2_rf_fe,
               r2_lm_step, r2_arbol_tuned, r2_rf_tuned) * 100,
  Accuracy = c(mat_lm_fe$overall["Accuracy"],
               mat_arbol_fe$overall["Accuracy"],
               mat_rf_fe$overall["Accuracy"],
               mat_lm_step$overall["Accuracy"],
               mat_arbol_tuned$overall["Accuracy"],
               mat_rf_tuned$overall["Accuracy"]) * 100,
  Fase     = c("Fase 1","Fase 1","Fase 1","Fase 2","Fase 2","Fase 2")
)

cat("\n\nTabla comparativa Fase 2 (FE + tuning):\n")
print(tabla_fase2 %>%
        mutate(RMSE     = formatC(RMSE, format="f", big.mark=",", digits=0),
               R2       = paste0(round(R2, 2), "%"),
               Accuracy = paste0(round(Accuracy, 2), "%")))

colores_fase2 <- c(
  "Reg. Lineal + FE"   = "#ffccbc", "Árbol + FE"   = "#c8e6c9", "RF + FE"   = "#b3e5fc",
  "Reg. Lineal Tuneada"= "#e64a19", "Árbol Tuneado"= "#2e7d32", "RF Tuneado"= "#0277bd"
)

tabla_fase2$Modelo <- factor(
  tabla_fase2$Modelo,
  levels = tabla_fase2$Modelo[order(tabla_fase2$RMSE, decreasing = TRUE)]
)

ggplot(tabla_fase2, aes(x = Modelo, y = RMSE, fill = Modelo)) +
  geom_bar(stat = "identity", alpha = 0.9, color = "black", width = 0.65) +
  geom_text(aes(label = formatC(RMSE, format="f", big.mark=",", digits=0)),
            hjust = -0.1, fontface = "bold", size = 3.5) +
  coord_flip() +
  scale_fill_manual(values = colores_fase2) +
  scale_y_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.15))) +
  labs(
    title    = "Fase 2 — Impacto del Tuning sobre modelos con FE",
    subtitle = "Colores claros = FE sin tuning | Colores sólidos = FE + tuning",
    x = NULL, y = "RMSE"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14), legend.position = "none")


# ============================================================
# ██████████████████████████████████████████████████████████
# FASE 3 — MODELOS AVANZADOS

cat("\n\n========== FASE 3: MODELOS AVANZADOS ==========\n")

# Lasso 
cat("\n--- Lasso ---\n")

x_train <- model.matrix(streams ~ ., data = datos_train_fe)[, -1]
y_train <- datos_train_fe$streams
x_test  <- model.matrix(streams ~ ., data = datos_test_fe)[, -1]
y_test  <- datos_test_fe$streams

set.seed(42)
cv_lasso     <- cv.glmnet(x_train, y_train, alpha = 1, nfolds = 5)
lambda_lasso <- cv_lasso$lambda.min

cat("Lambda óptimo:", round(lambda_lasso, 0), "\n")
cat("Variables eliminadas por Lasso:\n")
coef_lasso <- coef(cv_lasso, s = "lambda.min")
vars_cero  <- sum(coef_lasso[-1] == 0)
cat(" ", vars_cero, "de", nrow(coef_lasso) - 1, "variables llevadas a cero\n")

plot(cv_lasso, main = "Lasso — Selección de Lambda por CV")

pred_lasso   <- as.vector(predict(cv_lasso, s = lambda_lasso, newx = x_test))
rmse_lasso   <- rmse(y_test, pred_lasso)
r2_lasso     <- r2_calc(y_test, pred_lasso)

cat("RMSE:", formatC(rmse_lasso, format="f", big.mark=",", digits=0),
    "| R²:", round(r2_lasso * 100, 2), "%\n")

mat_lasso <- confusionMatrix(
  factor(clasificar_exito(pred_lasso),  levels = c("Bajo","Medio","Alto","Viral")),
  factor(clasificar_exito(y_test),      levels = c("Bajo","Medio","Alto","Viral"))
)

#Ridge
cat("\n--- Ridge ---\n")

set.seed(42)
cv_ridge     <- cv.glmnet(x_train, y_train, alpha = 0, nfolds = 5)
lambda_ridge <- cv_ridge$lambda.min

cat("Lambda óptimo:", round(lambda_ridge, 0), "\n")
plot(cv_ridge, main = "Ridge — Selección de Lambda por CV")

pred_ridge   <- as.vector(predict(cv_ridge, s = lambda_ridge, newx = x_test))
rmse_ridge   <- rmse(y_test, pred_ridge)
r2_ridge     <- r2_calc(y_test, pred_ridge)

cat("RMSE:", formatC(rmse_ridge, format="f", big.mark=",", digits=0),
    "| R²:", round(r2_ridge * 100, 2), "%\n")

mat_ridge <- confusionMatrix(
  factor(clasificar_exito(pred_ridge), levels = c("Bajo","Medio","Alto","Viral")),
  factor(clasificar_exito(y_test),     levels = c("Bajo","Medio","Alto","Viral"))
)

# Comparativa de coeficientes Lasso vs Ridge
coef_l <- as.data.frame(as.matrix(coef(cv_lasso, s = "lambda.min")))
coef_r <- as.data.frame(as.matrix(coef(cv_ridge, s = "lambda.min")))

coef_comp <- data.frame(
  Variable    = rownames(coef_l)[-1],
  Lasso       = coef_l[-1, 1],
  Ridge       = coef_r[-1, 1]
) %>%
  pivot_longer(cols = c(Lasso, Ridge), names_to = "Modelo", values_to = "Coeficiente")

ggplot(coef_comp, aes(x = reorder(Variable, abs(Coeficiente)),
                      y = Coeficiente, fill = Modelo)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.85, color = "black") +
  coord_flip() +
  scale_fill_manual(values = c("Lasso" = "#ff7043", "Ridge" = "#0277bd")) +
  labs(
    title    = "Coeficientes: Lasso vs Ridge",
    subtitle = "Barras ausentes en Lasso = variables eliminadas (coeficiente = 0)",
    x = "Variable", y = "Coeficiente"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

#XGBoost
cat("\n--- XGBoost ---\n")

escala_xgb    <- 1e6
streams_train <- datos_train_fe$streams / escala_xgb
streams_test  <- datos_test_fe$streams  / escala_xgb

x_train_mat <- as.matrix(datos_train_fe %>% select(-streams))
x_test_mat  <- as.matrix(datos_test_fe  %>% select(-streams))

dtrain <- xgb.DMatrix(data = x_train_mat, label = streams_train)
dtest  <- xgb.DMatrix(data = x_test_mat,  label = streams_test)

# Fase A: nrounds óptimo con CV
params_cv <- list(
  objective        = "reg:squarederror",
  eval_metric      = "rmse",
  max_depth        = 5,
  eta              = 0.05,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 1
)

set.seed(42)
cv_xgb <- xgb.cv(
  params                = params_cv,
  data                  = dtrain,
  nrounds               = 500,
  nfold                 = 5,
  early_stopping_rounds = 30,
  verbose               = 0
)

mejor_nrounds <- which.min(cv_xgb$evaluation_log$test_rmse_mean)
cat("nrounds óptimo (CV):", mejor_nrounds, "\n")

# Curva de aprendizaje
cv_log <- cv_xgb$evaluation_log
ggplot(cv_log, aes(x = iter)) +
  geom_line(aes(y = train_rmse_mean, color = "Train"), linewidth = 0.8) +
  geom_line(aes(y = test_rmse_mean,  color = "CV Test"), linewidth = 0.8) +
  geom_vline(xintercept = mejor_nrounds, color = "black",
             linetype = "dashed", linewidth = 0.7) +
  scale_color_manual(values = c("Train" = "#1DB954", "CV Test" = "#ff6f00")) +
  labs(
    title    = "XGBoost — Curva de Aprendizaje",
    subtitle = paste0("nrounds óptimo: ", mejor_nrounds),
    x = "Número de Árboles", y = "RMSE (millones de streams)", color = NULL
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

# Fase B: grid search max_depth y eta
grid_xgb <- expand.grid(max_depth = c(3, 5, 7), eta = c(0.01, 0.05, 0.1))
resultados_grid_xgb <- data.frame()

for (i in seq_len(nrow(grid_xgb))) {
  p_i <- list(
    objective        = "reg:squarederror",
    eval_metric      = "rmse",
    max_depth        = grid_xgb$max_depth[i],
    eta              = grid_xgb$eta[i],
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 1
  )
  set.seed(42)
  cv_i <- xgb.cv(params = p_i, data = dtrain, nrounds = 300,
                 nfold = 5, early_stopping_rounds = 20, verbose = 0)
  resultados_grid_xgb <- rbind(resultados_grid_xgb, data.frame(
    max_depth    = grid_xgb$max_depth[i],
    eta          = grid_xgb$eta[i],
    best_nrounds = which.min(cv_i$evaluation_log$test_rmse_mean),
    rmse_cv      = min(cv_i$evaluation_log$test_rmse_mean)
  ))
  cat("  depth=", grid_xgb$max_depth[i], "| eta=", grid_xgb$eta[i],
      "| nrounds=", which.min(cv_i$evaluation_log$test_rmse_mean),
      "| RMSE:", round(min(cv_i$evaluation_log$test_rmse_mean), 4), "\n")
}

mejor_xgb <- resultados_grid_xgb[which.min(resultados_grid_xgb$rmse_cv), ]
cat("\nMejor configuración XGBoost:\n"); print(mejor_xgb)

ggplot(resultados_grid_xgb,
       aes(x = factor(eta), y = rmse_cv,
           color = factor(max_depth), group = factor(max_depth))) +
  geom_line(linewidth = 1) + geom_point(size = 3) +
  scale_color_manual(values = c("3" = "#1DB954", "5" = "#0277bd", "7" = "#ff6f00")) +
  labs(title = "XGBoost — Grid Search: max_depth vs eta",
       subtitle = "Menor RMSE CV = mejor combinación",
       x = "Learning Rate (eta)", y = "RMSE CV (M streams)", color = "max_depth") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

# Modelo final XGBoost
params_xgb <- list(
  objective        = "reg:squarederror",
  eval_metric      = "rmse",
  max_depth        = mejor_xgb$max_depth,
  eta              = mejor_xgb$eta,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 1
)

set.seed(42)
modelo_xgb <- xgb.train(
  params    = params_xgb,
  data      = dtrain,
  nrounds   = mejor_xgb$best_nrounds,
  watchlist = list(train = dtrain, test = dtest),
  verbose   = 0
)

pred_xgb   <- predict(modelo_xgb, dtest) * escala_xgb
rmse_xgb   <- rmse(datos_test_fe$streams, pred_xgb)
r2_xgb     <- r2_calc(datos_test_fe$streams, pred_xgb)

cat("\nXGBoost Final\n")
cat("RMSE:", formatC(rmse_xgb, format="f", big.mark=",", digits=0),
    "| R²:", round(r2_xgb * 100, 2), "%\n")

ggplot(data.frame(Real = datos_test_fe$streams, Predicho = pred_xgb),
       aes(x = Real, y = Predicho)) +
  geom_point(alpha = 0.5, color = "#ff6f00", size = 2) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
  scale_x_continuous(labels = scales::comma) +
  scale_y_continuous(labels = scales::comma) +
  labs(title = "XGBoost: Valores Reales vs Predichos",
       subtitle = paste0("RMSE: ", formatC(rmse_xgb, format="f", big.mark=",", digits=0),
                         "  |  R²: ", round(r2_xgb * 100, 2), "%"),
       x = "Streams Reales", y = "Streams Predichos") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

imp_xgb <- xgb.importance(feature_names = colnames(x_train_mat), model = modelo_xgb)
ggplot(head(imp_xgb, 12), aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "#ff6f00", alpha = 0.85, color = "black") +
  coord_flip() +
  labs(title = "XGBoost — Importancia de Variables (Gain)",
       x = NULL, y = "Gain") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

mat_xgb <- confusionMatrix(
  factor(clasificar_exito(pred_xgb),           levels = c("Bajo","Medio","Alto","Viral")),
  factor(clasificar_exito(datos_test_fe$streams), levels = c("Bajo","Medio","Alto","Viral"))
)
print(mat_xgb)

ggplot(as.data.frame(mat_xgb$table), aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white", linewidth = 0.8) +
  geom_text(aes(label = Freq), size = 6, fontface = "bold", color = "black") +
  scale_fill_gradient(low = "#fff3e0", high = "#ff6f00") +
  labs(title = "Matriz de Confusión — XGBoost",
       subtitle = paste0("Accuracy: ", round(mat_xgb$overall["Accuracy"] * 100, 2), "%"),
       x = "Categoría Real", y = "Categoría Predicha", fill = "Frecuencia") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

# SVM 
cat("\n--- SVM ---\n")

# Escalamos las variables numéricas (SVM es sensible a la escala)
prep_svm <- preProcess(datos_train_fe %>% select(-streams),
                       method = c("center", "scale"))

train_svm <- predict(prep_svm, datos_train_fe)
test_svm  <- predict(prep_svm, datos_test_fe)

# Tuning: buscamos el mejor costo y gamma con CV
grid_svm <- expand.grid(
  C     = c(0.1, 1, 10, 100),
  sigma = c(0.001, 0.01, 0.1)
)

set.seed(42)
modelo_svm <- train(
  streams ~ .,
  data      = train_svm,
  method    = "svmRadial",
  tuneGrid  = grid_svm,
  trControl = control_cv,
  metric    = "RMSE"
)

cat("Mejores parámetros SVM:\n")
print(modelo_svm$bestTune)

ggplot(modelo_svm) +
  labs(title = "SVM — Tuning de C y sigma por CV",
       x = "Costo (C)", y = "RMSE (Cross-Validation)") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

pred_svm   <- predict(modelo_svm, test_svm)
rmse_svm   <- rmse(datos_test_fe$streams, pred_svm)
r2_svm     <- r2_calc(datos_test_fe$streams, pred_svm)

cat("RMSE:", formatC(rmse_svm, format="f", big.mark=",", digits=0),
    "| R²:", round(r2_svm * 100, 2), "%\n")

mat_svm <- confusionMatrix(
  factor(clasificar_exito(pred_svm),           levels = c("Bajo","Medio","Alto","Viral")),
  factor(clasificar_exito(datos_test_fe$streams), levels = c("Bajo","Medio","Alto","Viral"))
)
print(mat_svm)

ggplot(as.data.frame(mat_svm$table), aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white", linewidth = 0.8) +
  geom_text(aes(label = Freq), size = 6, fontface = "bold", color = "black") +
  scale_fill_gradient(low = "#f3e5f5", high = "#6a1b9a") +
  labs(title = "Matriz de Confusión — SVM",
       subtitle = paste0("Accuracy: ", round(mat_svm$overall["Accuracy"] * 100, 2), "%"),
       x = "Categoría Real", y = "Categoría Predicha", fill = "Frecuencia") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

# Stacking
cat("\n--- Stacking ---\n")

# Generamos predicciones out-of-fold (OOF) del nivel 1 sobre train
set.seed(42)
folds_stack <- createFolds(datos_train_fe$streams, k = 5, list = TRUE)

oof_rf  <- numeric(nrow(datos_train_fe))
oof_xgb <- numeric(nrow(datos_train_fe))
oof_svm <- numeric(nrow(datos_train_fe))

for (fold in folds_stack) {
  tr_idx <- setdiff(seq_len(nrow(datos_train_fe)), fold)
  val_idx <- fold

  # RF
  rf_fold <- randomForest(
    streams ~ .,
    data  = datos_train_fe[tr_idx, ],
    ntree = 300,
    mtry  = modelo_rf_tuned$bestTune$mtry
  )
  oof_rf[val_idx] <- predict(rf_fold, datos_train_fe[val_idx, ])

  # XGBoost
  d_tr  <- xgb.DMatrix(
    data  = as.matrix(datos_train_fe[tr_idx,  ] %>% select(-streams)),
    label = datos_train_fe$streams[tr_idx] / escala_xgb
  )
  d_val <- xgb.DMatrix(
    data = as.matrix(datos_train_fe[val_idx, ] %>% select(-streams))
  )
  xgb_fold <- xgb.train(params = params_xgb, data = d_tr,
                         nrounds = mejor_xgb$best_nrounds, verbose = 0)
  oof_xgb[val_idx] <- predict(xgb_fold, d_val) * escala_xgb

  # SVM
  prep_fold <- preProcess(datos_train_fe[tr_idx, ] %>% select(-streams),
                           method = c("center", "scale"))
  tr_s  <- predict(prep_fold, datos_train_fe[tr_idx,  ])
  val_s <- predict(prep_fold, datos_train_fe[val_idx, ])
  svm_fold <- svm(streams ~ ., data = tr_s,
                  kernel = "radial",
                  cost   = modelo_svm$bestTune$C,
                  gamma  = modelo_svm$bestTune$sigma)
  oof_svm[val_idx] <- predict(svm_fold, val_s)
}

cat("Predicciones OOF generadas.\n")

# Meta-features de entrenamiento
meta_train <- data.frame(
  pred_rf  = oof_rf,
  pred_xgb = oof_xgb,
  pred_svm = oof_svm,
  real     = datos_train_fe$streams
)

# Meta-learner: regresión lineal sobre predicciones OOF
meta_model <- lm(real ~ pred_rf + pred_xgb + pred_svm, data = meta_train)
cat("Pesos del meta-learner:\n")
print(round(coef(meta_model), 4))

# Predicciones del nivel 1 sobre test
meta_test <- data.frame(
  pred_rf  = predict(modelo_rf_tuned,    datos_test_fe),
  pred_xgb = pred_xgb,
  pred_svm = predict(modelo_svm, test_svm)
)

# Predicción final del stacking
pred_stack   <- predict(meta_model, meta_test)
rmse_stack   <- rmse(datos_test_fe$streams, pred_stack)
r2_stack     <- r2_calc(datos_test_fe$streams, pred_stack)

cat("\nStacking (RF + XGBoost + SVM)\n")
cat("RMSE:", formatC(rmse_stack, format="f", big.mark=",", digits=0),
    "| R²:", round(r2_stack * 100, 2), "%\n")

mat_stack <- confusionMatrix(
  factor(clasificar_exito(pred_stack),             levels = c("Bajo","Medio","Alto","Viral")),
  factor(clasificar_exito(datos_test_fe$streams),  levels = c("Bajo","Medio","Alto","Viral"))
)
print(mat_stack)

ggplot(as.data.frame(mat_stack$table), aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white", linewidth = 0.8) +
  geom_text(aes(label = Freq), size = 6, fontface = "bold", color = "black") +
  scale_fill_gradient(low = "#e8eaf6", high = "#283593") +
  labs(title = "Matriz de Confusión — Stacking",
       subtitle = paste0("Accuracy: ", round(mat_stack$overall["Accuracy"] * 100, 2), "%"),
       x = "Categoría Real", y = "Categoría Predicha", fill = "Frecuencia") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))


# TORNEO FINAL — LAS TRES FASES

cat("\n\n========== TORNEO FINAL ==========\n")

tabla_torneo <- data.frame(
  Fase   = c(
    "Parte 1","Parte 1","Parte 1",
    "Fase 1", "Fase 1", "Fase 1",
    "Fase 2", "Fase 2", "Fase 2",
    "Fase 3", "Fase 3", "Fase 3", "Fase 3", "Fase 3"
  ),
  Modelo = c(
    "Reg. Lineal P1","Árbol P1","RF P1",
    "Reg. Lineal + FE","Árbol + FE","RF + FE",
    "Reg. Lineal Tuneada","Árbol Tuneado","RF Tuneado",
    "Lasso","Ridge","XGBoost","SVM","Stacking"
  ),
  RMSE = c(
    resultados_p1$lineal$rmse, resultados_p1$arbol$rmse, resultados_p1$rf$rmse,
    rmse_lm_fe,   rmse_arbol_fe,   rmse_rf_fe,
    rmse_lm_step, rmse_arbol_tuned, rmse_rf_tuned,
    rmse_lasso, rmse_ridge, rmse_xgb, rmse_svm, rmse_stack
  ),
  R2 = c(
    resultados_p1$lineal$r2, resultados_p1$arbol$r2, resultados_p1$rf$r2,
    r2_lm_fe,   r2_arbol_fe,   r2_rf_fe,
    r2_lm_step, r2_arbol_tuned, r2_rf_tuned,
    r2_lasso, r2_ridge, r2_xgb, r2_svm, r2_stack
  ) * 100,
  Accuracy = c(
    resultados_p1$lineal$accuracy, resultados_p1$arbol$accuracy, resultados_p1$rf$accuracy,
    mat_lm_fe$overall["Accuracy"],
    mat_arbol_fe$overall["Accuracy"],
    mat_rf_fe$overall["Accuracy"],
    mat_lm_step$overall["Accuracy"],
    mat_arbol_tuned$overall["Accuracy"],
    mat_rf_tuned$overall["Accuracy"],
    mat_lasso$overall["Accuracy"],
    mat_ridge$overall["Accuracy"],
    mat_xgb$overall["Accuracy"],
    mat_svm$overall["Accuracy"],
    mat_stack$overall["Accuracy"]
  ) * 100
)

cat("\nTabla de posiciones completa:\n")
print(tabla_torneo %>%
        mutate(RMSE     = formatC(RMSE, format="f", big.mark=",", digits=0),
               R2       = paste0(round(R2, 2), "%"),
               Accuracy = paste0(round(Accuracy, 2), "%")) %>%
        arrange(RMSE))

# Paleta de colores por fase
colores_torneo <- c(
  "Reg. Lineal P1"      = "#ffccbc",
  "Árbol P1"            = "#c8e6c9",
  "RF P1"               = "#b3e5fc",
  "Reg. Lineal + FE"    = "#ef9a9a",
  "Árbol + FE"          = "#a5d6a7",
  "RF + FE"             = "#90caf9",
  "Reg. Lineal Tuneada" = "#e53935",
  "Árbol Tuneado"       = "#2e7d32",
  "RF Tuneado"          = "#0277bd",
  "Lasso"               = "#ff7043",
  "Ridge"               = "#5c6bc0",
  "XGBoost"             = "#ff6f00",
  "SVM"                 = "#6a1b9a",
  "Stacking"            = "#1a237e"
)

tabla_torneo$Modelo <- factor(
  tabla_torneo$Modelo,
  levels = tabla_torneo$Modelo[order(tabla_torneo$RMSE, decreasing = TRUE)]
)

# Gráfico RMSE
ggplot(tabla_torneo, aes(x = Modelo, y = RMSE, fill = Modelo)) +
  geom_bar(stat = "identity", alpha = 0.9, color = "black", width = 0.7) +
  geom_text(aes(label = formatC(RMSE, format="f", big.mark=",", digits=0)),
            hjust = -0.05, fontface = "bold", size = 2.8) +
  coord_flip() +
  scale_fill_manual(values = colores_torneo) +
  scale_y_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.18))) +
  labs(
    title    = "Torneo Final — Comparativa de RMSE (Parte 1 · Fase 1 · Fase 2 · Fase 3)",
    subtitle = "Menor RMSE = Mejor modelo",
    x = NULL, y = "RMSE"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 13), legend.position = "none")

# Gráfico R²
tabla_r2 <- tabla_torneo %>%
  mutate(Modelo = factor(Modelo, levels = Modelo[order(R2)]))

ggplot(tabla_r2, aes(x = Modelo, y = R2, fill = Modelo)) +
  geom_bar(stat = "identity", alpha = 0.9, color = "black", width = 0.7) +
  geom_text(aes(label = paste0(round(R2, 2), "%")),
            hjust = -0.05, fontface = "bold", size = 2.8) +
  coord_flip() +
  scale_fill_manual(values = colores_torneo) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  labs(
    title    = "Torneo Final — Comparativa de R²",
    subtitle = "Mayor R² = Mayor varianza explicada",
    x = NULL, y = "R² (%)"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 13), legend.position = "none")

# Gráfico Accuracy
tabla_acc <- tabla_torneo %>%
  mutate(Modelo = factor(Modelo, levels = Modelo[order(Accuracy)]))

ggplot(tabla_acc, aes(x = Modelo, y = Accuracy, fill = Modelo)) +
  geom_bar(stat = "identity", alpha = 0.9, color = "black", width = 0.7) +
  geom_text(aes(label = paste0(round(Accuracy, 2), "%")),
            hjust = -0.05, fontface = "bold", size = 2.8) +
  coord_flip() +
  scale_fill_manual(values = colores_torneo) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  labs(
    title    = "Torneo Final — Comparativa de Accuracy",
    subtitle = "Mayor Accuracy = Mejor clasificación por rangos de éxito",
    x = NULL, y = "Accuracy (%)"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 13), legend.position = "none")

# Resumen ejecutivo
ganador <- tabla_torneo[which.min(tabla_torneo$RMSE), ]
cat("\n============================================\n")
cat("RESUMEN FINAL\n")
cat("============================================\n")
cat("Modelo ganador: ", as.character(ganador$Modelo), "\n")
cat("RMSE:           ", formatC(ganador$RMSE, format="f", big.mark=",", digits=0), "\n")
cat("R²:             ", round(ganador$R2, 2), "%\n")
cat("Accuracy:       ", round(ganador$Accuracy, 2), "%\n")
cat("\nMejora total sobre RF Parte 1:\n")
cat("  RMSE: -", formatC(resultados_p1$rf$rmse - ganador$RMSE,
                         format="f", big.mark=",", digits=0), "streams\n")
cat("  R²:   +", round(ganador$R2 - resultados_p1$rf$r2 * 100, 2), "pp\n")



