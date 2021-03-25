library(rpart.plot)
library(randomForest)
library(modeldata)
library(pROC)
library(ranger)
library(leaflet)
library(rsample)
library(tidyverse)
library(skimr)
library(AmesHousing)
library(vip)
library(GGally)
library(patchwork)
library(glmnet)
library(plotmo)
library(rpart)
library (forcats)
library (readxl)
library(knitr)
library (funModeling)

# Análise inicial de dados -----------------------------------------------------

dados <- read_excel("C:\\Users\\dougl\\Documents\\PADS\\Atividade integradora\\output.xlsx")


# Tratamento inicial de dados que será feito no Python --------------------

# Precisamos tirar a variável "TIPO" pois ela esta toda igual
dados <- subset(dados, select = -TIPO)

# Esqueci de tirar estas variáveis do DF como combinamos
dados <- subset(dados, select = -F2)
dados <- subset(dados, select = -F4)
dados <- subset(dados, select = -F6)
dados <- subset(dados, select = -Q3)
dados <- subset(dados, select = -Q4)
dados <- subset(dados, select = -Q6)
dados <- subset(dados, select = -Q7)
dados <- subset(dados, select = -Q8a)


dados<-subset(dados, J1<11)


# Preparação dos dos dados para o trabalho --------------------------------

# Transformando o J1 em duas classes

dados <- dados %>% 
  mutate(J1 = factor(ifelse(J1 >= 8, "Alta", "Baixa"), 
                          levels = c("Baixa", "Alta")))

plot_num(dados)

# Trasnformar algumas variáveis em factor

dados <- dados %>% 
  mutate_at(vars (Q5, Q9, D1, Q9, A1_1, A1_2, A1_3, A1_4, F1, F3), ~as.factor(.))

str(dados)

# Treinamento e teste ----------------------------------------------------------

set.seed(123)

idx <- sample(nrow(dados), size = .75*nrow(dados), replace = FALSE)

treino <- dados[idx,]
teste <- dados[-idx,]


# Definição de corte proposto ---------------------------------------------

corte <- 0.50

# comparação dos modelos --------------------------------------------------

resultados <- tibble(modelo = c("logistica", "ridge", "lasso", "elastic-net", "Arvore", "Floresta Aleatória", "Bagging"), 
                     acuracia = NA, 
                     auc = NA)

# Modelo de regressão logística ---------------------

fit_tr <- glm(J1 ~ ., data = treino, family = "binomial")

prob_logistica <- predict(fit_tr, teste, type = "response")

resultados$acuracia[resultados$modelo == "logistica"] <-
  mean(dados$J1[-idx] == ifelse(prob_logistica >= corte, "Baixa", "Alta"))

resultados$auc[resultados$modelo == "logistica"] <- roc(dados$J1[-idx], prob_logistica)$auc

# Plotagem da curva ROC

estimativas_test_RL <- tibble(observado = teste$J1, 
                           probabilidade = prob_logistica)

roc_fit_RL <- estimativas_test_RL %>% 
  roc(response = observado, predictor = probabilidade)

plot(roc_fit_RL, print.auc = TRUE, print.auc.x = .5, print.auc.y = .5)

# entradas para glmnet ----------------------------------------------------------------

X_treino <- model.matrix(J1 ~ ., data = dados)[idx, -1]
y_treino <- dados$J1[idx]

X_test <- model.matrix(J1 ~ ., data = dados)[-idx,-1]
y_test <- dados$J1[-idx]


# ridge ----------------------------------------------------------------

ridge <- glmnet(X_treino, y_treino, alpha = 0, family = "binomial")

plot_glmnet(ridge, lwd = 2, cex.lab = 1.3)

cv_ridge <- cv.glmnet(X_treino, y_treino, alpha = 0, family = "binomial")

plot(cv_ridge, cex.lab = 1.3)

lambda_ridge <- cv_ridge$lambda.1se

prob_ridge <- as.numeric(predict(ridge, newx = X_test, type = "response", s = lambda_ridge))

resultados$acuracia[resultados$modelo == "elastic-net"] <- mean(y_test == ifelse(prob_ridge >= corte, "Baixa", "Alta"))

resultados$auc[resultados$modelo == "elastic-net"] <- roc(dados$J1[-idx], prob_ridge)$auc


# lasso ----------------------------------------------------------------

lasso <- glmnet(X_treino, y_treino, alpha = 1, family = "binomial", nlambda = 1000)

cv_lasso <- cv.glmnet(X_treino, y_treino, alpha = 1, family = "binomial")

plot(cv_lasso, cex.lab = 1.3)

lambda_lasso <- cv_lasso$lambda.1se

prob_lasso <- as.numeric(predict(lasso, newx = X_test, type = "response", s = lambda_lasso))

resultados$acuracia[resultados$modelo == "lasso"] <- mean(y_test == ifelse(prob_lasso > corte, "Baixa", "Alta"))

resultados$auc[resultados$modelo == "lasso"] <- roc(dados$J1[-idx], prob_lasso)$auc


# elastic net ----------------------------------------------------------------

elastic_net <- glmnet(X_treino, y_treino, alpha = 0.5, family = "binomial", nlambda = 1000)

cv_elastic_net <- cv.glmnet(X_treino, y_treino, alpha = 0.5, family = "binomial")

plot(cv_elastic_net, cex.lab = 1.3)

lambda_elastic_net <- cv_elastic_net$lambda.1se

prob_elastic_net <- as.numeric(predict(elastic_net, newx = X_test, type = "response", s = lambda_elastic_net))

resultados$acuracia[resultados$modelo == "elastic-net"] <- mean(y_test == ifelse(prob_elastic_net > corte, "Baixa", "Alta"))

resultados$auc[resultados$modelo == "elastic-net"] <- roc(dados$J1[-idx], prob_elastic_net)$auc

# árvore --------------------------------------------------------------

arvore <- rpart(J1 ~ ., treino)

rpart.plot(arvore, roundint = FALSE)

pred_arvore <- predict(arvore, teste, type = "prob")

resultados$acuracia[resultados$modelo == "Arvore"] <- mean(y_test == ifelse(pred_arvore > corte, "Baixa", "Alta"))

resultados$auc[resultados$modelo == "Arvore"] <- roc(dados$J1[-idx], pred_arvore[,2])$auc

# floresta aleatória ---------------------------------------------------

rf <- randomForest(J1 ~ ., data = treino)

predito_rf <- predict(rf, teste, type = "prob")

resultados$acuracia[resultados$modelo == "Floresta Aleatória"] <- mean(y_test == ifelse(predito_rf > corte, "Baixa", "Alta"))

resultados$auc[resultados$modelo == "Floresta Aleatória"] <- roc(dados$J1[-idx], predito_rf[,2])$auc

tibble(arvore = 1:nrow(rf$err.rate), 
       oob = rf$err.rate[,1]) %>% 
  ggplot(aes(arvore, oob)) + 
  geom_line(color = "#5B5FFF", size = 1.2) + 
  labs(x = "Número de Árvores", y = "Erro de Classificação (OOB)") + 
  theme_bw()


# bagging ---------------------------------------------------

bag <- randomForest(J1~., treino, mtry=ncol(dados)-1)

predito_bag <- predict(bag, teste, type = "prob" )

resultados$acuracia[resultados$modelo == "Bagging"] <- mean(y_test == ifelse(predito_bag > corte, "Baixa", "Alta"))

resultados$auc[resultados$modelo == "Bagging"] <- roc(dados$J1[-idx], predito_bag[,2])$auc

tibble(arvore = 1:nrow(bag$err.rate), 
       oob = bag$err.rate[,1]) %>% 
  ggplot(aes(arvore, oob)) + 
  geom_line(color = "#5B5FFF", size = 1.2) + 
  labs(x = "Número de Árvores", y = "Erro de Classificação (OOB)") + 
  theme_bw()



# Medidas de desempenho para o melhor modelo --------------------------------------

desempenho <- roc(dados$J1[-idx], pred_arvore[,2]) 

coords(desempenho, ret = c("1-accuracy", "sensitivity",               
                               "specificity", "ppv", "npv"))

# melhor modelo preditivo

resposta <- resultados %>%
  filter(mse == min(mse))

erro_medio <- sqrt(resposta$mse)

# comparação de importância de vaiáveis 

g1 <- vip(fit_lm, mapping = aes(fill = Sign)) + 
  labs(subtitle = "LM")

g2 <- vip(ridge, mapping = aes(fill = Sign)) + 
  labs(subtitle = "Ridge")

g3 <- vip(lasso, mapping = aes(fill = Sign)) + 
  labs(subtitle = "LASSO")

g4 <- vip(elastic_net, mapping = aes(fill = Sign)) + 
  labs(subtitle = "Elastic Net")

g5 <- vip(rf) + 
  labs(subtitle = "Randon Florest")

g6 <- vip(bag) + 
  labs(subtitle = "Bagging")


(g1 + g2) + plot_layout(guides = "collect")

(g3 + g4) + plot_layout(guides = "collect") 
  
(g5 + g6) + plot_layout(guides = "collect") 

dados %>% 
  ggplot(aes(Overall_Qual, Sale_Price))+ 
  geom_point(color = "red")

# Modelo final --------------------------------------------------------------

modelo_final <- randomForest(Sale_Price ~ ., data = dados)

