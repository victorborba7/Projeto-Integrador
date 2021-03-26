library(rpart.plot)
library(randomForest)
library(modeldata)
library(pROC)
library(ranger)
library(leaflet)
library(rsample)
library(tidyverse)
library(skimr)
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

dados_2 <- read_excel("C:\\Users\\dougl\\Documents\\PADS\\Atividade integradora\\output.xlsx")


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
                          levels = c("Alta", "Baixa")))

plot_num(dados)


# Análise inicial de balanceamento ----------------------------------------

table(dados$J1)/nrow(dados)

# Trasnformar algumas variáveis em factor

dados <- dados %>% 
  mutate_at(vars (Q5, Q9, D1, Q9, A1_1, A1_2, A1_3, A1_4, F1, F3), ~as.factor(.))

str(dados)


# Definição de um corte inicial -------------------------------------------

corte <- 0.5

# Treinamento e teste ----------------------------------------------------------

set.seed(123)

idx <- sample(nrow(dados), size = .75*nrow(dados), replace = FALSE)

treino <- dados[idx,]
teste <- dados[-idx,]


# comparação dos modelos --------------------------------------------------

resultados <- tibble(modelo = c("R. logistica", "ridge", "lasso", "elastic-net", "Arvore", "Floresta Aleatória", "Bagging"),
                    Acuracia = NA,
                    auc = NA)

# Modelo de regressão logística ---------------------

fit_tr <- glm(J1 ~ ., data = treino, family = "binomial")

prob_logistica <- predict(fit_tr, teste, type = "response")

resultados$Acuracia[resultados$modelo == "R. logistica"] <-mean(dados$J1[-idx] == ifelse(prob_logistica >= corte, "Alta", "Baixa"))

resultados$auc[resultados$modelo == "R. logistica"] <- roc(dados$J1[-idx], prob_logistica)$auc

## Plotagem da curva ROC

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

resultados$Acuracia[resultados$modelo == "ridge"] <- mean(y_test == ifelse(prob_ridge >= corte, "Alta", "Baixa"))

resultados$auc[resultados$modelo == "ridge"] <- roc(dados$J1[-idx], prob_ridge)$auc


# lasso ----------------------------------------------------------------

lasso <- glmnet(X_treino, y_treino, alpha = 1, family = "binomial", nlambda = 1000)

cv_lasso <- cv.glmnet(X_treino, y_treino, alpha = 1, family = "binomial")

plot(cv_lasso, cex.lab = 1.3)

lambda_lasso <- cv_lasso$lambda.1se

prob_lasso <- as.numeric(predict(lasso, newx = X_test, type = "response", s = lambda_lasso))

resultados$Acuracia[resultados$modelo == "lasso"] <- mean(y_test == ifelse(prob_lasso > corte, "Alta", "Baixa"))

resultados$auc[resultados$modelo == "lasso"] <- roc(dados$J1[-idx], prob_lasso)$auc


# elastic net ----------------------------------------------------------------

elastic_net <- glmnet(X_treino, y_treino, alpha = 0.5, family = "binomial", nlambda = 1000)

cv_elastic_net <- cv.glmnet(X_treino, y_treino, alpha = 0.5, family = "binomial")

plot(cv_elastic_net, cex.lab = 1.3)

lambda_elastic_net <- cv_elastic_net$lambda.1se

prob_elastic_net <- as.numeric(predict(elastic_net, newx = X_test, type = "response", s = lambda_elastic_net))

resultados$Acuracia[resultados$modelo == "elastic-net"] <- mean(y_test == ifelse(prob_elastic_net > corte, "Alta", "Baixa"))

resultados$auc[resultados$modelo == "elastic-net"] <- roc(dados$J1[-idx], prob_elastic_net)$auc

# árvore --------------------------------------------------------------

arvore <- rpart(J1 ~ ., treino)

rpart.plot(arvore, roundint = FALSE)

pred_arvore <- predict(arvore, teste, type = "prob")

resultados$Acuracia[resultados$modelo == "Arvore"] <- mean(y_test == ifelse(pred_arvore[,2] > corte, "Alta", "Baixa"))

resultados$auc[resultados$modelo == "Arvore"] <- roc(dados$J1[-idx], pred_arvore[,2])$auc


# floresta aleatória ---------------------------------------------------

rf <- randomForest(J1 ~ ., data = treino)

predito_rf <- predict(rf, teste, type = "prob")

resultados$Acuracia[resultados$modelo == "Floresta Aleatória"] <- mean(y_test == ifelse(predito_rf[,2] > corte, "Alta", "Baixa"))

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

resultados$Acuracia[resultados$modelo == "Bagging"] <- mean(y_test == ifelse(predito_bag[,2] > corte, "Alta", "Baixa"))

resultados$auc[resultados$modelo == "Bagging"] <- roc(dados$J1[-idx], predito_bag[,2])$auc

tibble(arvore = 1:nrow(bag$err.rate), 
       oob = bag$err.rate[,1]) %>% 
  ggplot(aes(arvore, oob)) + 
  geom_line(color = "#5B5FFF", size = 1.2) + 
  labs(x = "Número de Árvores", y = "Erro de Classificação (OOB)") + 
  theme_bw()


# Comparação variando o número de preditoras sorteadas a cada nó ----------

rf_5 <- randomForest(J1~ .,mtry =  5, data = treino, ntree = 300)
rf_10 <- randomForest(J1~ .,mtry =  10, data = treino, ntree = 300)
rf_15 <- randomForest(J1~ ., mtry = 15, data = treino, ntree = 300)
rf_20 <- randomForest(J1~ ., mtry = 20, data = treino, ntree = 300)
rf_25 <- randomForest(J1~ ., mtry = 25, data = treino, ntree = 300)

resultados_mtry <- tibble(mtry = 5, arvore = 1:nrow(rf_5$err.rate), 
                     oob = rf_5$err.rate[,1]) %>% 
  bind_rows(tibble(mtry = 10, arvore = 1:nrow(rf_10$err.rate), 
                   oob = rf_10$err.rate[,1])) %>%
  bind_rows(tibble(mtry = 15, arvore = 1:nrow(rf_15$err.rate), 
                   oob = rf_15$err.rate[,1])) %>% 
  bind_rows(tibble(mtry = 20, arvore = 1:nrow(rf_20$err.rate), 
                   oob = rf_20$err.rate[,1])) %>% 
  bind_rows(tibble(mtry = 25, arvore = 1:nrow(rf_25$err.rate), 
                   oob = rf_25$err.rate[,1]))

resultados_mtry %>% 
  mutate(mtry = factor(mtry)) %>% 
  ggplot(aes(arvore, oob, group = mtry, color = mtry)) + 
  geom_line( size = 1.2) + 
  labs(x = "Número de Árvores", y = "Erro de Classificação (OOB)", title = "Variação do erro de acordo com diferentes 'mtry'") + 
  theme_bw()


# Variable importance -----------------------------------------------------

g1 <- vip(rf) + 
  labs(subtitle = "Randon Florest")

g1 

dados %>% 
  sample_n(500) %>% 
  filter(B1_2 < 11) %>% 
  filter(OPERADORA == 'VIVO') %>% 
  ggplot(aes(B1_2))+ 
  geom_histogram(color = "black")+
  labs(x = "Nota B1_2", y = "Quantidade de respostas", title = "Amostra de resposta B1_2 para Vivo ao longo dos anos") + 
  theme_bw()

# Medidas de desempenho para o melhor modelo --------------------------------------

desempenho <- roc(dados$J1[-idx], predito_rf[,2]) 

tabela_desempenho <- coords(desempenho, ret = c("threshold","accuracy","1-accuracy", "sensitivity",               
                               "specificity", "ppv", "npv"))

resposta <- tabela_desempenho %>%
  filter(accuracy == max (accuracy))


# Modelo final considerando todos os dados --------------------------------

rf_final <- randomForest(J1 ~ ., data = dados)







