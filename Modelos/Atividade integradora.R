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

freq(df)

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

##dados %>% skim()

##dados <- dados %>% 
  ##na.omit()

plot_num(dados)

dados<-subset(dados, J1<11)

dados_2 <- dados %>% 
  mutate(J1 = factor(ifelse(J1 >= 6, "Alta", "Baixa"), 
                          levels = c("Baixa", "Alta")))


# Treinamento e teste ----------------------------------------------------------

set.seed(123)

idx <- sample(nrow(dados), size = .75*nrow(dados), replace = FALSE)

treino_1 <- dados[idx,]
teste_1 <- dados[-idx,]


y <- dados$J1


# resultados dos modelos para comparação ----------------------------------------------------------------

resultados <- tibble(metodo = c("lm", "ridge", "lasso", "elastic-net", "Àrvore de regressão", "bagging", "Floresta aleatória"), 
              mse = NA)


# Modelo linear ---------------------

fit_lm <- lm(J1 ~ ., data = treino_1)

y_lm <- predict(fit_lm, teste_1)

resultados$mse[resultados$metodo == "lm"] <- mean((y[-idx] - y_lm)^2)


# entradas para glmnet ----------------------------------------------------------------

entrada <- model.matrix(J1 ~ ., data = dados)[,-1]


# ridge ----------------------------------------------------------------

ridge <- glmnet(entrada[idx,], y[idx], alpha = 0)

plot_glmnet(ridge, lwd = 2, cex.lab = 1.3)

cv_ridge <- cv.glmnet(entrada[idx,], y[idx], alpha = 0)

plot(cv_ridge, cex.lab = 1.3)

y_ridge <- predict(ridge, newx = entrada[-idx,], s = cv_ridge$lambda.1se)

resultados$mse[resultados$metodo == "ridge"] <- mean((y[-idx] - y_ridge)^2)

# lasso ----------------------------------------------------------------

lasso <- glmnet(entrada[idx,], y[idx], alpha = 1, nlambda = 1000)

plot_glmnet(lasso, lwd = 2, cex.lab = 1.3, xvar = "lambda")

cv_lasso <- cv.glmnet(entrada[idx,], y[idx], alpha = 1, lambda = lasso$lambda)

plot(cv_lasso, cex.lab = 1.3)

y_lasso <- predict(lasso, newx = entrada[-idx,], s = cv_lasso$lambda.min)

resultados$mse[resultados$metodo == "lasso"] <- mean((y[-idx] - y_lasso)^2)

# elastic net ----------------------------------------------------------------

elastic_net <- glmnet(entrada[idx,], y[idx], alpha = 0.5, nlambda = 1000)

plot_glmnet(elastic_net, lwd = 2, cex.lab = 1.3, xvar = "lambda")

cv_elastic_net <- cv.glmnet(entrada[idx,], y[idx], alpha = 0.5)

plot(cv_elastic_net, cex.lab = 1.3)

y_elastic_net <- predict(elastic_net, newx = entrada[-idx,], s = cv_elastic_net$lambda.min)

resultados$mse[resultados$metodo == "elastic-net"] <- mean((y[-idx] - y_elastic_net)^2)

# árvore --------------------------------------------------------------

arvore <- rpart(J1 ~ ., treino_1)

rpart.plot(arvore, roundint = FALSE)

predito_ar <- predict(arvore, teste_1)

resultados$mse[resultados$metodo == "Àrvore de regressão"] <- mean((y[-idx] - predito_ar)^2)

# floresta aleatória ---------------------------------------------------

rf <- randomForest(J1 ~ ., data = treino_1, ntree=300, do.trace = 5)

predito_rf <- predict(rf, teste_1)

resultados$mse[resultados$metodo == "Floresta aleatória"] <- mean((y[-idx] - predito_rf)^2)

tibble(arvore = 1:length(rf$mse), 
       mse = rf$mse) %>% 
  ggplot(aes(arvore, mse)) + 
  geom_line(color = "red", size = 1.2) + 
  ylab("MSE (OOB)") + 
  xlab("Número de Árvores") +
  ggtitle("Erro quadrático médio em função do número de árvores (Floresta Aleatória)") +
  theme_bw()

# bagging ---------------------------------------------------

bag <- randomForest(Sale_Price~., treino_1, ntree=200,
                    mtry=ncol(dados)-1)

predito_bag <- predict(bag, teste_1)

resultados$mse[resultados$metodo == "bagging"] <- mean((y[-idx] - predito_bag)^2)

tibble(arvore = 1:length(bag$mse), 
       mse = bag$mse) %>% 
  ggplot(aes(arvore, mse)) + 
  geom_line(color = "blue", size = 1.2) + 
  ylab("MSE (OOB)") + 
  xlab("Número de Árvores") + 
  ggtitle("Erro quadrático médio em função do número de árvores (Bagging)") +
  theme_bw()

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

