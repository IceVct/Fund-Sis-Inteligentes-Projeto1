%% Projeto 1 de FCS 1/2018
% Alvaro Torres Vieira - 14/0079661
% Victor Araujo Vieira - 14/0032801

%% Carregando o dataset fisher iris e inicializando as variaveis
load fisheriris;
medidas = meas;
respostas = species;

proporcaoTreino = 0.7; % proporcao da amostra que sera usada para treino
numAmostras = size(species, 1);
numCadaClasse = numAmostras/3;

% Numero de amostras pra treino
nTreino = numAmostras*proporcaoTreino;
nTeste = numAmostras - nTreino;

% Criacao das variaveis que serao usadas para gravar os dados de treino e
% teste
medidasTreino = zeros(nTreino, size(medidas, 2));
medidasTeste = zeros(nTeste, size(medidas, 2));

% Gerando 150 numeros diferentes e aleatorios de 1 a 150 que serao usados
% tanto para o treino quanto para o teste
numerosAleatorios = randperm(150);

% Pega os 105 primeiros numeros aleatorios para usar para treinar os
% modelos
medidasTreino(1:nTreino, :) = medidas(numerosAleatorios(1:nTreino), :);
% Pega os outros 45 numeros aleatorios restantes para serem usados para
% testar os modelos treinados
medidasTeste(1:nTeste, :) = medidas(numerosAleatorios(nTreino + 1 : numAmostras), :);

% Gerando as variaveis que serao usadas para treino e teste dos modelos, e
% contem as respostas de cada uma das etapas
respostasTreino(1:nTreino, :) = respostas(numerosAleatorios(1:nTreino), :);
respostasTeste(1:nTeste, :) = respostas(numerosAleatorios(nTreino + 1 : numAmostras), :);

%% Criando os modelos para o treinamento usando o metodo Knn
% O standardize eh para padronizar as escalas, de forma a nao atrapalhar no
% treinamento e consequentemente na predicao

% Treinamento com K = 1
modeloK1 = fitcknn(medidasTreino, respostasTreino, 'NumNeighbors', 1, 'Standardize',1);
% Predicao com K = 1
respostaK1 = predict(modeloK1, medidasTeste);

% Treinamento com K = 5
modeloK5 = fitcknn(medidasTreino, respostasTreino, 'NumNeighbors', 5, 'Standardize',1);
% Predicao com K = 5
respostaK5 = predict(modeloK5, medidasTeste);

% Treinamento com K = 10
modeloK10 = fitcknn(medidasTreino, respostasTreino, 'NumNeighbors', 10, 'Standardize',1);
% Predicao com K = 10
respostaK10 = predict(modeloK10, medidasTeste);

% Treinamento com K = 30
modeloK30 = fitcknn(medidasTreino, respostasTreino, 'NumNeighbors', 30, 'Standardize',1);
% Predicao com K = 30
respostaK30 = predict(modeloK30, medidasTeste);

% Treinamento com K = 50
modeloK50 = fitcknn(medidasTreino, respostasTreino, 'NumNeighbors', 50, 'Standardize',1);
% Predicao com K = 50
respostaK50 = predict(modeloK50, medidasTeste);

%% Gerando e imprimindo as matrizes de confusao para cada K usado para treinar o Knn

%K = 1
geraMatConf(respostasTeste, respostaK1, medidasTeste);

%K = 5
geraMatConf(respostasTeste, respostaK5, medidasTeste);

%K = 10
geraMatConf(respostasTeste, respostaK10, medidasTeste);

%K = 30
geraMatConf(respostasTeste, respostaK30, medidasTeste);

%K = 50
geraMatConf(respostasTeste, respostaK50, medidasTeste);

