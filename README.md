  # Projeto de Reconhecimento de Letras em LIBRAS — Versão 0

  Este projeto é a **primeira versão funcional (v0)** de um sistema de reconhecimento de **letras do alfabeto em LIBRAS**, utilizando **MediaPipe**, **TensorFlow/Keras** e **Python 3.9**.  
  O objetivo desta versão é construir um pipeline básico que:

  - captura landmarks das mãos em tempo real  
  - organiza um dataset próprio (A–J)  
  - treina um modelo simples de classificação  
  - realiza predições em tempo real pela webcam  

  Essa é uma **versão de estudo**, servindo como base para evoluções futuras — como reconhecer movimentos (H, J), melhorar arquitetura do modelo, aumentar o dataset e avançar para frases ou palavras.

  ---

  ## Requisitos

  - Python **3.9.x**  
  - Ambiente virtual configurado  
  - Webcam funcionando  
  - Dependências instaladas:

```

pip install -r requirements.txt

```

Bibliotecas principais:

- `mediapipe==0.10.14`
- `tensorflow==2.12.0`
- `numpy`
- `opencv-python`

> ⚠️ Importante: versões mais novas do MediaPipe e Protobuf geram erros.  
> Por isso usamos versões compatíveis.

---

## 1. Construção do Dataset

O dataset é criado coletando 63 pontos (21 landmarks × xyz) da mão em poses estáticas para cada letra.

Para coletar:

```

python src/build_dataset.py

```

Este script:

- ativa a webcam  
- captura landmarks da mão  
- salva cada captura como `.npy` dentro da pasta da letra  
- no final, gera:

```

data/dataset_final.npz

```

com:

- `X`: matriz de features  
- `y`: labels numéricos  
- `labels`: nomes das classes  

---

## 2. Treinando o Modelo

Depois do dataset estar pronto, rode:

```

python src/train_model.py

```

O modelo gera:

```

models/model_libras.h5

```

Durante o treino, são exibidos:

- número de amostras  
- precisão  
- perda  
- métricas por época  

---

## 3. Testando em Tempo Real

Para executar o reconhecimento via webcam:

```

python src/test_model.py

```

A janela exibe:

- a mão com landmarks desenhados  
- a letra prevista (`Pred: A`, por exemplo)

Pressione **Q** para sair.

---

## 4. Avaliação do Modelo (Métricas)

Após treinar o modelo, você pode avaliar a performance real dele com:

```

python src/evaluate_model.py

```

Esse script calcula:

- **Acurácia geral**
- **Precision, Recall e F1-score por classe**
- **Matriz de confusão (visual)**

Os resultados são salvos automaticamente em:

```

metrics/

```

Arquivos gerados:

- `confusion_matrix.png` — imagem da matriz de confusão  
- `report.txt` — relatório completo com todas as métricas  

Essa avaliação permite identificar:

- classes que o modelo acerta mais  
- letras que estão sendo confundidas  
- onde é necessário aumentar o dataset  
- se o modelo sofre com sobreajuste  

---

## Como o Modelo Funciona

✔ Modelo denso (MLP)  
✔ Entrada: 63 valores (landmarks da mão normalizados)  
✔ Saída: uma das letras A–J  
✔ Treinado com dados do próprio usuário  
✔ Mede apenas **poses estáticas** (por isso H e J não são 100% precisas ainda)

---

## Próximas Melhorias (Versão 1)

- Captura de **sequências de movimento** para letras dinâmicas  
- Implementar **LSTM/GRU** para gestos contínuos  
- Dataset maior e com mais variações  
- Normalização por mão esquerda/direita  
- Melhorar estabilidade das predições  
- Interface mais avançada (Painel, Dashboard, WebApp)  
- Suporte a **mais letras e palavras**

---

## Objetivo Acadêmico

Este projeto representa um **estudo prático de visão computacional aplicada**, unindo:

- machine learning  
- deep learning  
- pré-processamento de landmarks  
- coleta e organização de dataset próprio  
- integração com webcam  
- pipeline completo de um mini-sistema real  

Além de servir como base para experimentos com modelos mais complexos.

---

## Autoria

Projeto criado como parte de um estudo pessoal sobre **aprendizado de máquina**, **visão computacional** e comunicação acessível usando LIBRAS.

---

## ⭐ Se este projeto te ajudou, considere marcá-lo com uma estrela!
```

