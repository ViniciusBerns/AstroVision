# AstroVision — Classificação Inteligente de Eventos Climáticos

# Grupo 
 
- Vinícius Almeida Bernardino de Souza - RM97888 
- João Pedro Borsato Cruz - RM550294 
- Maria Fernanda Vieira de Camargo - RM97956 
- Pedro Lucas de Andrade Nunes - RM550366 
- Fernanda Kaory Saito - RM551104 
 
## Sobre o Projeto 
 
O **AstroVision** é uma solução baseada em Inteligência Artificial e Visão Computacional desenvolvida para auxiliar na identificação automática de eventos climáticos extremos a partir de imagens. 
 
O projeto utiliza uma **Rede Neural Convolucional (CNN)** treinada do zero para classificar imagens em duas categorias: 
 
- Flood (Enchentes) 
- Wildfire (Incêndios Florestais) 
 
A proposta está alinhada ao **ODS 13 — Ação Contra a Mudança Global do Clima**. 
 
--- 
 
# Problema 
 
Eventos climáticos extremos vêm aumentando em frequência e intensidade nos últimos anos. 
 
O monitoramento manual de grandes volumes de imagens provenientes de satélites, drones e sistemas de observação ambiental pode ser lento e suscetível a erros. 
 
--- 
 
# Solução Proposta 
 
O AstroVision utiliza uma arquitetura de Deep Learning baseada em **Convolutional Neural Networks (CNNs)** para realizar a classificação automática de imagens ambientais. 
 
--- 
 
# Tecnologias Utilizadas 
 
- Python 3.13 
- TensorFlow / Keras 
- OpenCV 
- NumPy 
- Matplotlib 
- Seaborn 
- Scikit-Learn 
- Git & GitHub 
 
--- 
 
# Dataset 
 
| Classe | Quantidade | 
|----------|----------| 
| Flood | 51 imagens | 
| Wildfire | 50 imagens | 
| Total | 101 imagens | 
 
--- 
 
# Tratamento e Preparação dos Dados 
 
1. Leitura das imagens com OpenCV 
2. Conversão BGR para RGB 
3. Redimensionamento para 224x224 pixels 
4. Normalização dos pixels (0 a 1) 
5. Divisão em Treino (70%), Validação (15%) e Teste (15%) 
 
--- 
 
# Arquitetura da Rede Neural 
 
- Input: 224x224x3 
- Data Augmentation 
- Conv2D (32 filtros) + MaxPooling 
- Conv2D (64 filtros) + MaxPooling 
- Conv2D (128 filtros) + MaxPooling 
- Flatten 
- Dense (128) 
- Dropout (0.5) 
- Dense (1) com Sigmoid 
 
--- 
 
# Estratégias de Treinamento 
 
- Adam Optimizer 
- Binary Crossentropy 
- Data Augmentation 
- Early Stopping (patience=5) 
 
--- 
 
# Resultados Obtidos 
 
**Accuracy Final:** 81.25% 
 
## Classification Report 
 
| Classe | Precision | Recall | F1-Score | 
|----------|----------|----------|----------| 
| Flood | 0.73 | 1.00 | 0.84 | 
| Wildfire | 1.00 | 0.62 | 0.77 | 
 
--- 
 
# Avaliação Visual 
 
Arquivos gerados: 
 
- img/confusion_matrix.png 
- img/accuracy.png 
- img/loss.png 
 
--- 
 
# ODS Relacionado 
 
## ODS 13 — Ação Contra a Mudança Global do Clima 
 

