Faces Verification
==============================

Proyecto para el master de IA Avanzada de la UV 2020-2021. Alumno: Paul Van Branteghem

En este proyecto se busca realizar un modelo de verificación de caras mediante redes siamesas.



Project Organization
------------

    ├── LICENSE
    ├── Makefile           
    ├── README.md          
    ├── data
    │   └── raw            <- Datos de ORL face database en training/test
    │
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── research           <- Jupyter notebooks.
    │   └── notebooks
    │   │      └── 01.pvbl-modeling-ss.ipynb          <- Notebook principal del proyecto donde se desarrollan
    │   │      └── 02.pvbl-testing-camera.ipynb       <- Notebook de testing con la cámara
    │   └── scripts/lib    <- Scripts que contienen todo el código importado en los notebooks
    │          └── data    <- funciones de Data loaders
    │          └── models  <- funciones de los Modelos
    │          └── visualization <- funciones de visualización
    │          └── helpers <- funciones de ayuda
    │
    │
    ├── conda.yaml         <- requerimientos conda.yaml
    │
    ├── setup.py           <- 
    └─ 


--------
# Introducción
**Datos Utilizados**

ORL face database: Es una base de datos compuesta de caras de 40 personas con múltiples imágenes de cada una de ellas. Se han introducido también imágenes propias en la parte de testing.


**Principales Herramientas**
- Pytorch
- Pytorch Lighting
- Ray
- MLFlow

**Metodología**

*Training*

* Notebook:01.pvbl-modeling-ss.ipynb

1. Se cargan los datos de ORL face con un DataLoader
2. Se utilizan redes Neuronales Convolucionales para construir la red Siamesa. 
3. Se hace un testing basado en las métricas: loss, pairwise_distance entre una determinada imagen con respecto a las demás del test (dimisiliaridad). Esta última permite hacer dos cosas: un histograma y por otro lado, a partir de un treshold (en este caso 0.7), definir si la cara de la imagen de referencia es la misma que la de la cara a testear. Definiendo este 0-1 podemos definir una especie de accuracy.
4. Probamos una red más compleja propia para evaluar si mejoran los resultados SiameseNetworkV2.
5. Hacemos un hyperparameter tunning con la primera red (debido a que es computacionalmente menos pesada) para evaluar los mejores hiperparámetros del modelo. Debido a que tenemos muy pocos datos, evaluamos sobre el training de 4 caras los hiperparámetros (practica no recomendada pero que tiene como finalidad usar Ray para evaluar hiperparámetros).
6. Del mejor modelo, levantamos una API con mlflow y probamos a hacer peticiones con otras imágenes.

*Evaluación en producción*

* Notebook:02.pvbl-testing-camera.ipynb
1. Hay dos funciones principales:
    * take_photos: Saca dos fotos y evalua la similitud de las mismas
    * take_videos: Respecto a una foto base, saca un video y a tiempo real evalua la similitud del frame respecto a la persona de la foto base. Por lo tanto, es una verificación a tiempo real de dos personas.


**Bibiliografía**
- https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch/

Quickstart
----------
- Descargar el repo
- Instalar el environment de conda: conda env create -f conda.yaml
- ejecutar source .env
- Si se quiere reproducir de nuevo el notebook, ir a research/notebooks/01.pvbl-modeling-ss.ipynb y ejecutarlo
- Si se quiere devolver una API con uno de los modelos generados: mlflow run <path_id>
