# Clasificador de gestos motrices utilizando vectores de atributos distancia DTW sobre series de tiempo en representación SAX

Este trabajo presenta una revisión del análisis de series de tiempo aplicado a la clasificación y reconocimiento de patrones. El artículo indaga en 4 métodos y compara su desempeño para la tarea específica del reconocimiento de gestos motrices, tarea que trabaja con series de tiempo multivariables derivadas de la medición en la aceleración en los 3 ejes espaciales sobre un dispositivo sensor acelerómetro. Como aportación se plantea un nuevo método capaz de superar al estado del arte y la técnica de referencia con un 94.06\% de precisión en el conjunto de datos \textit{GesturePeeble} disponible en \textit{The UCR Time Series Archive}; El método propuesto implementa el algoritmo \textit{Dynamic Time Warping} para la conformación de vectores con atributos de distancia entre series de tiempo respresentadas con \textit{Symbolic Aggregate Approximation}, posibilitando para las fases posteriores de entrenamiento y predicción su ingreso en el modelo elegido. El método permite la elección del clasificador a conveniencia de la aplicación, más los resultados que se reportan se obtuvieron optando por un modelo clásico de Máquinas de Soporte Vectorial.

## Dependencias

El lenguaje de programación utilizado para la implementación es Python en su versión 3.10.
Las bibliotecas requeridas para la ejecución de los modelos o archivos de prueba son:

* Numpy

* Matplotlib

* Pandas

* scikit-learn

* resampy
  
  > Se ocupa *Pipenv* para la virtualización del ambiente por lo que en la sección de Código se pueden instalar las dependencias con:
  
  ```bash
  pipenv install
  ```



## Estructura del repositorio

- **Artículo**
  
  Este apartado almacena el artículo del proyecto y los archivos requeridos para la compilación del documento en LaTex

- **Código**
  
  Archivos fuente y carpetas relacionadas con la ejecución de las funciones y métodos reportados en el artículo
  
  - **Data**
    
    Archivos con los datos utilizados durante el desarrollo y pruebas de las técnicas y modelos
  
  - **Libraries**
    
    Archivos fuentes que implementa los algoritmo y técnicas usadas en los modelos(DTW,SAX,Optimización del parámetro *w*,...)
    
    > > *Particularmente esta biblioteca es funcional cuando se manejan arreglos numpy(secuencias temporales de longitud fija)*
  
  - **LLibraries**
    
    Archivos fuentes que implementa los algoritmo y técnicas usadas en los modelos(DTW,SAX,Optimización del parámetro *w*,...)
    
    > > *Particularmente esta biblioteca es funcional cuando se manejan listas de arreglos numpy(secuencias temporales de longitud variable)*
  
  - **Results**
    
    Hojas de cálculo utilizadas para realizar el registro de los resultados de cada modelo
    
    - Models results
  
  - **Tests**
    
    Archivo fuentes utilizados durante el desarrollo como sencillos programas que prueban la funcionalidad de las funciones
