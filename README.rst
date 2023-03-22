|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/intsystems/ProjectTemplate/tree/master
    :alt: Test status
    
.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master
    :target: https://app.codecov.io/gh/intsystems/ProjectTemplate
    :alt: Test coverage
    
.. |docs| image:: https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intsystems.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Метрический анализ пространства параметров глубоких нейросетей
    :Тип научной работы: M1P
    :Автор: Эрнест Рашидович Насыров
    :Научный руководитель: доктор ф.-м. наук, Стрижов Вадим Викторович
    :Научный консультант**???**: степень, Фамилия Имя Отчество


**Читать** `статью <http://www.machinelearning.ru/wiki/index.php?title=%D0%9D%D0%B0%D1%83%D1%87%D0%BD%D0%BE-%D0%B8%D1%81%D1%81%D0%BB%D0%B5%D0%B4%D0%BE%D0%B2%D0%B0%D1%82%D0%B5%D0%BB%D1%8C%D1%81%D0%BA%D0%B0%D1%8F_%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0_%28%D1%80%D0%B5%D0%BA%D0%BE%D0%BC%D0%B5%D0%BD%D0%B4%D0%B0%D1%86%D0%B8%D0%B8%29>`__
про то, как заниматься наукой

**Диипломные** `работы <http://www.machinelearning.ru/wiki/index.php?title=%D0%A3%D1%87%D0%B0%D1%81%D1%82%D0%BD%D0%B8%D0%BA:Strijov>`__ кафедры.

Abstract
========

Исследуется проблема снижения размерности пространства параметров модели машинного обучения. Решается задача восстановления временного ряда. Для восстановления используются авторегресионные модели: линейные, автоенкодеры, реккурентные сети - с непрерывным и дискретным временем. Проводится метрический анализ пространства параметров модели.  Предполагается, что отдельные параметры модели, случайные величины, собираются в векторы, многомерные случайные величины, анализ взаимного расположения которых в пространстве и представляет предмет исследования нашей работы.  Этот анализ снижает число параметров модели, оценивает значимости параметров, отбирая их. Для определения положения вектора параметров в пространстве оцениваются его матожидание и матрица ковариации с помощью методов \textit{бутстрэпа} и \textit{вариационного вывода}. Эксперименты проводятся на задачах восстановления синтетических временных рядов, квазипериодических показаний акселерометра, периодических видеоданных. Для восстановления применяются модели SSA, нелинейного PCA, RNN, Neural ODE.


`LinkReview <https://docs.google.com/document/d/197ZZ3pAftQzLtEjYcW8KKgALDledXuotjdYXJnXwgH0/edit?usp=sharing>`__,
`Paper <https://github.com/intsystems/2023-Project-141/blob/master/paper/Nasyrov2023_metric_analysis.pdf>`__,
`Slides <https://github.com/intsystems/2023-Project-141/blob/master/slides/Nasyrov2023Presentation.pdf>`__.

Software modules developed as part of the study
======================================================
1. Код экспериментов по восстановлению временного ряда с SSA `здесь <https://github.com/intsystems/2023-Project-141/blob/master/code/ssa_experiment.ipynb>`_.
	На текущий момент использовались данные синтетического $\sin$, `colab <http://colab.research.google.com/github/intsystems/2023-Project-141/blob/master/code/ssa_experiment.ipynb>`_-версия.
2. Код экспериментов по восстановлению временного ряда с 2NN `здесь <https://github.com/intsystems/2023-Project-141/blob/master/code/2nn_experiment.ipynb>`_. 
	На текущий момент использовались данные синтетического $\sin$, `colab <http://colab.research.google.com/github/intsystems/2023-Project-141/blob/master/code/2nn_experiment.ipynb>`_-версия.
	
Presentations at conferences on the topic of research
================================================
1. Планируется участие в конференции МФТИ. Дедлайн - 15 марта.




Research publications
===============================
1. 



Problem statment
======================================================
Problem 141
* __Title__: Metric analysis of deep network space parameters
* __Problem__: The structure of a neural work is exhaustive. The dimensionality of the parameter space should be reduced. The autoencoder in the subject of the investigation. Due to the continuous-time nature of the data, we analyze several types of autoencoders. We reconstruct spatial-time data, minimizing the error. 
* __Data__: 
	* Synthetic data sine for 2D visualizaion of the parameter distributions
	* Accelerometer quasiperiodic data
	* Limb movement quasiperiodic data (if any)
	* Video periodic data (cartoon, walking persona)
	* Video, fMRI, ECoG from the s41597-022-01173-0 
* __References__: 
	* [SSA and Hankel matrix construction](http://strijov.com/papers/Grabovoy2019QuasiPeriodicTimeSeries.pdf) or in [wiki](https://en.wikipedia.org/wiki/Singular_spectrum_analysis)
	* [Open multimodal iEEG-fMRI dataset from naturalistic stimulation](https://www.nature.com/articles/s41597-022-01173-0)
	* [Variational autoencoders to estimate parameters](https://arxiv.org/pdf/1606.05908.pdf)
	* RNN in the [5G book](https://arxiv.org/abs/2104.13478)
	* [Neural CDE](https://bit.ly/NeuroCDE)
* __Baseline__: RNN-like variational autoencoder in the criteria: error vs. complexity (number of parameters)
* __Roadmap__:
	* Prepare data so that the reconstruction work on a basic model (like SSA)
	* Estimate expectation and covariance of parameters (using VAE or else, to be discussed)
	* Reduce dimensionality, plot the error/complexity, plot the covariance
	* Run RNN-like model, plot
	* Assign the expectation and covariation matrix to each neuron of the model
	* Plot the parameter space regarding covariance as its metric tensor (end of minimum part)
	* Suggest a dimensionality reduction algorithm (naive part)
	* Run Neuro ODE/CDE model and plot the parameter space
	* Analyse the data distribution as the normalized flow 
	* Suggest the parameter space modification in terms of  the normalized flow (paradoxical part, diffusion model is needed)
	* Compare all models according to the criterion error/complexity (max part)
	* Construct the decoder model for any pair of data like fMRI-ECoG tensor and neuro CDE (supermax part)
* __Proposed solution__: description of the idea to implement in the project
* __Novelty__: Continous-time models are supposed to be simple due to their periodic nature. Since they approximate the vector fields, these models are universal. The model selection for the continuous time is not considered now, but at the time, it is acute for wearable multimedia devices for metaverse and augmented reality. 
* __Supergoal__ To join two encoders in a signal decoding model to reveal the connection between video and fMRI, between fMRI and ECoG.
* __Authors__: Expert Strijov, consultant ?

