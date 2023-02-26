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

Abstract
========

Исследуется проблема уменьшения количества параметров нейросети. Для этого в работе проводится метрический анализ пространства параметров нейросети. Авторы исследуют пространство параметров автоенкодера, как средства преобразования данных, непрерывных во времени. Более конкретно, оценивается матожидание и ковариация параметров модели, с использованием методов bootstrap, variational inference и др., проведится графический анализ распределения параметров, изучается их структура сообществ. Анализ проводится для автоенкодеров типа RNN, CNN, SSA на задачах предсказания синтетических временных рядов, квазипериодических показаний акселерометра, периодических видеоданных.

Алгоритмы оценки параметров модели
==================================
1. Bootstrap 
	~ то же, что и crossvalidation, но только мы семплируем из из нашей выборки.
2. Cross-validation
	Мы бьем выборку на фолды, обучаем модель на всех фолдах без одного (мб даже leave-one-out техника?). Т.е. получаем для каждого разбиения свой оптимальный набор параметров. Далее уже вычисляем E, cov.

3. Variational Inference
	Что-то сложное. Не знаю, как работает. Почитать!!!
	
	**Найти** библиотеку, делающую эту работу за нас.


Beginner's talk
===============
**Цель работы:** уменьшить количество параметров нейросети. Мотивация: непрерывные данные часто являются сильно скоррелированными (так как сигнал зависит от сигнала в предыдущие моменты времени). Поэтому и модели, применяемые для предсказания сигнала могут переобучаться, имея скрытое пространство большой размерности.

**Основная идея:** исследовать матрицу корреляции параметров сети.
Традиционный способ сделать это - смотреть на корелляции между каждой парой параметров (слоя сети), что не учитывает внутреннюю простую структуру сети: композиция линейных и простых нелинейных операций.

Мы же применим другой подход, который заключается в исследовании пространства, сопряженного к пространству признаков. Для примера: в $y=\sigma(Wx)$ строки W - линейные функционалы - формируют сопряженное пространство. Мы займемся строками W: каждая строка является случайной величиной (мы предпологаем ~ нормальной из-за большого количества данных по ЦПТ). Каждая такая строка имеет матожидание и матрицу ковариации. Строки и являются объектом исследования.

**Ожидаемый результат:** Ожидаем увилеть простую структуру пространства, наличие векторов, имеющих схожие функции (одинаковые cov матрицы, матожидание), наличие структуры сообществ векторов.


Research publications
===============================
1. 

Presentations at conferences on the topic of research
================================================
1. 

Software modules developed as part of the study
======================================================
1. A python package *mylib* with all implementation `here <https://github.com/intsystems/ProjectTemplate/tree/master/src>`_.
2. A code with all experiment visualisation `here <https://github.comintsystems/ProjectTemplate/blob/master/code/main.ipynb>`_. Can use `colab <http://colab.research.google.com/github/intsystems/ProjectTemplate/blob/master/code/main.ipynb>`_.



Problem statment
======================================================
## Problem 141
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
