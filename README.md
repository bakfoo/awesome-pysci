# Awesome Python Science Stack [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

Pythonの科学スタックをまとめました．開発コードの更新が二年以上ない重要でないライブラリ・パッケージは基本的に排除するようにしています．

Inspired by [awesome-python](https://github.com/vinta/awesome-python).

- [Awesome Python Science Stack](#awesome-python)
    - [基本](#基本)
    - [データ操作，データ構造保持](#データ操作，データ構造保持)
    - [分散データ処理](#分散データ処理)
    - [統計解析・統計モデル](#統計解析・統計モデル)
    - [機械学習](#機械学習)
    - [深層学習](#深層学習)
    - [謎系脳型計算](#謎系脳型計算)
    - [ベイズ推定，確率論的プログラミング](#ベイズ推定，確率論的プログラミング)
    - [時系列解析](#時系列解析)
    	- [カルマンフィルタ](#カルマンフィルタ)
    - [ネットワーク解析](#ネットワーク解析)
    - [最適化ライブラリ](#最適化ライブラリ)
		- [数理最適化モデラー](#数理最適化モデラー)
		- [方程式ソルバー](#方程式ソルバー)
		- [グラフ理論](#グラフ理論)
    	- [Matrix Factorization](#Matrix Factorization)
    	- [Factorization Machine](#Factorization Machine)
    	- [ベイズ最適化](#ベイズ最適化)
    - [自然言語解析](#自然言語解析)
    	- [日本語処理](#日本語処理)
    - [専門分野](#専門分野)
    	- [物理学](#物理学)
    	- [気象学・気候学](#気象学・気候学)
    	- [心理学実験](#心理学実験)
    	- [分子生物学・バイオインフォマテックス](#分子生物学・バイオインフォマテックス)
    	- [神経科学・脳科学](#神経科学・脳科学)
    	- [化学・分子動力学・ケモインフォマテックス](#化学・分子動力学・ケモインフォマテックス)
    	- [疫学](#疫学)
    	- [金融工学・BI](#金融工学・BI)
    	- [信号処理](#信号処理)
    	- [制御工学](#制御工学)
    - [数学](#数学)
    - [地理空間情報](#地理空間情報)
    - [コンピュータビジョン](#コンピュータビジョン)
    - [音声認識](#音声認識)
    - [可視化](#可視化)
    - [データセット](#データセット)
    - [リプロダクション](#リプロダクション)
    - [高速化](#高速化)
    - [外部インタフェース](#外部インタフェース)
- [Resources](#resources)
    - [Websites](#websites)
    - [Weekly](#weekly)
    - [Twitter](#twitter)
    - [Podcasts](#podcasts)
- [Other Awesome Lists](#other-awesome-lists)
- [Contributing](#contributing) 

- - -
# Awesome Python Science Stack


## 基本

* [NumPy](http://www.numpy.org/) - Python科学スタックの礎．多次元配列オブジェクトndarrayによる行列・ベクトル演算を基礎に，データの配置と演算の基盤となっている．
* [SciPy](http://www.scipy.org/) - NumPyの上に構築された科学計算系の数値計算ライブラリのコレクション．方程式ソルバー，最適化ライブラリ，統計モデル，積分，補完，疎行列，フーリエ変換，信号処理などが頻繁に利用される．非常に重要な役割を担っているのに，頻繁にオンラインリファレンスにアクセスできない事態になる．リファレンスを手元にダウンロードして置くのは必至．
* [Jupyter Notebook (IPython)](https://jupyter.org) - Mathematica, Mapleに影響を受けたWebブラウザをインタフェースとする対話型実行環境．コード，文章，数式，グラフ，画像，ビデオ，アニメーション，音声などを一つのドキュメントとして統合できるために，教育目的にでも科学成果のリブロダクションの目的にも最適なツールとなっている．今ではJupyter NotebookがあるからこそPythonを利用する大きな理由になっているといっても良い．


## データ操作，データ構造保持

* [Pandas](http://pandas.pydata.org/) - Rのデータフレームに影響を受けたデータ構造保持・データ操作のライブラリ．主要な演算がCythonで実装されているので高速な動作と簡単な操作を実現している．データ可視化，データ解析の基本ツールである．
* [PyTables](http://www.pytables.org/) - HDF5を扱うためのライブラリ．最近では大きなデータはHDF5で配布されていることが多く，そのデータを取り扱うための入力・出力操作を担う．

## 分散データ処理

* [Blaze](http://blaze.pydata.org/) - Numpy本家のContinuum Analytics社謹製の一連の分散ライブラリコレクション．Blaze, Dask, Datashape, DyND, Odoというパッケージからなっている．そのなかの[Blaze](http://blaze.readthedocs.io/en/latest/index.html)はメモリに収まらない大きなNumpy配列を扱うことができる．
* [Dask](http://dask.pydata.org/en/latest/) - Numpy, pandasと同じ操作を分散・Out-of-coreにできるようにしたデータ操作・保持ライブラリ．単一マシンでも複数マシンでも処理を分散できる．マシン分散にはsshやhttpを使うdask.distributedを利用し，PySparkなどのMapReduce系のマシン環境構築よりかなりお手軽．
* [DistArray](http://docs.enthought.com/distarray/) - Numpyと同じ操作をOut-of-coreにできるようにしたデータ操作・保持ライブラリ．マシン分散にはMPIを使ったIPython.parallel(現在のipyparallel)を利用する．
* [PySpark](http://spark.apache.org/docs/latest/programming-guide.html) - 本家SparkのPythonインタフェース．
* [bolt](http://bolt-project.org/docs/index.html) - PySparkとは違う，numpyフレンドリーなsparkインタフェースを実装した分散データ操作・保持ライブラリ．
* [dpark](https://github.com/douban/dpark) - SparkのPythonクローン．分散にはMesosクラスタを利用．手続き処理風に書いてMapReduceの処理を実行することができる．
* [luigi](https://github.com/spotify/luigi) - HiveやPigにかわるバッチデータパイプラインライブラリ．複雑なバッチデータワークフロー
をできるだけ簡単に書くことを目指している．
* [mrjob](https://github.com/Yelp/mrjob) - MapReduceをお手軽にAmazon EMRとGoogle Dataprocで実行することを目指したライブラリ．自分で立てたHadoopクラスタを使うこともできるが，EMR, Dataprocを利用することに特化したライブラリといっていい．
* [streamparse](https://github.com/Parsely/streamparse) - [Apache Storm](http://storm.apache.org/)にPythonで書かれたジョブを流すインタフェースライブラリ．
* [joblib](https://pythonhosted.org/joblib/) 単一マシンでプロセスをmultiprocessingよりずっとお手軽に並列実行するためのライブラリ．既存の関数を数行で並列実行できるようになる．
* [ipyparallel](https://ipyparallel.readthedocs.io/en/latest/) - IPython(Jupyter)を利用して分散計算を行うライブラリ．昔はIPythonに統合されていたが，Jupyterになって独立化された．主にMPIを利用してマシン間の通信を行う．環境セットアップが面倒．
* [StarCluster](http://star.mit.edu/cluster/docs/latest/plugins/ipython.html) - AWS EC2上でクラスタリングを簡単に行うフレームワークがStarCluster．そのStarClusterを利用して，IPython(Jupyter)から分散マシン処理を行うことができる．現在では実装もやり方も少し古いかもしれない．


## 統計解析・統計モデル

* [scipy.stats](http://docs.scipy.org/doc/scipy/reference/stats.html) - SciPyライブラリの中の統計解析手法，統計モデルコレクション．通常の統計分布はだいたいこのライブラリで事足りる．
* [Statsmodels](http://statsmodels.sourceforge.net/) - Pythonにおける統合的な統計解析，統計モデルライブラリ．以前はRに較べてかなり見劣りしていたが，最近は統計学の研究者でない限り問題ない充実度になってきた．
* [patsy](https://github.com/pydata/patsy) - R風("y ~ x1 + x2")に統計モデルを記述するユーティリティ．


## 機械学習

* [scikit-learn](http://scikit-learn.org/stable/) - 最も充実している汎用機械学習ライブラリ
* [vowpal_porpoise](https://github.com/josephreisinger/vowpal_porpoise) - 高速機械学習ライブラリ[Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit/)のPythonラッパー
* [XGBoost](http://xgboost.readthedocs.io/en/latest/python/python_intro.html) - 大きくスケールするアンサンブル学習(勾配ブースティング + ランダムフォレスト)分類器．Kaggleなどの機械学習プログラミング競技において絶大な支持を得ている．
* [Shogun](http://www.shogun-toolbox.org/) - SVMなどのカーネルメソッドに特化した機械学習ライブラリ
* [orange](http://orange.biolab.si/) - ビジュアルプログラミングに特色あるデータマイニングを中心とした老舗機械学習ライブラリ＋スタンドアローンアプリケーション．リュブリャナ大学が長年開発している．
* [MDP](http://mdp-toolkit.sourceforge.net/index.html) - 13年以上前から開発されてきた，データ処理フレームワークに，主成分分析，因子解析などの統計解析，機械学習のアルゴリズムを実装．既にこのライブラリの役割は追えていて，scikit-learnを使うのが通常である．
* [libsvm](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) - 本家livsvmが提供するPythonインタフェースは本体に含まれている．
* [milk](https://github.com/luispedro/milk/) - libsvmを使用したSVMやランダムフォレット，Non-negative matrix factorisationなどが使える，軽量機械学習ライブラリ．
* [PyMVPA](https://github.com/PyMVPA/PyMVPA) - fMRIなどの巨大データを統計解析するために作られたライブラリ．[OpenFMRI.org](https://openfmri.org/)のデータセットを直説読むことができるなど，fMRIのデータ解析に強い．機械学習はscikit-learnやShogunと組み合わせる
* [bolt](https://github.com/pprett/bolt) - SVMとロジステック回帰に特化した機械学習ライブラリ．コアの計算はC実装．古くて5年以上更新されていない．
* [pyGPs](https://github.com/marionmari/pyGPs) - ガウス過程による回帰・判別機械学習ライブラリ．
* [libcluster](https://github.com/dsteinberg/libcluster) - 階層ベイズによる判別問題を解く手法（変分ディリクレ過程，混合ベイズガウス過程など）を集めたライブラリ．C++実装だが，Pythonバインディングがある．
* [scikits系](http://scikits.appspot.com/scikits) - "scikit"の名の下に雑多の科学スタックを集めた．scikit-learn, scikit-image, statsmodels以外は死屍累々．


## 深層学習

* [Theano](https://github.com/Theano/Theano) - Bengio門下のモントリオール大謹製，深層学習ライブラリとしては老舗だが，比較的高速で現役．自動微分でも有名．
* [Tensorflow](https://www.tensorflow.org/) - Googleが満を持して出した分散深層学習フレームワーク．この分野では一番の人気で，エコシステムが豊か．ただし，分散計算はGoogleのデータセンタや専用プロセッサでないと最高のパフォーマンスがでないという根本的な問題がある．
* [caffe](http://caffe.berkeleyvision.org/) - 専門家以外への深層学習応用を加速したC++ライブラリ．設定ファイルとしてNNを記述する．Pythonのインタフェースがある．[Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)として学習済みモデルが多数あり，他のライブラリもこの学習済みモデルのインタフェースを設けるほど．
* [chainer](https://github.com/pfnet/chainer) - [PFN](https://www.preferred-networks.jp/ja/)が出したネットワークを動的に組み替え可能な深層学習フレームワーク．Cudaに渡す直前までPythonなので，デバグが楽でインタラクティブにネットワークの設計を行うことができる．日本では一番使われているように思われる．
* [mxnet](http://myungjun-youn-demo.readthedocs.io/en/latest/index.html) - parameter serverで分散計算ができることが売りの深層学習ライブラリ．
* [neon](https://github.com/NervanaSystems/neon) - （Intelに買収された）[Nervana system](https://www.nervanasys.com/)の深層学習フレームワーク．現在単一マシンで最速をほこる．CUDAとのインタフェースが独自実装で，ここの部分が速いらしい．
* [paddle](https://github.com/baidu/Paddle) - Baiduの深層学習フレームワーク．Caffe的なネットワーク設定をPythonで書いて，入力データソースと繋いで実行する．
* [keras](https://keras.io/) - 計算エンジンをTensorFlow, Theanoと切り替えられるインタフェースライブラリ．わかりやすいネットワーク実装と簡易な記法が売り．
* [Lasagne](https://github.com/Lasagne/Lasagne) - 人気があるTheanoの薄いラッパーライブラリ．
* [blocs](https://github.com/mila-udem/blocks) - 人気があるTheanoの薄いラッパーライブラリ．
* [tflearn](https://github.com/tflearn/tflearn) - scikit-learn風コードスタイルのTensorFlowのインタフェースライブラリ．同じようなライブラリに[skflow](https://github.com/tensorflow/skflow)があったが，こちらはTensorFlow本体に取り込まれた．
* [nolearn](https://github.com/dnouri/nolearn) - Lasagneをscikit-learn風コードスタイルにしたライブラリ．TensorFlow革命の後も奇跡的にまだ開発が続いている．
* [pylearn2](https://github.com/lisa-lab/pylearn2) - TheanoベースのBensio一門からでてきた深層学習ライブラリ．kerasの登場でほぼ死亡．クリエイターの現在の所属はTensorFlowのGoogle Brainだし．
* [hebel](https://github.com/hannes-brt/hebel) - PyCudaの上に書かれた独自深層学習ライブラリ．一時はそれなりに人気はあったが，TensorFlowの登場により死亡．
* [Neupy](http://neupy.com/pages/home.html) - ミニマリズムなニューラルネットライブラリ．もともと古典的なニューラルネットのライブラリをTheanoを使うことで深層学習に対応．


## 謎系脳型計算
* [NuPIC](https://github.com/numenta/nupic) - Palmのクリエイター ジェフ・ホーキンスが率いる謎の会社Numentaの謎系脳型計算ライブラリ．ただの畳み込みニューラルネットとベイジアンネットを組み合わせただけじゃないか疑惑が…．


## ベイズ推定，確率論的プログラミング
* [PyMC3](https://github.com/pymc-devs/pymc3) - MCMCによるベイズ推定ライブラリ．Theanoベース．
* [PyStan](https://pystan.readthedocs.io/en/latest/) - 最も進んだ汎用ベイズ推定計算ライブラリStanへのインタフェース．
* [edward](https://github.com/blei-lab/edward) - バックエンドにTensorFlowやStanなどを選べるイケてる変分ベイズ推定ライブラリ
* [emcee](http://dan.iel.fm/emcee/current/) - 分散MCMCアンサンブルサンプラー．もともと宇宙物理学の研究で使われていたライブラリを汎用化．"The MCMC Hammer"というサブタイトルはもちろん[M.C.ハマー](https://en.wikipedia.org/wiki/MC_Hammer)にちなむ．
* [BayesPy](https://github.com/bayespy/bayespy) - 変分ベイズ推定のピュアPython実装．将来は変分近似，ラブラス近似，MCMCサンプリングなどを実装するそう．


## 時系列解析
* [pyflux](https://pyflux.readthedocs.io/) - 時系列解析統合環境．Pythonの時系列解析ライブラリでは一番充実している．(公式サイトアクセス不能につき暫定的に変更)
* [cesium](http://cesium-ml.org/) - 主に脳波データ(EEG)の時系列データの特化した時系列解析・機械学習ライブラリ．実装されている機械学習の手法は回帰とランダムフォレスト．FFT/Wavelet法などの時間周波数解析は視野に入れてない．

### カルマンフィルタ
  * [pyKalman](https://github.com/pykalman/pykalman) EMアルゴリズムによるカルマンフィルタのPython実装．
  * [filterpy](https://pypi.python.org/pypi/filterpy/) 教科書["Kalman and Bayesian Filters in Python"](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/)の使用ライブラリ．速度の上で大きなアドバンテージはないが，教科書と共に非常に教育的．
  
## ネットワーク解析
* [NetworkX](https://networkx.github.io/) - ネットワーク解析におけるディファクトライブラリ．このライブラリを使うためにPythonを使う人達も．


## 最適化ライブラリ
* [cvxopt](http://cvxopt.org/) - 凸最適化問題を解くライブラリ．後段に説明するOpenOptやPuLPの数理最適化の文脈というより，機械学習の文脈の最適化問題を解くライブラリになっている．
* [autograd](https://github.com/HIPS/autograd) - Numpyベースで自動微分を行うライブラリ．Theanoなどがバックグラウンドで自動で行っている処理を明示的に実行することができる．ニューラルネットのような逐次最適化問題の損失関数を自分で設計するときに使う．
* [pycosat](https://pypi.python.org/pypi/pycosat) - Cで書かれた充足可能性問題ソルバー[PicoSAT](http://fmv.jku.at/picosat/)のPythonバインディング．
* [minfx](http://gna.org/projects/minfx/) - 直線探索，信頼領域探索，共役勾配法などを利用して数値最適化をするライブラリ．
* [kmpfit](http://www.astro.rug.nl/software/kapteyn/kmpfittutorial.html) - 最小二乗法に特化した単純な高速最適化ライブラリ．計算コアはC実装でCythonでPythonと繋いでいる．
* [Pymanopt](https://pymanopt.github.io/) - 多様体上の最適化ライブラリ．MatlabのManoptのPython実装．自動微分，自動ヘッセ行列計算．

### 数理最適化モデラー
*数理最適化問題をPythonから解くには: Pythonラッパー -> モデラー -> ソルバーという手順になる．モデラーやソルバーは大抵C/C++実装で別途インストールが必要である．*

* [OpenOpt](https://pypi.python.org/pypi/openopt) - 数々のソルバーと接続可能な数理最適化モデラー．主に非線形最適化問題に用いることが多い．myopenoptというPythonラッパーを利用してアクセスする．よくホームページが落ちているので心配になる．
* [PuLP](https://pypi.python.org/pypi/PuLP) -  数々のソルバーと接続可能な数理最適化モデラー．混合整数最適化モデリング（線形最適化問題，整数最適化問題に使う）を行う．mypulpというPythonラッパーを利用してアクセスする．
* [Pymo](http://www.pyomo.org/) - 数々のソルバーと接続可能な数理最適化モデラー．オブジェクト指向風にモデリングする．以前として活発に開発が続いている．

### 方程式ソルバー
* [SfePy](http://sfepy.org/doc-devel/index.html) - 有限要素法による偏微方程式のソルバー．1次元，2次元，3次元を扱う．
* [fipy](https://github.com/usnistgov/fipy) - 楕円，双曲線，放物線偏微分方程式のソルバー．READMEやホームページが文字化けやリンク切れできちんとしてないので実装の疑念が湧くが，一応使えるみたい．
* [FEniCS](https://fenicsproject.org/download/) - かなり多くの偏微分方程式に対応したソルバー．多くのソルバーがほぼ開発を止めたり速度を落とす中，未だに開発が継続されている．偏微分方程式ソルバー界の不死鳥．コアの実装はC++．

### グラフ理論
* [APGL](https://github.com/charanpald/APGL) - グラフ解析のためのグラフライブラリ．有向グラフ，無向グラフ，マルチグラフに対応．
* [graphillion](https://github.com/takemaru/graphillion) - 巨大グラフ，グラフ集合に対する，検索・最適化及び列挙を扱うためのグラフライブラリ．超高速なグラフ検索と列挙を特徴としている．

### Matrix Factorization
  * [PyMF](https://github.com/cthurau/pymf) - cvxoptベースのMatrix Factorizationライブラリ．いまではScikit-learnのMatrix Factorizationを使うことが多い．
  * [nimfa](https://github.com/marinkaz/nimfa) - Nonnegative Matrix FactorizationのピュアPython実装

### Factorization Machine
  * [pyFM](https://github.com/coreylynch/pyFM) - Cython実装のFactorization Machine．SGD W/ 自動正則化．
  * [fastFM](https://github.com/ibayer/fastFM) - scikit-learn風に書くFactorization Machineライブラリ．コアの計算はC実装のライブラリで高速化．
  * [pywFM](https://github.com/jfloff/pywFM) - Steffen Rendleの[libFM](http://libfm.org/)のPythonラッパー
  
### ベイズ最適化
  * [BayesOpt](https://bitbucket.org/rmcantin/bayesopt) - C++実装のベイズ最適化ライブラリ．pythonインタフェースがある．
  * [pybo](https://github.com/mwhoffman/pybo) - Python実装のベイズ最適化ライブラリ


## 自然言語解析

* [NLTK](http://www.nltk.org/) - 自然言語処理の統合ライブラリコレクション．
* [Pattern](http://www.clips.ua.ac.be/pattern) - Webマイニング用のテキスト解析．
* [gensim](http://radimrehurek.com/gensim/) - トピックモデリング，word2vecなどのベクトル空間モデリング
* [TextBlob](https://github.com/sloria/TextBlob) - Providing a consistent API for diving into common NLP tasks.
* [TextGrocery](https://github.com/2shou/TextGrocery) - A simple, efficient short-text classification tool based on LibLinear and Jieba.
* [langid.py](https://github.com/saffsd/langid.py) - Stand-alone language identification system.
* [quepy](http://idiap.github.io/bob/) - 自然言語をSQLなどのデータベースのクエリに変換するライブラリ．

### 日本語処理
  * [MeCab](http://taku910.github.io/mecab/) - C++実装の形態素解析エンジン．CRFを用いたパラメータ推定で判別精度が高く高速．Pythonバインディングは本体に付属している．
  * [PyKNP](http://lotus.kuee.kyoto-u.ac.jp/~john/pyknp.html) - 形態素解析エンジンJUMAN/KNPのPythonバインディング．
  * [ChaSen](http://chasen-legacy.osdn.jp/) - 比較的昔からある形態素解析エンジン．Pythonバインディングは本体に付属（ビルド時にオプション指定）．
  * [Igo-python](https://pypi.python.org/pypi/igo-python/) - Java/Common Lispで書かれた形態素解析ライブラリigoのPythonポーティング．
  * [Mykytea-python](https://github.com/chezou/Mykytea-python) -  形態素解析エンジンKyTeaのPythonバインディング
  * [janome](https://github.com/mocobeta/janome) - ピュアPythonの形態素解析ライブラリ
  * [neologd](https://github.com/neologd/mecab-ipadic-neologd) - MeCab用の新語辞書．IPA辞書の拡張．
  * [CaboCha](https://taku910.github.io/cabocha/) - SVMを用いた係り受け解析エンジン．Pythonバインディングは本体に付属している．
  * [pykakasi](https://pypi.python.org/pypi/pykakasi) - 漢字→かな(ローマ字)変換プログラム[KAKASHI](http://kakasi.namazu.org/index.html.ja)のPythonバインディング

## 専門分野

### 物理学
  * [QuTip](http://qutip.org/) - 量子物理学系の数値シミュレーションユーティリティ
  * [nbodykit](https://github.com/bccp/nbodykit) - MPIクラスタを利用して大規模並列N体シミュレーションをするライブラリ．主にスーパーコンピュータや，並列グリッドコンピュータなどのHPC環境で利用する．
  * [PyDy](http://www.pydy.org/) - 多体力学問題計算，可視化．運動方程式はSymPyを利用する．
  * [simPEG](http://simpeg.xyz/) - 地球物理学データの入出力，可視化，解析，シミュレーションライブラリ．
  * [SpacePy](http://spacepy.lanl.gov/) - 空間物理学データの入出力，可視化，解析ライブラリ．
  
### 宇宙物理学・天文学
  * [astropy](http://www.astropy.org/) - 宇宙物理学・天文学のデータの入出力，可視化，統計解析，機械学習ライブラリを集めたプロジェクト．
  * [SunPy](http://sunpy.org/) - 太陽物理学のデータの入出力，可視化，解析ライブラリ．astropyベース．
  * [gwpy](https://gwpy.github.io/) - 重力波解析，可視化ライブラリ．重力波解析ではLIGOのツールが有名だが，それをもっと使いやすくPythonicにしている．
  * [yt](http://yt-project.org/) - 天文学データのような大きなデータを入出力，可視化，解析する統合環境．Anaconda Pythonをカスタマイズした独自実行環境を利用する．
  * [halotools](https://github.com/astropy/halotools) - 銀河のハローの観測結果と（ハローとダークマターを含んだ）銀河の力学モデルのN体シミュレーションの結果の整合性をみるパッケージ．astropyプロジェクトのサブプロジェクト．
  * [PyRAF](http://bit.ly/py-RAF) - 天文学画像データの標準フォーマットであるIRAFの入出力インタフェース．

### 気象学・海洋学・気候学
  * [MetPy](https://github.com/metpy/MetPy) - 気象学データの入出力，可視化，解析ライブラリ．
  * [iris](http://scitools.org.uk/iris/docs/latest/index.html) - 気象学データの入出力，可視化，解析ライブラリ．
  * [PyNIO/PyNGL](http://pyaos.johnny-lin.com/) - 海洋データの入出力，可視化，解析ライブラリ．
  * [ulmo](https://github.com/ulmo-dev/ulmo) - 水文学，気候学のパブリックデータへのアクセスインタフェース．
  * [cdo-bindings](https://github.com/Try2Code/cdo-bindings) - 気候学のディファクトデータ入出力コマンド・ライブラリCDO(Climate Data Operators)のPythonバインディング．
  
### 地震学
  * [obspy](https://github.com/obspy/obspy)  - 地震波データの入出力，可視化，解析をPython科学スタックで行うライブラリ．現在の主流．
  * [pyasdf](http://seismicdata.github.io/pyasdf/) - 地震波データのフォーマットASDFを入出力するインタフェースライブラリ．obspyに取り込まれている．
  * [Pyrocko](http://emolch.github.io/pyrocko/) - 地震波データの入出力，可視化，解析ライブラリ．今は開発が止まり，実質obspyに置き換わられた．
  * [MSNoise](http://www.msnoise.org/)  -  Webインタフェースをもつ地震波の入出力，可視化，モニタリングライブラリ．

### 心理学実験
  * [PsychoPy](http://www.psychopy.org/index.html) - 心理学・認知心理学実験の刺激提示-データ収集ライブラリ．

### 分子生物学・バイオインフォマテックス
  * [Biopython](http://biopython.org/wiki/Biopython) - 分子生物学・分子遺伝学分野のデータ収集・可視化，解析ライブラリ．Next Generation Sequencer(NGS)興隆の今ではR/BioCunductorよりは生きづらいライブラリに…．
  * [CRISPResso](http://crispresso.rocks/) - CRISPR/CAS9設計支援・解析ライブラリ．
  * [bcbio-nextgen](https://github.com/chapmanb/bcbio-nextgen) - 新進気鋭のNGS対応データ収集，解析ライブラリ．NGSから出力される膨大なゲノムシーケンスデータ，その解析をマルチコアで並列実行したり，IPython parallelを利用してAWSインスタンスで並列実行する機能がある．


### 神経科学・脳科学
  * [nilearn](https://github.com/nilearn/nilearn) - 神経科学データの入出力，可視化，解析，機械学習ライブラリ．scikit-learnベース．
  * [tomopy](https://github.com/tomopy/tomopy) - X線CTなどの断層写真合成可視化ライブラリ．
  * [NIPY](http://nipy.org) - ニューロイメージングのツール集．

### 化学・分子動力学・ケモインフォマテックス
  * [cclib](http://cclib.github.io/) - 計算機化学データ可視化．解析ライブラリ．
  * [RDKit](http://www.rdkit.org/) - C++実装のケモインフォマテックスの解析，機械学習ライブラリ．Pythonインタフェースを持つ．
  * [PyQuante](http://pyquante.sourceforge.net/) - 量子化学

### 疫学
  * [epipy](http://cmrivers.github.io/epipy/) フィールド疫学データ入出力，可視化ライブラリ．
  
### 工学
  * [bob](http://idiap.github.io/bob/) - CとPythonで書かれた信号処理ライブラリ．音声，静止画像，動画映像のそれぞれに対応した関数群をそろえている．SVMやEMアルゴリズム，ブースティングなどの機械学習アルゴリズムも実装されている．
  * [python-control](https://pypi.python.org/pypi/control/0.7.0) - Matlabクローンを目指している制御工学理論ライブラリ．最近は開発が止まっている…．
  * [scikit-aero](https://scikits.appspot.com/scikit-aero) - 航空工学
  * [PyNE](http://pyne.io/) - 原子力工学

### 金融工学・BI
  * [Open Mining](https://github.com/mining/mining) - PythonベースのBIシステム． pandasベースで解析を行う．
  * [zipline](https://github.com/quantopian/zipline) - アルゴリズミックトレーディングのライブラリ．イベントドリブンでトレーディングタスクを実行するため，Cythonにより高速化されている．Blaze環境で並列実行もできる．
  
  
## 数学

* [Sage](http://www.sagemath.org/) - ライブラリというよりは世の中の定評ある計算機数学ライブラリ，GP/PARI, GAP, Singularなど，をCythonで統合し，Pythonインタフェースで使えるようにした計算機数学統合環境．Jupyterシェルのようなシェル，Jupyter notebookのようなWebインタフェース，クラウドコンピューティングの[SageMathCloud](https://cloud.sagemath.com/)などをエコシステムとして含んでいる．
* [SymPy](http://www.sympygamma.com/) - 数式処理，離散数学のための一連のツールを提供する．
* [slepc4py](https://bitbucket.org/slepc/slepc4py/src) - 巨大行列の固有値問題ソルバー [SLEPc](http://slepc.upv.es/)のPythonバインディング．
* [mpmath](http://mpmath.org/) - 任意の精度の浮動小数点を扱うライブラリ．SymPyやSageに組みこまれている．
* [SnaPy](http://www.math.uic.edu/t3m/SnapPy/) - 3次元多様体のトポロジーと幾何学を扱うライブラリ．
* [quaternionarray](https://github.com/zonca/quaternionarray) - 4元数配列を操作するライブラリ．
* [logpy](https://github.com/logpy/logpy) - Pythonでロジックプログラミング


## 地理空間情報(GIS)

* [Mapnik](http://mapnik.org/) - GISデータから地図をレンダリングするエンジン．実装はC++だが，ほとんどのAPIを呼び出すことできるPythonインタフェースを持っている．
* [GeoPandas](http://geopandas.org/) - pandasのGeo版．測地情報を直接操作できる．
* [Pyproj](https://github.com/jswhit/pyproj) - 測地情報（緯度経度）と地図のネイティブ座標の相互変換をするライブラリ
* [GDAL](http://www.gdal.org/) - ラスターデータフォーマットのメタデータの検索や、データフォーマットの変換を行うライブラリ．Pythonバインディングがある．
* [OGR](http://gdal.org/1.11/ogr/) - 地理空間ベクターデータ（シェープファイルなど）のメタデータの検索や、データフォーマットの変換を行うライブラリ．Pythonバインディングがある．GDALの一部である．
* [Shapely](https://github.com/Toblerity/Shapely) - 空間オブジェクトを幾何学的に操作したり，幾何学的性質を調べたりするライブラリ．広く使われているGEOSのPythonポーティング．
* [pysal](http://pysal.readthedocs.io/en/latest/index.html) - 地理空間データの入出力，可視化，解析ライブラリ．
* [OSGeo](http://bit.ly/osgeo-lib) - GISデータの入出力と解析
* [Basemap](http://matplotlib.org/basemap/) - 測地情報の2Dマッピング
* [GeoDjango](https://docs.djangoproject.com/en/dev/ref/contrib/gis/) - Djangoベースのジオサーバ．よく使われるC++実装の[MapServer](https://github.com/mapserver/mapserver)やJava実装の[GeoServer](https://github.com/geoserver/geoserver)と同じジオサーバのPython版．
* [geojson](https://github.com/frewsxcv/python-geojson) - GeoJsonを扱うライブラリ．
* [geopy](https://github.com/geopy/geopy) - 住所と測地情報（緯度経度）を結びつけるジオコーディングを行うライブラリ．
* [GeoIP2-python](https://github.com/maxmind/GeoIP2-python) - IPアドレスから地域を特定する有料データベース[MaxMind GeoIP2 DB](https://www.maxmind.com/en/geoip2-databases)にアクセスするためのライブラリ．
* [pygeoip](https://github.com/appliedsec/pygeoip) - IPアドレスから地域を特定する[geoip-api-c](https://github.com/maxmind/geoip-api-c)のPythonラッパー．
* [django-countries](https://github.com/SmileyChris/django-countries) - HTMLのフォームの選択肢として国名を供給数するDjangoアプリ．国旗アイコンや国ごとの住所フォーマットも供給する．


## コンピュータビジョン

* [OpenCV](http://opencv.org/) - 最も有名なオープンソースのコンピュータビジョンフレームワーク．巨大なフレームワークで，画像変換，画像解析から機械学習までなんでも揃う．しかし，丸一日以上かけてビルドする羽目になることが多くて，いつもビルドに苦労する．
* [SimpleCV](http://simplecv.org/) - OpenCVベースのコンピュータビジョンフレームワークで，OpenCVよりPythonicにコードを書くことができる．このフレームワークもOpenCVとこのフレームワークが使うライブラリの整合性がかなりピーキーで，OpenCVより更にビルドが地獄．



## 音声認識
*[PyHTK](https://github.com/danijel3/PyHTK) - 音声認識エンジン[HTK](http://htk.eng.cam.ac.uk/)をバックエンドに使った音声認識モデリング，解析ライブラリ．


## 可視化

* [matplotlib](http://matplotlib.org) - Pythonにおける二次元プロットライブラリの老舗．いまだに王者の風格が衰えず支配的．
* [Bokeh](http://bokeh.pydata.org) - ブラウザに表示するための高機能プロットライブラリ．インタラクティブな操作も可能．Continuum謹製．
* [d3py](https://github.com/mikedewar/d3py) - pythonからd3.jsをお気軽に使うライブラリ．
* [ggplot](https://github.com/yhat/ggplot) - Rのggplot2のサブクローン．
* [plotly](https://plot.ly) プロットWebサービスとの連携を視野に置いた，インタラクティブなプロットライブラリ．
* [seaborn](http://web.stanford.edu/~mwaskom/software/seaborn/) - matplotlibベースにグラフ種類の増加と審美的な改善を図った野心的プロットライブラリ
* [vincent](https://github.com/wrobstory/vincent) - A Python to Vega translator.
* [pygal](http://www.pygal.org/en/latest/) - A Python SVG Charts Creator.
* [pygraphviz](https://pypi.python.org/pypi/pygraphviz) - Python interface to [Graphviz](http://www.graphviz.org/).
* [PyQtGraph](http://www.pyqtgraph.org/) - Interactive and realtime 2D/3D/Image plotting and science/engineering widgets.
* [SnakeViz](http://jiffyclub.github.io/snakeviz/) - A browser based graphical viewer for the output of Python's cProfile module.
* [VisPy](http://vispy.org/) - High-performance scientific visualization based on OpenGL.


## データセット
* [PyDataset](https://github.com/iamaziz/PyDataset) - アヤメデータやボストンの自動車公害などの有名な統計データセットをpandasの形式で手に入れることができるデータコレクション．


## リプロダクション
* [rep](https://github.com/yandex/rep) - scikit-learnやXGBoostなどを使った研究成果の再現環境を構築．
* [Sumatra](https://github.com/open-research/sumatra) - 数値実験・シミュレーション研究の再現環境を構築．数値実験界の「自動化ラボノート」．
* [reprozip](https://vida-nyu.github.io/reprozip/) - 研究環境が全てPython科学スタックで実装さていることを前提に，dockerまたはvagrantを利用してマシンから実行環境からパラメータを含めて全てをリプロダクションさせる野心的プロジェクト．


## 高速化

* [Cython](http://cython.org/) - Optimizing Static Compiler for Python. Uses type mixins to compile Python into C or C++ modules resulting in large performance gains.
* [blaze](https://github.com/blaze/blaze) - NumPy and Pandas interface to Big Data.
* [Numba](http://numba.pydata.org/) - Python JIT (just in time) complier to LLVM aimed at scientific Python by the developers of Cython and NumPy.
* [PeachPy](https://github.com/Maratyszcza/PeachPy) - x86-64 assembler embedded in Python. Can be used as inline assembler for Python or as a stand-alone assembler for Windows, Linux, OS X, Native Client and Go.
* [PyPy](http://pypy.org/) - An implementation of Python in Python. The interpreter uses black magic to make Python very fast without having to add in additional type information.
* [Pyston](https://github.com/dropbox/pyston) - A Python implementation built using LLVM and modern JIT techniques with the goal of achieving good performance.
* [Stackless Python](https://bitbucket.org/stackless-dev/stackless/overview) - An enhanced version of the Python.


## 外部インタフェース

* [cffi](https://pypi.python.org/pypi/cffi) - Foreign Function Interface for Python calling C code.
* [ctypes](https://docs.python.org/2/library/ctypes.html) - (Python standard library) Foreign Function Interface for Python calling C code.
* [SWIG](http://www.swig.org/Doc1.3/Python.html) - Simplified Wrapper and Interface Generator.
* [PyCUDA](https://mathema.tician.de/software/pycuda/) - A Python wrapper for Nvidia's CUDA API.
* [cupy](http://docs.chainer.org/en/stable/cupy-reference/) - Chainerの派生プロジェクト．Numpy APIのサブセットをCUDAのAPIに接続する．ChainerがPython実装なのに比較的高速である理由がこのライブラリにある．



# Resources

Python科学スタックのためのリソース情報


## Websites

* [IBIS Wiki Python](http://ibisforest.org/index.php?python) 神嶌敏弘さんのPython科学スタックリンク集．日本のデータソースリンクの草分け．
* [Scipy Lecture Notes](http://www.scipy-lectures.org/) Numpy, SciPyの基礎を学ぶ．[邦訳](http://www.turbare.net/transl/scipy-lecture-notes/)．
* [CME 193: Introduction to Scientific Python](https://web.stanford.edu/~arbenson/cme193.html) スタンフォードのPython科学スタックの講義
* [Introduction to Python for Data Science](https://www.edx.org/course/introduction-python-data-science-microsoft-dat208x-3) - edXのPython科学スタックの講義
* [r/Python](https://www.reddit.com/r/python)
* [/r/CoolGithubProjects](https://www.reddit.com/r/coolgithubprojects/)
* [Full Stack Python](https://www.fullstackpython.com/)
* [Trending Python repositories on GitHub today](https://github.com/trending?l=python)
* [PyPI Ranking](http://pypi-ranking.info/alltime)
* [Planet Python](http://planetpython.org/)
* [Awesome Python @LibHunt](http://python.libhunt.com)


## Weekly

* [CodeTengu Weekly](http://weekly.codetengu.com/)
* [Import Python Newsletter](http://importpython.com/newsletter/)
* [Pycoder's Weekly](http://pycoders.com/)
* [Python Weekly](http://www.pythonweekly.com/)


## Twitter

* [@SciPyTip](https://twitter.com/scipytip)
* [@getpy](https://twitter.com/getpy)
* [@importpython](https://twitter.com/importpython)
* [@planetpython](https://twitter.com/planetpython)
* [@pycoders](https://twitter.com/pycoders)
* [@pypi](https://twitter.com/pypi)
* [@pythontrending](https://twitter.com/pythontrending)
* [@PythonWeekly](https://twitter.com/PythonWeekly)


## Podcasts

* [Podcast.init](http://podcastinit.com/)
* [Talk Python To Me](https://talkpython.fm/)

