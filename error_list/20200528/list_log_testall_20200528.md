## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py


### Error 1, [Traceback at line 43](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L43)<br />43..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783/mlmodels/model_keras//charcnn_zhang.py", line 261, in <module>
<br />    test(pars_choice="json", data_path=f"dataset/json/refactor/charcnn_zhang.json")
<br />  File "https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783/mlmodels/model_keras//charcnn_zhang.py", line 222, in test
<br />    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783/mlmodels/model_keras//charcnn_zhang.py", line 151, in get_params
<br />    cf = json.load(open(data_path, mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783/mlmodels/dataset/json/refactor/charcnn_zhang.json'



### Error 2, [Traceback at line 86](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L86)<br />86..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783/mlmodels/model_keras//charcnn.py", line 373, in <module>
<br />    test(pars_choice="json", data_path= f"dataset/json/refactor/charcnn.json")
<br />  File "https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783/mlmodels/model_keras//charcnn.py", line 330, in test
<br />    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783/mlmodels/model_keras//charcnn.py", line 266, in get_params
<br />    cf = json.load(open(data_path, mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783/mlmodels/dataset/json/refactor/charcnn.json'



### Error 3, [Traceback at line 132](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L132)<br />132..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783/mlmodels/model_keras//namentity_crm_bilstm.py", line 306, in <module>
<br />    test_module(model_uri=MODEL_URI, param_pars=param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783/mlmodels/models.py", line 257, in test_module
<br />    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783/mlmodels/model_keras/namentity_crm_bilstm.py", line 197, in get_params
<br />    cf = json.load(open(data_path, mode="r"))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783/mlmodels/dataset/json/refactor/namentity_crm_bilstm_dataloader.json'



### Error 4, [Traceback at line 178](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L178)<br />178..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783/mlmodels/model_keras//textcnn.py", line 258, in <module>
<br />    test_module(model_uri = MODEL_URI, param_pars= param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783/mlmodels/models.py", line 257, in test_module
<br />    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783/mlmodels/model_keras/textcnn.py", line 165, in get_params
<br />    cf = json.load(open(data_path, mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783/mlmodels/dataset/json/refactor/textcnn_keras.json'



### Error 5, [Traceback at line 216](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L216)<br />216..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783/mlmodels/model_dev//temporal_fusion_google.py", line 17, in <module>
<br />    from mlmodels.mode_tf.raw  import temporal_fusion_google
<br />ModuleNotFoundError: No module named 'mlmodels.mode_tf'



### Error 6, [Traceback at line 573](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L573)<br />573..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783/mlmodels/model_gluon//fb_prophet.py", line 160, in <module>
<br />    test(data_path = "model_fb/fbprophet.json", choice="json" )
<br />TypeError: test() got an unexpected keyword argument 'choice'



### Error 7, [Traceback at line 1641](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L1641)<br />1641..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783/mlmodels/model_tch//torchhub.py", line 431, in <module>
<br />    test(data_path="dataset/json/refactor/resnet18_benchmark_mnist.json", pars_choice="json", config_mode="test")
<br />  File "https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783/mlmodels/model_tch//torchhub.py", line 362, in test
<br />    model, session = fit(model, data_pars, compute_pars, out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783/mlmodels/model_tch//torchhub.py", line 231, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783/mlmodels/model_tch//torchhub.py", line 50, in _train
<br />    image, target = image.to(device), target.to(device)
<br />AttributeError: 'tuple' object has no attribute 'to'



### Error 8, [Traceback at line 1689](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L1689)<br />1689..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783/mlmodels/model_tch//transformer_sentence.py", line 488, in <module>
<br />    test(pars_choice="json", data_path="model_tch/transformer_sentence.json", config_mode="test")
<br />  File "https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783/mlmodels/model_tch//transformer_sentence.py", line 448, in test
<br />    model, session = fit(model, data_pars, model_pars, compute_pars, out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783/mlmodels/model_tch//transformer_sentence.py", line 139, in fit
<br />    train_data       = SentencesDataset(train_reader.get_examples(train_fname),  model=model.model)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/sentence_transformers/readers/NLIDataReader.py", line 21, in get_examples
<br />    mode="rt", encoding="utf-8").readlines()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/gzip.py", line 53, in open
<br />    binary_file = GzipFile(filename, gz_mode, compresslevel)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/gzip.py", line 163, in __init__
<br />    fileobj = self.myfileobj = builtins.open(filename, mode or 'rb')
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/82ca0cabe4779c98bad687c53f6357fc6efdf783/mlmodels/dataset/text/AllNLI/s1.train.gz'
