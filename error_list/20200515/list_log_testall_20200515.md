## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-15-16-10_169ff9dd8baf94be9a49cc5b3e3dcd3c926c4077.py


### Error 1, [Traceback at line 37](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-15-16-10_169ff9dd8baf94be9a49cc5b3e3dcd3c926c4077.py#L37)<br />37..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/169ff9dd8baf94be9a49cc5b3e3dcd3c926c4077/mlmodels/model_keras//keras_gan.py", line 31, in <module>
<br />    'AAE' : kg.aae.aae,
<br />AttributeError: module 'mlmodels.model_keras.raw.keras_gan' has no attribute 'aae'



### Error 2, [Traceback at line 81](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-15-16-10_169ff9dd8baf94be9a49cc5b3e3dcd3c926c4077.py#L81)<br />81..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/169ff9dd8baf94be9a49cc5b3e3dcd3c926c4077/mlmodels/model_keras//textcnn_dataloader.py", line 275, in <module>
<br />    test_module(model_uri = MODEL_URI, param_pars= param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/169ff9dd8baf94be9a49cc5b3e3dcd3c926c4077/mlmodels/models.py", line 257, in test_module
<br />    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/169ff9dd8baf94be9a49cc5b3e3dcd3c926c4077/mlmodels/model_keras/textcnn_dataloader.py", line 182, in get_params
<br />    cf = json.load(open(data_path, mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/169ff9dd8baf94be9a49cc5b3e3dcd3c926c4077/mlmodels/dataset/json/refactor/textcnn_keras.json'



### Error 3, [Traceback at line 128](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-15-16-10_169ff9dd8baf94be9a49cc5b3e3dcd3c926c4077.py#L128)<br />128..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/169ff9dd8baf94be9a49cc5b3e3dcd3c926c4077/mlmodels/model_keras//nbeats.py", line 315, in <module>
<br />    test(pars_choice="test01")
<br />  File "https://github.com/arita37/mlmodels/tree/169ff9dd8baf94be9a49cc5b3e3dcd3c926c4077/mlmodels/model_keras//nbeats.py", line 278, in test
<br />    Xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/169ff9dd8baf94be9a49cc5b3e3dcd3c926c4077/mlmodels/model_keras//nbeats.py", line 172, in get_dataset
<br />    train_data = Data(data_source= path_norm( data_pars["train_data_source"]) ,
<br />NameError: name 'Data' is not defined
