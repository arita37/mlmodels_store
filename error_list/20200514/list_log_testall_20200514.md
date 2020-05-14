## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-14-12-12_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py


### Error 1, [Traceback at line 37](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-14-12-12_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L37)<br />37..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//keras_gan.py", line 31, in <module>
<br />    'AAE' : kg.aae.aae,
<br />AttributeError: module 'mlmodels.model_keras.raw.keras_gan' has no attribute 'aae'



### Error 2, [Traceback at line 87](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-14-12-12_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L87)<br />87..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//textcnn_dataloader.py", line 275, in <module>
<br />    test_module(model_uri = MODEL_URI, param_pars= param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/models.py", line 257, in test_module
<br />    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras/textcnn_dataloader.py", line 182, in get_params
<br />    cf = json.load(open(data_path, mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/dataset/json/refactor/textcnn_keras.json'



### Error 3, [Traceback at line 134](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-14-12-12_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L134)<br />134..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//nbeats.py", line 315, in <module>
<br />    test(pars_choice="test01")
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//nbeats.py", line 278, in test
<br />    Xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//nbeats.py", line 172, in get_dataset
<br />    train_data = Data(data_source= path_norm( data_pars["train_data_source"]) ,
<br />NameError: name 'Data' is not defined
