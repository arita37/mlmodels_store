
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '6ca6da91408244e26c157e9e6467cc18ede43e71', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/6ca6da91408244e26c157e9e6467cc18ede43e71

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71

 ************************************************************************************************************************

  ############Check model ################################ 





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py --do test  

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "dataset":"mlmodels\/dataset\/text\/ag_news_csv",
    "train":true,
    "alphabet_size":69,
    "alphabet":"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"\/\\|_@#$%^&*~`+-=<>()[]{}",
    "input_size":1014,
    "num_of_classes":4
  },
  "preprocessors":[
    {
      "name":"loader",
      "uri":"mlmodels.preprocess.generic:pandasDataset",
      "args":{
        "colX":[
          "colX"
        ],
        "coly":[
          "coly"
        ],
        "encoding":"'ISO-8859-1'",
        "read_csv_parm":{
          "usecols":[
            0,
            1
          ],
          "names":[
            "coly",
            "colX"
          ]
        }
      }
    },
    {
      "name":"tokenizer",
      "uri":"mlmodels.model_keras.raw.char_cnn.data_utils:Data",
      "args":{
        "data_source":"",
        "alphabet":"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"\/\\|_@#$%^&*~`+-=<>()[]{}",
        "input_size":1014,
        "num_of_classes":4
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'> 
cls_name : pandasDataset
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn.json [Errno 2] File b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv' does not exist: b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv'

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn_zhang.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "dataset":"mlmodels\/dataset\/text\/ag_news_csv",
    "train":true,
    "alphabet_size":69,
    "alphabet":"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"\/\\|_@#$%^&*~`+-=<>()[]{}",
    "input_size":1014,
    "num_of_classes":4
  },
  "preprocessors":[
    {
      "name":"loader",
      "uri":"mlmodels.preprocess.generic:pandasDataset",
      "args":{
        "colX":[
          "colX"
        ],
        "coly":[
          "coly"
        ],
        "encoding":"'ISO-8859-1'",
        "read_csv_parm":{
          "usecols":[
            0,
            1
          ],
          "names":[
            "coly",
            "colX"
          ]
        }
      }
    },
    {
      "name":"tokenizer",
      "uri":"mlmodels.model_keras.raw.char_cnn.data_utils:Data",
      "args":{
        "data_source":"",
        "alphabet":"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"\/\\|_@#$%^&*~`+-=<>()[]{}",
        "input_size":1014,
        "num_of_classes":4
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'> 
cls_name : pandasDataset
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn_zhang.json [Errno 2] File b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv' does not exist: b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv'

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/textcnn.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "dataset":"mlmodels\/dataset\/text\/imdb",
    "pass_data_pars":false,
    "train":true,
    "maxlen":40,
    "max_features":5
  },
  "preprocessors":[
    {
      "name":"loader",
      "uri":"mlmodels.preprocess.generic:NumpyDataset",
      "args":{
        "numpy_loader_args":{
          "allow_pickle":true
        },
        "encoding":"'ISO-8859-1'"
      }
    },
    {
      "name":"imdb_process",
      "uri":"mlmodels.preprocess.text_keras:IMDBDataset",
      "args":{
        "num_words":5
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:NumpyDataset {'numpy_loader_args': {'allow_pickle': True}, 'encoding': "'ISO-8859-1'"} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.NumpyDataset'> 
cls_name : NumpyDataset
Dataset File path :  train/mlmodels/dataset/text/imdb.npz
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/textcnn.json [Errno 2] No such file or directory: 'train/mlmodels/dataset/text/imdb.npz'

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/text\/",
    "dataset":"ner_dataset.csv",
    "pass_data_pars":false,
    "train":true
  },
  "preprocessors":[
    {
      "name":"loader",
      "uri":"mlmodels.preprocess.generic:pandasDataset",
      "args":{
        "read_csv_parm":{
          "encoding":"ISO-8859-1"
        },
        "colX":[

        ],
        "coly":[

        ]
      }
    },
    {
      "uri":"mlmodels.preprocess.text_keras:Preprocess_namentity",
      "args":{
        "max_len":75
      },
      "internal_states":[
        "word_count"
      ]
    },
    {
      "name":"split_xy",
      "uri":"mlmodels.dataloader:split_xy_from_dict",
      "args":{
        "col_Xinput":[
          "X"
        ],
        "col_yinput":[
          "y"
        ]
      }
    },
    {
      "name":"split_train_test",
      "uri":"sklearn.model_selection:train_test_split",
      "args":{
        "test_size":0.5
      }
    },
    {
      "name":"saver",
      "uri":"mlmodels.dataloader:pickle_dump",
      "args":{
        "path":"mlmodels\/ztest\/ml_keras\/namentity_crm_bilstm\/data.pkl"
      }
    }
  ],
  "output":{
    "shape":[
      [
        75
      ],
      [
        75,
        18
      ]
    ],
    "max_len":75
  }
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:pandasDataset {'read_csv_parm': {'encoding': 'ISO-8859-1'}, 'colX': [], 'coly': []} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'> 
cls_name : pandasDataset

  URL:  mlmodels.preprocess.text_keras:Preprocess_namentity {'max_len': 75} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.text_keras.Preprocess_namentity'> 
cls_name : Preprocess_namentity

  
 Object Creation 

  
 Object Compute 

  
 Object get_data 

  URL:  mlmodels.dataloader:split_xy_from_dict {'col_Xinput': ['X'], 'col_yinput': ['y']} 

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f847c6eb620> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f847c6eb620> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f84e739d400> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f84e739d400> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f85065b9ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f85065b9ea0> 
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm.json [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/torchhub_cnn_dataloader.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/vision\/MNIST",
    "dataset":"MNIST",
    "data_type":"tch_dataset",
    "batch_size":10,
    "train":true
  },
  "preprocessors":[
    {
      "name":"tch_dataset_start",
      "uri":"mlmodels.preprocess.generic:get_dataset_torch",
      "args":{
        "dataloader":"torchvision.datasets:MNIST",
        "to_image":true,
        "transform":{
          "uri":"mlmodels.preprocess.image:torch_transform_mnist",
          "pass_data_pars":false,
          "arg":{
            "fixed_size":256,
            "path":"dataset\/vision\/MNIST\/"
          }
        },
        "shuffle":true,
        "download":true
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f84946d80d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f84946d80d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f84946d80d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 258, in compute
    obj_preprocessor = preprocessor_func(**args, data_info=self.data_info)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/preprocess/generic.py", line 502, in __init__
    df = pd.read_csv(file_path, **args.get("read_csv_parm",{}))
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
    self._reader = parsers.TextReader(src, **kwds)
  File "pandas/_libs/parsers.pyx", line 382, in pandas._libs.parsers.TextReader.__cinit__
  File "pandas/_libs/parsers.pyx", line 689, in pandas._libs.parsers.TextReader._setup_parser_source
FileNotFoundError: [Errno 2] File b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv' does not exist: b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv'
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 258, in compute
    obj_preprocessor = preprocessor_func(**args, data_info=self.data_info)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/preprocess/generic.py", line 502, in __init__
    df = pd.read_csv(file_path, **args.get("read_csv_parm",{}))
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
    self._reader = parsers.TextReader(src, **kwds)
  File "pandas/_libs/parsers.pyx", line 382, in pandas._libs.parsers.TextReader.__cinit__
  File "pandas/_libs/parsers.pyx", line 689, in pandas._libs.parsers.TextReader._setup_parser_source
FileNotFoundError: [Errno 2] File b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv' does not exist: b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv'
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 258, in compute
    obj_preprocessor = preprocessor_func(**args, data_info=self.data_info)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/preprocess/generic.py", line 607, in __init__
    data            = np.load( file_path,**args.get("numpy_loader_args", {}))
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 416, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
FileNotFoundError: [Errno 2] No such file or directory: 'train/mlmodels/dataset/text/imdb.npz'
Using TensorFlow backend.
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 301, in compute
    out_tmp = preprocessor_func(input_tmp, **args)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 93, in pickle_dump
    with open(kwargs["path"], "wb") as fi:
FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'
0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:21, 451233.99it/s] 61%|██████    | 6037504/9912422 [00:00<00:06, 642544.29it/s]9920512it [00:00, 38449764.61it/s]                           
0it [00:00, ?it/s]32768it [00:00, 1238300.33it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1014816.60it/s]1654784it [00:00, 12408872.65it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 11556.80it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Processing...
Done!

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8492bd2780>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8492874a20>), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/model_list_CIFAR.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/vision\/cifar10\/",
    "dataset":"CIFAR10",
    "data_type":"tf_dataset",
    "batch_size":10,
    "train":true
  },
  "preprocessors":[
    {
      "name":"tf_dataset_start",
      "uri":"mlmodels.preprocess.generic:tf_dataset_download",
      "arg":{
        "train_samples":2000,
        "test_samples":500,
        "shuffle":true,
        "download":true
      }
    },
    {
      "uri":"mlmodels.preprocess.generic:get_dataset_torch",
      "args":{
        "dataloader":"mlmodels.preprocess.generic:NumpyDataset",
        "to_image":true,
        "transform":{
          "uri":"mlmodels.preprocess.image:torch_transform_generic",
          "pass_data_pars":false,
          "arg":{
            "fixed_size":256
          }
        },
        "shuffle":true,
        "download":true
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:tf_dataset_download {} 

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f84946d0d08> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f84946d0d08> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f84946d0d08> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:988: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:01<?, ? MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:01<03:47,  1.41s/ MiB][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:01<03:47,  1.41s/ MiB][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   1%|          | 2/162 [00:01<02:44,  1.03s/ MiB][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:01<02:44,  1.03s/ MiB][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   2%|▏         | 3/162 [00:01<01:59,  1.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:01<01:59,  1.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:01<01:58,  1.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   3%|▎         | 5/162 [00:01<01:26,  1.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:01<01:26,  1.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:01<01:25,  1.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:01<01:03,  2.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:01<01:03,  2.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:02<01:02,  2.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:02<00:47,  3.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:02<00:47,  3.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:02<00:46,  3.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:   7%|▋         | 11/162 [00:02<00:35,  4.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:02<00:35,  4.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:02<00:34,  4.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:02<00:34,  4.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:   9%|▊         | 14/162 [00:02<00:26,  5.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:02<00:26,  5.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:02<00:25,  5.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:02<00:25,  5.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:02<00:19,  7.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:02<00:19,  7.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:02<00:19,  7.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:02<00:19,  7.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:02<00:19,  7.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:02<00:14,  9.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:02<00:14,  9.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:02<00:14,  9.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:02<00:14,  9.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:02<00:14,  9.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:02<00:13,  9.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:02<00:10, 12.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:02<00:10, 12.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:02<00:10, 12.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:02<00:10, 12.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:02<00:10, 12.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:02<00:10, 12.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:02<00:10, 12.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:02<00:07, 16.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:02<00:07, 16.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:02<00:07, 16.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:02<00:07, 16.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:02<00:07, 16.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:02<00:07, 16.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:02<00:07, 16.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:02<00:05, 20.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:02<00:05, 20.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:02<00:05, 20.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:02<00:05, 20.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:02<00:05, 20.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:02<00:05, 20.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:02<00:05, 20.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:02<00:05, 20.86 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:03<00:05, 20.86 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  28%|██▊       | 46/162 [00:03<00:04, 26.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:03<00:04, 26.54 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:03<00:04, 26.54 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:03<00:04, 26.54 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:03<00:04, 26.54 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:03<00:04, 26.54 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:03<00:04, 26.54 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:03<00:04, 26.54 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:03<00:04, 26.54 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:03<00:03, 33.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:03<00:03, 33.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:03<00:03, 33.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:03<00:03, 33.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:03<00:03, 33.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:03<00:03, 33.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:03<00:03, 33.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:03<00:03, 33.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:03<00:03, 33.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:03<00:03, 33.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:03<00:02, 40.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:03<00:02, 40.38 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:03<00:02, 40.38 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:03<00:02, 40.38 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:03<00:02, 40.38 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:03<00:02, 40.38 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:03<00:02, 40.38 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:03<00:02, 40.38 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:03<00:02, 40.38 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:03<00:01, 47.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:03<00:01, 47.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:03<00:01, 47.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:03<00:01, 47.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:03<00:01, 47.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:03<00:01, 47.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:03<00:01, 47.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:03<00:01, 47.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:03<00:01, 47.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:03<00:01, 47.35 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:03<00:01, 54.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:03<00:01, 54.25 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:03<00:01, 54.25 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:03<00:01, 54.25 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:03<00:01, 54.25 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:03<00:01, 54.25 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:03<00:01, 54.25 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:03<00:01, 54.25 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:03<00:01, 54.25 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:03<00:01, 59.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:03<00:01, 59.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:03<00:01, 59.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:03<00:01, 59.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:03<00:01, 59.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:03<00:01, 59.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:03<00:01, 59.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:03<00:01, 59.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:03<00:01, 59.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:03<00:01, 64.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:03<00:01, 64.55 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:03<00:01, 64.55 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:03<00:00, 64.55 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:03<00:00, 64.55 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:03<00:00, 64.55 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:03<00:00, 64.55 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:03<00:00, 64.55 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:03<00:00, 64.55 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:03<00:00, 68.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:03<00:00, 68.01 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:03<00:00, 68.01 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:03<00:00, 68.01 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:03<00:00, 68.01 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:03<00:00, 68.01 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:03<00:00, 68.01 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:03<00:00, 68.01 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:03<00:00, 68.01 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:03<00:00, 68.01 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:03<00:00, 72.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:03<00:00, 72.06 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:03<00:00, 72.06 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:03<00:00, 72.06 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:03<00:00, 72.06 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:03<00:00, 72.06 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:03<00:00, 72.06 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:03<00:00, 72.06 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:03<00:00, 72.06 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:03<00:00, 72.06 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:03<00:00, 75.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:03<00:00, 75.11 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:03<00:00, 75.11 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:03<00:00, 75.11 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:04<00:00, 75.11 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:04<00:00, 75.11 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:04<00:00, 75.11 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:04<00:00, 75.11 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:04<00:00, 75.11 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:04<00:00, 75.11 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:04<00:00, 75.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:04<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:04<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:04<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:04<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:04<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:04<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:04<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:04<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:04<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:04<00:00, 77.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:04<00:00, 77.31 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:04<00:00, 77.31 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:04<00:00, 77.31 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:04<00:00, 77.31 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:04<00:00, 77.31 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:04<00:00, 77.31 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:04<00:00, 77.31 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:04<00:00, 77.31 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:04<00:00, 76.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:04<00:00, 76.81 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:04<00:00, 76.81 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:04<00:00, 76.81 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:04<00:00, 76.81 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:04<00:00, 76.81 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:04<00:00, 76.81 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:04<00:00, 76.81 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:04<00:00, 76.81 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:04<00:00, 77.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:04<00:00, 77.68 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:04<00:00, 77.68 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:04<00:00, 77.68 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:04<00:00, 77.68 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:04<00:00, 77.68 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:04<00:00, 77.68 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 77.68 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.50s/ url]Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.50s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 77.68 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.50s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 77.68 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:04<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.92s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:06<00:00,  4.50s/ url]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 77.68 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.92s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.92s/ file]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 23.40 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:06<00:00,  6.92s/ url]
0 examples [00:00, ? examples/s]2020-08-18 12:07:02.538683: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-18 12:07:02.553025: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-08-18 12:07:02.553274: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e1d4bf1560 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-18 12:07:02.553293: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  3.02 examples/s]96 examples [00:00,  4.30 examples/s]201 examples [00:00,  6.14 examples/s]297 examples [00:00,  8.75 examples/s]393 examples [00:00, 12.44 examples/s]489 examples [00:00, 17.68 examples/s]587 examples [00:00, 25.06 examples/s]683 examples [00:01, 35.40 examples/s]785 examples [00:01, 49.83 examples/s]879 examples [00:01, 69.58 examples/s]972 examples [00:01, 96.07 examples/s]1069 examples [00:01, 131.61 examples/s]1165 examples [00:01, 177.54 examples/s]1265 examples [00:01, 235.57 examples/s]1361 examples [00:01, 301.40 examples/s]1454 examples [00:01, 377.15 examples/s]1547 examples [00:01, 458.35 examples/s]1648 examples [00:02, 547.48 examples/s]1743 examples [00:02, 625.99 examples/s]1838 examples [00:02, 692.80 examples/s]1932 examples [00:02, 738.74 examples/s]2030 examples [00:02, 796.54 examples/s]2129 examples [00:02, 845.75 examples/s]2230 examples [00:02, 887.25 examples/s]2327 examples [00:02, 903.36 examples/s]2425 examples [00:02, 922.66 examples/s]2522 examples [00:02, 911.81 examples/s]2619 examples [00:03, 925.68 examples/s]2719 examples [00:03, 943.06 examples/s]2816 examples [00:03, 949.62 examples/s]2913 examples [00:03, 946.78 examples/s]3014 examples [00:03, 964.51 examples/s]3117 examples [00:03, 981.93 examples/s]3216 examples [00:03, 978.06 examples/s]3315 examples [00:03, 952.38 examples/s]3411 examples [00:03, 941.38 examples/s]3506 examples [00:04, 928.41 examples/s]3601 examples [00:04, 931.27 examples/s]3695 examples [00:04, 933.56 examples/s]3791 examples [00:04, 941.10 examples/s]3886 examples [00:04, 931.07 examples/s]3986 examples [00:04, 948.59 examples/s]4084 examples [00:04, 956.51 examples/s]4182 examples [00:04, 960.36 examples/s]4279 examples [00:04, 949.56 examples/s]4378 examples [00:04, 960.03 examples/s]4475 examples [00:05, 960.33 examples/s]4572 examples [00:05, 959.62 examples/s]4669 examples [00:05, 959.73 examples/s]4766 examples [00:05, 950.39 examples/s]4862 examples [00:05, 930.49 examples/s]4956 examples [00:05, 909.28 examples/s]5052 examples [00:05, 923.28 examples/s]5147 examples [00:05, 929.66 examples/s]5241 examples [00:05, 920.16 examples/s]5334 examples [00:05, 920.47 examples/s]5427 examples [00:06, 921.24 examples/s]5522 examples [00:06, 928.77 examples/s]5617 examples [00:06, 934.99 examples/s]5711 examples [00:06, 924.69 examples/s]5806 examples [00:06, 931.98 examples/s]5902 examples [00:06, 936.40 examples/s]5996 examples [00:06, 925.10 examples/s]6094 examples [00:06, 938.82 examples/s]6188 examples [00:06, 933.14 examples/s]6285 examples [00:06, 941.43 examples/s]6382 examples [00:07, 949.25 examples/s]6477 examples [00:07, 947.99 examples/s]6576 examples [00:07, 959.09 examples/s]6672 examples [00:07, 936.44 examples/s]6766 examples [00:07, 937.24 examples/s]6861 examples [00:07, 938.14 examples/s]6955 examples [00:07, 937.71 examples/s]7049 examples [00:07, 928.20 examples/s]7142 examples [00:07, 925.62 examples/s]7238 examples [00:07, 932.59 examples/s]7332 examples [00:08, 925.21 examples/s]7425 examples [00:08, 905.10 examples/s]7516 examples [00:08, 891.16 examples/s]7606 examples [00:08, 887.04 examples/s]7700 examples [00:08, 900.37 examples/s]7793 examples [00:08, 908.43 examples/s]7884 examples [00:08, 908.74 examples/s]7975 examples [00:08, 897.28 examples/s]8065 examples [00:08, 889.40 examples/s]8161 examples [00:09, 907.68 examples/s]8257 examples [00:09, 921.27 examples/s]8352 examples [00:09, 926.62 examples/s]8445 examples [00:09, 924.81 examples/s]8539 examples [00:09, 926.86 examples/s]8640 examples [00:09, 948.30 examples/s]8739 examples [00:09, 957.66 examples/s]8835 examples [00:09, 946.54 examples/s]8930 examples [00:09, 931.08 examples/s]9024 examples [00:09, 892.55 examples/s]9115 examples [00:10, 896.00 examples/s]9212 examples [00:10, 915.37 examples/s]9310 examples [00:10, 931.89 examples/s]9404 examples [00:10, 924.08 examples/s]9497 examples [00:10, 921.36 examples/s]9599 examples [00:10, 948.38 examples/s]9702 examples [00:10, 969.09 examples/s]9802 examples [00:10, 977.82 examples/s]9901 examples [00:10, 977.46 examples/s]9999 examples [00:10, 973.32 examples/s]10097 examples [00:11, 920.27 examples/s]10201 examples [00:11, 951.53 examples/s]10300 examples [00:11, 962.20 examples/s]10397 examples [00:11, 943.24 examples/s]10494 examples [00:11, 950.40 examples/s]10590 examples [00:11, 950.10 examples/s]10688 examples [00:11, 958.60 examples/s]10785 examples [00:11, 953.43 examples/s]10881 examples [00:11, 928.69 examples/s]10978 examples [00:12, 938.49 examples/s]11073 examples [00:12, 927.45 examples/s]11171 examples [00:12, 941.01 examples/s]11271 examples [00:12, 956.64 examples/s]11367 examples [00:12, 940.37 examples/s]11468 examples [00:12, 957.90 examples/s]11565 examples [00:12, 959.86 examples/s]11663 examples [00:12, 965.54 examples/s]11766 examples [00:12, 983.89 examples/s]11865 examples [00:12, 965.00 examples/s]11962 examples [00:13, 959.15 examples/s]12059 examples [00:13, 956.83 examples/s]12157 examples [00:13, 960.74 examples/s]12254 examples [00:13, 961.84 examples/s]12356 examples [00:13, 978.06 examples/s]12462 examples [00:13, 1000.66 examples/s]12563 examples [00:13, 998.50 examples/s] 12670 examples [00:13, 1016.73 examples/s]12772 examples [00:13, 1009.27 examples/s]12874 examples [00:13, 980.73 examples/s] 12973 examples [00:14, 959.75 examples/s]13075 examples [00:14, 974.43 examples/s]13173 examples [00:14, 968.95 examples/s]13273 examples [00:14, 976.19 examples/s]13371 examples [00:14, 958.57 examples/s]13468 examples [00:14, 947.37 examples/s]13567 examples [00:14, 957.56 examples/s]13668 examples [00:14, 972.16 examples/s]13767 examples [00:14, 975.93 examples/s]13865 examples [00:14, 945.14 examples/s]13966 examples [00:15, 962.87 examples/s]14066 examples [00:15, 973.11 examples/s]14164 examples [00:15, 970.38 examples/s]14262 examples [00:15, 964.19 examples/s]14359 examples [00:15, 943.86 examples/s]14454 examples [00:15, 945.52 examples/s]14554 examples [00:15, 960.09 examples/s]14653 examples [00:15, 967.78 examples/s]14754 examples [00:15, 979.60 examples/s]14853 examples [00:16, 974.67 examples/s]14954 examples [00:16, 982.38 examples/s]15053 examples [00:16, 981.54 examples/s]15152 examples [00:16, 980.24 examples/s]15251 examples [00:16, 971.40 examples/s]15349 examples [00:16, 961.53 examples/s]15453 examples [00:16, 981.52 examples/s]15554 examples [00:16, 989.72 examples/s]15654 examples [00:16, 975.13 examples/s]15752 examples [00:16, 959.55 examples/s]15849 examples [00:17, 952.45 examples/s]15946 examples [00:17, 956.19 examples/s]16045 examples [00:17, 964.79 examples/s]16142 examples [00:17, 963.88 examples/s]16239 examples [00:17, 965.22 examples/s]16336 examples [00:17, 960.61 examples/s]16438 examples [00:17, 976.29 examples/s]16539 examples [00:17, 983.26 examples/s]16638 examples [00:17, 964.09 examples/s]16735 examples [00:17, 960.33 examples/s]16832 examples [00:18, 953.87 examples/s]16928 examples [00:18, 954.13 examples/s]17024 examples [00:18, 949.03 examples/s]17121 examples [00:18, 952.46 examples/s]17217 examples [00:18, 936.49 examples/s]17311 examples [00:18, 929.07 examples/s]17408 examples [00:18, 940.91 examples/s]17506 examples [00:18, 951.34 examples/s]17609 examples [00:18, 970.89 examples/s]17707 examples [00:18, 967.38 examples/s]17804 examples [00:19, 954.11 examples/s]17907 examples [00:19, 972.77 examples/s]18005 examples [00:19, 964.97 examples/s]18102 examples [00:19, 962.66 examples/s]18200 examples [00:19, 965.26 examples/s]18297 examples [00:19, 953.83 examples/s]18393 examples [00:19, 950.90 examples/s]18489 examples [00:19, 944.85 examples/s]18585 examples [00:19, 948.23 examples/s]18683 examples [00:19, 956.38 examples/s]18779 examples [00:20, 947.53 examples/s]18875 examples [00:20, 948.71 examples/s]18974 examples [00:20, 959.30 examples/s]19070 examples [00:20, 949.15 examples/s]19165 examples [00:20, 935.85 examples/s]19259 examples [00:20, 920.88 examples/s]19356 examples [00:20, 932.97 examples/s]19450 examples [00:20, 929.07 examples/s]19543 examples [00:20, 923.52 examples/s]19636 examples [00:21, 922.57 examples/s]19729 examples [00:21, 924.26 examples/s]19824 examples [00:21, 931.65 examples/s]19920 examples [00:21, 937.83 examples/s]20014 examples [00:21, 885.88 examples/s]20110 examples [00:21, 906.83 examples/s]20205 examples [00:21, 916.83 examples/s]20302 examples [00:21, 929.31 examples/s]20396 examples [00:21, 905.92 examples/s]20487 examples [00:21, 906.98 examples/s]20581 examples [00:22, 912.77 examples/s]20678 examples [00:22, 928.64 examples/s]20774 examples [00:22, 937.45 examples/s]20871 examples [00:22, 944.63 examples/s]20971 examples [00:22, 959.67 examples/s]21068 examples [00:22, 951.99 examples/s]21164 examples [00:22, 952.66 examples/s]21260 examples [00:22, 951.95 examples/s]21356 examples [00:22, 950.69 examples/s]21462 examples [00:22, 980.21 examples/s]21561 examples [00:23, 971.59 examples/s]21661 examples [00:23, 979.41 examples/s]21761 examples [00:23, 984.74 examples/s]21860 examples [00:23, 976.40 examples/s]21958 examples [00:23, 976.49 examples/s]22056 examples [00:23, 959.62 examples/s]22153 examples [00:23, 952.74 examples/s]22251 examples [00:23, 957.72 examples/s]22347 examples [00:23, 941.11 examples/s]22445 examples [00:23, 949.84 examples/s]22541 examples [00:24, 946.74 examples/s]22642 examples [00:24, 964.15 examples/s]22739 examples [00:24, 964.89 examples/s]22841 examples [00:24, 979.33 examples/s]22940 examples [00:24, 966.13 examples/s]23037 examples [00:24, 942.97 examples/s]23139 examples [00:24, 962.52 examples/s]23236 examples [00:24, 947.31 examples/s]23331 examples [00:24, 940.79 examples/s]23427 examples [00:25, 945.22 examples/s]23522 examples [00:25, 935.79 examples/s]23617 examples [00:25, 939.30 examples/s]23712 examples [00:25, 940.36 examples/s]23813 examples [00:25, 958.87 examples/s]23910 examples [00:25, 959.55 examples/s]24007 examples [00:25, 955.24 examples/s]24107 examples [00:25, 966.10 examples/s]24204 examples [00:25, 939.33 examples/s]24300 examples [00:25, 945.07 examples/s]24395 examples [00:26, 940.10 examples/s]24490 examples [00:26, 935.54 examples/s]24587 examples [00:26, 944.64 examples/s]24683 examples [00:26, 949.07 examples/s]24778 examples [00:26, 940.69 examples/s]24873 examples [00:26, 937.84 examples/s]24969 examples [00:26, 944.10 examples/s]25065 examples [00:26, 945.97 examples/s]25160 examples [00:26, 915.97 examples/s]25252 examples [00:26, 914.77 examples/s]25346 examples [00:27, 920.39 examples/s]25439 examples [00:27, 921.31 examples/s]25532 examples [00:27, 913.84 examples/s]25626 examples [00:27, 919.74 examples/s]25720 examples [00:27, 923.13 examples/s]25819 examples [00:27, 941.02 examples/s]25914 examples [00:27, 928.66 examples/s]26007 examples [00:27, 922.60 examples/s]26100 examples [00:27, 908.61 examples/s]26195 examples [00:27, 918.29 examples/s]26287 examples [00:28, 891.05 examples/s]26384 examples [00:28, 911.84 examples/s]26484 examples [00:28, 934.21 examples/s]26585 examples [00:28, 955.33 examples/s]26689 examples [00:28, 978.14 examples/s]26790 examples [00:28, 985.36 examples/s]26889 examples [00:28, 971.21 examples/s]26987 examples [00:28, 957.34 examples/s]27086 examples [00:28, 965.95 examples/s]27183 examples [00:29, 950.62 examples/s]27279 examples [00:29, 943.71 examples/s]27374 examples [00:29, 945.18 examples/s]27475 examples [00:29, 962.69 examples/s]27572 examples [00:29, 958.98 examples/s]27669 examples [00:29, 955.82 examples/s]27765 examples [00:29, 953.73 examples/s]27862 examples [00:29, 958.06 examples/s]27958 examples [00:29, 938.69 examples/s]28052 examples [00:29, 924.63 examples/s]28151 examples [00:30, 942.06 examples/s]28249 examples [00:30, 949.14 examples/s]28349 examples [00:30, 962.81 examples/s]28454 examples [00:30, 985.94 examples/s]28559 examples [00:30, 1002.40 examples/s]28662 examples [00:30, 1008.93 examples/s]28764 examples [00:30, 992.13 examples/s] 28867 examples [00:30, 1001.99 examples/s]28968 examples [00:30, 999.09 examples/s] 29069 examples [00:30, 999.96 examples/s]29170 examples [00:31, 990.77 examples/s]29270 examples [00:31, 968.30 examples/s]29368 examples [00:31, 969.15 examples/s]29466 examples [00:31, 956.42 examples/s]29566 examples [00:31, 968.36 examples/s]29666 examples [00:31, 976.88 examples/s]29764 examples [00:31, 973.59 examples/s]29862 examples [00:31, 949.81 examples/s]29958 examples [00:31, 929.49 examples/s]30052 examples [00:32, 880.60 examples/s]30151 examples [00:32, 909.89 examples/s]30247 examples [00:32, 921.92 examples/s]30344 examples [00:32, 934.54 examples/s]30444 examples [00:32, 951.56 examples/s]30540 examples [00:32, 950.30 examples/s]30641 examples [00:32, 965.98 examples/s]30738 examples [00:32, 962.86 examples/s]30835 examples [00:32, 938.73 examples/s]30935 examples [00:32, 954.80 examples/s]31035 examples [00:33, 967.13 examples/s]31136 examples [00:33, 976.83 examples/s]31234 examples [00:33, 968.02 examples/s]31334 examples [00:33, 977.26 examples/s]31437 examples [00:33, 991.83 examples/s]31537 examples [00:33, 989.50 examples/s]31640 examples [00:33, 999.74 examples/s]31741 examples [00:33, 991.95 examples/s]31841 examples [00:33, 983.56 examples/s]31940 examples [00:33, 977.61 examples/s]32039 examples [00:34, 980.16 examples/s]32138 examples [00:34, 981.66 examples/s]32237 examples [00:34, 958.23 examples/s]32335 examples [00:34, 962.71 examples/s]32432 examples [00:34, 952.10 examples/s]32535 examples [00:34, 970.80 examples/s]32636 examples [00:34, 982.00 examples/s]32735 examples [00:34, 972.49 examples/s]32833 examples [00:34, 970.98 examples/s]32933 examples [00:34, 979.03 examples/s]33035 examples [00:35, 990.01 examples/s]33135 examples [00:35, 989.59 examples/s]33235 examples [00:35, 986.16 examples/s]33340 examples [00:35, 1002.17 examples/s]33441 examples [00:35, 995.04 examples/s] 33542 examples [00:35, 998.50 examples/s]33644 examples [00:35, 1002.05 examples/s]33745 examples [00:35, 977.16 examples/s] 33843 examples [00:35, 977.48 examples/s]33941 examples [00:35, 966.78 examples/s]34043 examples [00:36, 981.68 examples/s]34142 examples [00:36, 978.79 examples/s]34240 examples [00:36, 956.16 examples/s]34342 examples [00:36, 971.94 examples/s]34442 examples [00:36, 978.54 examples/s]34542 examples [00:36, 981.04 examples/s]34641 examples [00:36, 977.15 examples/s]34739 examples [00:36, 972.89 examples/s]34837 examples [00:36, 964.88 examples/s]34938 examples [00:37, 977.83 examples/s]35041 examples [00:37, 992.60 examples/s]35142 examples [00:37, 995.84 examples/s]35242 examples [00:37, 993.68 examples/s]35342 examples [00:37, 992.80 examples/s]35442 examples [00:37, 989.58 examples/s]35542 examples [00:37, 990.80 examples/s]35644 examples [00:37, 997.57 examples/s]35744 examples [00:37, 987.56 examples/s]35843 examples [00:37, 986.46 examples/s]35942 examples [00:38, 983.88 examples/s]36042 examples [00:38, 988.15 examples/s]36142 examples [00:38, 989.87 examples/s]36242 examples [00:38, 970.93 examples/s]36340 examples [00:38, 968.14 examples/s]36437 examples [00:38, 963.68 examples/s]36536 examples [00:38, 970.13 examples/s]36634 examples [00:38, 969.44 examples/s]36731 examples [00:38, 942.47 examples/s]36830 examples [00:38, 953.94 examples/s]36926 examples [00:39, 953.52 examples/s]37022 examples [00:39, 953.61 examples/s]37120 examples [00:39, 961.19 examples/s]37217 examples [00:39, 956.91 examples/s]37316 examples [00:39, 965.49 examples/s]37415 examples [00:39, 971.32 examples/s]37513 examples [00:39, 970.28 examples/s]37612 examples [00:39, 973.49 examples/s]37710 examples [00:39, 962.90 examples/s]37807 examples [00:39, 962.61 examples/s]37906 examples [00:40, 969.15 examples/s]38003 examples [00:40, 966.17 examples/s]38104 examples [00:40, 976.73 examples/s]38202 examples [00:40, 954.82 examples/s]38301 examples [00:40, 964.36 examples/s]38398 examples [00:40, 957.18 examples/s]38494 examples [00:40, 954.77 examples/s]38590 examples [00:40, 952.22 examples/s]38686 examples [00:40, 939.82 examples/s]38781 examples [00:40, 931.09 examples/s]38875 examples [00:41, 917.73 examples/s]38970 examples [00:41, 924.68 examples/s]39066 examples [00:41, 933.68 examples/s]39160 examples [00:41, 928.97 examples/s]39257 examples [00:41, 939.50 examples/s]39352 examples [00:41, 939.86 examples/s]39448 examples [00:41, 945.59 examples/s]39547 examples [00:41, 958.43 examples/s]39643 examples [00:41, 948.84 examples/s]39738 examples [00:41, 940.95 examples/s]39838 examples [00:42, 955.23 examples/s]39935 examples [00:42, 958.09 examples/s]40031 examples [00:42, 891.92 examples/s]40122 examples [00:42, 870.41 examples/s]40210 examples [00:42, 860.95 examples/s]40306 examples [00:42, 888.11 examples/s]40396 examples [00:42, 883.76 examples/s]40488 examples [00:42, 892.51 examples/s]40578 examples [00:42, 876.99 examples/s]40673 examples [00:43, 895.63 examples/s]40763 examples [00:43, 889.81 examples/s]40860 examples [00:43, 910.62 examples/s]40961 examples [00:43, 935.61 examples/s]41056 examples [00:43, 937.12 examples/s]41154 examples [00:43, 949.19 examples/s]41253 examples [00:43, 960.85 examples/s]41356 examples [00:43, 979.90 examples/s]41457 examples [00:43, 988.56 examples/s]41557 examples [00:43, 960.40 examples/s]41657 examples [00:44, 970.63 examples/s]41762 examples [00:44, 992.75 examples/s]41862 examples [00:44, 976.47 examples/s]41960 examples [00:44, 972.56 examples/s]42058 examples [00:44, 972.88 examples/s]42156 examples [00:44, 963.58 examples/s]42258 examples [00:44, 977.95 examples/s]42361 examples [00:44, 991.77 examples/s]42461 examples [00:44, 980.82 examples/s]42560 examples [00:44, 974.03 examples/s]42658 examples [00:45, 970.70 examples/s]42757 examples [00:45, 974.43 examples/s]42857 examples [00:45, 979.76 examples/s]42959 examples [00:45, 989.11 examples/s]43058 examples [00:45, 978.06 examples/s]43156 examples [00:45, 978.00 examples/s]43256 examples [00:45, 984.33 examples/s]43358 examples [00:45, 992.77 examples/s]43458 examples [00:45, 994.31 examples/s]43558 examples [00:46, 972.58 examples/s]43656 examples [00:46, 973.25 examples/s]43754 examples [00:46, 954.14 examples/s]43850 examples [00:46, 949.91 examples/s]43946 examples [00:46, 950.27 examples/s]44042 examples [00:46, 939.24 examples/s]44137 examples [00:46, 925.15 examples/s]44233 examples [00:46, 934.06 examples/s]44328 examples [00:46, 937.73 examples/s]44425 examples [00:46, 945.84 examples/s]44523 examples [00:47, 954.20 examples/s]44619 examples [00:47, 929.21 examples/s]44718 examples [00:47, 944.65 examples/s]44816 examples [00:47, 953.09 examples/s]44916 examples [00:47, 964.69 examples/s]45013 examples [00:47, 962.39 examples/s]45111 examples [00:47, 966.32 examples/s]45209 examples [00:47, 968.46 examples/s]45308 examples [00:47, 973.48 examples/s]45406 examples [00:47, 968.60 examples/s]45503 examples [00:48, 941.20 examples/s]45603 examples [00:48, 955.31 examples/s]45699 examples [00:48, 945.25 examples/s]45796 examples [00:48, 950.16 examples/s]45892 examples [00:48, 946.59 examples/s]45987 examples [00:48, 939.30 examples/s]46082 examples [00:48, 938.61 examples/s]46179 examples [00:48, 946.13 examples/s]46276 examples [00:48, 952.68 examples/s]46379 examples [00:48, 972.31 examples/s]46477 examples [00:49, 964.61 examples/s]46576 examples [00:49, 970.93 examples/s]46674 examples [00:49, 971.14 examples/s]46773 examples [00:49, 975.28 examples/s]46873 examples [00:49, 981.58 examples/s]46972 examples [00:49, 969.46 examples/s]47072 examples [00:49, 977.67 examples/s]47176 examples [00:49, 994.97 examples/s]47276 examples [00:49, 976.90 examples/s]47374 examples [00:49, 967.38 examples/s]47471 examples [00:50, 955.22 examples/s]47567 examples [00:50, 954.13 examples/s]47663 examples [00:50, 944.48 examples/s]47761 examples [00:50, 954.51 examples/s]47859 examples [00:50, 959.69 examples/s]47956 examples [00:50, 953.17 examples/s]48053 examples [00:50, 957.78 examples/s]48151 examples [00:50, 962.22 examples/s]48248 examples [00:50, 961.83 examples/s]48345 examples [00:51, 932.25 examples/s]48439 examples [00:51, 934.31 examples/s]48534 examples [00:51, 938.41 examples/s]48631 examples [00:51, 946.96 examples/s]48726 examples [00:51, 942.95 examples/s]48823 examples [00:51, 950.15 examples/s]48922 examples [00:51, 961.67 examples/s]49024 examples [00:51, 976.18 examples/s]49122 examples [00:51, 972.92 examples/s]49220 examples [00:51, 946.15 examples/s]49319 examples [00:52, 957.28 examples/s]49419 examples [00:52, 968.34 examples/s]49522 examples [00:52, 985.92 examples/s]49621 examples [00:52, 986.76 examples/s]49720 examples [00:52, 980.80 examples/s]49819 examples [00:52, 979.63 examples/s]49918 examples [00:52, 963.21 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 7235/50000 [00:00<00:00, 72349.72 examples/s] 40%|███▉      | 19849/50000 [00:00<00:00, 82959.35 examples/s] 65%|██████▍   | 32339/50000 [00:00<00:00, 92251.77 examples/s] 87%|████████▋ | 43726/50000 [00:00<00:00, 97822.48 examples/s]                                                               0 examples [00:00, ? examples/s]78 examples [00:00, 776.20 examples/s]178 examples [00:00, 830.07 examples/s]274 examples [00:00, 863.03 examples/s]362 examples [00:00, 865.72 examples/s]462 examples [00:00, 900.68 examples/s]562 examples [00:00, 927.62 examples/s]662 examples [00:00, 947.08 examples/s]756 examples [00:00, 942.23 examples/s]847 examples [00:00, 929.69 examples/s]949 examples [00:01, 953.43 examples/s]1050 examples [00:01, 968.93 examples/s]1152 examples [00:01, 982.52 examples/s]1256 examples [00:01, 998.67 examples/s]1356 examples [00:01, 959.26 examples/s]1455 examples [00:01, 963.15 examples/s]1559 examples [00:01, 984.84 examples/s]1659 examples [00:01, 988.62 examples/s]1762 examples [00:01, 1000.42 examples/s]1863 examples [00:01, 982.63 examples/s] 1962 examples [00:02, 977.59 examples/s]2066 examples [00:02, 993.23 examples/s]2166 examples [00:02, 991.70 examples/s]2269 examples [00:02, 1002.35 examples/s]2370 examples [00:02, 989.47 examples/s] 2472 examples [00:02, 997.45 examples/s]2572 examples [00:02, 986.88 examples/s]2673 examples [00:02, 991.00 examples/s]2773 examples [00:02, 984.01 examples/s]2872 examples [00:02, 958.06 examples/s]2968 examples [00:03, 954.84 examples/s]3064 examples [00:03, 956.32 examples/s]3165 examples [00:03, 969.51 examples/s]3270 examples [00:03, 991.82 examples/s]3370 examples [00:03, 959.95 examples/s]3471 examples [00:03, 972.34 examples/s]3570 examples [00:03, 975.53 examples/s]3668 examples [00:03, 976.01 examples/s]3766 examples [00:03, 974.35 examples/s]3864 examples [00:03, 971.32 examples/s]3964 examples [00:04, 978.01 examples/s]4067 examples [00:04, 990.63 examples/s]4169 examples [00:04, 999.06 examples/s]4269 examples [00:04, 991.24 examples/s]4369 examples [00:04, 979.59 examples/s]4470 examples [00:04, 986.56 examples/s]4572 examples [00:04, 995.55 examples/s]4674 examples [00:04, 1002.67 examples/s]4775 examples [00:04, 983.35 examples/s] 4874 examples [00:05, 941.58 examples/s]4969 examples [00:05, 928.10 examples/s]5063 examples [00:05, 905.45 examples/s]5157 examples [00:05, 912.76 examples/s]5253 examples [00:05, 924.73 examples/s]5351 examples [00:05, 938.80 examples/s]5452 examples [00:05, 956.96 examples/s]5553 examples [00:05, 969.83 examples/s]5652 examples [00:05, 974.36 examples/s]5750 examples [00:05, 970.16 examples/s]5848 examples [00:06, 963.90 examples/s]5945 examples [00:06, 964.36 examples/s]6043 examples [00:06, 968.33 examples/s]6144 examples [00:06, 979.45 examples/s]6243 examples [00:06, 974.75 examples/s]6341 examples [00:06, 970.50 examples/s]6439 examples [00:06, 970.75 examples/s]6537 examples [00:06, 960.93 examples/s]6637 examples [00:06, 970.64 examples/s]6735 examples [00:06, 968.38 examples/s]6835 examples [00:07, 975.14 examples/s]6933 examples [00:07, 964.45 examples/s]7033 examples [00:07, 973.08 examples/s]7138 examples [00:07, 992.20 examples/s]7238 examples [00:07, 975.43 examples/s]7336 examples [00:07, 972.52 examples/s]7438 examples [00:07, 985.30 examples/s]7543 examples [00:07, 1003.44 examples/s]7646 examples [00:07, 1011.00 examples/s]7748 examples [00:07, 973.77 examples/s] 7850 examples [00:08, 987.06 examples/s]7950 examples [00:08, 978.95 examples/s]8049 examples [00:08, 973.70 examples/s]8151 examples [00:08, 986.44 examples/s]8250 examples [00:08, 972.29 examples/s]8353 examples [00:08, 987.54 examples/s]8452 examples [00:08, 971.67 examples/s]8550 examples [00:08, 968.50 examples/s]8647 examples [00:08, 961.44 examples/s]8744 examples [00:09, 954.66 examples/s]8841 examples [00:09, 957.88 examples/s]8939 examples [00:09, 964.12 examples/s]9039 examples [00:09, 972.98 examples/s]9137 examples [00:09, 956.64 examples/s]9233 examples [00:09, 955.97 examples/s]9332 examples [00:09, 964.56 examples/s]9434 examples [00:09, 977.88 examples/s]9532 examples [00:09, 966.19 examples/s]9629 examples [00:09, 950.77 examples/s]9725 examples [00:10, 949.84 examples/s]9826 examples [00:10, 964.22 examples/s]9924 examples [00:10, 968.14 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteS8ZBFA/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteS8ZBFA/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f84946d80d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f84946d80d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f84946d80d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f84e00b7828>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f841a9ca390>), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/resnet34_benchmark_mnist.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/vision\/MNIST\/",
    "dataset":"MNIST",
    "data_type":"tch_dataset",
    "batch_size":10,
    "train":true
  },
  "preprocessors":[
    {
      "name":"tch_dataset_start",
      "uri":"mlmodels.preprocess.generic:get_dataset_torch",
      "args":{
        "dataloader":"torchvision.datasets:MNIST",
        "to_image":true,
        "transform":{
          "uri":"mlmodels.preprocess.image:torch_transform_mnist",
          "pass_data_pars":false,
          "arg":{

          }
        },
        "shuffle":true,
        "download":true
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f84946d80d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f84946d80d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f84946d80d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8492874320>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8492874908>), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/textcnn.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/recommender\/",
    "dataset":"IMDB_sample.txt",
    "data_type":"csv_dataset",
    "batch_size":64,
    "train":true
  },
  "preprocessors":[
    {
      "uri":"mlmodels.model_tch.textcnn:split_train_valid",
      "args":{
        "frac":0.99
      }
    },
    {
      "uri":"mlmodels.model_tch.textcnn:create_tabular_dataset",
      "args":{
        "lang":"en",
        "pretrained_emb":"glove.6B.300d"
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.model_tch.textcnn:split_train_valid {'frac': 0.99} 
Error /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/textcnn.json Module ['mlmodels.model_tch.textcnn', 'split_train_valid'] notfound, libtorch_cpu.so: cannot open shared object file: No such file or directory, tuple index out of range

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/transformer_sentence.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{

  },
  "preprocessors":[
    {
      "name":"DataReader",
      "uri":"mlmodels.model_tch.util_transformer:TransformerDataReader",
      "args":{
        "train":{
          "uri":"sentence_transformers.readers:NLIDataReader",
          "dataset":"dataset\/text\/AllNLI\/train"
        },
        "test":{
          "uri":"sentence_transformers.readers:STSBenchmarkDataReader",
          "dataset":"dataset\/text\/stsbenchmark\/val"
        }
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.model_tch.util_transformer:TransformerDataReader {'train': {'uri': 'sentence_transformers.readers:NLIDataReader', 'dataset': 'dataset/text/AllNLI/train'}, 'test': {'uri': 'sentence_transformers.readers:STSBenchmarkDataReader', 'dataset': 'dataset/text/stsbenchmark/val'}} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.model_tch.util_transformer.TransformerDataReader'> 
cls_name : TransformerDataReader

  
 Object Creation 

  
 Object Compute 
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/transformer_sentence.json Module ['sentence_transformers.readers', 'STSBenchmarkDataReader'] notfound, module 'sentence_transformers.readers' has no attribute 'STSBenchmarkDataReader', tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 672, in load_function_uri
    return  getattr(importlib.import_module(package), name)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 24, in <module>
    import torchtext
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 42, in <module>
    _init_extension()
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 38, in _init_extension
    torch.ops.load_library(ext_specs.origin)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torch/_ops.py", line 106, in load_library
    ctypes.CDLL(path)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/ctypes/__init__.py", line 348, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: libtorch_cpu.so: cannot open shared object file: No such file or directory

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 683, in load_function_uri
    package_name = str(Path(package).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 248, in compute
    preprocessor_func = load_function(uri)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 688, in load_function_uri
    raise NameError(f"Module {pkg} notfound, {e1}, {e2}")
NameError: Module ['mlmodels.model_tch.textcnn', 'split_train_valid'] notfound, libtorch_cpu.so: cannot open shared object file: No such file or directory, tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 672, in load_function_uri
    return  getattr(importlib.import_module(package), name)
AttributeError: module 'sentence_transformers.readers' has no attribute 'STSBenchmarkDataReader'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 683, in load_function_uri
    package_name = str(Path(package).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 275, in compute
    obj_preprocessor.compute(input_tmp)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/util_transformer.py", line 275, in compute
    test_func = load_function(self.test_reader)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 688, in load_function_uri
    raise NameError(f"Module {pkg} notfound, {e1}, {e2}")
NameError: Module ['sentence_transformers.readers', 'STSBenchmarkDataReader'] notfound, module 'sentence_transformers.readers' has no attribute 'STSBenchmarkDataReader', tuple index out of range





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/preprocess/generic.py --do test  

  #### Test unit Dataloader/Dataset   #################################### 

  


 #################### tf_dataset_download 

  tf_dataset_download mlmodels/preprocess/generic:tf_dataset_download {'train_samples': 500, 'test_samples': 500} 

  MNIST 

  Dataset Name is :  mnist 
WARNING:absl:Dataset mnist is hosted on GCS. It will automatically be downloaded to your
local data directory. If you'd instead prefer to read directly from our public
GCS bucket (recommended if you're running on GCP), you can instead set
data_dir=gs://tfds-data/datasets.

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.22 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.22 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  9.22 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.48 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.48 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.13 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.13 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.00 file/s]2020-08-18 12:08:13.071616: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-18 12:08:13.076012: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-08-18 12:08:13.076194: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ceefcac1e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-18 12:08:13.076213: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['mnist2', 'train', 'test', 'cifar10', 'fashion-mnist_small.npy', 'mnist_dataset_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:20, 470451.49it/s] 93%|█████████▎| 9175040/9912422 [00:00<00:01, 670591.87it/s]9920512it [00:00, 46551712.63it/s]                           
0it [00:00, ?it/s]32768it [00:00, 608454.65it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 463544.04it/s]1654784it [00:00, 12194580.00it/s]                         
0it [00:00, ?it/s]8192it [00:00, 247229.71it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/raw
Processing...
Done!

  


 #################### PandasDataset 

  PandasDataset mlmodels/preprocess/generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': 'ISO-8859-1', 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX'], 'encoding': 'ISO-8859-1'}} 

  


 #################### NumpyDataset 

  NumpyDataset mlmodels/preprocess/generic:NumpyDataset {'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'numpy_loader_args': {}} 

Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/train/mnist.npz
