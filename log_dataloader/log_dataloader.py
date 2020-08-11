
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fc20d999598> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fc20d999598> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fc278649488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fc278649488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fc297867ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fc297867ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fc225984158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fc225984158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fc225984158> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 19%|█▊        | 1843200/9912422 [00:00<00:00, 18429249.43it/s]9920512it [00:00, 27289145.55it/s]                             
0it [00:00, ?it/s]32768it [00:00, 540054.28it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 151948.85it/s]1654784it [00:00, 10498982.20it/s]                         
0it [00:00, ?it/s]8192it [00:00, 171817.59it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc22597c4a8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc223b2f9e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fc22597ad90> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fc22597ad90> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fc22597ad90> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:988: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:32,  1.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:32,  1.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:32,  1.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:31,  1.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:30,  1.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:30,  1.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:29,  1.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<01:03,  2.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:03,  2.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:02,  2.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:02,  2.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:02,  2.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:01,  2.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:01,  2.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:00,  2.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▊         | 14/162 [00:00<00:43,  3.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:43,  3.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:42,  3.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:42,  3.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:42,  3.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:41,  3.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:41,  3.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:00<00:29,  4.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:29,  4.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:29,  4.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:29,  4.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:29,  4.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:28,  4.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:28,  4.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:28,  4.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:01<00:20,  6.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:20,  6.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:20,  6.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:20,  6.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:19,  6.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:19,  6.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:19,  6.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:19,  6.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:01<00:14,  9.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:14,  9.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:13,  9.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:13,  9.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:13,  9.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:13,  9.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:13,  9.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:13,  9.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:01<00:09, 12.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:09, 12.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:09, 12.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:09, 12.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:09, 12.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:09, 12.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:09, 12.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:07, 15.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:07, 15.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:07, 15.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:07, 15.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:07, 15.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:06, 15.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 15.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 15.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 20.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 20.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 20.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 20.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 20.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 20.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 20.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 25.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 25.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 25.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 25.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 25.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 25.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 25.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 25.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 30.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 30.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 30.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 30.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 30.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 30.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 30.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 30.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 36.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 36.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 36.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 36.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 36.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 36.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 36.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 36.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 41.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 41.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 41.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 41.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 41.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 41.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 41.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 41.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 47.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 47.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 47.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 47.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 47.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 47.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 47.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 47.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 47.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 51.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 51.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 51.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 51.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 51.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 51.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 51.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 51.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 51.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 56.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 56.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 56.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:00, 56.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:00, 56.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 56.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 56.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 56.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 58.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 58.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 58.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 58.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 58.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 58.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 58.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 58.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 60.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 60.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 60.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 60.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 60.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 60.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 60.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 60.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 61.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 61.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 61.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 61.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 61.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 61.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 61.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 61.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 61.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 64.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 64.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 64.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 64.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 64.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 64.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 64.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 64.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 65.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 65.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 65.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 65.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 65.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 65.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 65.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 65.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 66.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 66.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 66.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 66.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 66.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 66.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 66.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 66.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 66.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 66.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 66.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 66.18 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 66.18 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 66.18 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 66.18 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 66.18 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 65.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 65.68 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 65.68 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.10s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.10s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 65.68 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.10s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 65.68 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.51s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  3.10s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 65.68 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.51s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.51s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 29.39 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.51s/ url]
0 examples [00:00, ? examples/s]2020-08-11 18:07:51.517347: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-11 18:07:51.531145: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-08-11 18:07:51.531343: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x563844fdc0a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-11 18:07:51.531364: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
5 examples [00:00, 49.92 examples/s]92 examples [00:00, 69.59 examples/s]180 examples [00:00, 96.14 examples/s]268 examples [00:00, 131.17 examples/s]360 examples [00:00, 176.51 examples/s]450 examples [00:00, 232.60 examples/s]540 examples [00:00, 299.15 examples/s]629 examples [00:00, 373.39 examples/s]715 examples [00:00, 449.19 examples/s]808 examples [00:01, 531.23 examples/s]900 examples [00:01, 608.22 examples/s]988 examples [00:01, 661.59 examples/s]1075 examples [00:01, 709.30 examples/s]1162 examples [00:01, 748.13 examples/s]1248 examples [00:01, 776.75 examples/s]1334 examples [00:01, 798.18 examples/s]1420 examples [00:01, 811.46 examples/s]1508 examples [00:01, 828.69 examples/s]1598 examples [00:01, 846.19 examples/s]1687 examples [00:02, 857.99 examples/s]1775 examples [00:02, 861.49 examples/s]1863 examples [00:02, 865.62 examples/s]1951 examples [00:02, 861.73 examples/s]2038 examples [00:02, 860.56 examples/s]2125 examples [00:02, 852.09 examples/s]2213 examples [00:02, 858.40 examples/s]2300 examples [00:02, 853.87 examples/s]2390 examples [00:02, 864.86 examples/s]2477 examples [00:02, 864.68 examples/s]2570 examples [00:03, 881.37 examples/s]2661 examples [00:03, 889.38 examples/s]2751 examples [00:03, 882.38 examples/s]2840 examples [00:03, 876.92 examples/s]2928 examples [00:03, 854.45 examples/s]3015 examples [00:03, 858.42 examples/s]3107 examples [00:03, 873.51 examples/s]3199 examples [00:03, 886.60 examples/s]3289 examples [00:03, 889.54 examples/s]3379 examples [00:03, 887.12 examples/s]3468 examples [00:04, 878.14 examples/s]3558 examples [00:04, 883.13 examples/s]3647 examples [00:04, 882.65 examples/s]3736 examples [00:04, 884.09 examples/s]3834 examples [00:04, 908.45 examples/s]3926 examples [00:04, 882.83 examples/s]4016 examples [00:04, 884.99 examples/s]4105 examples [00:04, 880.89 examples/s]4194 examples [00:04, 871.36 examples/s]4282 examples [00:04, 869.75 examples/s]4370 examples [00:05, 861.95 examples/s]4457 examples [00:05, 863.05 examples/s]4544 examples [00:05, 859.12 examples/s]4634 examples [00:05, 870.74 examples/s]4723 examples [00:05, 874.47 examples/s]4811 examples [00:05, 827.61 examples/s]4897 examples [00:05, 836.98 examples/s]4984 examples [00:05, 846.00 examples/s]5072 examples [00:05, 855.91 examples/s]5167 examples [00:06, 879.55 examples/s]5256 examples [00:06, 871.88 examples/s]5345 examples [00:06, 875.83 examples/s]5435 examples [00:06, 880.79 examples/s]5527 examples [00:06, 890.73 examples/s]5617 examples [00:06, 892.78 examples/s]5711 examples [00:06, 904.41 examples/s]5802 examples [00:06, 904.81 examples/s]5899 examples [00:06, 922.03 examples/s]5996 examples [00:06, 935.19 examples/s]6094 examples [00:07, 946.38 examples/s]6190 examples [00:07, 949.49 examples/s]6286 examples [00:07, 916.45 examples/s]6378 examples [00:07, 906.74 examples/s]6469 examples [00:07, 892.94 examples/s]6559 examples [00:07, 857.67 examples/s]6646 examples [00:07, 844.66 examples/s]6731 examples [00:07, 845.52 examples/s]6816 examples [00:07, 840.93 examples/s]6901 examples [00:07, 831.00 examples/s]6989 examples [00:08, 844.92 examples/s]7076 examples [00:08, 850.40 examples/s]7171 examples [00:08, 876.27 examples/s]7263 examples [00:08, 888.66 examples/s]7353 examples [00:08, 879.22 examples/s]7448 examples [00:08, 899.24 examples/s]7541 examples [00:08, 905.76 examples/s]7640 examples [00:08, 928.12 examples/s]7734 examples [00:08, 930.03 examples/s]7828 examples [00:08, 927.59 examples/s]7921 examples [00:09, 928.07 examples/s]8015 examples [00:09, 928.71 examples/s]8108 examples [00:09, 927.96 examples/s]8201 examples [00:09, 921.75 examples/s]8294 examples [00:09, 912.70 examples/s]8386 examples [00:09, 914.75 examples/s]8478 examples [00:09, 903.40 examples/s]8572 examples [00:09, 912.01 examples/s]8668 examples [00:09, 922.66 examples/s]8762 examples [00:09, 927.14 examples/s]8855 examples [00:10, 919.66 examples/s]8948 examples [00:10, 921.07 examples/s]9045 examples [00:10, 935.01 examples/s]9139 examples [00:10, 920.12 examples/s]9232 examples [00:10, 915.61 examples/s]9324 examples [00:10, 907.29 examples/s]9415 examples [00:10, 896.93 examples/s]9505 examples [00:10, 887.50 examples/s]9595 examples [00:10, 888.44 examples/s]9684 examples [00:11, 883.52 examples/s]9776 examples [00:11, 892.07 examples/s]9866 examples [00:11, 885.83 examples/s]9955 examples [00:11, 881.45 examples/s]10044 examples [00:11, 837.04 examples/s]10130 examples [00:11, 842.95 examples/s]10215 examples [00:11, 845.01 examples/s]10306 examples [00:11, 861.00 examples/s]10393 examples [00:11, 859.49 examples/s]10481 examples [00:11, 863.58 examples/s]10571 examples [00:12, 873.38 examples/s]10661 examples [00:12, 878.86 examples/s]10749 examples [00:12, 862.10 examples/s]10836 examples [00:12, 851.95 examples/s]10929 examples [00:12, 872.22 examples/s]11019 examples [00:12, 877.98 examples/s]11110 examples [00:12, 884.30 examples/s]11202 examples [00:12, 890.23 examples/s]11295 examples [00:12, 899.29 examples/s]11387 examples [00:12, 900.45 examples/s]11478 examples [00:13, 896.78 examples/s]11568 examples [00:13, 882.17 examples/s]11658 examples [00:13, 884.64 examples/s]11751 examples [00:13, 895.67 examples/s]11844 examples [00:13, 905.62 examples/s]11935 examples [00:13, 896.57 examples/s]12025 examples [00:13, 860.92 examples/s]12113 examples [00:13, 865.22 examples/s]12203 examples [00:13, 872.68 examples/s]12296 examples [00:14, 887.47 examples/s]12390 examples [00:14, 900.95 examples/s]12482 examples [00:14, 904.04 examples/s]12575 examples [00:14, 909.23 examples/s]12667 examples [00:14, 901.08 examples/s]12759 examples [00:14, 905.81 examples/s]12854 examples [00:14, 916.70 examples/s]12947 examples [00:14, 918.49 examples/s]13039 examples [00:14, 918.06 examples/s]13131 examples [00:14, 908.51 examples/s]13224 examples [00:15, 912.61 examples/s]13316 examples [00:15, 902.61 examples/s]13407 examples [00:15, 893.19 examples/s]13497 examples [00:15, 890.02 examples/s]13587 examples [00:15, 885.99 examples/s]13676 examples [00:15, 882.59 examples/s]13765 examples [00:15, 846.43 examples/s]13850 examples [00:15, 805.31 examples/s]13937 examples [00:15, 821.71 examples/s]14024 examples [00:15, 835.42 examples/s]14111 examples [00:16, 844.74 examples/s]14198 examples [00:16, 849.63 examples/s]14284 examples [00:16, 852.57 examples/s]14376 examples [00:16, 870.95 examples/s]14474 examples [00:16, 899.29 examples/s]14569 examples [00:16, 913.25 examples/s]14661 examples [00:16, 914.09 examples/s]14753 examples [00:16, 914.06 examples/s]14849 examples [00:16, 925.08 examples/s]14942 examples [00:16, 912.89 examples/s]15035 examples [00:17, 917.25 examples/s]15127 examples [00:17, 909.90 examples/s]15219 examples [00:17, 909.28 examples/s]15313 examples [00:17, 917.77 examples/s]15410 examples [00:17, 931.28 examples/s]15504 examples [00:17, 930.23 examples/s]15598 examples [00:17, 905.74 examples/s]15689 examples [00:17, 894.86 examples/s]15779 examples [00:17, 888.24 examples/s]15871 examples [00:18, 895.20 examples/s]15962 examples [00:18, 898.25 examples/s]16052 examples [00:18, 898.29 examples/s]16142 examples [00:18, 897.17 examples/s]16236 examples [00:18, 907.79 examples/s]16332 examples [00:18, 920.03 examples/s]16425 examples [00:18, 918.52 examples/s]16519 examples [00:18, 924.83 examples/s]16612 examples [00:18, 879.35 examples/s]16701 examples [00:18, 851.01 examples/s]16787 examples [00:19, 818.59 examples/s]16873 examples [00:19, 830.38 examples/s]16962 examples [00:19, 846.84 examples/s]17051 examples [00:19, 857.68 examples/s]17140 examples [00:19, 866.84 examples/s]17238 examples [00:19, 895.78 examples/s]17332 examples [00:19, 907.45 examples/s]17424 examples [00:19, 897.03 examples/s]17514 examples [00:19, 886.51 examples/s]17603 examples [00:19, 882.23 examples/s]17692 examples [00:20, 879.90 examples/s]17785 examples [00:20, 894.27 examples/s]17879 examples [00:20, 907.06 examples/s]17975 examples [00:20, 920.76 examples/s]18068 examples [00:20, 920.24 examples/s]18162 examples [00:20, 923.91 examples/s]18255 examples [00:20, 909.96 examples/s]18347 examples [00:20, 833.54 examples/s]18436 examples [00:20, 848.81 examples/s]18524 examples [00:21, 856.71 examples/s]18613 examples [00:21, 865.09 examples/s]18701 examples [00:21, 863.25 examples/s]18789 examples [00:21, 867.86 examples/s]18884 examples [00:21, 888.61 examples/s]18981 examples [00:21, 908.86 examples/s]19077 examples [00:21, 922.19 examples/s]19171 examples [00:21, 925.74 examples/s]19265 examples [00:21, 927.47 examples/s]19358 examples [00:21, 919.70 examples/s]19452 examples [00:22, 924.65 examples/s]19545 examples [00:22, 907.70 examples/s]19636 examples [00:22, 887.51 examples/s]19725 examples [00:22, 887.12 examples/s]19814 examples [00:22, 884.68 examples/s]19906 examples [00:22, 894.44 examples/s]19996 examples [00:22, 884.51 examples/s]20085 examples [00:22, 840.93 examples/s]20173 examples [00:22, 852.24 examples/s]20262 examples [00:22, 861.97 examples/s]20355 examples [00:23, 880.33 examples/s]20451 examples [00:23, 900.89 examples/s]20545 examples [00:23, 910.90 examples/s]20637 examples [00:23, 908.65 examples/s]20729 examples [00:23, 893.83 examples/s]20819 examples [00:23, 884.46 examples/s]20908 examples [00:23, 855.33 examples/s]20995 examples [00:23, 858.05 examples/s]21082 examples [00:23, 834.74 examples/s]21166 examples [00:24, 824.83 examples/s]21249 examples [00:24, 818.65 examples/s]21337 examples [00:24, 834.80 examples/s]21425 examples [00:24, 845.63 examples/s]21510 examples [00:24, 845.82 examples/s]21595 examples [00:24, 840.15 examples/s]21682 examples [00:24, 848.33 examples/s]21772 examples [00:24, 862.78 examples/s]21860 examples [00:24, 867.69 examples/s]21947 examples [00:24, 835.38 examples/s]22033 examples [00:25, 842.58 examples/s]22124 examples [00:25, 861.20 examples/s]22211 examples [00:25, 846.73 examples/s]22297 examples [00:25, 848.31 examples/s]22386 examples [00:25, 858.82 examples/s]22476 examples [00:25, 869.76 examples/s]22568 examples [00:25, 880.44 examples/s]22657 examples [00:25, 854.30 examples/s]22743 examples [00:25, 844.44 examples/s]22829 examples [00:25, 848.81 examples/s]22915 examples [00:26, 851.75 examples/s]23004 examples [00:26, 862.68 examples/s]23097 examples [00:26, 879.33 examples/s]23187 examples [00:26, 883.65 examples/s]23276 examples [00:26, 879.81 examples/s]23365 examples [00:26, 882.66 examples/s]23454 examples [00:26, 880.78 examples/s]23543 examples [00:26, 882.81 examples/s]23635 examples [00:26, 892.48 examples/s]23728 examples [00:26, 903.37 examples/s]23821 examples [00:27, 910.71 examples/s]23914 examples [00:27, 914.02 examples/s]24009 examples [00:27, 923.21 examples/s]24105 examples [00:27, 931.14 examples/s]24199 examples [00:27, 929.20 examples/s]24292 examples [00:27, 918.01 examples/s]24384 examples [00:27, 898.84 examples/s]24475 examples [00:27, 885.91 examples/s]24564 examples [00:27, 884.00 examples/s]24653 examples [00:27, 870.79 examples/s]24747 examples [00:28, 889.45 examples/s]24837 examples [00:28, 888.48 examples/s]24926 examples [00:28, 886.35 examples/s]25015 examples [00:28, 880.75 examples/s]25104 examples [00:28, 871.47 examples/s]25192 examples [00:28, 872.48 examples/s]25282 examples [00:28, 879.26 examples/s]25374 examples [00:28, 890.68 examples/s]25466 examples [00:28, 898.76 examples/s]25558 examples [00:28, 902.82 examples/s]25649 examples [00:29, 879.62 examples/s]25738 examples [00:29, 865.26 examples/s]25825 examples [00:29, 863.19 examples/s]25912 examples [00:29, 840.84 examples/s]25997 examples [00:29, 823.26 examples/s]26090 examples [00:29, 850.33 examples/s]26177 examples [00:29, 853.73 examples/s]26263 examples [00:29, 852.70 examples/s]26349 examples [00:29, 854.14 examples/s]26435 examples [00:30, 851.80 examples/s]26530 examples [00:30, 878.36 examples/s]26622 examples [00:30, 889.71 examples/s]26716 examples [00:30, 901.95 examples/s]26810 examples [00:30, 911.68 examples/s]26902 examples [00:30, 913.81 examples/s]26998 examples [00:30, 924.78 examples/s]27091 examples [00:30, 916.76 examples/s]27183 examples [00:30, 899.62 examples/s]27274 examples [00:30, 894.90 examples/s]27364 examples [00:31, 874.94 examples/s]27452 examples [00:31, 860.14 examples/s]27539 examples [00:31, 822.92 examples/s]27627 examples [00:31, 838.01 examples/s]27712 examples [00:31, 841.53 examples/s]27797 examples [00:31, 840.60 examples/s]27887 examples [00:31, 857.25 examples/s]27973 examples [00:31, 835.91 examples/s]28063 examples [00:31, 853.99 examples/s]28153 examples [00:31, 864.75 examples/s]28242 examples [00:32, 869.84 examples/s]28330 examples [00:32, 861.13 examples/s]28421 examples [00:32, 873.73 examples/s]28509 examples [00:32, 870.56 examples/s]28597 examples [00:32, 853.04 examples/s]28683 examples [00:32, 853.94 examples/s]28769 examples [00:32, 850.72 examples/s]28855 examples [00:32, 829.25 examples/s]28939 examples [00:32, 826.56 examples/s]29023 examples [00:33, 828.17 examples/s]29110 examples [00:33, 838.93 examples/s]29199 examples [00:33, 852.29 examples/s]29293 examples [00:33, 876.68 examples/s]29381 examples [00:33, 863.29 examples/s]29468 examples [00:33, 856.09 examples/s]29554 examples [00:33, 851.27 examples/s]29643 examples [00:33, 860.51 examples/s]29733 examples [00:33, 870.39 examples/s]29821 examples [00:33, 859.70 examples/s]29908 examples [00:34, 856.92 examples/s]29997 examples [00:34, 864.92 examples/s]30084 examples [00:34, 827.41 examples/s]30175 examples [00:34, 850.44 examples/s]30269 examples [00:34, 875.27 examples/s]30362 examples [00:34, 890.16 examples/s]30455 examples [00:34, 901.31 examples/s]30552 examples [00:34, 918.63 examples/s]30649 examples [00:34, 931.73 examples/s]30743 examples [00:34, 916.46 examples/s]30835 examples [00:35, 907.72 examples/s]30926 examples [00:35, 893.85 examples/s]31016 examples [00:35, 889.70 examples/s]31106 examples [00:35, 870.70 examples/s]31194 examples [00:35, 861.49 examples/s]31281 examples [00:35, 852.43 examples/s]31368 examples [00:35, 856.82 examples/s]31454 examples [00:35, 844.63 examples/s]31553 examples [00:35, 882.90 examples/s]31649 examples [00:36, 902.95 examples/s]31740 examples [00:36, 899.10 examples/s]31831 examples [00:36, 893.85 examples/s]31921 examples [00:36, 887.18 examples/s]32014 examples [00:36, 898.01 examples/s]32108 examples [00:36, 906.85 examples/s]32200 examples [00:36, 910.06 examples/s]32292 examples [00:36, 905.52 examples/s]32383 examples [00:36, 898.88 examples/s]32473 examples [00:36, 881.62 examples/s]32562 examples [00:37, 877.08 examples/s]32650 examples [00:37, 868.06 examples/s]32737 examples [00:37, 864.93 examples/s]32824 examples [00:37, 865.97 examples/s]32917 examples [00:37, 882.01 examples/s]33006 examples [00:37, 884.18 examples/s]33098 examples [00:37, 892.53 examples/s]33193 examples [00:37, 908.12 examples/s]33289 examples [00:37, 920.67 examples/s]33382 examples [00:37, 918.75 examples/s]33474 examples [00:38, 909.39 examples/s]33566 examples [00:38, 898.64 examples/s]33656 examples [00:38, 876.26 examples/s]33744 examples [00:38, 848.58 examples/s]33830 examples [00:38, 844.78 examples/s]33916 examples [00:38, 848.21 examples/s]34009 examples [00:38, 869.83 examples/s]34106 examples [00:38, 895.20 examples/s]34201 examples [00:38, 910.02 examples/s]34298 examples [00:38, 926.02 examples/s]34394 examples [00:39, 935.10 examples/s]34488 examples [00:39, 935.41 examples/s]34582 examples [00:39, 935.65 examples/s]34676 examples [00:39, 932.95 examples/s]34770 examples [00:39, 928.02 examples/s]34863 examples [00:39, 918.18 examples/s]34958 examples [00:39, 924.77 examples/s]35053 examples [00:39, 929.70 examples/s]35147 examples [00:39, 926.42 examples/s]35240 examples [00:39, 927.36 examples/s]35334 examples [00:40, 928.86 examples/s]35427 examples [00:40, 918.89 examples/s]35519 examples [00:40, 901.46 examples/s]35610 examples [00:40, 874.89 examples/s]35704 examples [00:40, 892.60 examples/s]35796 examples [00:40, 900.58 examples/s]35887 examples [00:40, 902.86 examples/s]35978 examples [00:40, 877.40 examples/s]36073 examples [00:40, 896.93 examples/s]36163 examples [00:41, 897.35 examples/s]36260 examples [00:41, 916.53 examples/s]36354 examples [00:41, 921.71 examples/s]36453 examples [00:41, 940.68 examples/s]36548 examples [00:41, 939.04 examples/s]36643 examples [00:41, 941.51 examples/s]36739 examples [00:41, 944.16 examples/s]36834 examples [00:41, 943.80 examples/s]36929 examples [00:41, 930.07 examples/s]37026 examples [00:41, 939.36 examples/s]37121 examples [00:42, 918.36 examples/s]37213 examples [00:42, 898.88 examples/s]37304 examples [00:42, 894.79 examples/s]37397 examples [00:42, 904.19 examples/s]37491 examples [00:42, 914.31 examples/s]37587 examples [00:42, 925.58 examples/s]37683 examples [00:42, 932.82 examples/s]37781 examples [00:42, 943.80 examples/s]37876 examples [00:42, 944.16 examples/s]37971 examples [00:42, 923.24 examples/s]38064 examples [00:43, 908.71 examples/s]38156 examples [00:43, 900.10 examples/s]38247 examples [00:43, 895.99 examples/s]38337 examples [00:43, 896.43 examples/s]38431 examples [00:43, 906.43 examples/s]38528 examples [00:43, 922.46 examples/s]38621 examples [00:43, 923.55 examples/s]38714 examples [00:43, 873.23 examples/s]38809 examples [00:43, 892.68 examples/s]38903 examples [00:44, 904.70 examples/s]39000 examples [00:44, 921.01 examples/s]39095 examples [00:44, 926.78 examples/s]39192 examples [00:44, 936.79 examples/s]39288 examples [00:44, 941.93 examples/s]39383 examples [00:44, 933.85 examples/s]39477 examples [00:44, 913.28 examples/s]39569 examples [00:44, 907.33 examples/s]39662 examples [00:44, 912.07 examples/s]39754 examples [00:44, 896.84 examples/s]39844 examples [00:45, 885.85 examples/s]39933 examples [00:45, 886.51 examples/s]40022 examples [00:45, 843.93 examples/s]40113 examples [00:45, 861.38 examples/s]40207 examples [00:45, 882.76 examples/s]40302 examples [00:45, 899.54 examples/s]40398 examples [00:45, 915.70 examples/s]40491 examples [00:45, 918.59 examples/s]40584 examples [00:45, 905.49 examples/s]40675 examples [00:45, 900.01 examples/s]40766 examples [00:46, 893.37 examples/s]40858 examples [00:46, 900.47 examples/s]40950 examples [00:46, 903.69 examples/s]41045 examples [00:46, 916.80 examples/s]41140 examples [00:46, 924.32 examples/s]41233 examples [00:46, 920.37 examples/s]41326 examples [00:46, 920.83 examples/s]41420 examples [00:46, 925.79 examples/s]41513 examples [00:46, 871.63 examples/s]41610 examples [00:47, 898.60 examples/s]41706 examples [00:47, 916.10 examples/s]41801 examples [00:47, 924.87 examples/s]41894 examples [00:47, 892.65 examples/s]41986 examples [00:47, 899.41 examples/s]42078 examples [00:47, 904.68 examples/s]42169 examples [00:47, 901.93 examples/s]42260 examples [00:47, 890.49 examples/s]42350 examples [00:47, 889.74 examples/s]42440 examples [00:47, 889.33 examples/s]42530 examples [00:48, 876.86 examples/s]42618 examples [00:48, 875.83 examples/s]42706 examples [00:48, 868.26 examples/s]42799 examples [00:48, 884.11 examples/s]42891 examples [00:48, 894.43 examples/s]42984 examples [00:48, 904.38 examples/s]43077 examples [00:48, 910.15 examples/s]43171 examples [00:48, 916.15 examples/s]43263 examples [00:48, 899.98 examples/s]43357 examples [00:48, 908.91 examples/s]43448 examples [00:49, 886.02 examples/s]43541 examples [00:49, 898.53 examples/s]43632 examples [00:49, 893.97 examples/s]43722 examples [00:49, 873.32 examples/s]43810 examples [00:49, 859.01 examples/s]43897 examples [00:49, 856.70 examples/s]43987 examples [00:49, 867.22 examples/s]44074 examples [00:49, 854.79 examples/s]44163 examples [00:49, 862.77 examples/s]44250 examples [00:49, 858.12 examples/s]44340 examples [00:50, 869.59 examples/s]44428 examples [00:50, 864.04 examples/s]44518 examples [00:50, 872.48 examples/s]44606 examples [00:50, 874.08 examples/s]44697 examples [00:50, 882.01 examples/s]44790 examples [00:50, 893.44 examples/s]44880 examples [00:50, 888.10 examples/s]44975 examples [00:50, 903.55 examples/s]45071 examples [00:50, 917.35 examples/s]45163 examples [00:50, 908.06 examples/s]45254 examples [00:51, 889.27 examples/s]45344 examples [00:51, 883.42 examples/s]45433 examples [00:51, 875.83 examples/s]45524 examples [00:51, 885.80 examples/s]45618 examples [00:51, 899.51 examples/s]45712 examples [00:51, 911.28 examples/s]45809 examples [00:51, 926.70 examples/s]45902 examples [00:51, 921.84 examples/s]45995 examples [00:51, 921.32 examples/s]46088 examples [00:52, 888.15 examples/s]46178 examples [00:52, 879.50 examples/s]46267 examples [00:52, 862.08 examples/s]46354 examples [00:52, 837.77 examples/s]46439 examples [00:52, 837.39 examples/s]46528 examples [00:52, 850.21 examples/s]46614 examples [00:52, 841.96 examples/s]46699 examples [00:52, 844.10 examples/s]46784 examples [00:52, 832.24 examples/s]46870 examples [00:52, 840.22 examples/s]46959 examples [00:53, 852.23 examples/s]47049 examples [00:53, 863.93 examples/s]47142 examples [00:53, 880.63 examples/s]47236 examples [00:53, 895.04 examples/s]47330 examples [00:53, 906.63 examples/s]47422 examples [00:53, 909.52 examples/s]47515 examples [00:53, 915.00 examples/s]47607 examples [00:53, 908.31 examples/s]47699 examples [00:53, 911.27 examples/s]47793 examples [00:53, 918.44 examples/s]47887 examples [00:54, 924.25 examples/s]47981 examples [00:54, 926.95 examples/s]48074 examples [00:54, 927.04 examples/s]48167 examples [00:54, 919.02 examples/s]48259 examples [00:54, 916.76 examples/s]48351 examples [00:54, 898.35 examples/s]48449 examples [00:54, 920.50 examples/s]48546 examples [00:54, 933.00 examples/s]48640 examples [00:54, 928.66 examples/s]48733 examples [00:54, 906.50 examples/s]48825 examples [00:55, 907.80 examples/s]48921 examples [00:55, 922.59 examples/s]49019 examples [00:55, 938.33 examples/s]49114 examples [00:55, 934.76 examples/s]49210 examples [00:55, 940.41 examples/s]49305 examples [00:55, 922.19 examples/s]49398 examples [00:55, 913.09 examples/s]49490 examples [00:55, 898.58 examples/s]49580 examples [00:55, 836.77 examples/s]49666 examples [00:56, 841.47 examples/s]49757 examples [00:56, 858.47 examples/s]49847 examples [00:56, 869.13 examples/s]49942 examples [00:56, 888.90 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▍        | 7253/50000 [00:00<00:00, 72528.51 examples/s] 39%|███▊      | 19371/50000 [00:00<00:00, 82460.30 examples/s] 62%|██████▏   | 30758/50000 [00:00<00:00, 89899.59 examples/s] 82%|████████▏ | 41234/50000 [00:00<00:00, 93894.00 examples/s]                                                               0 examples [00:00, ? examples/s]74 examples [00:00, 734.99 examples/s]162 examples [00:00, 771.70 examples/s]251 examples [00:00, 802.14 examples/s]341 examples [00:00, 828.65 examples/s]430 examples [00:00, 843.58 examples/s]520 examples [00:00, 859.02 examples/s]609 examples [00:00, 867.43 examples/s]705 examples [00:00, 892.73 examples/s]801 examples [00:00, 910.49 examples/s]890 examples [00:01, 897.65 examples/s]978 examples [00:01, 879.26 examples/s]1065 examples [00:01, 863.56 examples/s]1151 examples [00:01, 816.55 examples/s]1241 examples [00:01, 839.76 examples/s]1333 examples [00:01, 861.28 examples/s]1427 examples [00:01, 883.23 examples/s]1524 examples [00:01, 906.89 examples/s]1621 examples [00:01, 923.76 examples/s]1715 examples [00:01, 926.33 examples/s]1808 examples [00:02, 916.80 examples/s]1900 examples [00:02, 905.34 examples/s]1991 examples [00:02, 896.76 examples/s]2086 examples [00:02, 910.57 examples/s]2179 examples [00:02, 914.84 examples/s]2271 examples [00:02, 895.30 examples/s]2361 examples [00:02, 887.85 examples/s]2450 examples [00:02, 875.56 examples/s]2542 examples [00:02, 888.29 examples/s]2640 examples [00:02, 912.11 examples/s]2732 examples [00:03, 905.72 examples/s]2827 examples [00:03, 915.87 examples/s]2919 examples [00:03, 910.38 examples/s]3014 examples [00:03, 920.03 examples/s]3111 examples [00:03, 932.66 examples/s]3206 examples [00:03, 937.16 examples/s]3300 examples [00:03, 926.02 examples/s]3397 examples [00:03, 937.51 examples/s]3492 examples [00:03, 940.18 examples/s]3588 examples [00:03, 944.04 examples/s]3683 examples [00:04, 941.65 examples/s]3778 examples [00:04, 925.41 examples/s]3871 examples [00:04, 905.79 examples/s]3962 examples [00:04, 881.79 examples/s]4052 examples [00:04, 884.09 examples/s]4142 examples [00:04, 885.38 examples/s]4231 examples [00:04, 885.85 examples/s]4321 examples [00:04, 889.84 examples/s]4411 examples [00:04, 878.84 examples/s]4505 examples [00:05, 893.27 examples/s]4595 examples [00:05, 894.11 examples/s]4690 examples [00:05, 906.69 examples/s]4785 examples [00:05, 918.84 examples/s]4882 examples [00:05, 931.76 examples/s]4976 examples [00:05, 926.34 examples/s]5070 examples [00:05, 929.62 examples/s]5166 examples [00:05, 938.32 examples/s]5261 examples [00:05, 941.32 examples/s]5357 examples [00:05, 946.30 examples/s]5452 examples [00:06, 933.55 examples/s]5546 examples [00:06, 905.48 examples/s]5637 examples [00:06, 903.11 examples/s]5728 examples [00:06, 904.08 examples/s]5819 examples [00:06, 872.61 examples/s]5913 examples [00:06, 889.67 examples/s]6003 examples [00:06, 884.12 examples/s]6092 examples [00:06, 861.87 examples/s]6179 examples [00:06, 847.82 examples/s]6270 examples [00:06, 862.60 examples/s]6357 examples [00:07, 863.03 examples/s]6449 examples [00:07, 877.73 examples/s]6543 examples [00:07, 895.32 examples/s]6638 examples [00:07, 908.49 examples/s]6730 examples [00:07, 902.15 examples/s]6821 examples [00:07, 900.20 examples/s]6914 examples [00:07, 908.07 examples/s]7005 examples [00:07, 907.42 examples/s]7102 examples [00:07, 925.24 examples/s]7198 examples [00:07, 932.97 examples/s]7292 examples [00:08, 930.13 examples/s]7386 examples [00:08, 932.87 examples/s]7480 examples [00:08, 933.88 examples/s]7574 examples [00:08, 901.45 examples/s]7665 examples [00:08, 901.05 examples/s]7757 examples [00:08, 903.93 examples/s]7848 examples [00:08, 895.48 examples/s]7938 examples [00:08, 882.24 examples/s]8027 examples [00:08, 874.86 examples/s]8115 examples [00:09, 859.75 examples/s]8202 examples [00:09, 856.25 examples/s]8293 examples [00:09, 869.24 examples/s]8383 examples [00:09, 875.71 examples/s]8475 examples [00:09, 888.53 examples/s]8576 examples [00:09, 919.76 examples/s]8669 examples [00:09, 922.44 examples/s]8764 examples [00:09, 930.39 examples/s]8859 examples [00:09, 934.87 examples/s]8953 examples [00:09, 936.01 examples/s]9047 examples [00:10, 921.98 examples/s]9140 examples [00:10, 923.26 examples/s]9237 examples [00:10, 934.30 examples/s]9335 examples [00:10, 945.96 examples/s]9433 examples [00:10, 954.35 examples/s]9530 examples [00:10, 958.03 examples/s]9626 examples [00:10, 951.59 examples/s]9722 examples [00:10, 951.11 examples/s]9818 examples [00:10, 952.05 examples/s]9914 examples [00:10, 951.41 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteS3CC3V/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteS3CC3V/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fc225984158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fc225984158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fc225984158> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc2713610f0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc1e4a314e0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fc225984158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fc225984158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fc225984158> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc223b2f2b0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc223b2f048>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.02 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.02 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  9.02 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.70 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.70 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.07 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.07 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.44 file/s]2020-08-11 18:09:08.101365: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-11 18:09:08.105131: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-08-11 18:09:08.105305: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558925b24b70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-11 18:09:08.105323: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:09, 142185.39it/s] 88%|████████▊ | 8724480/9912422 [00:00<00:05, 202978.95it/s]9920512it [00:00, 44090342.18it/s]                           
0it [00:00, ?it/s]32768it [00:00, 438375.20it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 144564.84it/s]1654784it [00:00, 9840479.26it/s]                          
0it [00:00, ?it/s]8192it [00:00, 154591.85it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
