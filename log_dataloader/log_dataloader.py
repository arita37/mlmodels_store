
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/561e81504931f72fe00de6bf5a2e274589f15d68', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '561e81504931f72fe00de6bf5a2e274589f15d68', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/561e81504931f72fe00de6bf5a2e274589f15d68

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/561e81504931f72fe00de6bf5a2e274589f15d68

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/561e81504931f72fe00de6bf5a2e274589f15d68

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f2f06c216a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f2f06c216a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f2f721e90d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f2f721e90d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f2f8bf16ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f2f8bf16ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f2f1fa64950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f2f1fa64950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f2f1fa64950> , (data_info, **args) 

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
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 428, in load
    fid = open(os_fspath(file), "rb")
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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 30%|███       | 3022848/9912422 [00:00<00:00, 30225410.12it/s]9920512it [00:00, 33863805.59it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 291974.16it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 7693865.84it/s]          
0it [00:00, ?it/s]8192it [00:00, 205820.88it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f2f1cfa6a90>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f2f1cfaad30>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f2f1fa64598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f2f1fa64598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f2f1fa64598> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:986: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:34,  1.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:34,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:34,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:33,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:33,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:32,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:31,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<01:04,  2.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:04,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:04,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:03,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:03,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:03,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:02,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:02,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:01,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:01,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:43,  3.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:43,  3.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:42,  3.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:42,  3.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:42,  3.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:42,  3.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:41,  3.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:41,  3.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:00<00:29,  4.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:29,  4.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:29,  4.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:29,  4.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:28,  4.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:28,  4.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:28,  4.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:28,  4.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:27,  4.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:27,  4.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:01<00:19,  6.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:19,  6.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:19,  6.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:19,  6.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:19,  6.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:19,  6.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:18,  6.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:18,  6.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:18,  6.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:18,  6.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:01<00:13,  9.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:13,  9.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:13,  9.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:13,  9.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:12,  9.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:12,  9.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:12,  9.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:12,  9.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:12,  9.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:12,  9.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███       | 50/162 [00:01<00:09, 12.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:09, 12.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:08, 12.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:08, 12.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:08, 12.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:08, 12.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:08, 12.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:08, 12.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:08, 12.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 16.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 16.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 16.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 16.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 16.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 16.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 16.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 16.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 16.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 16.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 21.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 21.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 21.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 21.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 21.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 21.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 21.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 21.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 21.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 27.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 27.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 27.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 27.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 27.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 27.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 27.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 27.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 27.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 34.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 34.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 34.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 34.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 34.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 34.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 34.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 34.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 34.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 41.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 41.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 41.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 41.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 41.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 41.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 41.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 41.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 41.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 47.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 47.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 47.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 47.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 47.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 47.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 47.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 47.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 47.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 54.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 54.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 54.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 54.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 54.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 54.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 54.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 54.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 54.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 59.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 59.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 59.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 59.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 59.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 59.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 59.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 59.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 59.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 59.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 65.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 65.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 65.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 65.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 65.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 65.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 65.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 65.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 65.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 67.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 67.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 67.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 67.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 67.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 67.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 67.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 67.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 67.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 70.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 70.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 70.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 70.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 70.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 70.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 70.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 70.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 70.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 70.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 73.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 73.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 73.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 73.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 73.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 73.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 73.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 73.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 73.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 72.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 72.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 72.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 72.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 72.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 72.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 72.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.68s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.68s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 72.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.68s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 72.46 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.03s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.68s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 72.46 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.03s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.03s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 32.21 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.03s/ url]
0 examples [00:00, ? examples/s]2020-06-07 06:09:31.899867: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-07 06:09:31.914896: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-07 06:09:31.915127: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55693ce8a180 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-07 06:09:31.915148: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
18 examples [00:00, 179.50 examples/s]109 examples [00:00, 236.28 examples/s]199 examples [00:00, 303.39 examples/s]291 examples [00:00, 379.34 examples/s]382 examples [00:00, 459.66 examples/s]477 examples [00:00, 542.95 examples/s]571 examples [00:00, 621.39 examples/s]659 examples [00:00, 678.80 examples/s]743 examples [00:00, 680.37 examples/s]823 examples [00:01, 705.58 examples/s]912 examples [00:01, 750.82 examples/s]1002 examples [00:01, 788.16 examples/s]1094 examples [00:01, 823.25 examples/s]1185 examples [00:01, 845.49 examples/s]1279 examples [00:01, 867.26 examples/s]1372 examples [00:01, 883.75 examples/s]1464 examples [00:01, 893.68 examples/s]1557 examples [00:01, 902.24 examples/s]1649 examples [00:01, 904.44 examples/s]1742 examples [00:02, 909.43 examples/s]1834 examples [00:02, 911.27 examples/s]1926 examples [00:02, 907.19 examples/s]2017 examples [00:02, 890.87 examples/s]2107 examples [00:02, 869.59 examples/s]2195 examples [00:02, 870.29 examples/s]2283 examples [00:02, 849.29 examples/s]2371 examples [00:02, 857.39 examples/s]2464 examples [00:02, 875.83 examples/s]2555 examples [00:02, 885.51 examples/s]2644 examples [00:03, 873.98 examples/s]2737 examples [00:03, 889.03 examples/s]2830 examples [00:03, 900.57 examples/s]2921 examples [00:03, 902.09 examples/s]3012 examples [00:03, 891.42 examples/s]3102 examples [00:03, 881.02 examples/s]3191 examples [00:03, 874.56 examples/s]3279 examples [00:03, 861.46 examples/s]3366 examples [00:03, 850.63 examples/s]3452 examples [00:04, 850.79 examples/s]3542 examples [00:04, 862.89 examples/s]3633 examples [00:04, 875.55 examples/s]3724 examples [00:04, 884.36 examples/s]3815 examples [00:04, 889.85 examples/s]3905 examples [00:04, 881.42 examples/s]3995 examples [00:04, 886.61 examples/s]4084 examples [00:04, 882.17 examples/s]4175 examples [00:04, 889.63 examples/s]4266 examples [00:04, 893.65 examples/s]4356 examples [00:05, 880.36 examples/s]4445 examples [00:05, 874.62 examples/s]4533 examples [00:05, 868.31 examples/s]4622 examples [00:05, 872.37 examples/s]4710 examples [00:05, 851.63 examples/s]4796 examples [00:05, 836.89 examples/s]4883 examples [00:05, 846.32 examples/s]4977 examples [00:05, 870.14 examples/s]5071 examples [00:05, 889.17 examples/s]5161 examples [00:05, 876.66 examples/s]5249 examples [00:06, 858.75 examples/s]5338 examples [00:06, 866.73 examples/s]5429 examples [00:06, 877.36 examples/s]5525 examples [00:06, 898.58 examples/s]5616 examples [00:06, 899.84 examples/s]5707 examples [00:06, 890.25 examples/s]5797 examples [00:06, 878.17 examples/s]5888 examples [00:06, 887.41 examples/s]5977 examples [00:06, 877.66 examples/s]6072 examples [00:06, 895.72 examples/s]6165 examples [00:07, 903.93 examples/s]6259 examples [00:07, 913.78 examples/s]6354 examples [00:07, 922.83 examples/s]6447 examples [00:07, 920.87 examples/s]6540 examples [00:07, 908.18 examples/s]6631 examples [00:07, 901.17 examples/s]6722 examples [00:07, 877.21 examples/s]6810 examples [00:07, 873.03 examples/s]6902 examples [00:07, 884.11 examples/s]6991 examples [00:07, 882.18 examples/s]7080 examples [00:08, 883.20 examples/s]7170 examples [00:08, 886.05 examples/s]7259 examples [00:08, 879.85 examples/s]7348 examples [00:08, 878.68 examples/s]7436 examples [00:08, 833.00 examples/s]7526 examples [00:08, 851.05 examples/s]7613 examples [00:08, 855.84 examples/s]7699 examples [00:08, 856.89 examples/s]7785 examples [00:08, 848.73 examples/s]7872 examples [00:09, 854.57 examples/s]7963 examples [00:09, 868.57 examples/s]8052 examples [00:09, 872.56 examples/s]8141 examples [00:09, 874.37 examples/s]8229 examples [00:09, 873.02 examples/s]8317 examples [00:09, 872.92 examples/s]8405 examples [00:09, 860.42 examples/s]8496 examples [00:09, 872.20 examples/s]8584 examples [00:09, 863.02 examples/s]8671 examples [00:09, 834.24 examples/s]8757 examples [00:10, 839.18 examples/s]8847 examples [00:10, 855.65 examples/s]8933 examples [00:10, 851.39 examples/s]9025 examples [00:10, 868.86 examples/s]9115 examples [00:10, 876.80 examples/s]9206 examples [00:10, 884.73 examples/s]9299 examples [00:10, 895.62 examples/s]9391 examples [00:10, 901.87 examples/s]9484 examples [00:10, 908.65 examples/s]9575 examples [00:10, 908.49 examples/s]9666 examples [00:11, 869.88 examples/s]9754 examples [00:11, 850.19 examples/s]9845 examples [00:11, 865.94 examples/s]9940 examples [00:11, 887.37 examples/s]10030 examples [00:11, 830.58 examples/s]10115 examples [00:11, 818.58 examples/s]10204 examples [00:11, 837.09 examples/s]10297 examples [00:11, 862.67 examples/s]10387 examples [00:11, 871.88 examples/s]10477 examples [00:12, 877.68 examples/s]10566 examples [00:12, 862.46 examples/s]10655 examples [00:12, 869.86 examples/s]10743 examples [00:12, 856.82 examples/s]10831 examples [00:12, 862.65 examples/s]10918 examples [00:12, 862.18 examples/s]11005 examples [00:12, 855.59 examples/s]11092 examples [00:12, 857.60 examples/s]11180 examples [00:12, 862.53 examples/s]11267 examples [00:12, 850.09 examples/s]11358 examples [00:13, 865.61 examples/s]11446 examples [00:13, 868.78 examples/s]11538 examples [00:13, 883.16 examples/s]11627 examples [00:13, 875.29 examples/s]11715 examples [00:13, 874.87 examples/s]11805 examples [00:13, 881.66 examples/s]11896 examples [00:13, 889.88 examples/s]11986 examples [00:13, 888.04 examples/s]12075 examples [00:13, 865.68 examples/s]12162 examples [00:13, 852.53 examples/s]12254 examples [00:14, 870.95 examples/s]12342 examples [00:14, 863.06 examples/s]12436 examples [00:14, 883.76 examples/s]12532 examples [00:14, 901.63 examples/s]12627 examples [00:14, 914.67 examples/s]12719 examples [00:14, 888.69 examples/s]12809 examples [00:14, 882.37 examples/s]12904 examples [00:14, 900.80 examples/s]12995 examples [00:14, 887.60 examples/s]13087 examples [00:15, 896.36 examples/s]13177 examples [00:15, 879.95 examples/s]13266 examples [00:15, 874.91 examples/s]13355 examples [00:15, 878.63 examples/s]13443 examples [00:15, 869.26 examples/s]13534 examples [00:15, 880.90 examples/s]13627 examples [00:15, 892.43 examples/s]13724 examples [00:15, 914.07 examples/s]13820 examples [00:15, 925.04 examples/s]13913 examples [00:15, 911.90 examples/s]14005 examples [00:16, 910.90 examples/s]14097 examples [00:16, 903.75 examples/s]14188 examples [00:16, 844.82 examples/s]14275 examples [00:16, 851.20 examples/s]14365 examples [00:16, 865.25 examples/s]14455 examples [00:16, 872.70 examples/s]14544 examples [00:16, 876.30 examples/s]14633 examples [00:16, 878.49 examples/s]14722 examples [00:16, 871.09 examples/s]14815 examples [00:16, 885.49 examples/s]14907 examples [00:17, 893.63 examples/s]14997 examples [00:17, 847.05 examples/s]15090 examples [00:17, 868.40 examples/s]15179 examples [00:17, 872.87 examples/s]15275 examples [00:17, 894.76 examples/s]15365 examples [00:17, 889.83 examples/s]15455 examples [00:17, 867.05 examples/s]15543 examples [00:17, 859.30 examples/s]15630 examples [00:17, 821.98 examples/s]15713 examples [00:18, 803.21 examples/s]15804 examples [00:18, 832.07 examples/s]15888 examples [00:18, 833.44 examples/s]15972 examples [00:18, 835.07 examples/s]16062 examples [00:18, 852.06 examples/s]16152 examples [00:18, 863.19 examples/s]16244 examples [00:18, 877.57 examples/s]16332 examples [00:18, 876.28 examples/s]16422 examples [00:18, 882.10 examples/s]16511 examples [00:18, 873.76 examples/s]16601 examples [00:19, 880.93 examples/s]16690 examples [00:19, 876.65 examples/s]16778 examples [00:19, 870.69 examples/s]16868 examples [00:19, 877.97 examples/s]16956 examples [00:19, 858.49 examples/s]17045 examples [00:19, 865.98 examples/s]17132 examples [00:19, 813.35 examples/s]17215 examples [00:19, 807.00 examples/s]17305 examples [00:19, 830.94 examples/s]17395 examples [00:19, 849.96 examples/s]17489 examples [00:20, 874.29 examples/s]17577 examples [00:20, 869.32 examples/s]17665 examples [00:20, 868.47 examples/s]17757 examples [00:20, 882.10 examples/s]17846 examples [00:20, 878.90 examples/s]17935 examples [00:20, 868.22 examples/s]18027 examples [00:20, 880.56 examples/s]18116 examples [00:20, 827.24 examples/s]18208 examples [00:20, 852.10 examples/s]18300 examples [00:21, 869.92 examples/s]18390 examples [00:21, 876.65 examples/s]18479 examples [00:21, 862.22 examples/s]18566 examples [00:21, 846.61 examples/s]18655 examples [00:21, 856.91 examples/s]18747 examples [00:21, 873.29 examples/s]18838 examples [00:21, 883.72 examples/s]18927 examples [00:21, 882.80 examples/s]19017 examples [00:21, 886.97 examples/s]19106 examples [00:21, 861.66 examples/s]19198 examples [00:22, 876.57 examples/s]19286 examples [00:22, 872.05 examples/s]19374 examples [00:22, 872.27 examples/s]19470 examples [00:22, 894.41 examples/s]19566 examples [00:22, 911.14 examples/s]19660 examples [00:22, 918.30 examples/s]19753 examples [00:22, 909.38 examples/s]19845 examples [00:22, 911.54 examples/s]19937 examples [00:22, 907.66 examples/s]20028 examples [00:22, 845.74 examples/s]20114 examples [00:23, 832.36 examples/s]20198 examples [00:23, 834.63 examples/s]20290 examples [00:23, 857.41 examples/s]20386 examples [00:23, 883.52 examples/s]20475 examples [00:23, 874.35 examples/s]20574 examples [00:23, 905.23 examples/s]20671 examples [00:23, 921.61 examples/s]20764 examples [00:23, 888.14 examples/s]20854 examples [00:23, 888.15 examples/s]20944 examples [00:24, 876.78 examples/s]21035 examples [00:24, 883.99 examples/s]21124 examples [00:24, 882.38 examples/s]21213 examples [00:24, 878.26 examples/s]21307 examples [00:24, 894.53 examples/s]21399 examples [00:24, 901.57 examples/s]21491 examples [00:24, 906.90 examples/s]21582 examples [00:24, 880.58 examples/s]21671 examples [00:24, 847.77 examples/s]21757 examples [00:24, 835.07 examples/s]21847 examples [00:25, 853.26 examples/s]21934 examples [00:25, 857.81 examples/s]22021 examples [00:25, 847.88 examples/s]22106 examples [00:25, 824.77 examples/s]22199 examples [00:25, 853.44 examples/s]22287 examples [00:25, 859.30 examples/s]22382 examples [00:25, 883.53 examples/s]22477 examples [00:25, 900.08 examples/s]22568 examples [00:25, 899.75 examples/s]22661 examples [00:25, 908.15 examples/s]22753 examples [00:26, 902.67 examples/s]22847 examples [00:26, 911.79 examples/s]22939 examples [00:26, 898.32 examples/s]23029 examples [00:26, 897.77 examples/s]23119 examples [00:26, 860.47 examples/s]23206 examples [00:26, 836.87 examples/s]23293 examples [00:26, 844.08 examples/s]23380 examples [00:26, 848.41 examples/s]23466 examples [00:26, 837.84 examples/s]23550 examples [00:27, 828.39 examples/s]23635 examples [00:27, 834.30 examples/s]23724 examples [00:27, 848.48 examples/s]23809 examples [00:27, 843.54 examples/s]23895 examples [00:27, 846.32 examples/s]23991 examples [00:27, 876.13 examples/s]24079 examples [00:27, 873.30 examples/s]24168 examples [00:27, 872.56 examples/s]24256 examples [00:27, 873.20 examples/s]24344 examples [00:27, 868.87 examples/s]24435 examples [00:28, 878.18 examples/s]24523 examples [00:28, 847.67 examples/s]24613 examples [00:28, 861.96 examples/s]24700 examples [00:28, 846.37 examples/s]24791 examples [00:28, 863.46 examples/s]24880 examples [00:28, 869.83 examples/s]24971 examples [00:28, 880.10 examples/s]25065 examples [00:28, 895.40 examples/s]25155 examples [00:28, 888.41 examples/s]25246 examples [00:28, 893.30 examples/s]25336 examples [00:29, 894.73 examples/s]25430 examples [00:29, 906.05 examples/s]25524 examples [00:29, 915.82 examples/s]25616 examples [00:29, 901.19 examples/s]25708 examples [00:29, 905.51 examples/s]25799 examples [00:29, 901.70 examples/s]25893 examples [00:29, 910.74 examples/s]25985 examples [00:29, 912.58 examples/s]26077 examples [00:29, 878.05 examples/s]26166 examples [00:29, 878.76 examples/s]26255 examples [00:30, 865.92 examples/s]26342 examples [00:30, 854.43 examples/s]26430 examples [00:30, 860.79 examples/s]26517 examples [00:30, 827.22 examples/s]26605 examples [00:30, 841.88 examples/s]26694 examples [00:30, 855.74 examples/s]26786 examples [00:30, 871.35 examples/s]26880 examples [00:30, 890.25 examples/s]26970 examples [00:30, 871.83 examples/s]27059 examples [00:31, 876.16 examples/s]27150 examples [00:31, 885.14 examples/s]27239 examples [00:31, 857.23 examples/s]27328 examples [00:31, 866.60 examples/s]27415 examples [00:31, 854.39 examples/s]27501 examples [00:31, 811.25 examples/s]27586 examples [00:31, 820.28 examples/s]27678 examples [00:31, 847.80 examples/s]27771 examples [00:31, 869.77 examples/s]27861 examples [00:31, 877.66 examples/s]27953 examples [00:32, 889.78 examples/s]28043 examples [00:32, 885.94 examples/s]28134 examples [00:32, 890.97 examples/s]28224 examples [00:32, 891.63 examples/s]28314 examples [00:32, 877.89 examples/s]28402 examples [00:32, 877.54 examples/s]28490 examples [00:32, 856.26 examples/s]28578 examples [00:32, 862.38 examples/s]28665 examples [00:32, 857.15 examples/s]28752 examples [00:32, 857.99 examples/s]28847 examples [00:33, 878.31 examples/s]28935 examples [00:33, 859.27 examples/s]29022 examples [00:33, 842.51 examples/s]29115 examples [00:33, 864.43 examples/s]29206 examples [00:33, 877.51 examples/s]29298 examples [00:33, 886.92 examples/s]29391 examples [00:33, 898.83 examples/s]29482 examples [00:33, 843.35 examples/s]29568 examples [00:33, 805.19 examples/s]29657 examples [00:34, 828.23 examples/s]29749 examples [00:34, 851.76 examples/s]29840 examples [00:34, 867.65 examples/s]29933 examples [00:34, 883.67 examples/s]30022 examples [00:34, 823.19 examples/s]30115 examples [00:34, 851.70 examples/s]30208 examples [00:34, 871.97 examples/s]30299 examples [00:34, 882.74 examples/s]30388 examples [00:34, 833.51 examples/s]30475 examples [00:34, 842.48 examples/s]30567 examples [00:35, 863.11 examples/s]30656 examples [00:35, 868.73 examples/s]30746 examples [00:35, 876.96 examples/s]30835 examples [00:35, 874.89 examples/s]30923 examples [00:35, 869.24 examples/s]31011 examples [00:35, 829.82 examples/s]31103 examples [00:35, 853.17 examples/s]31195 examples [00:35, 870.38 examples/s]31283 examples [00:35, 861.14 examples/s]31370 examples [00:36, 845.85 examples/s]31466 examples [00:36, 874.84 examples/s]31556 examples [00:36, 879.88 examples/s]31645 examples [00:36, 870.27 examples/s]31736 examples [00:36, 880.93 examples/s]31825 examples [00:36, 834.40 examples/s]31915 examples [00:36, 851.19 examples/s]32003 examples [00:36, 857.52 examples/s]32094 examples [00:36, 870.06 examples/s]32182 examples [00:36, 857.17 examples/s]32273 examples [00:37, 871.94 examples/s]32363 examples [00:37, 874.61 examples/s]32451 examples [00:37, 872.81 examples/s]32539 examples [00:37, 872.26 examples/s]32629 examples [00:37, 879.18 examples/s]32720 examples [00:37, 887.39 examples/s]32809 examples [00:37, 868.07 examples/s]32901 examples [00:37, 881.67 examples/s]32990 examples [00:37, 884.04 examples/s]33079 examples [00:37, 871.05 examples/s]33169 examples [00:38, 878.76 examples/s]33257 examples [00:38, 852.99 examples/s]33349 examples [00:38, 870.55 examples/s]33437 examples [00:38, 862.43 examples/s]33532 examples [00:38, 886.14 examples/s]33626 examples [00:38, 899.89 examples/s]33719 examples [00:38, 907.80 examples/s]33810 examples [00:38, 905.93 examples/s]33901 examples [00:38, 898.68 examples/s]33991 examples [00:39, 873.81 examples/s]34082 examples [00:39, 883.02 examples/s]34171 examples [00:39, 882.07 examples/s]34264 examples [00:39, 893.58 examples/s]34354 examples [00:39, 886.63 examples/s]34449 examples [00:39, 902.60 examples/s]34543 examples [00:39, 911.65 examples/s]34638 examples [00:39, 920.67 examples/s]34733 examples [00:39, 927.15 examples/s]34826 examples [00:39, 880.14 examples/s]34915 examples [00:40, 866.09 examples/s]35008 examples [00:40, 883.70 examples/s]35102 examples [00:40, 899.40 examples/s]35195 examples [00:40, 905.66 examples/s]35291 examples [00:40, 918.70 examples/s]35384 examples [00:40, 906.91 examples/s]35477 examples [00:40, 911.03 examples/s]35569 examples [00:40, 910.54 examples/s]35661 examples [00:40, 911.65 examples/s]35753 examples [00:40, 909.97 examples/s]35845 examples [00:41, 902.83 examples/s]35937 examples [00:41, 907.50 examples/s]36028 examples [00:41, 896.64 examples/s]36118 examples [00:41, 888.10 examples/s]36208 examples [00:41, 890.50 examples/s]36298 examples [00:41, 864.54 examples/s]36385 examples [00:41, 865.03 examples/s]36481 examples [00:41, 884.87 examples/s]36572 examples [00:41, 889.54 examples/s]36662 examples [00:41, 862.61 examples/s]36749 examples [00:42, 855.80 examples/s]36835 examples [00:42, 856.12 examples/s]36921 examples [00:42, 845.99 examples/s]37006 examples [00:42, 834.03 examples/s]37090 examples [00:42, 801.85 examples/s]37180 examples [00:42, 827.48 examples/s]37267 examples [00:42, 837.97 examples/s]37359 examples [00:42, 859.75 examples/s]37446 examples [00:42, 857.10 examples/s]37532 examples [00:43, 850.76 examples/s]37624 examples [00:43, 867.86 examples/s]37712 examples [00:43, 854.57 examples/s]37798 examples [00:43, 821.41 examples/s]37882 examples [00:43, 825.49 examples/s]37970 examples [00:43, 838.71 examples/s]38063 examples [00:43, 862.20 examples/s]38151 examples [00:43, 865.09 examples/s]38238 examples [00:43, 844.55 examples/s]38331 examples [00:43, 866.70 examples/s]38421 examples [00:44, 875.50 examples/s]38513 examples [00:44, 886.66 examples/s]38604 examples [00:44, 892.12 examples/s]38694 examples [00:44, 888.11 examples/s]38788 examples [00:44, 900.39 examples/s]38882 examples [00:44, 911.30 examples/s]38975 examples [00:44, 914.51 examples/s]39067 examples [00:44, 907.69 examples/s]39161 examples [00:44, 915.45 examples/s]39253 examples [00:44, 899.92 examples/s]39344 examples [00:45, 874.78 examples/s]39433 examples [00:45, 878.57 examples/s]39527 examples [00:45, 894.37 examples/s]39617 examples [00:45, 889.71 examples/s]39707 examples [00:45, 880.62 examples/s]39796 examples [00:45, 860.74 examples/s]39886 examples [00:45, 870.09 examples/s]39975 examples [00:45, 875.09 examples/s]40063 examples [00:45, 813.73 examples/s]40153 examples [00:46, 836.76 examples/s]40238 examples [00:46, 829.93 examples/s]40325 examples [00:46, 839.50 examples/s]40410 examples [00:46, 832.74 examples/s]40498 examples [00:46, 845.45 examples/s]40591 examples [00:46, 868.93 examples/s]40681 examples [00:46, 877.66 examples/s]40770 examples [00:46, 873.61 examples/s]40859 examples [00:46, 877.70 examples/s]40947 examples [00:46, 873.27 examples/s]41036 examples [00:47, 875.74 examples/s]41124 examples [00:47, 869.84 examples/s]41212 examples [00:47, 857.96 examples/s]41300 examples [00:47, 863.45 examples/s]41387 examples [00:47, 862.23 examples/s]41474 examples [00:47, 864.43 examples/s]41561 examples [00:47, 861.10 examples/s]41652 examples [00:47, 873.23 examples/s]41741 examples [00:47, 876.73 examples/s]41829 examples [00:47, 863.89 examples/s]41916 examples [00:48, 857.39 examples/s]42002 examples [00:48, 856.10 examples/s]42089 examples [00:48, 858.66 examples/s]42175 examples [00:48, 829.87 examples/s]42259 examples [00:48, 819.32 examples/s]42342 examples [00:48, 804.73 examples/s]42431 examples [00:48, 826.26 examples/s]42515 examples [00:48, 829.66 examples/s]42604 examples [00:48, 846.70 examples/s]42692 examples [00:49, 856.23 examples/s]42779 examples [00:49, 859.69 examples/s]42866 examples [00:49, 862.32 examples/s]42956 examples [00:49, 870.73 examples/s]43048 examples [00:49, 883.75 examples/s]43145 examples [00:49, 906.96 examples/s]43241 examples [00:49, 921.54 examples/s]43334 examples [00:49, 919.08 examples/s]43430 examples [00:49, 929.81 examples/s]43524 examples [00:49, 906.34 examples/s]43615 examples [00:50, 899.55 examples/s]43706 examples [00:50, 890.18 examples/s]43796 examples [00:50, 868.74 examples/s]43884 examples [00:50, 858.87 examples/s]43971 examples [00:50, 848.67 examples/s]44057 examples [00:50, 844.60 examples/s]44143 examples [00:50, 849.00 examples/s]44228 examples [00:50, 797.23 examples/s]44315 examples [00:50, 817.52 examples/s]44405 examples [00:50, 837.90 examples/s]44496 examples [00:51, 855.90 examples/s]44583 examples [00:51, 850.31 examples/s]44669 examples [00:51, 852.29 examples/s]44758 examples [00:51, 861.54 examples/s]44847 examples [00:51, 869.61 examples/s]44939 examples [00:51, 882.38 examples/s]45028 examples [00:51, 844.44 examples/s]45113 examples [00:51, 819.74 examples/s]45202 examples [00:51, 837.27 examples/s]45287 examples [00:52, 840.98 examples/s]45378 examples [00:52, 859.66 examples/s]45466 examples [00:52, 864.24 examples/s]45557 examples [00:52, 877.39 examples/s]45645 examples [00:52, 877.23 examples/s]45733 examples [00:52, 868.66 examples/s]45825 examples [00:52, 879.53 examples/s]45914 examples [00:52, 859.01 examples/s]46004 examples [00:52, 868.92 examples/s]46094 examples [00:52, 877.52 examples/s]46185 examples [00:53, 884.75 examples/s]46283 examples [00:53, 910.12 examples/s]46375 examples [00:53, 906.11 examples/s]46466 examples [00:53, 906.67 examples/s]46557 examples [00:53, 867.55 examples/s]46645 examples [00:53, 844.29 examples/s]46731 examples [00:53, 847.11 examples/s]46819 examples [00:53, 854.69 examples/s]46905 examples [00:53, 819.33 examples/s]46994 examples [00:53, 836.33 examples/s]47084 examples [00:54, 853.93 examples/s]47173 examples [00:54, 862.34 examples/s]47262 examples [00:54, 869.04 examples/s]47357 examples [00:54, 890.33 examples/s]47447 examples [00:54, 888.84 examples/s]47537 examples [00:54, 889.65 examples/s]47627 examples [00:54, 874.06 examples/s]47717 examples [00:54, 881.33 examples/s]47807 examples [00:54, 884.98 examples/s]47896 examples [00:54, 877.71 examples/s]47985 examples [00:55, 879.90 examples/s]48074 examples [00:55, 846.50 examples/s]48159 examples [00:55, 817.41 examples/s]48242 examples [00:55, 808.19 examples/s]48335 examples [00:55, 840.07 examples/s]48429 examples [00:55, 867.53 examples/s]48521 examples [00:55, 882.36 examples/s]48610 examples [00:55, 871.88 examples/s]48698 examples [00:55, 869.92 examples/s]48787 examples [00:56, 874.55 examples/s]48878 examples [00:56, 884.84 examples/s]48967 examples [00:56, 885.59 examples/s]49057 examples [00:56, 889.09 examples/s]49149 examples [00:56, 896.29 examples/s]49239 examples [00:56, 874.93 examples/s]49327 examples [00:56, 873.16 examples/s]49420 examples [00:56, 887.00 examples/s]49509 examples [00:56, 839.13 examples/s]49594 examples [00:56, 829.82 examples/s]49681 examples [00:57, 839.65 examples/s]49769 examples [00:57, 850.40 examples/s]49860 examples [00:57, 865.78 examples/s]49947 examples [00:57, 858.37 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 11%|█         | 5605/50000 [00:00<00:00, 56048.05 examples/s] 33%|███▎      | 16369/50000 [00:00<00:00, 65459.07 examples/s] 54%|█████▍    | 27094/50000 [00:00<00:00, 74123.75 examples/s] 75%|███████▌  | 37604/50000 [00:00<00:00, 81312.28 examples/s] 98%|█████████▊| 48919/50000 [00:00<00:00, 88808.47 examples/s]                                                               0 examples [00:00, ? examples/s]67 examples [00:00, 665.16 examples/s]162 examples [00:00, 729.68 examples/s]256 examples [00:00, 781.34 examples/s]348 examples [00:00, 818.01 examples/s]443 examples [00:00, 850.79 examples/s]528 examples [00:00, 849.52 examples/s]617 examples [00:00, 859.67 examples/s]699 examples [00:00, 659.28 examples/s]795 examples [00:00, 726.75 examples/s]889 examples [00:01, 779.48 examples/s]979 examples [00:01, 811.37 examples/s]1069 examples [00:01, 834.76 examples/s]1157 examples [00:01, 845.30 examples/s]1250 examples [00:01, 867.19 examples/s]1345 examples [00:01, 888.42 examples/s]1440 examples [00:01, 905.17 examples/s]1536 examples [00:01, 919.39 examples/s]1629 examples [00:01, 916.38 examples/s]1722 examples [00:02, 911.12 examples/s]1816 examples [00:02, 918.70 examples/s]1909 examples [00:02, 878.68 examples/s]1999 examples [00:02, 883.36 examples/s]2088 examples [00:02, 870.05 examples/s]2178 examples [00:02, 877.79 examples/s]2272 examples [00:02, 892.72 examples/s]2368 examples [00:02, 911.58 examples/s]2460 examples [00:02, 911.75 examples/s]2552 examples [00:02, 899.82 examples/s]2643 examples [00:03, 893.82 examples/s]2737 examples [00:03, 905.66 examples/s]2831 examples [00:03, 914.84 examples/s]2924 examples [00:03, 918.54 examples/s]3016 examples [00:03, 898.79 examples/s]3107 examples [00:03, 892.65 examples/s]3198 examples [00:03, 897.41 examples/s]3288 examples [00:03, 890.14 examples/s]3379 examples [00:03, 894.21 examples/s]3469 examples [00:03, 878.58 examples/s]3557 examples [00:04, 864.77 examples/s]3647 examples [00:04, 874.29 examples/s]3735 examples [00:04, 875.48 examples/s]3827 examples [00:04, 886.35 examples/s]3916 examples [00:04, 884.48 examples/s]4006 examples [00:04, 885.78 examples/s]4096 examples [00:04, 889.86 examples/s]4190 examples [00:04, 904.32 examples/s]4284 examples [00:04, 913.42 examples/s]4376 examples [00:04, 900.04 examples/s]4467 examples [00:05, 897.82 examples/s]4559 examples [00:05, 903.81 examples/s]4651 examples [00:05, 907.27 examples/s]4742 examples [00:05, 893.44 examples/s]4832 examples [00:05, 882.04 examples/s]4921 examples [00:05, 873.80 examples/s]5009 examples [00:05, 852.55 examples/s]5099 examples [00:05, 866.09 examples/s]5186 examples [00:05, 853.86 examples/s]5272 examples [00:06, 855.33 examples/s]5367 examples [00:06, 879.97 examples/s]5461 examples [00:06, 895.31 examples/s]5559 examples [00:06, 917.29 examples/s]5652 examples [00:06, 919.52 examples/s]5745 examples [00:06, 907.23 examples/s]5840 examples [00:06, 918.79 examples/s]5936 examples [00:06, 930.05 examples/s]6030 examples [00:06, 929.86 examples/s]6124 examples [00:06, 930.84 examples/s]6218 examples [00:07, 924.86 examples/s]6313 examples [00:07, 931.64 examples/s]6407 examples [00:07, 926.45 examples/s]6500 examples [00:07, 916.57 examples/s]6592 examples [00:07, 904.37 examples/s]6683 examples [00:07, 874.54 examples/s]6771 examples [00:07, 869.65 examples/s]6861 examples [00:07, 875.50 examples/s]6951 examples [00:07, 880.14 examples/s]7041 examples [00:07, 884.99 examples/s]7130 examples [00:08, 885.63 examples/s]7222 examples [00:08, 895.50 examples/s]7312 examples [00:08, 877.05 examples/s]7402 examples [00:08, 882.06 examples/s]7496 examples [00:08, 897.77 examples/s]7586 examples [00:08, 890.63 examples/s]7678 examples [00:08, 897.46 examples/s]7768 examples [00:08, 881.12 examples/s]7858 examples [00:08, 886.44 examples/s]7947 examples [00:08, 846.90 examples/s]8033 examples [00:09, 807.69 examples/s]8118 examples [00:09, 817.85 examples/s]8210 examples [00:09, 843.85 examples/s]8300 examples [00:09, 859.39 examples/s]8390 examples [00:09, 869.43 examples/s]8481 examples [00:09, 878.50 examples/s]8573 examples [00:09, 888.73 examples/s]8663 examples [00:09, 884.50 examples/s]8753 examples [00:09, 887.58 examples/s]8842 examples [00:10, 882.80 examples/s]8933 examples [00:10, 889.51 examples/s]9023 examples [00:10, 892.07 examples/s]9115 examples [00:10, 898.43 examples/s]9212 examples [00:10, 917.21 examples/s]9304 examples [00:10, 917.75 examples/s]9396 examples [00:10, 896.80 examples/s]9486 examples [00:10, 891.24 examples/s]9576 examples [00:10, 851.47 examples/s]9668 examples [00:10, 870.71 examples/s]9757 examples [00:11, 874.28 examples/s]9849 examples [00:11, 886.39 examples/s]9940 examples [00:11, 891.15 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s] 94%|█████████▍| 9410/10000 [00:00<00:00, 94098.97 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteWG2OAH/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteWG2OAH/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f2f1fa64950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f2f1fa64950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f2f1fa64950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f2f85cf2ac8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f2ea8e31ef0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f2f1fa64950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f2f1fa64950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f2f1fa64950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f2f05e230f0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f2f05e230b8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f2e97498268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f2e97498268> 

  function with postional parmater data_info <function split_train_valid at 0x7f2e97498268> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f2e97498378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f2e97498378> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f2e97498378> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=c92f72c741a1690b0ce1096b99f09c771f2038499d5d173bbce5458dbba1c0a3
  Stored in directory: /tmp/pip-ephem-wheel-cache-ookb55bs/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.2.5
WARNING: You are using pip version 20.1; however, version 20.1.1 is available.
You should consider upgrading via the '/opt/hostedtoolcache/Python/3.6.10/x64/bin/python -m pip install --upgrade pip' command.
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<24:48:27, 9.65kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<17:35:52, 13.6kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<12:22:18, 19.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:40:03, 27.6kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<6:03:05, 39.4kB/s].vector_cache/glove.6B.zip:   1%|          | 9.36M/862M [00:01<4:12:34, 56.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.0M/862M [00:01<2:55:43, 80.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.7M/862M [00:01<2:02:17, 115kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 26.4M/862M [00:01<1:25:06, 164kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.4M/862M [00:02<59:18, 233kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 34.8M/862M [00:02<41:28, 332kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.5M/862M [00:02<28:57, 474kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.0M/862M [00:02<20:14, 673kB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.0M/862M [00:02<14:15, 953kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.3M/862M [00:02<10:01, 1.35MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:03<09:02, 1.49MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:05<08:13, 1.63MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:05<08:05, 1.66MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.9M/862M [00:05<06:13, 2.15MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:07<06:39, 2.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.2M/862M [00:07<06:14, 2.14MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:07<04:46, 2.79MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:09<06:03, 2.19MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:09<07:10, 1.85MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:09<05:46, 2.30MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:09<04:11, 3.16MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:11<15:29, 854kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:11<12:09, 1.09MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.1M/862M [00:11<08:46, 1.50MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.3M/862M [00:13<09:16, 1.42MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:13<07:49, 1.68MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.2M/862M [00:13<05:45, 2.28MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:15<07:08, 1.83MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:15<07:42, 1.70MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.3M/862M [00:15<05:58, 2.19MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.0M/862M [00:15<04:24, 2.96MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<07:20, 1.77MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.9M/862M [00:17<06:34, 1.98MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.3M/862M [00:17<04:54, 2.64MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:18<06:15, 2.07MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:19<07:14, 1.78MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.6M/862M [00:19<05:47, 2.23MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:19<04:11, 3.07MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<16:44, 769kB/s] .vector_cache/glove.6B.zip:  10%|█         | 90.2M/862M [00:21<13:06, 981kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.7M/862M [00:21<09:30, 1.35MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<09:27, 1.35MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:23<09:27, 1.35MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.9M/862M [00:23<07:12, 1.77MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:23<05:13, 2.44MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:24<08:35, 1.48MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.5M/862M [00:25<07:25, 1.71MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:25<05:32, 2.29MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:37, 1.91MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<07:27, 1.70MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<05:48, 2.18MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<04:13, 2.98MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<08:24, 1.50MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<07:17, 1.73MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<05:23, 2.33MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<06:30, 1.93MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<07:19, 1.71MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<05:42, 2.19MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<04:10, 2.99MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<08:07, 1.53MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<07:03, 1.76MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<05:13, 2.38MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<06:22, 1.94MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<07:12, 1.72MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<05:37, 2.20MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<04:05, 3.01MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<08:16, 1.49MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<07:09, 1.72MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:37<05:20, 2.30MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:23, 1.91MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<07:11, 1.70MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<05:42, 2.14MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<04:09, 2.93MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<17:40, 689kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<13:42, 888kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<09:52, 1.23MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<09:32, 1.27MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<09:21, 1.29MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<07:09, 1.69MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<05:08, 2.34MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<09:36, 1.25MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<08:03, 1.49MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<05:54, 2.03MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<06:44, 1.78MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<06:02, 1.98MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<04:32, 2.63MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:47, 2.06MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<06:40, 1.78MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<05:19, 2.23MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<03:51, 3.07MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<15:22, 769kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<12:04, 980kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<08:45, 1.35MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<08:41, 1.35MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<08:40, 1.35MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<06:43, 1.75MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<04:50, 2.41MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<15:49, 738kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<12:20, 947kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<08:56, 1.30MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<08:47, 1.32MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<08:43, 1.33MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<06:38, 1.75MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<04:47, 2.41MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<08:21, 1.38MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<07:07, 1.62MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<05:14, 2.20MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<06:11, 1.85MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<06:53, 1.67MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<05:27, 2.10MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<03:57, 2.88MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<16:36, 687kB/s] .vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<12:51, 887kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<09:17, 1.22MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<08:58, 1.26MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<07:31, 1.51MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:04<05:34, 2.03MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<06:20, 1.78MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<06:57, 1.62MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<05:29, 2.05MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<03:57, 2.83MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<14:46, 759kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<11:33, 969kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<08:22, 1.33MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<08:17, 1.34MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<08:15, 1.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<06:23, 1.74MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<04:35, 2.41MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<15:04, 734kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<11:46, 940kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<08:29, 1.30MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<08:19, 1.32MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<08:09, 1.35MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<06:11, 1.77MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:14<04:29, 2.44MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<07:20, 1.49MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<06:18, 1.73MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<04:42, 2.31MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<05:41, 1.91MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<05:10, 2.09MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<03:52, 2.80MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<05:04, 2.13MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<05:49, 1.85MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<04:40, 2.31MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<03:23, 3.15MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<18:48, 570kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<14:23, 744kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<10:18, 1.04MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<09:29, 1.12MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<08:54, 1.20MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<06:48, 1.56MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:24<04:52, 2.17MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<20:04, 527kB/s] .vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<15:12, 695kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<10:54, 966kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<09:54, 1.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<09:15, 1.13MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<07:03, 1.49MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:28<05:03, 2.07MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<13:18, 785kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<10:15, 1.02MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<07:23, 1.41MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:30<05:21, 1.94MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<13:52, 747kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<10:51, 955kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<07:49, 1.32MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<07:43, 1.33MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<07:43, 1.33MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<05:58, 1.72MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:34<04:17, 2.39MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<13:55, 735kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<10:49, 944kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<07:49, 1.30MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<07:45, 1.31MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<06:29, 1.56MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:38<04:48, 2.11MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<05:40, 1.78MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<05:02, 2.00MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<03:44, 2.69MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<04:52, 2.05MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<04:27, 2.25MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:42<03:20, 3.00MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<04:37, 2.15MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<04:20, 2.29MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<03:15, 3.04MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<04:24, 2.24MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<05:10, 1.91MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:03, 2.43MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<02:58, 3.30MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<06:19, 1.55MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<05:27, 1.80MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<04:01, 2.43MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<05:02, 1.94MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<05:40, 1.72MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:30, 2.16MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:50<03:15, 2.97MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<12:49, 754kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<09:52, 979kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<07:12, 1.34MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<07:06, 1.35MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<07:05, 1.35MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:29, 1.75MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:54<03:56, 2.42MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<13:00, 734kB/s] .vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<10:09, 939kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<07:21, 1.29MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<07:11, 1.32MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<06:05, 1.55MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<04:30, 2.09MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<05:11, 1.81MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<05:39, 1.66MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:22, 2.14MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<03:10, 2.94MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<06:53, 1.35MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<05:49, 1.60MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<04:18, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<05:05, 1.82MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:32, 2.04MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<03:24, 2.70MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:29, 2.05MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<05:16, 1.74MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:11, 2.19MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<03:02, 3.00MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<13:11, 692kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<10:14, 891kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<07:21, 1.24MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<07:06, 1.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<06:58, 1.30MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<05:19, 1.70MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<03:49, 2.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<07:54, 1.14MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:12<06:31, 1.38MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<04:48, 1.86MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<05:17, 1.68MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<05:35, 1.59MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:18, 2.06MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<03:07, 2.83MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<06:24, 1.38MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<05:28, 1.62MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<04:03, 2.17MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:44, 1.85MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:18, 2.03MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<03:15, 2.68MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:08, 2.10MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:44, 1.83MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<03:46, 2.30MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:20<02:43, 3.17MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<20:26, 423kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<15:12, 568kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<10:50, 794kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<09:29, 903kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<08:32, 1.00MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<06:26, 1.33MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:24<04:35, 1.85MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<12:18, 691kB/s] .vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<09:34, 887kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<06:53, 1.23MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<06:37, 1.27MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<06:25, 1.31MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<04:52, 1.72MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<03:30, 2.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<07:06, 1.18MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<05:53, 1.42MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<04:19, 1.93MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<04:48, 1.72MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<05:08, 1.61MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<04:01, 2.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:32<02:53, 2.84MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<19:38, 419kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<14:38, 561kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<10:25, 786kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<09:02, 902kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<08:02, 1.01MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<05:59, 1.36MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<04:16, 1.90MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<10:01, 807kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:38<07:54, 1.02MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<05:42, 1.41MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<05:42, 1.40MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<05:47, 1.38MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<04:29, 1.78MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<03:13, 2.47MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<10:45, 738kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<08:24, 944kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<06:03, 1.31MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<05:56, 1.33MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<05:53, 1.34MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<04:33, 1.73MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<03:16, 2.38MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<11:34, 674kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<08:49, 883kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<06:22, 1.22MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<06:08, 1.26MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<05:09, 1.50MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<03:46, 2.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:19, 1.77MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:44, 1.62MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<03:44, 2.05MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<02:42, 2.82MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<10:36, 717kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<08:13, 924kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<05:54, 1.28MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<05:49, 1.29MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<05:40, 1.33MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<04:17, 1.75MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<03:05, 2.42MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<06:10, 1.21MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<04:58, 1.50MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<03:40, 2.02MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<04:13, 1.75MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<04:31, 1.63MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<03:33, 2.08MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<02:33, 2.86MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<23:49, 307kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<17:18, 423kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<12:16, 594kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<10:11, 712kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<08:39, 837kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<06:22, 1.14MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<04:32, 1.59MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<06:34, 1.09MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<05:23, 1.33MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<03:55, 1.82MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:18, 1.65MB/s].vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:05<04:31, 1.57MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:32, 2.01MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<02:32, 2.77MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<16:49, 419kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<12:32, 561kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<08:56, 784kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<07:44, 901kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<06:58, 999kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<05:12, 1.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<03:44, 1.85MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<04:56, 1.40MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<04:12, 1.64MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<03:06, 2.22MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:38, 1.87MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<04:03, 1.68MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<03:13, 2.12MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:13<02:19, 2.92MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<08:52, 763kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<06:57, 973kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<05:00, 1.35MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<04:57, 1.35MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<04:56, 1.35MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<03:49, 1.74MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:17<02:44, 2.42MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<08:14, 805kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<06:27, 1.03MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<04:40, 1.41MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<04:44, 1.39MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<04:41, 1.40MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:37, 1.80MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:21<02:36, 2.49MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<21:21, 304kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<15:39, 414kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:23<11:05, 582kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<09:07, 704kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<07:05, 905kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<05:07, 1.25MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<04:57, 1.28MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<04:56, 1.29MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:47, 1.67MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:27<02:43, 2.31MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<09:31, 659kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<07:21, 853kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<05:18, 1.18MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<05:02, 1.23MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<04:54, 1.27MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:43, 1.67MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<02:40, 2.31MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<04:46, 1.29MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:59, 1.54MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:55, 2.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:24, 1.78MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:03, 1.98MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:17, 2.64MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<02:53, 2.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:21, 1.79MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:36, 2.29MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:37<01:56, 3.08MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:08, 1.89MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:50, 2.08MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<02:07, 2.78MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<02:45, 2.12MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<03:10, 1.85MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:29, 2.36MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:41<01:47, 3.23MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<05:09, 1.12MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<04:13, 1.37MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<03:04, 1.87MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:26, 1.67MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:37, 1.58MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<02:46, 2.05MB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:45<02:00, 2.82MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<04:06, 1.38MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<03:29, 1.62MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<02:35, 2.17MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:01, 1.85MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:43, 2.05MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<02:03, 2.70MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:38, 2.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:27, 2.24MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<01:51, 2.94MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:29, 2.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:56, 1.85MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:21, 2.31MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:53<01:42, 3.17MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<06:57, 774kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<05:27, 986kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<03:56, 1.36MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<03:53, 1.36MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<03:18, 1.60MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:27, 2.15MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:51, 1.84MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<03:10, 1.65MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:30, 2.08MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<01:48, 2.87MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<06:48, 761kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<05:19, 970kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<03:49, 1.34MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<03:46, 1.35MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<03:42, 1.37MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:49, 1.80MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<02:01, 2.49MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<04:01, 1.25MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 560M/862M [04:05<03:20, 1.50MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<02:27, 2.03MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<02:50, 1.75MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<03:02, 1.63MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:20, 2.11MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<01:41, 2.90MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<03:27, 1.42MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:56, 1.67MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:09, 2.26MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:35, 1.86MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:52, 1.67MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:11<02:16, 2.11MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<01:38, 2.91MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<06:15, 761kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<04:52, 975kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<03:29, 1.35MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<03:29, 1.34MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<03:32, 1.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<02:41, 1.74MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<01:57, 2.37MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:39, 1.74MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:22, 1.95MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<01:45, 2.61MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<02:12, 2.06MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<02:30, 1.81MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<01:57, 2.32MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<01:26, 3.14MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<02:34, 1.74MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:16, 1.97MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<01:40, 2.65MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:10, 2.03MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:29, 1.77MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<01:59, 2.21MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<01:25, 3.04MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<05:43, 758kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<04:27, 971kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<03:12, 1.35MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<03:11, 1.34MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<03:07, 1.36MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:22, 1.79MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<01:42, 2.47MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<03:19, 1.27MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<02:45, 1.52MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:01, 2.07MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:20, 1.77MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:31<02:32, 1.62MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:00, 2.05MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<01:26, 2.83MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<05:21, 759kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<04:11, 968kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<03:00, 1.34MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:57, 1.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:59, 1.33MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:18, 1.72MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<01:39, 2.38MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<05:55, 663kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<04:33, 861kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<03:15, 1.20MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<03:08, 1.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<03:03, 1.26MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:19, 1.65MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:41, 2.26MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:15, 1.68MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<01:59, 1.90MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:28, 2.55MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:49, 2.03MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:03, 1.80MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:36, 2.30MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<01:09, 3.16MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<03:32, 1.03MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<02:52, 1.27MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:05, 1.72MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:14, 1.60MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:56, 1.83MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:26, 2.47MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:46, 1.98MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<02:00, 1.74MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<01:34, 2.23MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:07, 3.05MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<02:19, 1.48MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:59, 1.73MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:28, 2.32MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:46, 1.89MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:57, 1.72MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:32, 2.17MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<01:06, 2.98MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<10:42, 308kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<07:51, 420kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<05:32, 590kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<04:32, 712kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<03:53, 830kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<02:53, 1.11MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<02:01, 1.56MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<04:48, 657kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<03:42, 850kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<02:39, 1.18MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<02:30, 1.23MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<02:04, 1.49MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:31, 2.02MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:44, 1.73MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:52, 1.62MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:26, 2.09MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:01, 2.88MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:33, 1.16MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:05, 1.41MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<01:31, 1.92MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:42, 1.69MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:48, 1.59MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:24, 2.05MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<01:00, 2.80MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:59, 1.42MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:42, 1.65MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<01:15, 2.22MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:27, 1.87MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:36, 1.70MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:16, 2.15MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:10<00:54, 2.96MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<08:41, 308kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<06:22, 420kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<04:29, 590kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<03:39, 712kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:50, 915kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<02:02, 1.26MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:58, 1.29MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:56, 1.31MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:28, 1.72MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<01:04, 2.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:25, 1.73MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:15, 1.96MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<00:55, 2.63MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:10, 2.03MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:23, 1.73MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:04, 2.22MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<00:46, 3.02MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:21, 1.72MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:11, 1.95MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<00:52, 2.62MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:07, 2.02MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:17, 1.76MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:01, 2.21MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<00:43, 3.02MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<03:10, 692kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<02:27, 891kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:45, 1.23MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:40, 1.27MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:37, 1.31MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:14, 1.70MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:28<00:52, 2.36MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<06:32, 315kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<04:47, 428kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<03:21, 602kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<02:44, 725kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<02:21, 843kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:43, 1.14MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:32<01:13, 1.58MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:26, 1.33MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:12, 1.57MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<00:53, 2.11MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:00, 1.82MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:07, 1.65MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:52, 2.08MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:36<00:37, 2.86MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<02:29, 715kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:53, 936kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:21, 1.29MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:18, 1.30MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:17, 1.31MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:58, 1.72MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:40<00:41, 2.39MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:26, 1.14MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:11, 1.38MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:51, 1.87MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:55, 1.69MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:58, 1.59MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:45, 2.07MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<00:32, 2.83MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:59, 1.51MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:51, 1.74MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:37, 2.35MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:44, 1.93MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:49, 1.74MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:37, 2.24MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:27, 3.03MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<00:45, 1.78MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:41, 1.98MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:30, 2.63MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:37, 2.06MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:43, 1.78MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:33, 2.27MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<00:24, 3.04MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:36, 1.98MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:33, 2.15MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:25, 2.83MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:32, 2.15MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:37, 1.83MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:29, 2.28MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:56<00:21, 3.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<01:33, 696kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<01:10, 914kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:51, 1.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<00:35, 1.75MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<01:51, 547kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<01:31, 666kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<01:05, 910kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:45, 1.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<01:02, 914kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:49, 1.14MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:35, 1.55MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:34, 1.50MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:35, 1.46MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:27, 1.90MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:19, 2.60MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:29, 1.63MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:25, 1.91MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:18, 2.51MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:21, 2.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:25, 1.75MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:19, 2.22MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:08<00:13, 3.04MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:54, 736kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:41, 947kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:29, 1.31MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:27, 1.32MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:26, 1.34MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:19, 1.77MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:13, 2.44MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:24, 1.30MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:20, 1.55MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:14, 2.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:15, 1.79MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:16, 1.63MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:12, 2.06MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<00:08, 2.83MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:34, 685kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:26, 883kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:18<00:17, 1.23MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:15, 1.27MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:14, 1.29MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:10, 1.67MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:20<00:06, 2.32MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:20, 719kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:15, 921kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:10, 1.27MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:08, 1.30MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:24<00:08, 1.32MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:05, 1.73MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:03, 2.40MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:05, 1.21MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:04, 1.45MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 1.97MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 1.74MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.63MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 2.08MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<48:25:02,  2.29it/s]  0%|          | 638/400000 [00:00<33:50:36,  3.28it/s]  0%|          | 1327/400000 [00:00<23:39:15,  4.68it/s]  0%|          | 1963/400000 [00:00<16:32:12,  6.69it/s]  1%|          | 2594/400000 [00:00<11:33:45,  9.55it/s]  1%|          | 3131/400000 [00:00<8:05:20, 13.63it/s]   1%|          | 3699/400000 [00:01<5:39:36, 19.45it/s]  1%|          | 4399/400000 [00:01<3:57:35, 27.75it/s]  1%|▏         | 5085/400000 [00:01<2:46:18, 39.58it/s]  1%|▏         | 5774/400000 [00:01<1:56:30, 56.40it/s]  2%|▏         | 6437/400000 [00:01<1:21:42, 80.28it/s]  2%|▏         | 7104/400000 [00:01<57:23, 114.09it/s]   2%|▏         | 7754/400000 [00:01<40:24, 161.76it/s]  2%|▏         | 8403/400000 [00:01<28:32, 228.65it/s]  2%|▏         | 9067/400000 [00:01<20:14, 321.88it/s]  2%|▏         | 9737/400000 [00:01<14:26, 450.55it/s]  3%|▎         | 10395/400000 [00:02<10:23, 624.65it/s]  3%|▎         | 11077/400000 [00:02<07:32, 858.62it/s]  3%|▎         | 11786/400000 [00:02<05:32, 1166.08it/s]  3%|▎         | 12461/400000 [00:02<04:11, 1543.31it/s]  3%|▎         | 13123/400000 [00:02<03:13, 2001.47it/s]  3%|▎         | 13823/400000 [00:02<02:31, 2547.00it/s]  4%|▎         | 14494/400000 [00:02<02:04, 3087.94it/s]  4%|▍         | 15148/400000 [00:02<01:44, 3667.16it/s]  4%|▍         | 15805/400000 [00:02<01:30, 4226.55it/s]  4%|▍         | 16459/400000 [00:02<01:21, 4690.89it/s]  4%|▍         | 17134/400000 [00:03<01:14, 5161.73it/s]  4%|▍         | 17835/400000 [00:03<01:08, 5604.45it/s]  5%|▍         | 18514/400000 [00:03<01:04, 5911.36it/s]  5%|▍         | 19261/400000 [00:03<01:00, 6304.64it/s]  5%|▍         | 19955/400000 [00:03<00:59, 6438.65it/s]  5%|▌         | 20644/400000 [00:03<01:00, 6288.14it/s]  5%|▌         | 21306/400000 [00:03<00:59, 6346.69it/s]  5%|▌         | 21986/400000 [00:03<00:58, 6475.48it/s]  6%|▌         | 22675/400000 [00:03<00:57, 6591.97it/s]  6%|▌         | 23372/400000 [00:03<00:56, 6700.09it/s]  6%|▌         | 24114/400000 [00:04<00:54, 6899.55it/s]  6%|▌         | 24861/400000 [00:04<00:53, 7060.52it/s]  6%|▋         | 25582/400000 [00:04<00:52, 7101.20it/s]  7%|▋         | 26370/400000 [00:04<00:51, 7317.59it/s]  7%|▋         | 27107/400000 [00:04<00:51, 7231.93it/s]  7%|▋         | 27834/400000 [00:04<00:52, 7035.30it/s]  7%|▋         | 28571/400000 [00:04<00:52, 7130.95it/s]  7%|▋         | 29320/400000 [00:04<00:51, 7234.92it/s]  8%|▊         | 30046/400000 [00:04<00:51, 7194.94it/s]  8%|▊         | 30768/400000 [00:05<00:51, 7169.35it/s]  8%|▊         | 31496/400000 [00:05<00:51, 7200.55it/s]  8%|▊         | 32227/400000 [00:05<00:50, 7231.23it/s]  8%|▊         | 32954/400000 [00:05<00:50, 7242.17it/s]  8%|▊         | 33679/400000 [00:05<00:50, 7205.64it/s]  9%|▊         | 34400/400000 [00:05<00:52, 6969.40it/s]  9%|▉         | 35099/400000 [00:05<00:52, 6931.60it/s]  9%|▉         | 35794/400000 [00:05<00:52, 6893.75it/s]  9%|▉         | 36485/400000 [00:05<00:53, 6835.81it/s]  9%|▉         | 37170/400000 [00:05<00:55, 6563.63it/s]  9%|▉         | 37830/400000 [00:06<00:56, 6460.11it/s] 10%|▉         | 38482/400000 [00:06<00:55, 6476.91it/s] 10%|▉         | 39164/400000 [00:06<00:54, 6575.65it/s] 10%|▉         | 39824/400000 [00:06<00:54, 6552.74it/s] 10%|█         | 40514/400000 [00:06<00:54, 6651.79it/s] 10%|█         | 41225/400000 [00:06<00:52, 6782.78it/s] 10%|█         | 41909/400000 [00:06<00:52, 6797.29it/s] 11%|█         | 42590/400000 [00:06<00:53, 6705.25it/s] 11%|█         | 43262/400000 [00:06<00:53, 6648.79it/s] 11%|█         | 43945/400000 [00:06<00:53, 6699.31it/s] 11%|█         | 44622/400000 [00:07<00:52, 6716.84it/s] 11%|█▏        | 45307/400000 [00:07<00:52, 6748.59it/s] 11%|█▏        | 45988/400000 [00:07<00:52, 6765.59it/s] 12%|█▏        | 46674/400000 [00:07<00:52, 6793.42it/s] 12%|█▏        | 47354/400000 [00:07<00:53, 6534.54it/s] 12%|█▏        | 48010/400000 [00:07<00:54, 6503.20it/s] 12%|█▏        | 48662/400000 [00:07<00:54, 6416.51it/s] 12%|█▏        | 49374/400000 [00:07<00:53, 6611.53it/s] 13%|█▎        | 50057/400000 [00:07<00:52, 6673.48it/s] 13%|█▎        | 50742/400000 [00:07<00:51, 6722.76it/s] 13%|█▎        | 51416/400000 [00:08<00:53, 6528.62it/s] 13%|█▎        | 52079/400000 [00:08<00:53, 6531.07it/s] 13%|█▎        | 52818/400000 [00:08<00:51, 6765.33it/s] 13%|█▎        | 53549/400000 [00:08<00:50, 6917.94it/s] 14%|█▎        | 54244/400000 [00:08<00:50, 6836.43it/s] 14%|█▎        | 54957/400000 [00:08<00:49, 6921.29it/s] 14%|█▍        | 55664/400000 [00:08<00:49, 6962.32it/s] 14%|█▍        | 56408/400000 [00:08<00:48, 7097.19it/s] 14%|█▍        | 57162/400000 [00:08<00:47, 7223.27it/s] 14%|█▍        | 57893/400000 [00:08<00:47, 7248.10it/s] 15%|█▍        | 58620/400000 [00:09<00:47, 7212.75it/s] 15%|█▍        | 59343/400000 [00:09<00:48, 7084.16it/s] 15%|█▌        | 60053/400000 [00:09<00:50, 6732.21it/s] 15%|█▌        | 60785/400000 [00:09<00:49, 6897.42it/s] 15%|█▌        | 61479/400000 [00:09<00:49, 6821.08it/s] 16%|█▌        | 62165/400000 [00:09<00:50, 6713.59it/s] 16%|█▌        | 62865/400000 [00:09<00:49, 6796.58it/s] 16%|█▌        | 63547/400000 [00:09<00:49, 6801.29it/s] 16%|█▌        | 64240/400000 [00:09<00:49, 6839.26it/s] 16%|█▌        | 64925/400000 [00:10<00:49, 6831.44it/s] 16%|█▋        | 65609/400000 [00:10<00:49, 6806.58it/s] 17%|█▋        | 66304/400000 [00:10<00:48, 6847.21it/s] 17%|█▋        | 66994/400000 [00:10<00:48, 6859.07it/s] 17%|█▋        | 67681/400000 [00:10<00:48, 6842.59it/s] 17%|█▋        | 68372/400000 [00:10<00:48, 6860.15it/s] 17%|█▋        | 69059/400000 [00:10<00:49, 6639.44it/s] 17%|█▋        | 69739/400000 [00:10<00:49, 6685.60it/s] 18%|█▊        | 70466/400000 [00:10<00:48, 6849.44it/s] 18%|█▊        | 71161/400000 [00:10<00:47, 6878.63it/s] 18%|█▊        | 71851/400000 [00:11<00:48, 6711.39it/s] 18%|█▊        | 72524/400000 [00:11<00:49, 6653.47it/s] 18%|█▊        | 73245/400000 [00:11<00:47, 6810.38it/s] 18%|█▊        | 73966/400000 [00:11<00:47, 6924.71it/s] 19%|█▊        | 74661/400000 [00:11<00:47, 6891.02it/s] 19%|█▉        | 75352/400000 [00:11<00:47, 6787.10it/s] 19%|█▉        | 76101/400000 [00:11<00:46, 6983.02it/s] 19%|█▉        | 76860/400000 [00:11<00:45, 7154.46it/s] 19%|█▉        | 77594/400000 [00:11<00:44, 7206.99it/s] 20%|█▉        | 78322/400000 [00:11<00:44, 7228.53it/s] 20%|█▉        | 79063/400000 [00:12<00:44, 7279.86it/s] 20%|█▉        | 79793/400000 [00:12<00:45, 7079.28it/s] 20%|██        | 80503/400000 [00:12<00:48, 6617.97it/s] 20%|██        | 81205/400000 [00:12<00:47, 6733.09it/s] 20%|██        | 81885/400000 [00:12<00:47, 6705.63it/s] 21%|██        | 82575/400000 [00:12<00:46, 6760.93it/s] 21%|██        | 83255/400000 [00:12<00:47, 6649.56it/s] 21%|██        | 83923/400000 [00:12<00:47, 6620.28it/s] 21%|██        | 84606/400000 [00:12<00:47, 6681.09it/s] 21%|██▏       | 85302/400000 [00:13<00:46, 6760.65it/s] 21%|██▏       | 85980/400000 [00:13<00:46, 6726.26it/s] 22%|██▏       | 86660/400000 [00:13<00:46, 6747.00it/s] 22%|██▏       | 87350/400000 [00:13<00:46, 6790.54it/s] 22%|██▏       | 88030/400000 [00:13<00:47, 6605.10it/s] 22%|██▏       | 88705/400000 [00:13<00:46, 6647.55it/s] 22%|██▏       | 89376/400000 [00:13<00:46, 6662.97it/s] 23%|██▎       | 90049/400000 [00:13<00:46, 6682.94it/s] 23%|██▎       | 90718/400000 [00:13<00:46, 6604.65it/s] 23%|██▎       | 91386/400000 [00:13<00:46, 6625.18it/s] 23%|██▎       | 92049/400000 [00:14<00:46, 6567.31it/s] 23%|██▎       | 92791/400000 [00:14<00:45, 6801.34it/s] 23%|██▎       | 93579/400000 [00:14<00:43, 7092.49it/s] 24%|██▎       | 94346/400000 [00:14<00:42, 7255.57it/s] 24%|██▍       | 95076/400000 [00:14<00:43, 6972.69it/s] 24%|██▍       | 95834/400000 [00:14<00:42, 7143.80it/s] 24%|██▍       | 96581/400000 [00:14<00:41, 7236.51it/s] 24%|██▍       | 97371/400000 [00:14<00:40, 7423.22it/s] 25%|██▍       | 98118/400000 [00:14<00:41, 7346.01it/s] 25%|██▍       | 98856/400000 [00:14<00:42, 7131.94it/s] 25%|██▍       | 99573/400000 [00:15<00:43, 6973.16it/s] 25%|██▌       | 100287/400000 [00:15<00:42, 7021.34it/s] 25%|██▌       | 101038/400000 [00:15<00:41, 7159.09it/s] 25%|██▌       | 101789/400000 [00:15<00:41, 7260.64it/s] 26%|██▌       | 102535/400000 [00:15<00:40, 7317.35it/s] 26%|██▌       | 103269/400000 [00:15<00:41, 7209.21it/s] 26%|██▌       | 103992/400000 [00:15<00:41, 7109.95it/s] 26%|██▌       | 104705/400000 [00:15<00:41, 7070.62it/s] 26%|██▋       | 105414/400000 [00:15<00:42, 6952.14it/s] 27%|██▋       | 106111/400000 [00:15<00:42, 6896.08it/s] 27%|██▋       | 106802/400000 [00:16<00:42, 6884.27it/s] 27%|██▋       | 107529/400000 [00:16<00:41, 6993.47it/s] 27%|██▋       | 108272/400000 [00:16<00:40, 7118.12it/s] 27%|██▋       | 109050/400000 [00:16<00:39, 7303.37it/s] 27%|██▋       | 109809/400000 [00:16<00:39, 7386.29it/s] 28%|██▊       | 110581/400000 [00:16<00:38, 7482.97it/s] 28%|██▊       | 111331/400000 [00:16<00:39, 7312.73it/s] 28%|██▊       | 112065/400000 [00:16<00:40, 7162.81it/s] 28%|██▊       | 112844/400000 [00:16<00:39, 7339.06it/s] 28%|██▊       | 113581/400000 [00:17<00:39, 7281.61it/s] 29%|██▊       | 114311/400000 [00:17<00:40, 7135.24it/s] 29%|██▉       | 115027/400000 [00:17<00:40, 6982.37it/s] 29%|██▉       | 115728/400000 [00:17<00:41, 6922.19it/s] 29%|██▉       | 116422/400000 [00:17<00:41, 6866.27it/s] 29%|██▉       | 117110/400000 [00:17<00:41, 6862.94it/s] 29%|██▉       | 117798/400000 [00:17<00:41, 6726.92it/s] 30%|██▉       | 118494/400000 [00:17<00:41, 6794.58it/s] 30%|██▉       | 119203/400000 [00:17<00:40, 6878.32it/s] 30%|██▉       | 119953/400000 [00:17<00:39, 7052.06it/s] 30%|███       | 120707/400000 [00:18<00:38, 7189.80it/s] 30%|███       | 121503/400000 [00:18<00:37, 7402.46it/s] 31%|███       | 122263/400000 [00:18<00:37, 7460.11it/s] 31%|███       | 123042/400000 [00:18<00:36, 7553.68it/s] 31%|███       | 123825/400000 [00:18<00:36, 7631.93it/s] 31%|███       | 124590/400000 [00:18<00:36, 7595.77it/s] 31%|███▏      | 125351/400000 [00:18<00:36, 7561.37it/s] 32%|███▏      | 126108/400000 [00:18<00:37, 7361.56it/s] 32%|███▏      | 126846/400000 [00:18<00:37, 7212.02it/s] 32%|███▏      | 127570/400000 [00:18<00:38, 7129.81it/s] 32%|███▏      | 128285/400000 [00:19<00:38, 7059.96it/s] 32%|███▏      | 128993/400000 [00:19<00:39, 6926.29it/s] 32%|███▏      | 129688/400000 [00:19<00:39, 6872.66it/s] 33%|███▎      | 130393/400000 [00:19<00:38, 6923.75it/s] 33%|███▎      | 131087/400000 [00:19<00:39, 6758.83it/s] 33%|███▎      | 131798/400000 [00:19<00:39, 6858.09it/s] 33%|███▎      | 132511/400000 [00:19<00:38, 6937.22it/s] 33%|███▎      | 133260/400000 [00:19<00:37, 7090.47it/s] 33%|███▎      | 133997/400000 [00:19<00:37, 7171.23it/s] 34%|███▎      | 134756/400000 [00:19<00:36, 7290.74it/s] 34%|███▍      | 135487/400000 [00:20<00:36, 7296.44it/s] 34%|███▍      | 136241/400000 [00:20<00:35, 7365.72it/s] 34%|███▍      | 137003/400000 [00:20<00:35, 7437.54it/s] 34%|███▍      | 137748/400000 [00:20<00:35, 7377.35it/s] 35%|███▍      | 138507/400000 [00:20<00:35, 7439.42it/s] 35%|███▍      | 139252/400000 [00:20<00:35, 7341.31it/s] 35%|███▍      | 139987/400000 [00:20<00:36, 7212.20it/s] 35%|███▌      | 140710/400000 [00:20<00:36, 7045.38it/s] 35%|███▌      | 141417/400000 [00:20<00:36, 7012.70it/s] 36%|███▌      | 142120/400000 [00:21<00:36, 7011.95it/s] 36%|███▌      | 142822/400000 [00:21<00:36, 6992.10it/s] 36%|███▌      | 143550/400000 [00:21<00:36, 7075.68it/s] 36%|███▌      | 144259/400000 [00:21<00:37, 6897.78it/s] 36%|███▌      | 144954/400000 [00:21<00:36, 6912.29it/s] 36%|███▋      | 145669/400000 [00:21<00:36, 6980.61it/s] 37%|███▋      | 146368/400000 [00:21<00:36, 6977.35it/s] 37%|███▋      | 147120/400000 [00:21<00:35, 7131.05it/s] 37%|███▋      | 147860/400000 [00:21<00:34, 7208.06it/s] 37%|███▋      | 148637/400000 [00:21<00:34, 7366.75it/s] 37%|███▋      | 149376/400000 [00:22<00:34, 7328.38it/s] 38%|███▊      | 150111/400000 [00:22<00:34, 7206.03it/s] 38%|███▊      | 150833/400000 [00:22<00:36, 6846.04it/s] 38%|███▊      | 151523/400000 [00:22<00:36, 6834.89it/s] 38%|███▊      | 152220/400000 [00:22<00:36, 6874.20it/s] 38%|███▊      | 152940/400000 [00:22<00:35, 6968.77it/s] 38%|███▊      | 153648/400000 [00:22<00:35, 6999.34it/s] 39%|███▊      | 154388/400000 [00:22<00:34, 7114.13it/s] 39%|███▉      | 155111/400000 [00:22<00:34, 7145.84it/s] 39%|███▉      | 155827/400000 [00:22<00:34, 7073.42it/s] 39%|███▉      | 156536/400000 [00:23<00:35, 6880.12it/s] 39%|███▉      | 157240/400000 [00:23<00:35, 6927.29it/s] 39%|███▉      | 157935/400000 [00:23<00:35, 6903.16it/s] 40%|███▉      | 158628/400000 [00:23<00:34, 6908.84it/s] 40%|███▉      | 159320/400000 [00:23<00:34, 6899.58it/s] 40%|████      | 160033/400000 [00:23<00:34, 6966.52it/s] 40%|████      | 160731/400000 [00:23<00:34, 6929.79it/s] 40%|████      | 161425/400000 [00:23<00:34, 6872.08it/s] 41%|████      | 162204/400000 [00:23<00:33, 7123.29it/s] 41%|████      | 162985/400000 [00:23<00:32, 7316.23it/s] 41%|████      | 163752/400000 [00:24<00:31, 7415.71it/s] 41%|████      | 164572/400000 [00:24<00:30, 7633.62it/s] 41%|████▏     | 165355/400000 [00:24<00:30, 7691.10it/s] 42%|████▏     | 166144/400000 [00:24<00:30, 7748.91it/s] 42%|████▏     | 166968/400000 [00:24<00:29, 7888.28it/s] 42%|████▏     | 167759/400000 [00:24<00:31, 7341.98it/s] 42%|████▏     | 168503/400000 [00:24<00:32, 7132.39it/s] 42%|████▏     | 169224/400000 [00:24<00:32, 7059.03it/s] 42%|████▏     | 169936/400000 [00:24<00:32, 7010.37it/s] 43%|████▎     | 170641/400000 [00:25<00:32, 7010.76it/s] 43%|████▎     | 171346/400000 [00:25<00:32, 7017.28it/s] 43%|████▎     | 172050/400000 [00:25<00:32, 6918.81it/s] 43%|████▎     | 172751/400000 [00:25<00:32, 6945.44it/s] 43%|████▎     | 173482/400000 [00:25<00:32, 7049.94it/s] 44%|████▎     | 174212/400000 [00:25<00:31, 7120.71it/s] 44%|████▎     | 174926/400000 [00:25<00:32, 7021.64it/s] 44%|████▍     | 175684/400000 [00:25<00:31, 7180.11it/s] 44%|████▍     | 176404/400000 [00:25<00:31, 7034.76it/s] 44%|████▍     | 177112/400000 [00:25<00:31, 7048.01it/s] 44%|████▍     | 177836/400000 [00:26<00:31, 7103.40it/s] 45%|████▍     | 178557/400000 [00:26<00:31, 7133.32it/s] 45%|████▍     | 179272/400000 [00:26<00:31, 7083.89it/s] 45%|████▍     | 179981/400000 [00:26<00:31, 7044.48it/s] 45%|████▌     | 180701/400000 [00:26<00:30, 7088.96it/s] 45%|████▌     | 181504/400000 [00:26<00:29, 7346.07it/s] 46%|████▌     | 182290/400000 [00:26<00:29, 7491.87it/s] 46%|████▌     | 183043/400000 [00:26<00:28, 7502.59it/s] 46%|████▌     | 183796/400000 [00:26<00:29, 7215.40it/s] 46%|████▌     | 184522/400000 [00:26<00:29, 7189.49it/s] 46%|████▋     | 185244/400000 [00:27<00:30, 7030.12it/s] 46%|████▋     | 185950/400000 [00:27<00:30, 6994.32it/s] 47%|████▋     | 186670/400000 [00:27<00:30, 7053.33it/s] 47%|████▋     | 187377/400000 [00:27<00:30, 6993.41it/s] 47%|████▋     | 188129/400000 [00:27<00:29, 7141.59it/s] 47%|████▋     | 188913/400000 [00:27<00:28, 7335.79it/s] 47%|████▋     | 189679/400000 [00:27<00:28, 7427.63it/s] 48%|████▊     | 190483/400000 [00:27<00:27, 7600.60it/s] 48%|████▊     | 191246/400000 [00:27<00:27, 7459.11it/s] 48%|████▊     | 191995/400000 [00:27<00:28, 7427.58it/s] 48%|████▊     | 192740/400000 [00:28<00:29, 6955.08it/s] 48%|████▊     | 193443/400000 [00:28<00:30, 6879.36it/s] 49%|████▊     | 194137/400000 [00:28<00:29, 6873.99it/s] 49%|████▊     | 194829/400000 [00:28<00:29, 6884.72it/s] 49%|████▉     | 195548/400000 [00:28<00:29, 6972.04it/s] 49%|████▉     | 196248/400000 [00:28<00:30, 6581.13it/s] 49%|████▉     | 196913/400000 [00:28<00:30, 6592.86it/s] 49%|████▉     | 197602/400000 [00:28<00:30, 6676.43it/s] 50%|████▉     | 198273/400000 [00:28<00:30, 6563.99it/s] 50%|████▉     | 199064/400000 [00:29<00:29, 6914.19it/s] 50%|████▉     | 199864/400000 [00:29<00:27, 7205.97it/s] 50%|█████     | 200650/400000 [00:29<00:26, 7389.65it/s] 50%|█████     | 201396/400000 [00:29<00:27, 7280.04it/s] 51%|█████     | 202130/400000 [00:29<00:27, 7142.14it/s] 51%|█████     | 202937/400000 [00:29<00:26, 7395.96it/s] 51%|█████     | 203692/400000 [00:29<00:26, 7440.61it/s] 51%|█████     | 204440/400000 [00:29<00:26, 7262.53it/s] 51%|█████▏    | 205193/400000 [00:29<00:26, 7338.06it/s] 51%|█████▏    | 205961/400000 [00:29<00:26, 7435.39it/s] 52%|█████▏    | 206707/400000 [00:30<00:26, 7317.38it/s] 52%|█████▏    | 207503/400000 [00:30<00:25, 7496.93it/s] 52%|█████▏    | 208309/400000 [00:30<00:25, 7654.57it/s] 52%|█████▏    | 209129/400000 [00:30<00:24, 7808.24it/s] 52%|█████▏    | 209970/400000 [00:30<00:23, 7977.35it/s] 53%|█████▎    | 210771/400000 [00:30<00:24, 7854.97it/s] 53%|█████▎    | 211559/400000 [00:30<00:24, 7739.85it/s] 53%|█████▎    | 212335/400000 [00:30<00:24, 7603.53it/s] 53%|█████▎    | 213098/400000 [00:30<00:25, 7384.29it/s] 53%|█████▎    | 213840/400000 [00:30<00:25, 7359.63it/s] 54%|█████▎    | 214578/400000 [00:31<00:25, 7252.12it/s] 54%|█████▍    | 215305/400000 [00:31<00:26, 6992.97it/s] 54%|█████▍    | 216008/400000 [00:31<00:27, 6787.00it/s] 54%|█████▍    | 216691/400000 [00:31<00:27, 6735.16it/s] 54%|█████▍    | 217368/400000 [00:31<00:27, 6691.13it/s] 55%|█████▍    | 218057/400000 [00:31<00:26, 6748.63it/s] 55%|█████▍    | 218756/400000 [00:31<00:26, 6818.44it/s] 55%|█████▍    | 219440/400000 [00:31<00:26, 6774.42it/s] 55%|█████▌    | 220119/400000 [00:31<00:26, 6778.45it/s] 55%|█████▌    | 220834/400000 [00:32<00:26, 6884.99it/s] 55%|█████▌    | 221533/400000 [00:32<00:25, 6914.80it/s] 56%|█████▌    | 222266/400000 [00:32<00:25, 7034.21it/s] 56%|█████▌    | 222971/400000 [00:32<00:25, 6951.23it/s] 56%|█████▌    | 223667/400000 [00:32<00:26, 6620.13it/s] 56%|█████▌    | 224333/400000 [00:32<00:27, 6465.99it/s] 56%|█████▋    | 225027/400000 [00:32<00:26, 6600.91it/s] 56%|█████▋    | 225769/400000 [00:32<00:25, 6826.51it/s] 57%|█████▋    | 226523/400000 [00:32<00:24, 7024.74it/s] 57%|█████▋    | 227230/400000 [00:32<00:25, 6891.69it/s] 57%|█████▋    | 227923/400000 [00:33<00:25, 6796.93it/s] 57%|█████▋    | 228670/400000 [00:33<00:24, 6983.53it/s] 57%|█████▋    | 229419/400000 [00:33<00:23, 7127.93it/s] 58%|█████▊    | 230175/400000 [00:33<00:23, 7249.95it/s] 58%|█████▊    | 230931/400000 [00:33<00:23, 7339.87it/s] 58%|█████▊    | 231668/400000 [00:33<00:23, 7190.26it/s] 58%|█████▊    | 232390/400000 [00:33<00:24, 6903.39it/s] 58%|█████▊    | 233100/400000 [00:33<00:23, 6959.05it/s] 58%|█████▊    | 233799/400000 [00:33<00:24, 6911.63it/s] 59%|█████▊    | 234493/400000 [00:33<00:24, 6872.29it/s] 59%|█████▉    | 235240/400000 [00:34<00:23, 7040.73it/s] 59%|█████▉    | 236006/400000 [00:34<00:22, 7213.73it/s] 59%|█████▉    | 236730/400000 [00:34<00:23, 6893.39it/s] 59%|█████▉    | 237425/400000 [00:34<00:23, 6883.88it/s] 60%|█████▉    | 238120/400000 [00:34<00:23, 6901.76it/s] 60%|█████▉    | 238813/400000 [00:34<00:23, 6881.03it/s] 60%|█████▉    | 239503/400000 [00:34<00:23, 6875.74it/s] 60%|██████    | 240239/400000 [00:34<00:22, 7013.54it/s] 60%|██████    | 240942/400000 [00:34<00:24, 6612.80it/s] 60%|██████    | 241659/400000 [00:35<00:23, 6769.72it/s] 61%|██████    | 242348/400000 [00:35<00:23, 6802.68it/s] 61%|██████    | 243038/400000 [00:35<00:22, 6829.30it/s] 61%|██████    | 243726/400000 [00:35<00:22, 6843.94it/s] 61%|██████    | 244423/400000 [00:35<00:22, 6879.62it/s] 61%|██████▏   | 245113/400000 [00:35<00:23, 6602.55it/s] 61%|██████▏   | 245777/400000 [00:35<00:23, 6575.51it/s] 62%|██████▏   | 246465/400000 [00:35<00:23, 6662.58it/s] 62%|██████▏   | 247170/400000 [00:35<00:22, 6772.15it/s] 62%|██████▏   | 247864/400000 [00:35<00:22, 6820.25it/s] 62%|██████▏   | 248548/400000 [00:36<00:22, 6791.86it/s] 62%|██████▏   | 249229/400000 [00:36<00:22, 6632.83it/s] 62%|██████▏   | 249910/400000 [00:36<00:22, 6682.74it/s] 63%|██████▎   | 250598/400000 [00:36<00:22, 6739.87it/s] 63%|██████▎   | 251310/400000 [00:36<00:21, 6848.43it/s] 63%|██████▎   | 252046/400000 [00:36<00:21, 6994.11it/s] 63%|██████▎   | 252747/400000 [00:36<00:21, 6980.32it/s] 63%|██████▎   | 253447/400000 [00:36<00:21, 6882.88it/s] 64%|██████▎   | 254137/400000 [00:36<00:21, 6850.33it/s] 64%|██████▎   | 254826/400000 [00:36<00:21, 6861.38it/s] 64%|██████▍   | 255513/400000 [00:37<00:21, 6816.01it/s] 64%|██████▍   | 256212/400000 [00:37<00:20, 6867.24it/s] 64%|██████▍   | 256900/400000 [00:37<00:20, 6828.78it/s] 64%|██████▍   | 257606/400000 [00:37<00:20, 6895.51it/s] 65%|██████▍   | 258343/400000 [00:37<00:20, 7029.33it/s] 65%|██████▍   | 259050/400000 [00:37<00:20, 7040.28it/s] 65%|██████▍   | 259755/400000 [00:37<00:20, 6949.25it/s] 65%|██████▌   | 260451/400000 [00:37<00:21, 6611.05it/s] 65%|██████▌   | 261206/400000 [00:37<00:20, 6857.99it/s] 65%|██████▌   | 261924/400000 [00:37<00:19, 6951.21it/s] 66%|██████▌   | 262624/400000 [00:38<00:20, 6803.72it/s] 66%|██████▌   | 263308/400000 [00:38<00:20, 6593.88it/s] 66%|██████▌   | 263987/400000 [00:38<00:20, 6650.23it/s] 66%|██████▌   | 264677/400000 [00:38<00:20, 6721.07it/s] 66%|██████▋   | 265358/400000 [00:38<00:19, 6744.56it/s] 67%|██████▋   | 266034/400000 [00:38<00:20, 6684.34it/s] 67%|██████▋   | 266707/400000 [00:38<00:19, 6697.42it/s] 67%|██████▋   | 267486/400000 [00:38<00:18, 6989.15it/s] 67%|██████▋   | 268223/400000 [00:38<00:18, 7098.07it/s] 67%|██████▋   | 268989/400000 [00:39<00:18, 7255.14it/s] 67%|██████▋   | 269737/400000 [00:39<00:17, 7318.48it/s] 68%|██████▊   | 270487/400000 [00:39<00:17, 7368.47it/s] 68%|██████▊   | 271258/400000 [00:39<00:17, 7466.77it/s] 68%|██████▊   | 272028/400000 [00:39<00:16, 7534.15it/s] 68%|██████▊   | 272791/400000 [00:39<00:16, 7560.72it/s] 68%|██████▊   | 273548/400000 [00:39<00:16, 7452.04it/s] 69%|██████▊   | 274301/400000 [00:39<00:16, 7473.47it/s] 69%|██████▉   | 275050/400000 [00:39<00:17, 7212.86it/s] 69%|██████▉   | 275774/400000 [00:39<00:17, 7032.38it/s] 69%|██████▉   | 276506/400000 [00:40<00:17, 7113.65it/s] 69%|██████▉   | 277255/400000 [00:40<00:16, 7222.09it/s] 69%|██████▉   | 277988/400000 [00:40<00:16, 7250.15it/s] 70%|██████▉   | 278726/400000 [00:40<00:16, 7286.72it/s] 70%|██████▉   | 279456/400000 [00:40<00:16, 7147.05it/s] 70%|███████   | 280172/400000 [00:40<00:17, 7045.98it/s] 70%|███████   | 280905/400000 [00:40<00:16, 7125.86it/s] 70%|███████   | 281619/400000 [00:40<00:16, 7037.87it/s] 71%|███████   | 282327/400000 [00:40<00:16, 7048.98it/s] 71%|███████   | 283033/400000 [00:40<00:16, 6997.13it/s] 71%|███████   | 283734/400000 [00:41<00:16, 6885.64it/s] 71%|███████   | 284440/400000 [00:41<00:16, 6935.14it/s] 71%|███████▏  | 285138/400000 [00:41<00:16, 6947.09it/s] 71%|███████▏  | 285834/400000 [00:41<00:16, 6858.94it/s] 72%|███████▏  | 286532/400000 [00:41<00:16, 6893.97it/s] 72%|███████▏  | 287222/400000 [00:41<00:17, 6633.95it/s] 72%|███████▏  | 287888/400000 [00:41<00:16, 6601.37it/s] 72%|███████▏  | 288604/400000 [00:41<00:16, 6759.13it/s] 72%|███████▏  | 289290/400000 [00:41<00:16, 6786.79it/s] 72%|███████▏  | 289977/400000 [00:41<00:16, 6811.21it/s] 73%|███████▎  | 290660/400000 [00:42<00:16, 6805.86it/s] 73%|███████▎  | 291342/400000 [00:42<00:15, 6801.28it/s] 73%|███████▎  | 292023/400000 [00:42<00:16, 6732.31it/s] 73%|███████▎  | 292697/400000 [00:42<00:16, 6694.31it/s] 73%|███████▎  | 293392/400000 [00:42<00:15, 6767.89it/s] 74%|███████▎  | 294070/400000 [00:42<00:15, 6755.85it/s] 74%|███████▎  | 294746/400000 [00:42<00:15, 6734.33it/s] 74%|███████▍  | 295420/400000 [00:42<00:15, 6676.99it/s] 74%|███████▍  | 296088/400000 [00:42<00:15, 6619.97it/s] 74%|███████▍  | 296751/400000 [00:43<00:16, 6416.19it/s] 74%|███████▍  | 297423/400000 [00:43<00:15, 6503.59it/s] 75%|███████▍  | 298075/400000 [00:43<00:15, 6398.96it/s] 75%|███████▍  | 298717/400000 [00:43<00:15, 6368.60it/s] 75%|███████▍  | 299377/400000 [00:43<00:15, 6434.34it/s] 75%|███████▌  | 300076/400000 [00:43<00:15, 6589.71it/s] 75%|███████▌  | 300737/400000 [00:43<00:15, 6550.04it/s] 75%|███████▌  | 301394/400000 [00:43<00:15, 6459.55it/s] 76%|███████▌  | 302086/400000 [00:43<00:14, 6589.26it/s] 76%|███████▌  | 302758/400000 [00:43<00:14, 6625.45it/s] 76%|███████▌  | 303451/400000 [00:44<00:14, 6713.62it/s] 76%|███████▌  | 304124/400000 [00:44<00:14, 6707.63it/s] 76%|███████▌  | 304796/400000 [00:44<00:14, 6686.74it/s] 76%|███████▋  | 305499/400000 [00:44<00:13, 6786.14it/s] 77%|███████▋  | 306179/400000 [00:44<00:13, 6778.22it/s] 77%|███████▋  | 306888/400000 [00:44<00:13, 6866.92it/s] 77%|███████▋  | 307576/400000 [00:44<00:13, 6793.80it/s] 77%|███████▋  | 308273/400000 [00:44<00:13, 6844.49it/s] 77%|███████▋  | 308958/400000 [00:44<00:13, 6644.36it/s] 77%|███████▋  | 309692/400000 [00:44<00:13, 6836.14it/s] 78%|███████▊  | 310437/400000 [00:45<00:12, 7007.71it/s] 78%|███████▊  | 311182/400000 [00:45<00:12, 7132.24it/s] 78%|███████▊  | 311938/400000 [00:45<00:12, 7254.46it/s] 78%|███████▊  | 312694/400000 [00:45<00:11, 7341.18it/s] 78%|███████▊  | 313430/400000 [00:45<00:11, 7283.32it/s] 79%|███████▊  | 314174/400000 [00:45<00:11, 7327.99it/s] 79%|███████▊  | 314908/400000 [00:45<00:11, 7307.44it/s] 79%|███████▉  | 315640/400000 [00:45<00:11, 7184.15it/s] 79%|███████▉  | 316360/400000 [00:45<00:11, 7060.00it/s] 79%|███████▉  | 317068/400000 [00:45<00:11, 7019.54it/s] 79%|███████▉  | 317771/400000 [00:46<00:11, 6866.61it/s] 80%|███████▉  | 318466/400000 [00:46<00:11, 6890.93it/s] 80%|███████▉  | 319157/400000 [00:46<00:11, 6829.39it/s] 80%|███████▉  | 319885/400000 [00:46<00:11, 6958.41it/s] 80%|████████  | 320641/400000 [00:46<00:11, 7126.42it/s] 80%|████████  | 321356/400000 [00:46<00:11, 6945.29it/s] 81%|████████  | 322053/400000 [00:46<00:11, 6666.75it/s] 81%|████████  | 322730/400000 [00:46<00:11, 6695.63it/s] 81%|████████  | 323410/400000 [00:46<00:11, 6724.55it/s] 81%|████████  | 324126/400000 [00:46<00:11, 6849.32it/s] 81%|████████  | 324838/400000 [00:47<00:10, 6926.14it/s] 81%|████████▏ | 325533/400000 [00:47<00:10, 6893.63it/s] 82%|████████▏ | 326227/400000 [00:47<00:10, 6906.18it/s] 82%|████████▏ | 326919/400000 [00:47<00:10, 6904.15it/s] 82%|████████▏ | 327610/400000 [00:47<00:10, 6876.57it/s] 82%|████████▏ | 328299/400000 [00:47<00:10, 6852.39it/s] 82%|████████▏ | 328985/400000 [00:47<00:10, 6850.63it/s] 82%|████████▏ | 329700/400000 [00:47<00:10, 6935.73it/s] 83%|████████▎ | 330394/400000 [00:47<00:10, 6894.32it/s] 83%|████████▎ | 331084/400000 [00:48<00:10, 6888.06it/s] 83%|████████▎ | 331787/400000 [00:48<00:09, 6928.38it/s] 83%|████████▎ | 332481/400000 [00:48<00:10, 6683.78it/s] 83%|████████▎ | 333152/400000 [00:48<00:10, 6619.92it/s] 83%|████████▎ | 333836/400000 [00:48<00:09, 6683.09it/s] 84%|████████▎ | 334578/400000 [00:48<00:09, 6885.10it/s] 84%|████████▍ | 335342/400000 [00:48<00:09, 7095.17it/s] 84%|████████▍ | 336095/400000 [00:48<00:08, 7220.00it/s] 84%|████████▍ | 336845/400000 [00:48<00:08, 7301.54it/s] 84%|████████▍ | 337578/400000 [00:48<00:08, 7232.51it/s] 85%|████████▍ | 338303/400000 [00:49<00:08, 7185.59it/s] 85%|████████▍ | 339023/400000 [00:49<00:08, 7072.08it/s] 85%|████████▍ | 339732/400000 [00:49<00:08, 7011.14it/s] 85%|████████▌ | 340496/400000 [00:49<00:08, 7188.50it/s] 85%|████████▌ | 341270/400000 [00:49<00:07, 7343.13it/s] 86%|████████▌ | 342015/400000 [00:49<00:07, 7372.94it/s] 86%|████████▌ | 342754/400000 [00:49<00:07, 7226.51it/s] 86%|████████▌ | 343479/400000 [00:49<00:07, 7108.20it/s] 86%|████████▌ | 344243/400000 [00:49<00:07, 7257.57it/s] 86%|████████▌ | 344971/400000 [00:49<00:07, 7146.71it/s] 86%|████████▋ | 345688/400000 [00:50<00:07, 6958.87it/s] 87%|████████▋ | 346387/400000 [00:50<00:07, 6955.60it/s] 87%|████████▋ | 347085/400000 [00:50<00:07, 6889.96it/s] 87%|████████▋ | 347781/400000 [00:50<00:07, 6910.20it/s] 87%|████████▋ | 348491/400000 [00:50<00:07, 6964.01it/s] 87%|████████▋ | 349190/400000 [00:50<00:07, 6969.51it/s] 87%|████████▋ | 349906/400000 [00:50<00:07, 7024.53it/s] 88%|████████▊ | 350609/400000 [00:50<00:07, 6933.36it/s] 88%|████████▊ | 351303/400000 [00:50<00:07, 6907.34it/s] 88%|████████▊ | 351995/400000 [00:50<00:07, 6684.42it/s] 88%|████████▊ | 352666/400000 [00:51<00:07, 6645.51it/s] 88%|████████▊ | 353332/400000 [00:51<00:07, 6573.02it/s] 88%|████████▊ | 353991/400000 [00:51<00:07, 6536.10it/s] 89%|████████▊ | 354646/400000 [00:51<00:07, 6237.94it/s] 89%|████████▉ | 355322/400000 [00:51<00:06, 6384.78it/s] 89%|████████▉ | 356051/400000 [00:51<00:06, 6630.86it/s] 89%|████████▉ | 356719/400000 [00:51<00:06, 6516.02it/s] 89%|████████▉ | 357432/400000 [00:51<00:06, 6687.35it/s] 90%|████████▉ | 358138/400000 [00:51<00:06, 6793.78it/s] 90%|████████▉ | 358825/400000 [00:52<00:06, 6816.43it/s] 90%|████████▉ | 359509/400000 [00:52<00:05, 6765.55it/s] 90%|█████████ | 360188/400000 [00:52<00:05, 6639.63it/s] 90%|█████████ | 360875/400000 [00:52<00:05, 6706.44it/s] 90%|█████████ | 361634/400000 [00:52<00:05, 6948.20it/s] 91%|█████████ | 362332/400000 [00:52<00:05, 6955.65it/s] 91%|█████████ | 363031/400000 [00:52<00:05, 6965.69it/s] 91%|█████████ | 363730/400000 [00:52<00:05, 6913.23it/s] 91%|█████████ | 364423/400000 [00:52<00:05, 6849.44it/s] 91%|█████████▏| 365109/400000 [00:52<00:05, 6822.90it/s] 91%|█████████▏| 365807/400000 [00:53<00:04, 6867.72it/s] 92%|█████████▏| 366504/400000 [00:53<00:04, 6895.67it/s] 92%|█████████▏| 367205/400000 [00:53<00:04, 6927.62it/s] 92%|█████████▏| 367899/400000 [00:53<00:04, 6648.03it/s] 92%|█████████▏| 368586/400000 [00:53<00:04, 6711.56it/s] 92%|█████████▏| 369350/400000 [00:53<00:04, 6964.00it/s] 93%|█████████▎| 370128/400000 [00:53<00:04, 7189.53it/s] 93%|█████████▎| 370895/400000 [00:53<00:03, 7325.95it/s] 93%|█████████▎| 371636/400000 [00:53<00:03, 7348.26it/s] 93%|█████████▎| 372403/400000 [00:53<00:03, 7439.62it/s] 93%|█████████▎| 373191/400000 [00:54<00:03, 7565.81it/s] 93%|█████████▎| 373969/400000 [00:54<00:03, 7626.69it/s] 94%|█████████▎| 374748/400000 [00:54<00:03, 7674.57it/s] 94%|█████████▍| 375534/400000 [00:54<00:03, 7726.56it/s] 94%|█████████▍| 376308/400000 [00:54<00:03, 7576.12it/s] 94%|█████████▍| 377067/400000 [00:54<00:03, 7533.64it/s] 94%|█████████▍| 377830/400000 [00:54<00:02, 7560.38it/s] 95%|█████████▍| 378595/400000 [00:54<00:02, 7583.76it/s] 95%|█████████▍| 379354/400000 [00:54<00:02, 7434.69it/s] 95%|█████████▌| 380101/400000 [00:54<00:02, 7442.63it/s] 95%|█████████▌| 380846/400000 [00:55<00:02, 7075.15it/s] 95%|█████████▌| 381558/400000 [00:55<00:02, 6854.17it/s] 96%|█████████▌| 382248/400000 [00:55<00:02, 6846.88it/s] 96%|█████████▌| 382952/400000 [00:55<00:02, 6902.50it/s] 96%|█████████▌| 383645/400000 [00:55<00:02, 6854.09it/s] 96%|█████████▌| 384333/400000 [00:55<00:02, 6857.15it/s] 96%|█████████▋| 385020/400000 [00:55<00:02, 6839.52it/s] 96%|█████████▋| 385705/400000 [00:55<00:02, 6793.28it/s] 97%|█████████▋| 386437/400000 [00:55<00:01, 6941.26it/s] 97%|█████████▋| 387133/400000 [00:56<00:01, 6884.50it/s] 97%|█████████▋| 387823/400000 [00:56<00:01, 6888.68it/s] 97%|█████████▋| 388513/400000 [00:56<00:01, 6863.96it/s] 97%|█████████▋| 389200/400000 [00:56<00:01, 6848.57it/s] 97%|█████████▋| 389886/400000 [00:56<00:01, 6774.22it/s] 98%|█████████▊| 390599/400000 [00:56<00:01, 6875.83it/s] 98%|█████████▊| 391384/400000 [00:56<00:01, 7140.26it/s] 98%|█████████▊| 392159/400000 [00:56<00:01, 7312.49it/s] 98%|█████████▊| 392937/400000 [00:56<00:00, 7443.34it/s] 98%|█████████▊| 393693/400000 [00:56<00:00, 7476.32it/s] 99%|█████████▊| 394443/400000 [00:57<00:00, 7468.44it/s] 99%|█████████▉| 395239/400000 [00:57<00:00, 7608.69it/s] 99%|█████████▉| 396002/400000 [00:57<00:00, 7540.45it/s] 99%|█████████▉| 396770/400000 [00:57<00:00, 7581.21it/s] 99%|█████████▉| 397530/400000 [00:57<00:00, 7470.94it/s]100%|█████████▉| 398279/400000 [00:57<00:00, 7294.43it/s]100%|█████████▉| 399011/400000 [00:57<00:00, 7192.70it/s]100%|█████████▉| 399732/400000 [00:57<00:00, 7142.64it/s]100%|█████████▉| 399999/400000 [00:57<00:00, 6924.12it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f2f8bc970f0>, <torchtext.data.dataset.TabularDataset object at 0x7f2f8bc97240>, <torchtext.vocab.Vocab object at 0x7f2f8bc97160>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  5.84 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  5.84 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  5.84 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.85 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.85 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  3.40 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  3.40 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.06 file/s]2020-06-07 06:19:06.481066: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-07 06:19:06.486171: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-07 06:19:06.486405: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x556d6851e6a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-07 06:19:06.486438: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['train', 'test', 'mnist2', 'cifar10', 'mnist_dataset_small.npy', 'fashion-mnist_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 29%|██▉       | 2899968/9912422 [00:00<00:00, 28999016.27it/s]9920512it [00:00, 34609895.61it/s]                             
0it [00:00, ?it/s]32768it [00:00, 357218.11it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 154235.86it/s]1654784it [00:00, 10955459.68it/s]                         
0it [00:00, ?it/s]8192it [00:00, 191267.85it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
