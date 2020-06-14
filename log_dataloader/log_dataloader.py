
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/525b5a33511bef1b32994ba341cc4ff09d4ffc92', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '525b5a33511bef1b32994ba341cc4ff09d4ffc92', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/525b5a33511bef1b32994ba341cc4ff09d4ffc92

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/525b5a33511bef1b32994ba341cc4ff09d4ffc92

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/525b5a33511bef1b32994ba341cc4ff09d4ffc92

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f3ad0da9620> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f3ad0da9620> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f3b3c3700d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f3b3c3700d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f3b5609cea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f3b5609cea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f3ae9bed950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f3ae9bed950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f3ae9bed950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:02, 157680.75it/s] 84%|████████▍ | 8355840/9912422 [00:00<00:06, 225074.89it/s]9920512it [00:00, 45755906.11it/s]                           
0it [00:00, ?it/s]32768it [00:00, 499206.19it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 460004.62it/s]1654784it [00:00, 10995133.70it/s]                         
0it [00:00, ?it/s]8192it [00:00, 224999.92it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3ae6e192b0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3ad0c77eb8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f3ae9bed598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f3ae9bed598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f3ae9bed598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:16,  2.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:16,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:15,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:15,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:14,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:14,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:13,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:52,  2.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:52,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:51,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:51,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:51,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:50,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:50,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:50,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:49,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:35,  4.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:35,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:34,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:34,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:34,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:34,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:33,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:33,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:33,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:33,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:00<00:23,  5.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:23,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:23,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:23,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:23,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:22,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:22,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:22,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:22,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:22,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:00<00:15,  8.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:15,  8.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:15,  8.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:15,  8.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:15,  8.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:15,  8.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:15,  8.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:15,  8.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:15,  8.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:14,  8.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:01<00:10, 11.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:10, 11.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:10, 11.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:10, 11.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:10, 11.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:10, 11.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 11.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 11.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 11.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███       | 50/162 [00:01<00:07, 14.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:07, 14.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:07, 14.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 14.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 14.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:07, 14.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 14.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:07, 14.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 14.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 19.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 19.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 19.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 19.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 19.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 19.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 19.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 19.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 19.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 25.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 25.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 25.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 25.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 25.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 25.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 25.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 25.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 25.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 31.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 31.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 31.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 31.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 31.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 31.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 31.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 31.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 31.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 38.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 38.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 38.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 38.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 38.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 38.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 38.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 38.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 38.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 38.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 45.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 45.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 45.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 45.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 45.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 45.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 45.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 45.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 45.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 52.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 52.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 52.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 52.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 52.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 52.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 52.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 52.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 52.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 58.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 58.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 58.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 58.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 58.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 58.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 58.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 58.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 58.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 58.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 63.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 63.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 63.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 63.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 63.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 63.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 63.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 63.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 63.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 63.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 68.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 68.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 68.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 68.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 68.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 68.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 68.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 68.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 68.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 69.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 69.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 69.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 69.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 69.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 69.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 69.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 69.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 69.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 72.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 72.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 72.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 72.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 72.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 72.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 72.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 72.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 72.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 72.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 74.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 74.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 74.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 74.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 74.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 74.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 74.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 74.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 74.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 76.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 76.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 76.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 76.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 76.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 76.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.54s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.54s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 76.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.54s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 76.10 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.64s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.54s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 76.10 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.64s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.64s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 34.91 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.64s/ url]
0 examples [00:00, ? examples/s]2020-06-14 18:09:01.217780: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-14 18:09:01.232524: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-06-14 18:09:01.232710: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56176aaf7b50 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-14 18:09:01.232728: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
21 examples [00:00, 209.68 examples/s]122 examples [00:00, 275.06 examples/s]231 examples [00:00, 354.31 examples/s]333 examples [00:00, 440.04 examples/s]434 examples [00:00, 529.23 examples/s]533 examples [00:00, 614.95 examples/s]633 examples [00:00, 694.55 examples/s]733 examples [00:00, 763.36 examples/s]826 examples [00:00, 773.67 examples/s]925 examples [00:01, 827.80 examples/s]1017 examples [00:01, 846.08 examples/s]1110 examples [00:01, 867.72 examples/s]1203 examples [00:01, 885.26 examples/s]1297 examples [00:01, 900.63 examples/s]1396 examples [00:01, 924.98 examples/s]1491 examples [00:01, 920.24 examples/s]1591 examples [00:01, 941.78 examples/s]1693 examples [00:01, 963.32 examples/s]1798 examples [00:01, 987.72 examples/s]1903 examples [00:02, 1003.26 examples/s]2004 examples [00:02, 989.72 examples/s] 2106 examples [00:02, 997.67 examples/s]2209 examples [00:02, 1004.47 examples/s]2311 examples [00:02, 1007.18 examples/s]2412 examples [00:02, 1002.04 examples/s]2515 examples [00:02, 1009.85 examples/s]2617 examples [00:02, 963.35 examples/s] 2714 examples [00:02, 957.79 examples/s]2811 examples [00:02, 956.46 examples/s]2915 examples [00:03, 977.98 examples/s]3014 examples [00:03, 973.89 examples/s]3112 examples [00:03, 967.85 examples/s]3209 examples [00:03, 962.89 examples/s]3311 examples [00:03, 976.77 examples/s]3414 examples [00:03, 989.95 examples/s]3514 examples [00:03, 989.46 examples/s]3614 examples [00:03, 983.10 examples/s]3713 examples [00:03, 980.97 examples/s]3816 examples [00:03, 994.90 examples/s]3922 examples [00:04, 1013.00 examples/s]4024 examples [00:04, 996.15 examples/s] 4126 examples [00:04, 1002.89 examples/s]4227 examples [00:04, 998.17 examples/s] 4335 examples [00:04, 1018.97 examples/s]4439 examples [00:04, 1022.40 examples/s]4542 examples [00:04, 992.26 examples/s] 4643 examples [00:04, 996.55 examples/s]4743 examples [00:04, 985.80 examples/s]4845 examples [00:05, 994.53 examples/s]4946 examples [00:05, 996.68 examples/s]5046 examples [00:05, 996.63 examples/s]5147 examples [00:05, 999.67 examples/s]5254 examples [00:05, 1017.84 examples/s]5362 examples [00:05, 1034.88 examples/s]5466 examples [00:05, 1021.72 examples/s]5569 examples [00:05, 1020.17 examples/s]5672 examples [00:05, 1011.48 examples/s]5776 examples [00:05, 1015.21 examples/s]5879 examples [00:06, 1017.27 examples/s]5984 examples [00:06, 1024.44 examples/s]6087 examples [00:06, 1018.34 examples/s]6189 examples [00:06, 1010.04 examples/s]6296 examples [00:06, 1026.22 examples/s]6399 examples [00:06, 1020.75 examples/s]6506 examples [00:06, 1033.48 examples/s]6610 examples [00:06, 1030.18 examples/s]6714 examples [00:06, 1029.30 examples/s]6819 examples [00:06, 1032.51 examples/s]6923 examples [00:07, 1029.29 examples/s]7026 examples [00:07, 1027.33 examples/s]7129 examples [00:07, 1007.78 examples/s]7237 examples [00:07, 1027.57 examples/s]7340 examples [00:07, 1014.25 examples/s]7442 examples [00:07, 1008.11 examples/s]7546 examples [00:07, 1015.06 examples/s]7648 examples [00:07, 1009.93 examples/s]7750 examples [00:07, 968.62 examples/s] 7848 examples [00:07, 960.48 examples/s]7953 examples [00:08, 983.35 examples/s]8057 examples [00:08, 998.73 examples/s]8158 examples [00:08, 981.98 examples/s]8261 examples [00:08, 995.24 examples/s]8364 examples [00:08, 1003.18 examples/s]8468 examples [00:08, 1013.41 examples/s]8570 examples [00:08, 1008.40 examples/s]8671 examples [00:08, 986.06 examples/s] 8772 examples [00:08, 990.35 examples/s]8877 examples [00:08, 1006.21 examples/s]8978 examples [00:09, 932.03 examples/s] 9073 examples [00:09, 935.68 examples/s]9177 examples [00:09, 963.57 examples/s]9279 examples [00:09, 979.23 examples/s]9382 examples [00:09, 993.85 examples/s]9483 examples [00:09, 998.33 examples/s]9584 examples [00:09, 979.29 examples/s]9684 examples [00:09, 983.53 examples/s]9786 examples [00:09, 994.06 examples/s]9895 examples [00:10, 1020.57 examples/s]9998 examples [00:10, 994.23 examples/s] 10098 examples [00:10, 940.10 examples/s]10198 examples [00:10, 955.33 examples/s]10300 examples [00:10, 971.18 examples/s]10405 examples [00:10, 992.66 examples/s]10506 examples [00:10, 995.47 examples/s]10609 examples [00:10, 1003.55 examples/s]10710 examples [00:10, 1002.76 examples/s]10813 examples [00:10, 1008.80 examples/s]10916 examples [00:11, 1012.43 examples/s]11018 examples [00:11, 1006.03 examples/s]11119 examples [00:11, 984.15 examples/s] 11225 examples [00:11, 1003.60 examples/s]11334 examples [00:11, 1025.91 examples/s]11437 examples [00:11, 1020.20 examples/s]11542 examples [00:11, 1027.29 examples/s]11645 examples [00:11, 1020.93 examples/s]11748 examples [00:11, 1020.50 examples/s]11855 examples [00:11, 1033.36 examples/s]11959 examples [00:12, 1018.73 examples/s]12061 examples [00:12, 1005.56 examples/s]12162 examples [00:12, 1002.37 examples/s]12263 examples [00:12, 996.99 examples/s] 12363 examples [00:12, 993.77 examples/s]12464 examples [00:12, 997.99 examples/s]12567 examples [00:12, 1005.32 examples/s]12668 examples [00:12, 989.12 examples/s] 12774 examples [00:12, 1007.53 examples/s]12875 examples [00:13, 977.83 examples/s] 12974 examples [00:13, 949.43 examples/s]13077 examples [00:13, 970.27 examples/s]13175 examples [00:13, 968.32 examples/s]13277 examples [00:13, 981.63 examples/s]13381 examples [00:13, 997.22 examples/s]13484 examples [00:13, 1004.84 examples/s]13585 examples [00:13, 996.53 examples/s] 13685 examples [00:13, 980.97 examples/s]13789 examples [00:13, 996.76 examples/s]13891 examples [00:14, 1002.72 examples/s]13998 examples [00:14, 1021.66 examples/s]14106 examples [00:14, 1037.30 examples/s]14210 examples [00:14, 1017.01 examples/s]14316 examples [00:14, 1027.22 examples/s]14420 examples [00:14, 1028.11 examples/s]14524 examples [00:14, 1030.55 examples/s]14628 examples [00:14, 1016.52 examples/s]14730 examples [00:14, 973.86 examples/s] 14839 examples [00:14, 1004.64 examples/s]14943 examples [00:15, 1014.68 examples/s]15045 examples [00:15, 1002.96 examples/s]15151 examples [00:15, 1015.71 examples/s]15253 examples [00:15, 998.30 examples/s] 15354 examples [00:15, 998.06 examples/s]15460 examples [00:15, 1013.46 examples/s]15565 examples [00:15, 1023.52 examples/s]15672 examples [00:15, 1036.89 examples/s]15777 examples [00:15, 1038.46 examples/s]15881 examples [00:15, 1037.23 examples/s]15987 examples [00:16, 1042.36 examples/s]16092 examples [00:16, 1042.71 examples/s]16197 examples [00:16, 1044.43 examples/s]16302 examples [00:16, 1028.22 examples/s]16405 examples [00:16, 994.33 examples/s] 16505 examples [00:16, 994.06 examples/s]16609 examples [00:16, 1006.59 examples/s]16715 examples [00:16, 1016.36 examples/s]16817 examples [00:16, 1005.51 examples/s]16918 examples [00:17, 1005.80 examples/s]17019 examples [00:17, 996.72 examples/s] 17121 examples [00:17, 1002.47 examples/s]17222 examples [00:17, 995.64 examples/s] 17325 examples [00:17, 1005.51 examples/s]17430 examples [00:17, 1017.46 examples/s]17532 examples [00:17, 1004.81 examples/s]17633 examples [00:17, 1004.65 examples/s]17736 examples [00:17, 1007.23 examples/s]17837 examples [00:17, 998.23 examples/s] 17944 examples [00:18, 1017.43 examples/s]18046 examples [00:18, 1001.28 examples/s]18147 examples [00:18, 945.69 examples/s] 18243 examples [00:18, 933.97 examples/s]18351 examples [00:18, 972.20 examples/s]18456 examples [00:18, 994.25 examples/s]18559 examples [00:18, 1004.33 examples/s]18660 examples [00:18, 979.22 examples/s] 18760 examples [00:18, 985.05 examples/s]18863 examples [00:18, 998.01 examples/s]18964 examples [00:19, 990.86 examples/s]19066 examples [00:19, 997.81 examples/s]19174 examples [00:19, 1018.39 examples/s]19277 examples [00:19, 1007.07 examples/s]19380 examples [00:19, 1012.46 examples/s]19487 examples [00:19, 1027.26 examples/s]19592 examples [00:19, 1031.15 examples/s]19696 examples [00:19, 1030.41 examples/s]19800 examples [00:19, 1031.13 examples/s]19904 examples [00:19, 1020.02 examples/s]20007 examples [00:20, 970.47 examples/s] 20107 examples [00:20, 979.11 examples/s]20210 examples [00:20, 991.19 examples/s]20310 examples [00:20, 987.36 examples/s]20414 examples [00:20, 1000.78 examples/s]20515 examples [00:20, 995.24 examples/s] 20620 examples [00:20, 1009.36 examples/s]20724 examples [00:20, 1016.38 examples/s]20826 examples [00:20, 1003.61 examples/s]20928 examples [00:21, 1007.75 examples/s]21030 examples [00:21, 1011.05 examples/s]21135 examples [00:21, 1021.99 examples/s]21238 examples [00:21, 989.59 examples/s] 21343 examples [00:21, 1004.43 examples/s]21452 examples [00:21, 1026.14 examples/s]21555 examples [00:21, 1016.74 examples/s]21661 examples [00:21, 1027.76 examples/s]21765 examples [00:21, 1029.10 examples/s]21869 examples [00:21, 1019.05 examples/s]21972 examples [00:22, 1020.27 examples/s]22076 examples [00:22, 1023.84 examples/s]22183 examples [00:22, 1036.01 examples/s]22292 examples [00:22, 1049.46 examples/s]22398 examples [00:22, 1037.15 examples/s]22502 examples [00:22, 1031.72 examples/s]22606 examples [00:22, 1022.05 examples/s]22709 examples [00:22, 1022.15 examples/s]22812 examples [00:22, 1023.44 examples/s]22915 examples [00:22, 1003.12 examples/s]23016 examples [00:23, 1001.90 examples/s]23117 examples [00:23, 1000.30 examples/s]23221 examples [00:23, 1011.87 examples/s]23323 examples [00:23, 978.62 examples/s] 23422 examples [00:23, 976.59 examples/s]23528 examples [00:23, 998.67 examples/s]23631 examples [00:23, 1005.89 examples/s]23732 examples [00:23, 1004.36 examples/s]23838 examples [00:23, 1018.35 examples/s]23940 examples [00:23, 1008.46 examples/s]24048 examples [00:24, 1027.85 examples/s]24155 examples [00:24, 1038.20 examples/s]24259 examples [00:24, 1024.47 examples/s]24362 examples [00:24, 998.92 examples/s] 24463 examples [00:24, 986.12 examples/s]24568 examples [00:24, 1004.20 examples/s]24672 examples [00:24, 1013.68 examples/s]24774 examples [00:24, 1012.70 examples/s]24879 examples [00:24, 1020.65 examples/s]24982 examples [00:25, 1012.37 examples/s]25086 examples [00:25, 1020.49 examples/s]25192 examples [00:25, 1029.38 examples/s]25296 examples [00:25, 1029.15 examples/s]25399 examples [00:25, 1008.04 examples/s]25500 examples [00:25, 1008.62 examples/s]25602 examples [00:25, 1011.44 examples/s]25706 examples [00:25, 1017.69 examples/s]25809 examples [00:25, 1018.34 examples/s]25911 examples [00:25, 1014.67 examples/s]26013 examples [00:26, 998.14 examples/s] 26113 examples [00:26, 996.75 examples/s]26213 examples [00:26, 996.12 examples/s]26315 examples [00:26, 1002.83 examples/s]26416 examples [00:26, 1003.95 examples/s]26517 examples [00:26, 998.75 examples/s] 26617 examples [00:26, 992.56 examples/s]26727 examples [00:26, 1021.96 examples/s]26833 examples [00:26, 1032.34 examples/s]26937 examples [00:26, 1032.40 examples/s]27041 examples [00:27, 1012.80 examples/s]27143 examples [00:27, 1005.26 examples/s]27245 examples [00:27, 1007.89 examples/s]27346 examples [00:27, 1002.02 examples/s]27447 examples [00:27, 977.49 examples/s] 27548 examples [00:27, 985.80 examples/s]27652 examples [00:27, 999.55 examples/s]27753 examples [00:27, 996.28 examples/s]27857 examples [00:27, 1007.98 examples/s]27958 examples [00:27, 1008.03 examples/s]28059 examples [00:28, 976.32 examples/s] 28160 examples [00:28, 983.62 examples/s]28263 examples [00:28, 995.21 examples/s]28363 examples [00:28, 991.24 examples/s]28467 examples [00:28, 1004.98 examples/s]28568 examples [00:28, 957.43 examples/s] 28665 examples [00:28, 952.80 examples/s]28763 examples [00:28, 959.79 examples/s]28865 examples [00:28, 976.09 examples/s]28966 examples [00:29, 983.20 examples/s]29069 examples [00:29, 996.22 examples/s]29171 examples [00:29, 1003.08 examples/s]29272 examples [00:29, 987.65 examples/s] 29376 examples [00:29, 1001.97 examples/s]29478 examples [00:29, 1004.86 examples/s]29581 examples [00:29, 1009.94 examples/s]29684 examples [00:29, 1013.47 examples/s]29786 examples [00:29, 985.38 examples/s] 29885 examples [00:29, 985.16 examples/s]29984 examples [00:30, 975.93 examples/s]30082 examples [00:30, 920.11 examples/s]30183 examples [00:30, 945.12 examples/s]30284 examples [00:30, 961.72 examples/s]30382 examples [00:30, 965.98 examples/s]30479 examples [00:30, 964.71 examples/s]30577 examples [00:30, 967.57 examples/s]30676 examples [00:30, 973.94 examples/s]30774 examples [00:30, 967.65 examples/s]30873 examples [00:30, 972.42 examples/s]30971 examples [00:31, 958.81 examples/s]31076 examples [00:31, 983.51 examples/s]31177 examples [00:31, 989.20 examples/s]31282 examples [00:31, 1005.56 examples/s]31384 examples [00:31, 1007.92 examples/s]31485 examples [00:31, 999.46 examples/s] 31588 examples [00:31, 1006.33 examples/s]31689 examples [00:31, 998.31 examples/s] 31792 examples [00:31, 1003.82 examples/s]31896 examples [00:31, 1012.28 examples/s]31998 examples [00:32, 995.90 examples/s] 32101 examples [00:32, 1004.74 examples/s]32205 examples [00:32, 1014.44 examples/s]32307 examples [00:32, 1015.86 examples/s]32410 examples [00:32, 1017.36 examples/s]32512 examples [00:32, 1012.56 examples/s]32614 examples [00:32, 1007.44 examples/s]32720 examples [00:32, 1022.49 examples/s]32823 examples [00:32, 1022.48 examples/s]32932 examples [00:32, 1041.39 examples/s]33037 examples [00:33, 1020.45 examples/s]33145 examples [00:33, 1035.28 examples/s]33254 examples [00:33, 1050.10 examples/s]33360 examples [00:33, 1044.49 examples/s]33471 examples [00:33, 1061.66 examples/s]33578 examples [00:33, 1026.00 examples/s]33681 examples [00:33, 960.62 examples/s] 33779 examples [00:33, 964.02 examples/s]33881 examples [00:33, 979.95 examples/s]33985 examples [00:34, 994.90 examples/s]34090 examples [00:34, 1008.90 examples/s]34192 examples [00:34, 1010.57 examples/s]34301 examples [00:34, 1030.92 examples/s]34409 examples [00:34, 1045.16 examples/s]34514 examples [00:34, 1040.49 examples/s]34619 examples [00:34, 1014.02 examples/s]34728 examples [00:34, 1035.07 examples/s]34835 examples [00:34, 1043.21 examples/s]34940 examples [00:34, 1037.64 examples/s]35044 examples [00:35, 1032.98 examples/s]35148 examples [00:35, 1015.08 examples/s]35250 examples [00:35, 1012.80 examples/s]35354 examples [00:35, 1020.59 examples/s]35460 examples [00:35, 1029.89 examples/s]35565 examples [00:35, 1033.08 examples/s]35669 examples [00:35, 1004.92 examples/s]35771 examples [00:35, 1008.17 examples/s]35873 examples [00:35, 1011.40 examples/s]35977 examples [00:35, 1018.60 examples/s]36081 examples [00:36, 1023.88 examples/s]36184 examples [00:36, 1001.64 examples/s]36285 examples [00:36, 1001.91 examples/s]36394 examples [00:36, 1024.17 examples/s]36497 examples [00:36, 1024.48 examples/s]36600 examples [00:36, 1025.72 examples/s]36703 examples [00:36, 985.16 examples/s] 36809 examples [00:36, 1004.78 examples/s]36912 examples [00:36, 1012.13 examples/s]37015 examples [00:36, 1015.44 examples/s]37118 examples [00:37, 1019.62 examples/s]37223 examples [00:37, 1026.69 examples/s]37330 examples [00:37, 1039.29 examples/s]37435 examples [00:37, 1030.42 examples/s]37539 examples [00:37, 1028.41 examples/s]37642 examples [00:37, 1027.27 examples/s]37745 examples [00:37, 1023.70 examples/s]37848 examples [00:37, 1020.62 examples/s]37956 examples [00:37, 1035.12 examples/s]38060 examples [00:38, 1014.99 examples/s]38168 examples [00:38, 1031.49 examples/s]38272 examples [00:38, 1021.62 examples/s]38377 examples [00:38, 1027.89 examples/s]38482 examples [00:38, 1033.70 examples/s]38586 examples [00:38, 1021.88 examples/s]38689 examples [00:38, 1006.56 examples/s]38790 examples [00:38, 970.86 examples/s] 38888 examples [00:38, 954.42 examples/s]38984 examples [00:38, 924.34 examples/s]39084 examples [00:39, 945.65 examples/s]39192 examples [00:39, 981.96 examples/s]39292 examples [00:39, 985.12 examples/s]39395 examples [00:39, 996.13 examples/s]39498 examples [00:39, 1003.16 examples/s]39599 examples [00:39, 994.91 examples/s] 39708 examples [00:39, 1019.74 examples/s]39811 examples [00:39, 986.53 examples/s] 39919 examples [00:39, 1010.78 examples/s]40021 examples [00:40, 964.41 examples/s] 40123 examples [00:40, 979.24 examples/s]40223 examples [00:40, 984.86 examples/s]40322 examples [00:40, 973.73 examples/s]40426 examples [00:40, 990.72 examples/s]40528 examples [00:40, 998.21 examples/s]40630 examples [00:40, 1004.53 examples/s]40732 examples [00:40, 1007.80 examples/s]40833 examples [00:40, 992.67 examples/s] 40934 examples [00:40, 995.69 examples/s]41034 examples [00:41, 991.71 examples/s]41137 examples [00:41, 1002.78 examples/s]41238 examples [00:41, 1004.57 examples/s]41339 examples [00:41, 990.77 examples/s] 41442 examples [00:41, 1000.50 examples/s]41551 examples [00:41, 1025.04 examples/s]41658 examples [00:41, 1037.75 examples/s]41763 examples [00:41, 1038.82 examples/s]41868 examples [00:41, 1023.70 examples/s]41973 examples [00:41, 1030.10 examples/s]42077 examples [00:42, 1032.35 examples/s]42181 examples [00:42, 1013.53 examples/s]42283 examples [00:42, 988.18 examples/s] 42386 examples [00:42, 998.20 examples/s]42490 examples [00:42, 1008.45 examples/s]42598 examples [00:42, 1026.45 examples/s]42701 examples [00:42, 1027.24 examples/s]42804 examples [00:42, 1025.13 examples/s]42907 examples [00:42, 1001.81 examples/s]43011 examples [00:42, 1011.99 examples/s]43113 examples [00:43, 1010.03 examples/s]43215 examples [00:43, 1011.18 examples/s]43319 examples [00:43, 1018.48 examples/s]43423 examples [00:43, 1023.22 examples/s]43526 examples [00:43, 1019.12 examples/s]43630 examples [00:43, 1025.15 examples/s]43733 examples [00:43, 1025.83 examples/s]43836 examples [00:43, 1002.90 examples/s]43937 examples [00:43, 988.15 examples/s] 44038 examples [00:43, 992.65 examples/s]44138 examples [00:44, 959.56 examples/s]44235 examples [00:44, 962.11 examples/s]44332 examples [00:44, 925.15 examples/s]44427 examples [00:44, 929.89 examples/s]44529 examples [00:44, 952.93 examples/s]44630 examples [00:44, 968.36 examples/s]44735 examples [00:44, 991.24 examples/s]44835 examples [00:44, 962.82 examples/s]44932 examples [00:44, 945.40 examples/s]45032 examples [00:45, 960.83 examples/s]45134 examples [00:45, 976.13 examples/s]45243 examples [00:45, 1006.45 examples/s]45348 examples [00:45, 1017.11 examples/s]45451 examples [00:45, 1012.52 examples/s]45554 examples [00:45, 1016.27 examples/s]45656 examples [00:45, 1014.58 examples/s]45758 examples [00:45, 1015.62 examples/s]45860 examples [00:45, 1004.00 examples/s]45961 examples [00:45, 975.62 examples/s] 46059 examples [00:46, 973.75 examples/s]46157 examples [00:46, 966.43 examples/s]46258 examples [00:46, 977.18 examples/s]46363 examples [00:46, 995.60 examples/s]46470 examples [00:46, 1015.54 examples/s]46575 examples [00:46, 1023.60 examples/s]46683 examples [00:46, 1039.13 examples/s]46788 examples [00:46, 1036.93 examples/s]46892 examples [00:46, 1016.53 examples/s]46994 examples [00:46, 998.94 examples/s] 47096 examples [00:47, 1002.57 examples/s]47199 examples [00:47, 1008.12 examples/s]47304 examples [00:47, 1020.08 examples/s]47407 examples [00:47, 1012.14 examples/s]47509 examples [00:47, 1009.51 examples/s]47611 examples [00:47, 1004.95 examples/s]47712 examples [00:47, 996.68 examples/s] 47814 examples [00:47, 1001.13 examples/s]47919 examples [00:47, 1015.08 examples/s]48021 examples [00:47, 1013.01 examples/s]48123 examples [00:48, 1007.77 examples/s]48224 examples [00:48, 1005.07 examples/s]48329 examples [00:48, 1017.55 examples/s]48434 examples [00:48, 1023.19 examples/s]48537 examples [00:48, 1013.27 examples/s]48639 examples [00:48, 1011.07 examples/s]48741 examples [00:48, 1004.10 examples/s]48842 examples [00:48, 1005.61 examples/s]48943 examples [00:48, 976.81 examples/s] 49050 examples [00:49, 1001.37 examples/s]49151 examples [00:49, 988.50 examples/s] 49251 examples [00:49, 952.23 examples/s]49347 examples [00:49, 942.82 examples/s]49443 examples [00:49, 946.50 examples/s]49547 examples [00:49, 971.00 examples/s]49656 examples [00:49, 1002.27 examples/s]49758 examples [00:49, 1005.87 examples/s]49859 examples [00:49, 1006.92 examples/s]49960 examples [00:49, 1002.76 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 16%|█▌        | 7850/50000 [00:00<00:00, 78498.95 examples/s] 40%|████      | 20236/50000 [00:00<00:00, 88186.94 examples/s] 65%|██████▌   | 32608/50000 [00:00<00:00, 96500.57 examples/s] 90%|█████████ | 45120/50000 [00:00<00:00, 103608.70 examples/s]                                                                0 examples [00:00, ? examples/s]82 examples [00:00, 813.69 examples/s]188 examples [00:00, 874.52 examples/s]291 examples [00:00, 915.15 examples/s]395 examples [00:00, 949.18 examples/s]507 examples [00:00, 993.26 examples/s]608 examples [00:00, 995.96 examples/s]712 examples [00:00, 1007.17 examples/s]821 examples [00:00, 1028.75 examples/s]924 examples [00:00, 1026.55 examples/s]1026 examples [00:01, 1022.94 examples/s]1127 examples [00:01, 1007.12 examples/s]1229 examples [00:01, 1009.74 examples/s]1337 examples [00:01, 1029.12 examples/s]1440 examples [00:01, 827.85 examples/s] 1534 examples [00:01, 857.34 examples/s]1639 examples [00:01, 905.08 examples/s]1742 examples [00:01, 937.98 examples/s]1844 examples [00:01, 959.00 examples/s]1948 examples [00:01, 980.64 examples/s]2050 examples [00:02, 991.00 examples/s]2154 examples [00:02, 1003.52 examples/s]2256 examples [00:02, 1006.74 examples/s]2361 examples [00:02, 1017.51 examples/s]2464 examples [00:02, 1020.74 examples/s]2567 examples [00:02, 1018.24 examples/s]2670 examples [00:02, 1016.75 examples/s]2772 examples [00:02, 1014.27 examples/s]2874 examples [00:02, 999.27 examples/s] 2980 examples [00:03, 1014.78 examples/s]3082 examples [00:03, 995.58 examples/s] 3186 examples [00:03, 1007.16 examples/s]3289 examples [00:03, 1012.34 examples/s]3391 examples [00:03, 980.12 examples/s] 3493 examples [00:03, 989.23 examples/s]3593 examples [00:03, 986.61 examples/s]3696 examples [00:03, 997.49 examples/s]3801 examples [00:03, 1012.34 examples/s]3903 examples [00:03, 999.43 examples/s] 4004 examples [00:04, 971.39 examples/s]4102 examples [00:04, 967.18 examples/s]4199 examples [00:04, 963.32 examples/s]4302 examples [00:04, 981.44 examples/s]4406 examples [00:04, 996.96 examples/s]4507 examples [00:04, 999.38 examples/s]4608 examples [00:04, 983.83 examples/s]4709 examples [00:04, 991.08 examples/s]4810 examples [00:04, 994.09 examples/s]4910 examples [00:04, 981.31 examples/s]5012 examples [00:05, 990.51 examples/s]5115 examples [00:05, 999.40 examples/s]5218 examples [00:05, 1006.44 examples/s]5329 examples [00:05, 1033.70 examples/s]5433 examples [00:05, 1006.25 examples/s]5537 examples [00:05, 1015.94 examples/s]5639 examples [00:05, 1007.33 examples/s]5748 examples [00:05, 1028.92 examples/s]5855 examples [00:05, 1039.00 examples/s]5961 examples [00:05, 1043.31 examples/s]6066 examples [00:06, 1044.93 examples/s]6171 examples [00:06, 1030.87 examples/s]6275 examples [00:06, 1032.46 examples/s]6382 examples [00:06, 1041.14 examples/s]6487 examples [00:06, 1032.80 examples/s]6591 examples [00:06, 1032.94 examples/s]6695 examples [00:06, 1011.06 examples/s]6807 examples [00:06, 1039.90 examples/s]6913 examples [00:06, 1044.74 examples/s]7018 examples [00:07, 1033.98 examples/s]7122 examples [00:07, 1033.46 examples/s]7226 examples [00:07, 1023.51 examples/s]7335 examples [00:07, 1041.99 examples/s]7440 examples [00:07, 1028.96 examples/s]7544 examples [00:07, 1021.70 examples/s]7647 examples [00:07, 1002.60 examples/s]7748 examples [00:07, 976.01 examples/s] 7849 examples [00:07, 984.34 examples/s]7950 examples [00:07, 989.62 examples/s]8053 examples [00:08, 998.95 examples/s]8160 examples [00:08, 1016.65 examples/s]8262 examples [00:08, 1010.80 examples/s]8364 examples [00:08, 1003.62 examples/s]8465 examples [00:08, 1001.85 examples/s]8570 examples [00:08, 1014.42 examples/s]8674 examples [00:08, 1020.86 examples/s]8778 examples [00:08, 1023.87 examples/s]8881 examples [00:08, 1022.75 examples/s]8984 examples [00:08, 1016.88 examples/s]9088 examples [00:09, 1021.34 examples/s]9191 examples [00:09, 982.04 examples/s] 9290 examples [00:09, 933.19 examples/s]9386 examples [00:09, 940.88 examples/s]9481 examples [00:09, 931.73 examples/s]9581 examples [00:09, 949.15 examples/s]9689 examples [00:09, 984.86 examples/s]9789 examples [00:09, 987.54 examples/s]9893 examples [00:09, 1002.20 examples/s]9994 examples [00:09, 988.86 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteBH8UYT/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteBH8UYT/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f3ae9bed950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f3ae9bed950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f3ae9bed950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3a7c135dd8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3a6ef7ca90>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f3ae9bed950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f3ae9bed950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f3ae9bed950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3a7c11e748>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3acffaa0b8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f3a615ac268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f3a615ac268> 

  function with postional parmater data_info <function split_train_valid at 0x7f3a615ac268> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f3a615ac378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f3a615ac378> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f3a615ac378> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=e7b0b9758759907b6f511633b261cbc04f2454e5991122c3ef2f24ff8d4d2d5d
  Stored in directory: /tmp/pip-ephem-wheel-cache-glldwi0d/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.2.5
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<22:15:25, 10.8kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:48:45, 15.1kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:07:22, 21.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 885k/862M [00:01<7:47:40, 30.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.56M/862M [00:01<5:26:34, 43.8kB/s].vector_cache/glove.6B.zip:   1%|          | 8.26M/862M [00:01<3:47:26, 62.6kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.0M/862M [00:01<2:38:38, 89.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.4M/862M [00:01<1:50:25, 128kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.7M/862M [00:01<1:17:07, 182kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.9M/862M [00:01<53:44, 259kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.2M/862M [00:01<37:35, 369kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.3M/862M [00:02<26:14, 526kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.7M/862M [00:02<18:25, 746kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.1M/862M [00:02<12:53, 1.06MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.2M/862M [00:02<09:07, 1.49MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.5M/862M [00:02<06:25, 2.11MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.9M/862M [00:02<07:47, 1.73MB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<07:20, 1.83MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:04<09:02, 1.48MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:04<07:07, 1.89MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.3M/862M [00:05<05:14, 2.56MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:05<04:52, 2.74MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:06<7:55:44, 28.1kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:06<5:32:57, 40.1kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:08<3:54:44, 56.7kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:08<2:46:48, 79.8kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:08<1:57:10, 113kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 67.2M/862M [00:08<1:21:56, 162kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.9M/862M [00:10<1:06:32, 199kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:10<49:30, 267kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:10<35:21, 374kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.7M/862M [00:10<24:50, 530kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.0M/862M [00:12<38:04, 346kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:12<27:59, 470kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.0M/862M [00:12<19:53, 660kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.2M/862M [00:14<16:57, 773kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:14<14:38, 894kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:14<10:55, 1.20MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:14<07:46, 1.68MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.3M/862M [00:16<1:28:51, 147kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:16<1:03:32, 205kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.2M/862M [00:16<44:43, 291kB/s]  .vector_cache/glove.6B.zip:  10%|▉         | 84.4M/862M [00:18<34:15, 378kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:18<25:17, 512kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.3M/862M [00:18<18:00, 718kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.5M/862M [00:20<15:35, 827kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:20<12:13, 1.05MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.5M/862M [00:20<08:52, 1.45MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.6M/862M [00:22<09:13, 1.39MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:22<07:47, 1.65MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.6M/862M [00:22<05:46, 2.22MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.7M/862M [00:24<07:02, 1.81MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:24<07:32, 1.69MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<05:55, 2.15MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<06:11, 2.05MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<05:39, 2.24MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<04:16, 2.96MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<05:56, 2.12MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<06:46, 1.86MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:20, 2.36MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<03:52, 3.24MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<10:14, 1.23MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<08:29, 1.48MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<06:15, 2.00MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<07:16, 1.71MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<07:42, 1.62MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:01, 2.07MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<06:12, 2.00MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:36, 2.21MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<04:14, 2.92MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<05:55, 2.08MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<04:59, 2.47MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<03:39, 3.37MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<08:44, 1.40MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<07:35, 1.62MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:40, 2.16MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<06:30, 1.88MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<07:26, 1.64MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:50, 2.09MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<04:14, 2.86MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<08:11, 1.48MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<06:58, 1.74MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<05:11, 2.33MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<06:25, 1.88MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<06:57, 1.74MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:28, 2.20MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<05:46, 2.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<05:18, 2.26MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<04:00, 2.98MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<05:35, 2.13MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<06:27, 1.85MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<05:07, 2.32MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<04:13, 2.81MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<7:03:56, 28.0kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<4:56:40, 40.0kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<3:29:01, 56.5kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<2:29:02, 79.2kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<1:44:51, 112kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:51<1:13:18, 160kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<56:49, 206kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<41:03, 286kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<29:00, 403kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<22:47, 512kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<18:19, 636kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<13:24, 869kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<11:13, 1.03MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<09:02, 1.28MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:57<06:36, 1.75MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<07:19, 1.57MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<07:28, 1.54MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<05:42, 2.02MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [00:59<04:08, 2.77MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<08:22, 1.37MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<07:03, 1.62MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:01<05:13, 2.19MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<06:18, 1.80MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<05:23, 2.11MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<04:07, 2.76MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:03<03:00, 3.77MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<44:54, 252kB/s] .vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<33:44, 336kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<24:09, 468kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<18:39, 603kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<14:12, 792kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<10:12, 1.10MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<09:44, 1.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<09:05, 1.23MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:50, 1.63MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:09<04:55, 2.26MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<09:39, 1.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<07:54, 1.40MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:48, 1.91MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<06:38, 1.66MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<06:54, 1.60MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<05:23, 2.05MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<05:32, 1.98MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<04:59, 2.20MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<03:46, 2.91MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<05:11, 2.10MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<04:44, 2.30MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<03:35, 3.04MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<05:04, 2.14MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<05:45, 1.88MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<04:29, 2.41MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<03:17, 3.28MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<07:53, 1.37MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<06:26, 1.67MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<04:43, 2.28MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:21<03:26, 3.11MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:22<45:32, 235kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<34:04, 314kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<24:22, 439kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<18:42, 569kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<14:10, 750kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<10:07, 1.05MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<09:33, 1.11MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<08:52, 1.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<06:40, 1.58MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<04:45, 2.21MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<28:05, 374kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<20:44, 506kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<14:43, 711kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<10:58, 951kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<7:02:16, 24.7kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<4:55:21, 35.3kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<3:27:45, 49.9kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<2:27:56, 70.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<1:44:03, 99.5kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:32<1:12:42, 142kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<56:27, 182kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<40:44, 253kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<28:42, 358kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<22:10, 462kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<17:40, 579kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<12:53, 792kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:36<09:06, 1.12MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<1:19:00, 129kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<56:19, 180kB/s]  .vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<39:35, 256kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<29:59, 337kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<22:00, 458kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<15:37, 644kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<13:16, 756kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<11:19, 884kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<08:21, 1.20MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:42<05:58, 1.67MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<08:38, 1.15MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<07:03, 1.41MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<05:11, 1.91MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<05:55, 1.67MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<06:09, 1.60MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:48, 2.05MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<04:56, 1.99MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<04:28, 2.19MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<03:22, 2.90MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<04:38, 2.10MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<04:14, 2.30MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<03:10, 3.06MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<04:30, 2.15MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<05:06, 1.89MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:00, 2.41MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:52<02:53, 3.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<37:27, 257kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<27:12, 353kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<19:14, 498kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<15:39, 610kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<12:54, 739kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<09:27, 1.01MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<06:41, 1.42MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<19:46, 479kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<14:47, 640kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<10:34, 894kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<09:34, 982kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<08:37, 1.09MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<06:26, 1.46MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<04:38, 2.02MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:01<06:58, 1.34MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<05:49, 1.60MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<04:18, 2.16MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<05:10, 1.79MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<04:33, 2.03MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<03:25, 2.70MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:33, 2.02MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:07, 2.23MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<03:07, 2.94MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:20, 2.11MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:53, 1.87MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<03:48, 2.40MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<02:47, 3.25MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<05:35, 1.62MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<04:51, 1.87MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<03:35, 2.52MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<03:06, 2.90MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<6:00:18, 25.0kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<4:12:32, 35.6kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:11<2:56:07, 50.9kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<2:07:02, 70.4kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<1:30:54, 98.3kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<1:04:00, 139kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:13<44:44, 199kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<35:10, 252kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<25:33, 347kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<18:03, 490kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<14:32, 605kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<11:05, 793kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<07:58, 1.10MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<07:34, 1.15MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<07:04, 1.23MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<05:23, 1.62MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<05:09, 1.68MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:30, 1.92MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<03:20, 2.58MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<04:19, 1.99MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<04:46, 1.80MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<03:46, 2.27MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:00, 2.12MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<03:41, 2.30MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<02:48, 3.03MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<03:54, 2.16MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:27, 1.89MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<03:33, 2.38MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<02:33, 3.28MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<1:02:33, 134kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<44:36, 188kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<31:20, 267kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<23:47, 350kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<18:20, 453kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<13:12, 629kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:31<09:16, 890kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<26:42, 309kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<19:32, 422kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<13:50, 594kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<11:34, 707kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<08:56, 915kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<06:24, 1.27MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<06:24, 1.27MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<06:08, 1.32MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:42, 1.72MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<03:22, 2.38MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<59:48, 135kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<42:39, 188kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<29:56, 268kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:40<22:44, 351kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<16:43, 477kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<11:52, 669kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<10:08, 780kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<07:54, 999kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<05:43, 1.38MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<05:50, 1.34MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<05:40, 1.38MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:18, 1.82MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<03:06, 2.51MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<06:51, 1.13MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<05:36, 1.39MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:05, 1.89MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<04:39, 1.65MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<04:02, 1.90MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:01, 2.54MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<03:54, 1.95MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:17, 1.78MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<03:23, 2.25MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<02:43, 2.78MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<4:42:10, 26.8kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<3:17:07, 38.3kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:52<2:17:16, 54.7kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<1:59:18, 62.9kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<1:24:56, 88.3kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<59:43, 125kB/s]   .vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<42:44, 174kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<30:40, 242kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<21:35, 343kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<16:44, 440kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<12:27, 591kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<08:52, 826kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<07:54, 923kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<07:01, 1.04MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<05:17, 1.38MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:00<03:45, 1.93MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<11:48, 613kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<08:59, 803kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<06:27, 1.11MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<06:10, 1.16MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<05:03, 1.42MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<03:40, 1.94MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<04:13, 1.68MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<04:24, 1.61MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<03:23, 2.08MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<02:27, 2.87MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<06:29, 1.08MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<05:17, 1.33MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<03:52, 1.81MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<04:19, 1.61MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<03:44, 1.86MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<02:47, 2.48MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<03:33, 1.94MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<03:11, 2.16MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<02:22, 2.89MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<03:16, 2.08MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<02:52, 2.37MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<02:08, 3.16MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:14<01:35, 4.25MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<26:37, 254kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<20:00, 337kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<14:16, 472kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<10:01, 668kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<10:34, 632kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<08:04, 827kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<05:46, 1.15MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<05:34, 1.19MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<04:34, 1.45MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:19, 1.98MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:52, 1.69MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<03:23, 1.93MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<02:31, 2.57MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<03:17, 1.97MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<02:58, 2.18MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<02:14, 2.88MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:04, 2.08MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:47, 2.29MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<02:06, 3.02MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:58, 2.13MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:43, 2.32MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:03, 3.06MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:55, 2.15MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:41, 2.33MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<02:01, 3.07MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<01:53, 3.29MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<3:54:42, 26.5kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<2:43:58, 37.8kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<1:55:01, 53.4kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<1:21:54, 75.0kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<57:33, 106kB/s]   .vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<40:09, 152kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<30:10, 201kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<21:45, 279kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<15:18, 395kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<12:00, 500kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<09:37, 623kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<06:59, 857kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:37<04:56, 1.21MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<07:10, 826kB/s] .vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<05:38, 1.05MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<04:05, 1.45MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<04:13, 1.39MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<04:08, 1.41MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<03:08, 1.86MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:41<02:17, 2.55MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<03:46, 1.53MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:15, 1.78MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<02:24, 2.39MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:00, 1.90MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:16, 1.75MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<02:34, 2.21MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<02:42, 2.09MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<02:28, 2.29MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<01:52, 3.01MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:36, 2.14MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:23, 2.33MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<01:48, 3.07MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:34, 2.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:21, 2.34MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<01:47, 3.08MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:32, 2.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:20, 2.33MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<01:45, 3.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<02:30, 2.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<02:17, 2.34MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<01:44, 3.08MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:28, 2.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<02:48, 1.89MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:11, 2.41MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<01:35, 3.30MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<04:02, 1.30MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<03:21, 1.56MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:27, 2.13MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:56, 1.77MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<03:04, 1.69MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:22, 2.17MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:29, 2.06MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<02:16, 2.24MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<01:42, 2.99MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:21, 2.13MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:41, 1.87MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:06, 2.39MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<01:31, 3.28MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<04:36, 1.08MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<03:43, 1.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:43, 1.81MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<03:02, 1.61MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:37, 1.86MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<01:57, 2.50MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:30, 1.93MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:43, 1.77MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<02:07, 2.26MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<01:42, 2.78MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<2:58:42, 26.7kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<2:04:45, 38.1kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<1:27:17, 54.0kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<1:02:21, 75.5kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<43:52, 107kB/s]   .vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:14<30:31, 153kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<23:27, 198kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<16:57, 273kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<11:56, 386kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<09:14, 495kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<07:23, 618kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<05:21, 849kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:18<03:46, 1.19MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<05:52, 766kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<04:34, 981kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<03:17, 1.36MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<03:19, 1.33MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<03:13, 1.37MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:28, 1.78MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:25, 1.80MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:08, 2.03MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<01:35, 2.72MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:06, 2.04MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:20, 1.83MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<01:50, 2.33MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:26<01:18, 3.23MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<21:39, 195kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<15:34, 271kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<10:55, 384kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<08:33, 486kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<06:20, 655kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<04:31, 913kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<04:06, 996kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<03:42, 1.10MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:47, 1.46MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:32<01:59, 2.02MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<30:06, 134kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<21:27, 187kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<15:01, 266kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<11:20, 349kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<08:45, 451kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<06:18, 624kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:36<04:24, 881kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<30:47, 126kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<21:55, 177kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<15:19, 251kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<11:30, 331kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:40<08:26, 452kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<05:56, 636kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<05:01, 747kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<04:16, 876kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<03:09, 1.18MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<02:12, 1.66MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<09:53, 372kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<07:17, 503kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<05:09, 706kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<04:25, 816kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<03:27, 1.04MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:29, 1.44MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:33, 1.39MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:08, 1.65MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:34, 2.22MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:55, 1.81MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:41, 2.04MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<01:16, 2.71MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:07, 3.05MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<2:12:23, 25.8kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<1:32:15, 36.8kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<1:04:16, 52.1kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<45:43, 73.1kB/s]  .vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<32:03, 104kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<22:20, 148kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<16:24, 200kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<11:48, 277kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<08:17, 392kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<06:27, 497kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<04:49, 662kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<03:25, 927kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<03:07, 1.01MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:30, 1.25MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:48, 1.72MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:58, 1.55MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:41, 1.81MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:15, 2.42MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:34, 1.90MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:24, 2.12MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:03, 2.81MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:25, 2.06MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:35, 1.84MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:15, 2.31MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:05<00:53, 3.19MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<21:19, 134kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<15:13, 188kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<10:44, 265kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<07:27, 377kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<06:32, 427kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<04:51, 574kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<03:25, 807kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<02:24, 1.13MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<17:55, 152kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<12:47, 213kB/s].vector_cache/glove.6B.zip:  81%|████████  | 701M/862M [05:11<08:56, 301kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:13<06:48, 391kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<05:17, 501kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<03:49, 691kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<03:02, 850kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<02:23, 1.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:43, 1.48MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:46, 1.42MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:45, 1.43MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:20, 1.87MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<00:57, 2.59MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<02:06, 1.16MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:43, 1.41MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:15, 1.92MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:25, 1.67MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<01:14, 1.92MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<00:55, 2.56MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:10, 1.96MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:17, 1.78MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:01, 2.25MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:03, 2.11MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<00:58, 2.31MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<00:43, 3.08MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:00, 2.16MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:08, 1.90MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<00:54, 2.37MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:27<00:38, 3.27MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<2:02:28, 17.2kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<1:25:41, 24.5kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:28<59:17, 35.0kB/s]  .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<41:07, 49.9kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<1:48:09, 19.0kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<1:15:03, 27.1kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<51:38, 38.4kB/s]  .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<36:37, 54.1kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<25:36, 76.9kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<17:33, 110kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<13:22, 143kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<09:32, 200kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<06:39, 283kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<04:57, 372kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<03:50, 480kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<02:45, 662kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<02:09, 820kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:41, 1.05MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<01:12, 1.44MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:13, 1.39MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:12, 1.41MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:55, 1.84MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:53, 1.84MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:47, 2.07MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:35, 2.74MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:46, 2.05MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:41, 2.25MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:31, 2.97MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:42, 2.12MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:48, 1.87MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:37, 2.35MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:39, 2.17MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:36, 2.36MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:27, 3.10MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<00:37, 2.17MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:34, 2.35MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:25, 3.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:35, 2.18MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:32, 2.36MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:24, 3.11MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:53<00:34, 2.16MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:38, 1.89MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:30, 2.38MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:31, 2.19MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:29, 2.37MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:21, 3.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:30, 2.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:33, 1.93MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:26, 2.44MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<00:18, 3.34MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<58:55, 17.3kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<41:05, 24.7kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<28:03, 35.3kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<19:08, 49.8kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<13:24, 70.6kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<09:08, 101kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<06:21, 139kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<04:30, 195kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<03:04, 276kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<02:15, 361kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<01:39, 490kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<01:08, 687kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:56, 797kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:43, 1.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:07<00:30, 1.40MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:23, 1.74MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<25:44, 26.8kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<17:32, 38.2kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<11:29, 54.0kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<08:10, 75.6kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<05:39, 107kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:11<03:41, 153kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<02:51, 193kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<02:02, 267kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<01:22, 378kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:59, 483kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:47, 605kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:33, 829kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:15<00:21, 1.17MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<02:04, 199kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<01:28, 277kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:58, 391kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:41, 494kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:33, 617kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:23, 844kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:16, 1.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:12, 1.26MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:08, 1.71MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:07, 1.56MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:06, 1.82MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:04, 2.44MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:04, 1.91MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:03, 2.13MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:02, 2.86MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 2.07MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.84MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.35MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 3.22MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 1.27MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<32:04:46,  3.46it/s]  0%|          | 842/400000 [00:00<22:24:44,  4.95it/s]  0%|          | 1696/400000 [00:00<15:39:32,  7.07it/s]  1%|          | 2390/400000 [00:00<10:56:49, 10.09it/s]  1%|          | 3172/400000 [00:00<7:39:07, 14.41it/s]   1%|          | 4011/400000 [00:00<5:20:56, 20.56it/s]  1%|          | 4883/400000 [00:00<3:44:23, 29.35it/s]  1%|▏         | 5749/400000 [00:00<2:36:57, 41.86it/s]  2%|▏         | 6605/400000 [00:01<1:49:51, 59.68it/s]  2%|▏         | 7434/400000 [00:01<1:16:58, 84.99it/s]  2%|▏         | 8273/400000 [00:01<54:00, 120.90it/s]   2%|▏         | 9147/400000 [00:01<37:56, 171.69it/s]  2%|▏         | 9980/400000 [00:01<26:44, 243.07it/s]  3%|▎         | 10819/400000 [00:01<18:54, 342.97it/s]  3%|▎         | 11654/400000 [00:01<13:26, 481.48it/s]  3%|▎         | 12486/400000 [00:01<09:37, 670.61it/s]  3%|▎         | 13319/400000 [00:01<06:57, 926.04it/s]  4%|▎         | 14157/400000 [00:01<05:05, 1263.05it/s]  4%|▎         | 14987/400000 [00:02<03:47, 1693.10it/s]  4%|▍         | 15815/400000 [00:02<02:54, 2202.43it/s]  4%|▍         | 16616/400000 [00:02<02:16, 2804.59it/s]  4%|▍         | 17448/400000 [00:02<01:49, 3500.15it/s]  5%|▍         | 18306/400000 [00:02<01:29, 4255.54it/s]  5%|▍         | 19140/400000 [00:02<01:16, 4988.31it/s]  5%|▍         | 19965/400000 [00:02<01:07, 5596.85it/s]  5%|▌         | 20803/400000 [00:02<01:01, 6216.17it/s]  5%|▌         | 21631/400000 [00:02<00:56, 6717.80it/s]  6%|▌         | 22486/400000 [00:03<00:52, 7177.58it/s]  6%|▌         | 23343/400000 [00:03<00:49, 7544.35it/s]  6%|▌         | 24183/400000 [00:03<00:48, 7779.76it/s]  6%|▋         | 25030/400000 [00:03<00:47, 7972.77it/s]  6%|▋         | 25872/400000 [00:03<00:46, 8041.78it/s]  7%|▋         | 26713/400000 [00:03<00:45, 8147.53it/s]  7%|▋         | 27550/400000 [00:03<00:45, 8170.29it/s]  7%|▋         | 28383/400000 [00:03<00:45, 8101.20it/s]  7%|▋         | 29238/400000 [00:03<00:45, 8229.73it/s]  8%|▊         | 30070/400000 [00:03<00:44, 8248.62it/s]  8%|▊         | 30918/400000 [00:04<00:44, 8316.58it/s]  8%|▊         | 31754/400000 [00:04<00:44, 8240.41it/s]  8%|▊         | 32582/400000 [00:04<00:44, 8249.04it/s]  8%|▊         | 33440/400000 [00:04<00:43, 8343.21it/s]  9%|▊         | 34277/400000 [00:04<00:43, 8325.46it/s]  9%|▉         | 35129/400000 [00:04<00:43, 8380.60it/s]  9%|▉         | 35981/400000 [00:04<00:43, 8420.84it/s]  9%|▉         | 36824/400000 [00:04<00:43, 8363.92it/s]  9%|▉         | 37668/400000 [00:04<00:43, 8385.40it/s] 10%|▉         | 38507/400000 [00:04<00:43, 8345.51it/s] 10%|▉         | 39354/400000 [00:05<00:43, 8378.88it/s] 10%|█         | 40204/400000 [00:05<00:42, 8412.49it/s] 10%|█         | 41046/400000 [00:05<00:42, 8391.50it/s] 10%|█         | 41902/400000 [00:05<00:42, 8441.16it/s] 11%|█         | 42747/400000 [00:05<00:42, 8409.12it/s] 11%|█         | 43589/400000 [00:05<00:42, 8384.48it/s] 11%|█         | 44438/400000 [00:05<00:42, 8415.26it/s] 11%|█▏        | 45280/400000 [00:05<00:45, 7732.05it/s] 12%|█▏        | 46116/400000 [00:05<00:44, 7907.77it/s] 12%|█▏        | 46968/400000 [00:05<00:43, 8081.61it/s] 12%|█▏        | 47784/400000 [00:06<00:44, 7994.46it/s] 12%|█▏        | 48600/400000 [00:06<00:43, 8043.39it/s] 12%|█▏        | 49408/400000 [00:06<00:43, 7978.64it/s] 13%|█▎        | 50269/400000 [00:06<00:42, 8156.61it/s] 13%|█▎        | 51128/400000 [00:06<00:42, 8281.68it/s] 13%|█▎        | 51959/400000 [00:06<00:42, 8221.99it/s] 13%|█▎        | 52820/400000 [00:06<00:41, 8333.67it/s] 13%|█▎        | 53656/400000 [00:06<00:42, 8192.33it/s] 14%|█▎        | 54519/400000 [00:06<00:41, 8318.40it/s] 14%|█▍        | 55386/400000 [00:06<00:40, 8419.46it/s] 14%|█▍        | 56230/400000 [00:07<00:41, 8379.60it/s] 14%|█▍        | 57091/400000 [00:07<00:40, 8445.21it/s] 14%|█▍        | 57937/400000 [00:07<00:41, 8239.68it/s] 15%|█▍        | 58777/400000 [00:07<00:41, 8286.21it/s] 15%|█▍        | 59624/400000 [00:07<00:40, 8338.34it/s] 15%|█▌        | 60459/400000 [00:07<00:40, 8341.23it/s] 15%|█▌        | 61318/400000 [00:07<00:40, 8413.81it/s] 16%|█▌        | 62161/400000 [00:07<00:41, 8181.86it/s] 16%|█▌        | 63028/400000 [00:07<00:40, 8319.75it/s] 16%|█▌        | 63884/400000 [00:08<00:40, 8388.13it/s] 16%|█▌        | 64725/400000 [00:08<00:40, 8348.54it/s] 16%|█▋        | 65588/400000 [00:08<00:39, 8429.00it/s] 17%|█▋        | 66432/400000 [00:08<00:40, 8249.12it/s] 17%|█▋        | 67275/400000 [00:08<00:40, 8300.32it/s] 17%|█▋        | 68139/400000 [00:08<00:39, 8397.65it/s] 17%|█▋        | 68984/400000 [00:08<00:39, 8411.72it/s] 17%|█▋        | 69844/400000 [00:08<00:38, 8465.71it/s] 18%|█▊        | 70692/400000 [00:08<00:39, 8251.39it/s] 18%|█▊        | 71542/400000 [00:08<00:39, 8323.70it/s] 18%|█▊        | 72397/400000 [00:09<00:39, 8388.06it/s] 18%|█▊        | 73237/400000 [00:09<00:39, 8368.36it/s] 19%|█▊        | 74094/400000 [00:09<00:38, 8425.17it/s] 19%|█▊        | 74938/400000 [00:09<00:39, 8188.07it/s] 19%|█▉        | 75807/400000 [00:09<00:38, 8331.83it/s] 19%|█▉        | 76669/400000 [00:09<00:38, 8413.44it/s] 19%|█▉        | 77522/400000 [00:09<00:38, 8446.46it/s] 20%|█▉        | 78380/400000 [00:09<00:37, 8485.67it/s] 20%|█▉        | 79230/400000 [00:09<00:38, 8358.46it/s] 20%|██        | 80104/400000 [00:09<00:37, 8467.56it/s] 20%|██        | 80975/400000 [00:10<00:37, 8535.98it/s] 20%|██        | 81830/400000 [00:10<00:37, 8454.39it/s] 21%|██        | 82702/400000 [00:10<00:37, 8529.70it/s] 21%|██        | 83556/400000 [00:10<00:37, 8349.80it/s] 21%|██        | 84422/400000 [00:10<00:37, 8440.05it/s] 21%|██▏       | 85299/400000 [00:10<00:36, 8534.16it/s] 22%|██▏       | 86154/400000 [00:10<00:37, 8478.91it/s] 22%|██▏       | 87026/400000 [00:10<00:36, 8548.82it/s] 22%|██▏       | 87882/400000 [00:10<00:37, 8425.98it/s] 22%|██▏       | 88726/400000 [00:10<00:37, 8326.02it/s] 22%|██▏       | 89590/400000 [00:11<00:36, 8416.47it/s] 23%|██▎       | 90448/400000 [00:11<00:36, 8461.61it/s] 23%|██▎       | 91319/400000 [00:11<00:36, 8532.25it/s] 23%|██▎       | 92173/400000 [00:11<00:36, 8387.62it/s] 23%|██▎       | 93048/400000 [00:11<00:36, 8493.09it/s] 23%|██▎       | 93925/400000 [00:11<00:35, 8572.28it/s] 24%|██▎       | 94784/400000 [00:11<00:35, 8524.85it/s] 24%|██▍       | 95638/400000 [00:11<00:36, 8410.25it/s] 24%|██▍       | 96480/400000 [00:11<00:36, 8379.45it/s] 24%|██▍       | 97322/400000 [00:11<00:36, 8388.81it/s] 25%|██▍       | 98167/400000 [00:12<00:35, 8404.71it/s] 25%|██▍       | 99009/400000 [00:12<00:35, 8408.40it/s] 25%|██▍       | 99883/400000 [00:12<00:35, 8503.20it/s] 25%|██▌       | 100734/400000 [00:12<00:35, 8354.76it/s] 25%|██▌       | 101595/400000 [00:12<00:35, 8427.08it/s] 26%|██▌       | 102459/400000 [00:12<00:35, 8487.21it/s] 26%|██▌       | 103309/400000 [00:12<00:35, 8472.86it/s] 26%|██▌       | 104182/400000 [00:12<00:34, 8548.08it/s] 26%|██▋       | 105038/400000 [00:12<00:35, 8387.33it/s] 26%|██▋       | 105900/400000 [00:12<00:34, 8453.40it/s] 27%|██▋       | 106754/400000 [00:13<00:34, 8478.60it/s] 27%|██▋       | 107609/400000 [00:13<00:34, 8498.84it/s] 27%|██▋       | 108460/400000 [00:13<00:34, 8455.62it/s] 27%|██▋       | 109306/400000 [00:13<00:35, 8300.05it/s] 28%|██▊       | 110183/400000 [00:13<00:34, 8434.81it/s] 28%|██▊       | 111060/400000 [00:13<00:33, 8532.08it/s] 28%|██▊       | 111927/400000 [00:13<00:33, 8570.49it/s] 28%|██▊       | 112795/400000 [00:13<00:33, 8602.89it/s] 28%|██▊       | 113656/400000 [00:13<00:33, 8518.77it/s] 29%|██▊       | 114509/400000 [00:14<00:33, 8462.19it/s] 29%|██▉       | 115378/400000 [00:14<00:33, 8528.86it/s] 29%|██▉       | 116234/400000 [00:14<00:33, 8537.50it/s] 29%|██▉       | 117089/400000 [00:14<00:33, 8539.25it/s] 29%|██▉       | 117958/400000 [00:14<00:32, 8583.13it/s] 30%|██▉       | 118817/400000 [00:14<00:33, 8387.60it/s] 30%|██▉       | 119693/400000 [00:14<00:32, 8495.46it/s] 30%|███       | 120544/400000 [00:14<00:32, 8490.39it/s] 30%|███       | 121403/400000 [00:14<00:32, 8518.21it/s] 31%|███       | 122271/400000 [00:14<00:32, 8565.88it/s] 31%|███       | 123129/400000 [00:15<00:33, 8364.71it/s] 31%|███       | 123996/400000 [00:15<00:32, 8453.77it/s] 31%|███       | 124856/400000 [00:15<00:32, 8494.10it/s] 31%|███▏      | 125718/400000 [00:15<00:32, 8529.70it/s] 32%|███▏      | 126588/400000 [00:15<00:31, 8579.08it/s] 32%|███▏      | 127447/400000 [00:15<00:32, 8388.84it/s] 32%|███▏      | 128321/400000 [00:15<00:32, 8488.94it/s] 32%|███▏      | 129178/400000 [00:15<00:31, 8511.38it/s] 33%|███▎      | 130031/400000 [00:15<00:31, 8515.27it/s] 33%|███▎      | 130893/400000 [00:15<00:31, 8544.59it/s] 33%|███▎      | 131748/400000 [00:16<00:32, 8241.31it/s] 33%|███▎      | 132585/400000 [00:16<00:32, 8278.26it/s] 33%|███▎      | 133415/400000 [00:16<00:32, 8283.46it/s] 34%|███▎      | 134272/400000 [00:16<00:31, 8365.82it/s] 34%|███▍      | 135145/400000 [00:16<00:31, 8470.07it/s] 34%|███▍      | 135994/400000 [00:16<00:31, 8329.50it/s] 34%|███▍      | 136870/400000 [00:16<00:31, 8452.89it/s] 34%|███▍      | 137731/400000 [00:16<00:30, 8497.23it/s] 35%|███▍      | 138594/400000 [00:16<00:30, 8535.51it/s] 35%|███▍      | 139471/400000 [00:16<00:30, 8602.24it/s] 35%|███▌      | 140332/400000 [00:17<00:30, 8429.17it/s] 35%|███▌      | 141195/400000 [00:17<00:30, 8488.20it/s] 36%|███▌      | 142045/400000 [00:17<00:30, 8421.70it/s] 36%|███▌      | 142920/400000 [00:17<00:30, 8517.36it/s] 36%|███▌      | 143782/400000 [00:17<00:29, 8546.40it/s] 36%|███▌      | 144638/400000 [00:17<00:30, 8404.39it/s] 36%|███▋      | 145517/400000 [00:17<00:29, 8515.74it/s] 37%|███▋      | 146379/400000 [00:17<00:29, 8544.08it/s] 37%|███▋      | 147253/400000 [00:17<00:29, 8601.29it/s] 37%|███▋      | 148127/400000 [00:17<00:29, 8641.04it/s] 37%|███▋      | 148992/400000 [00:18<00:30, 8333.39it/s] 37%|███▋      | 149870/400000 [00:18<00:29, 8461.91it/s] 38%|███▊      | 150719/400000 [00:18<00:29, 8449.82it/s] 38%|███▊      | 151587/400000 [00:18<00:29, 8516.67it/s] 38%|███▊      | 152456/400000 [00:18<00:28, 8565.34it/s] 38%|███▊      | 153314/400000 [00:18<00:29, 8335.01it/s] 39%|███▊      | 154194/400000 [00:18<00:29, 8468.25it/s] 39%|███▉      | 155043/400000 [00:18<00:28, 8461.59it/s] 39%|███▉      | 155921/400000 [00:18<00:28, 8554.14it/s] 39%|███▉      | 156793/400000 [00:18<00:28, 8602.26it/s] 39%|███▉      | 157655/400000 [00:19<00:28, 8378.04it/s] 40%|███▉      | 158535/400000 [00:19<00:28, 8499.95it/s] 40%|███▉      | 159387/400000 [00:19<00:28, 8400.11it/s] 40%|████      | 160263/400000 [00:19<00:28, 8503.64it/s] 40%|████      | 161144/400000 [00:19<00:27, 8593.16it/s] 41%|████      | 162005/400000 [00:19<00:28, 8413.97it/s] 41%|████      | 162882/400000 [00:19<00:27, 8517.68it/s] 41%|████      | 163736/400000 [00:19<00:27, 8486.50it/s] 41%|████      | 164602/400000 [00:19<00:27, 8537.08it/s] 41%|████▏     | 165464/400000 [00:20<00:27, 8560.69it/s] 42%|████▏     | 166321/400000 [00:20<00:28, 8342.71it/s] 42%|████▏     | 167196/400000 [00:20<00:27, 8459.20it/s] 42%|████▏     | 168044/400000 [00:20<00:27, 8461.99it/s] 42%|████▏     | 168921/400000 [00:20<00:27, 8549.55it/s] 42%|████▏     | 169796/400000 [00:20<00:26, 8580.44it/s] 43%|████▎     | 170655/400000 [00:20<00:27, 8446.08it/s] 43%|████▎     | 171530/400000 [00:20<00:26, 8533.38it/s] 43%|████▎     | 172385/400000 [00:20<00:26, 8535.39it/s] 43%|████▎     | 173261/400000 [00:20<00:26, 8599.78it/s] 44%|████▎     | 174140/400000 [00:21<00:26, 8654.18it/s] 44%|████▍     | 175006/400000 [00:21<00:26, 8394.16it/s] 44%|████▍     | 175848/400000 [00:21<00:26, 8321.79it/s] 44%|████▍     | 176682/400000 [00:21<00:27, 8239.52it/s] 44%|████▍     | 177552/400000 [00:21<00:26, 8372.08it/s] 45%|████▍     | 178391/400000 [00:21<00:26, 8344.50it/s] 45%|████▍     | 179232/400000 [00:21<00:26, 8362.65it/s] 45%|████▌     | 180090/400000 [00:21<00:26, 8425.42it/s] 45%|████▌     | 180941/400000 [00:21<00:25, 8449.79it/s] 45%|████▌     | 181796/400000 [00:21<00:25, 8478.97it/s] 46%|████▌     | 182645/400000 [00:22<00:25, 8413.52it/s] 46%|████▌     | 183487/400000 [00:22<00:26, 8200.96it/s] 46%|████▌     | 184339/400000 [00:22<00:26, 8293.93it/s] 46%|████▋     | 185172/400000 [00:22<00:25, 8303.71it/s] 47%|████▋     | 186043/400000 [00:22<00:25, 8419.92it/s] 47%|████▋     | 186887/400000 [00:22<00:25, 8396.99it/s] 47%|████▋     | 187728/400000 [00:22<00:25, 8365.71it/s] 47%|████▋     | 188605/400000 [00:22<00:24, 8481.98it/s] 47%|████▋     | 189454/400000 [00:22<00:24, 8466.78it/s] 48%|████▊     | 190328/400000 [00:22<00:24, 8546.40it/s] 48%|████▊     | 191184/400000 [00:23<00:24, 8469.02it/s] 48%|████▊     | 192032/400000 [00:23<00:24, 8444.98it/s] 48%|████▊     | 192902/400000 [00:23<00:24, 8517.26it/s] 48%|████▊     | 193755/400000 [00:23<00:24, 8488.99it/s] 49%|████▊     | 194629/400000 [00:23<00:23, 8562.11it/s] 49%|████▉     | 195504/400000 [00:23<00:23, 8614.22it/s] 49%|████▉     | 196366/400000 [00:23<00:24, 8398.24it/s] 49%|████▉     | 197238/400000 [00:23<00:23, 8490.12it/s] 50%|████▉     | 198097/400000 [00:23<00:23, 8518.90it/s] 50%|████▉     | 198955/400000 [00:23<00:23, 8536.49it/s] 50%|████▉     | 199810/400000 [00:24<00:23, 8407.24it/s] 50%|█████     | 200652/400000 [00:24<00:23, 8324.18it/s] 50%|█████     | 201530/400000 [00:24<00:23, 8454.39it/s] 51%|█████     | 202377/400000 [00:24<00:23, 8451.39it/s] 51%|█████     | 203255/400000 [00:24<00:23, 8545.03it/s] 51%|█████     | 204118/400000 [00:24<00:22, 8568.16it/s] 51%|█████     | 204976/400000 [00:24<00:23, 8439.00it/s] 51%|█████▏    | 205853/400000 [00:24<00:22, 8533.86it/s] 52%|█████▏    | 206709/400000 [00:24<00:22, 8539.30it/s] 52%|█████▏    | 207579/400000 [00:24<00:22, 8584.98it/s] 52%|█████▏    | 208438/400000 [00:25<00:22, 8533.12it/s] 52%|█████▏    | 209292/400000 [00:25<00:22, 8331.84it/s] 53%|█████▎    | 210166/400000 [00:25<00:22, 8447.72it/s] 53%|█████▎    | 211013/400000 [00:25<00:22, 8353.22it/s] 53%|█████▎    | 211869/400000 [00:25<00:22, 8413.30it/s] 53%|█████▎    | 212712/400000 [00:25<00:22, 8387.23it/s] 53%|█████▎    | 213552/400000 [00:25<00:22, 8378.01it/s] 54%|█████▎    | 214428/400000 [00:25<00:21, 8487.47it/s] 54%|█████▍    | 215283/400000 [00:25<00:21, 8505.32it/s] 54%|█████▍    | 216162/400000 [00:26<00:21, 8586.97it/s] 54%|█████▍    | 217022/400000 [00:26<00:21, 8483.71it/s] 54%|█████▍    | 217872/400000 [00:26<00:21, 8454.86it/s] 55%|█████▍    | 218740/400000 [00:26<00:21, 8520.88it/s] 55%|█████▍    | 219593/400000 [00:26<00:21, 8381.37it/s] 55%|█████▌    | 220435/400000 [00:26<00:21, 8390.36it/s] 55%|█████▌    | 221282/400000 [00:26<00:21, 8412.12it/s] 56%|█████▌    | 222124/400000 [00:26<00:21, 8346.43it/s] 56%|█████▌    | 222999/400000 [00:26<00:20, 8462.32it/s] 56%|█████▌    | 223847/400000 [00:26<00:20, 8466.76it/s] 56%|█████▌    | 224696/400000 [00:27<00:20, 8471.04it/s] 56%|█████▋    | 225544/400000 [00:27<00:20, 8323.64it/s] 57%|█████▋    | 226378/400000 [00:27<00:20, 8309.82it/s] 57%|█████▋    | 227250/400000 [00:27<00:20, 8426.16it/s] 57%|█████▋    | 228094/400000 [00:27<00:20, 8417.14it/s] 57%|█████▋    | 228971/400000 [00:27<00:20, 8518.01it/s] 57%|█████▋    | 229824/400000 [00:27<00:20, 8384.47it/s] 58%|█████▊    | 230698/400000 [00:27<00:19, 8485.84it/s] 58%|█████▊    | 231574/400000 [00:27<00:19, 8564.31it/s] 58%|█████▊    | 232432/400000 [00:27<00:19, 8539.02it/s] 58%|█████▊    | 233306/400000 [00:28<00:19, 8592.59it/s] 59%|█████▊    | 234166/400000 [00:28<00:19, 8487.20it/s] 59%|█████▉    | 235016/400000 [00:28<00:19, 8436.04it/s] 59%|█████▉    | 235886/400000 [00:28<00:19, 8511.83it/s] 59%|█████▉    | 236738/400000 [00:28<00:19, 8385.75it/s] 59%|█████▉    | 237590/400000 [00:28<00:19, 8423.16it/s] 60%|█████▉    | 238433/400000 [00:28<00:19, 8315.27it/s] 60%|█████▉    | 239306/400000 [00:28<00:19, 8432.99it/s] 60%|██████    | 240185/400000 [00:28<00:18, 8535.57it/s] 60%|██████    | 241040/400000 [00:28<00:18, 8537.59it/s] 60%|██████    | 241895/400000 [00:29<00:18, 8520.04it/s] 61%|██████    | 242748/400000 [00:29<00:18, 8419.37it/s] 61%|██████    | 243591/400000 [00:29<00:18, 8374.20it/s] 61%|██████    | 244467/400000 [00:29<00:18, 8485.09it/s] 61%|██████▏   | 245317/400000 [00:29<00:18, 8483.01it/s] 62%|██████▏   | 246191/400000 [00:29<00:17, 8557.97it/s] 62%|██████▏   | 247048/400000 [00:29<00:18, 8374.82it/s] 62%|██████▏   | 247907/400000 [00:29<00:18, 8436.37it/s] 62%|██████▏   | 248769/400000 [00:29<00:17, 8490.27it/s] 62%|██████▏   | 249623/400000 [00:29<00:17, 8504.10it/s] 63%|██████▎   | 250485/400000 [00:30<00:17, 8531.77it/s] 63%|██████▎   | 251339/400000 [00:30<00:17, 8414.01it/s] 63%|██████▎   | 252182/400000 [00:30<00:17, 8401.66it/s] 63%|██████▎   | 253050/400000 [00:30<00:17, 8481.01it/s] 63%|██████▎   | 253906/400000 [00:30<00:17, 8504.49it/s] 64%|██████▎   | 254784/400000 [00:30<00:16, 8583.24it/s] 64%|██████▍   | 255661/400000 [00:30<00:16, 8636.01it/s] 64%|██████▍   | 256525/400000 [00:30<00:17, 8426.67it/s] 64%|██████▍   | 257370/400000 [00:30<00:17, 8348.20it/s] 65%|██████▍   | 258206/400000 [00:30<00:17, 8307.59it/s] 65%|██████▍   | 259074/400000 [00:31<00:16, 8414.33it/s] 65%|██████▍   | 259941/400000 [00:31<00:16, 8488.84it/s] 65%|██████▌   | 260805/400000 [00:31<00:16, 8533.37it/s] 65%|██████▌   | 261679/400000 [00:31<00:16, 8592.93it/s] 66%|██████▌   | 262539/400000 [00:31<00:16, 8272.41it/s] 66%|██████▌   | 263370/400000 [00:31<00:16, 8167.03it/s] 66%|██████▌   | 264213/400000 [00:31<00:16, 8243.08it/s] 66%|██████▋   | 265057/400000 [00:31<00:16, 8301.18it/s] 66%|██████▋   | 265934/400000 [00:31<00:15, 8435.51it/s] 67%|██████▋   | 266780/400000 [00:32<00:15, 8380.91it/s] 67%|██████▋   | 267643/400000 [00:32<00:15, 8452.01it/s] 67%|██████▋   | 268498/400000 [00:32<00:15, 8481.12it/s] 67%|██████▋   | 269347/400000 [00:32<00:15, 8444.23it/s] 68%|██████▊   | 270219/400000 [00:32<00:15, 8524.43it/s] 68%|██████▊   | 271072/400000 [00:32<00:15, 8501.70it/s] 68%|██████▊   | 271944/400000 [00:32<00:14, 8564.58it/s] 68%|██████▊   | 272818/400000 [00:32<00:14, 8615.07it/s] 68%|██████▊   | 273680/400000 [00:32<00:14, 8581.02it/s] 69%|██████▊   | 274557/400000 [00:32<00:14, 8636.35it/s] 69%|██████▉   | 275421/400000 [00:33<00:14, 8615.85it/s] 69%|██████▉   | 276297/400000 [00:33<00:14, 8656.91it/s] 69%|██████▉   | 277164/400000 [00:33<00:14, 8658.26it/s] 70%|██████▉   | 278030/400000 [00:33<00:14, 8603.59it/s] 70%|██████▉   | 278894/400000 [00:33<00:14, 8612.19it/s] 70%|██████▉   | 279756/400000 [00:33<00:14, 8569.98it/s] 70%|███████   | 280628/400000 [00:33<00:13, 8614.32it/s] 70%|███████   | 281503/400000 [00:33<00:13, 8653.14it/s] 71%|███████   | 282369/400000 [00:33<00:13, 8606.54it/s] 71%|███████   | 283237/400000 [00:33<00:13, 8625.25it/s] 71%|███████   | 284100/400000 [00:34<00:13, 8592.88it/s] 71%|███████   | 284968/400000 [00:34<00:13, 8617.30it/s] 71%|███████▏  | 285838/400000 [00:34<00:13, 8641.09it/s] 72%|███████▏  | 286703/400000 [00:34<00:13, 8609.01it/s] 72%|███████▏  | 287566/400000 [00:34<00:13, 8613.97it/s] 72%|███████▏  | 288428/400000 [00:34<00:12, 8600.90it/s] 72%|███████▏  | 289298/400000 [00:34<00:12, 8629.11it/s] 73%|███████▎  | 290174/400000 [00:34<00:12, 8667.47it/s] 73%|███████▎  | 291041/400000 [00:34<00:12, 8639.64it/s] 73%|███████▎  | 291906/400000 [00:34<00:12, 8587.03it/s] 73%|███████▎  | 292778/400000 [00:35<00:12, 8626.06it/s] 73%|███████▎  | 293650/400000 [00:35<00:12, 8653.75it/s] 74%|███████▎  | 294516/400000 [00:35<00:12, 8351.69it/s] 74%|███████▍  | 295354/400000 [00:35<00:12, 8302.72it/s] 74%|███████▍  | 296205/400000 [00:35<00:12, 8362.10it/s] 74%|███████▍  | 297062/400000 [00:35<00:12, 8421.58it/s] 74%|███████▍  | 297938/400000 [00:35<00:11, 8518.97it/s] 75%|███████▍  | 298796/400000 [00:35<00:11, 8535.00it/s] 75%|███████▍  | 299651/400000 [00:35<00:12, 8151.60it/s] 75%|███████▌  | 300510/400000 [00:35<00:12, 8276.86it/s] 75%|███████▌  | 301342/400000 [00:36<00:12, 8165.32it/s] 76%|███████▌  | 302198/400000 [00:36<00:11, 8275.78it/s] 76%|███████▌  | 303058/400000 [00:36<00:11, 8368.15it/s] 76%|███████▌  | 303922/400000 [00:36<00:11, 8446.49it/s] 76%|███████▌  | 304777/400000 [00:36<00:11, 8476.74it/s] 76%|███████▋  | 305648/400000 [00:36<00:11, 8544.90it/s] 77%|███████▋  | 306504/400000 [00:36<00:11, 8297.95it/s] 77%|███████▋  | 307336/400000 [00:36<00:11, 8205.65it/s] 77%|███████▋  | 308203/400000 [00:36<00:11, 8337.06it/s] 77%|███████▋  | 309064/400000 [00:36<00:10, 8416.43it/s] 77%|███████▋  | 309912/400000 [00:37<00:10, 8433.07it/s] 78%|███████▊  | 310788/400000 [00:37<00:10, 8526.97it/s] 78%|███████▊  | 311662/400000 [00:37<00:10, 8588.91it/s] 78%|███████▊  | 312522/400000 [00:37<00:10, 8590.15it/s] 78%|███████▊  | 313382/400000 [00:37<00:10, 8557.31it/s] 79%|███████▊  | 314255/400000 [00:37<00:09, 8606.13it/s] 79%|███████▉  | 315116/400000 [00:37<00:09, 8587.01it/s] 79%|███████▉  | 315988/400000 [00:37<00:09, 8624.87it/s] 79%|███████▉  | 316851/400000 [00:37<00:09, 8603.77it/s] 79%|███████▉  | 317712/400000 [00:37<00:09, 8592.75it/s] 80%|███████▉  | 318586/400000 [00:38<00:09, 8634.07it/s] 80%|███████▉  | 319457/400000 [00:38<00:09, 8653.81it/s] 80%|████████  | 320326/400000 [00:38<00:09, 8662.66it/s] 80%|████████  | 321193/400000 [00:38<00:09, 8607.27it/s] 81%|████████  | 322054/400000 [00:38<00:09, 8579.04it/s] 81%|████████  | 322913/400000 [00:38<00:09, 8564.20it/s] 81%|████████  | 323774/400000 [00:38<00:08, 8575.17it/s] 81%|████████  | 324649/400000 [00:38<00:08, 8625.62it/s] 81%|████████▏ | 325512/400000 [00:38<00:08, 8618.28it/s] 82%|████████▏ | 326374/400000 [00:38<00:08, 8546.77it/s] 82%|████████▏ | 327244/400000 [00:39<00:08, 8591.62it/s] 82%|████████▏ | 328104/400000 [00:39<00:08, 8592.49it/s] 82%|████████▏ | 328964/400000 [00:39<00:08, 8511.17it/s] 82%|████████▏ | 329816/400000 [00:39<00:08, 8502.83it/s] 83%|████████▎ | 330667/400000 [00:39<00:08, 8450.21it/s] 83%|████████▎ | 331533/400000 [00:39<00:08, 8509.44it/s] 83%|████████▎ | 332389/400000 [00:39<00:07, 8523.21it/s] 83%|████████▎ | 333254/400000 [00:39<00:07, 8560.47it/s] 84%|████████▎ | 334115/400000 [00:39<00:07, 8573.38it/s] 84%|████████▎ | 334973/400000 [00:39<00:07, 8505.99it/s] 84%|████████▍ | 335841/400000 [00:40<00:07, 8556.81it/s] 84%|████████▍ | 336709/400000 [00:40<00:07, 8590.94it/s] 84%|████████▍ | 337584/400000 [00:40<00:07, 8637.89it/s] 85%|████████▍ | 338448/400000 [00:40<00:07, 8605.04it/s] 85%|████████▍ | 339309/400000 [00:40<00:07, 8506.29it/s] 85%|████████▌ | 340173/400000 [00:40<00:07, 8544.39it/s] 85%|████████▌ | 341028/400000 [00:40<00:06, 8473.93it/s] 85%|████████▌ | 341894/400000 [00:40<00:06, 8527.50it/s] 86%|████████▌ | 342748/400000 [00:40<00:06, 8508.58it/s] 86%|████████▌ | 343600/400000 [00:41<00:06, 8467.60it/s] 86%|████████▌ | 344468/400000 [00:41<00:06, 8527.76it/s] 86%|████████▋ | 345344/400000 [00:41<00:06, 8595.01it/s] 87%|████████▋ | 346216/400000 [00:41<00:06, 8631.32it/s] 87%|████████▋ | 347080/400000 [00:41<00:06, 8589.86it/s] 87%|████████▋ | 347940/400000 [00:41<00:06, 8516.65it/s] 87%|████████▋ | 348816/400000 [00:41<00:05, 8586.65it/s] 87%|████████▋ | 349691/400000 [00:41<00:05, 8634.36it/s] 88%|████████▊ | 350555/400000 [00:41<00:05, 8268.63it/s] 88%|████████▊ | 351386/400000 [00:41<00:06, 7734.61it/s] 88%|████████▊ | 352245/400000 [00:42<00:05, 7972.04it/s] 88%|████████▊ | 353110/400000 [00:42<00:05, 8163.22it/s] 88%|████████▊ | 353984/400000 [00:42<00:05, 8326.75it/s] 89%|████████▊ | 354858/400000 [00:42<00:05, 8444.50it/s] 89%|████████▉ | 355707/400000 [00:42<00:05, 8296.14it/s] 89%|████████▉ | 356541/400000 [00:42<00:05, 8231.78it/s] 89%|████████▉ | 357395/400000 [00:42<00:05, 8319.71it/s] 90%|████████▉ | 358269/400000 [00:42<00:04, 8441.20it/s] 90%|████████▉ | 359146/400000 [00:42<00:04, 8536.05it/s] 90%|█████████ | 360002/400000 [00:42<00:04, 8252.45it/s] 90%|█████████ | 360852/400000 [00:43<00:04, 8323.67it/s] 90%|█████████ | 361725/400000 [00:43<00:04, 8441.11it/s] 91%|█████████ | 362589/400000 [00:43<00:04, 8497.57it/s] 91%|█████████ | 363454/400000 [00:43<00:04, 8541.55it/s] 91%|█████████ | 364310/400000 [00:43<00:04, 8290.62it/s] 91%|█████████▏| 365183/400000 [00:43<00:04, 8416.06it/s] 92%|█████████▏| 366055/400000 [00:43<00:03, 8504.69it/s] 92%|█████████▏| 366916/400000 [00:43<00:03, 8535.91it/s] 92%|█████████▏| 367789/400000 [00:43<00:03, 8591.20it/s] 92%|█████████▏| 368650/400000 [00:43<00:03, 8323.91it/s] 92%|█████████▏| 369521/400000 [00:44<00:03, 8435.18it/s] 93%|█████████▎| 370394/400000 [00:44<00:03, 8520.62it/s] 93%|█████████▎| 371269/400000 [00:44<00:03, 8587.16it/s] 93%|█████████▎| 372142/400000 [00:44<00:03, 8628.97it/s] 93%|█████████▎| 373006/400000 [00:44<00:03, 8404.64it/s] 93%|█████████▎| 373872/400000 [00:44<00:03, 8479.34it/s] 94%|█████████▎| 374734/400000 [00:44<00:02, 8519.34it/s] 94%|█████████▍| 375604/400000 [00:44<00:02, 8569.91it/s] 94%|█████████▍| 376472/400000 [00:44<00:02, 8600.84it/s] 94%|█████████▍| 377333/400000 [00:45<00:02, 8258.81it/s] 95%|█████████▍| 378194/400000 [00:45<00:02, 8360.80it/s] 95%|█████████▍| 379066/400000 [00:45<00:02, 8463.11it/s] 95%|█████████▍| 379943/400000 [00:45<00:02, 8549.13it/s] 95%|█████████▌| 380820/400000 [00:45<00:02, 8613.69it/s] 95%|█████████▌| 381683/400000 [00:45<00:02, 8509.53it/s] 96%|█████████▌| 382560/400000 [00:45<00:02, 8584.94it/s] 96%|█████████▌| 383432/400000 [00:45<00:01, 8624.86it/s] 96%|█████████▌| 384307/400000 [00:45<00:01, 8660.78it/s] 96%|█████████▋| 385182/400000 [00:45<00:01, 8685.78it/s] 97%|█████████▋| 386051/400000 [00:46<00:01, 8506.97it/s] 97%|█████████▋| 386903/400000 [00:46<00:01, 8481.32it/s] 97%|█████████▋| 387780/400000 [00:46<00:01, 8564.04it/s] 97%|█████████▋| 388660/400000 [00:46<00:01, 8630.80it/s] 97%|█████████▋| 389535/400000 [00:46<00:01, 8666.14it/s] 98%|█████████▊| 390403/400000 [00:46<00:01, 8530.69it/s] 98%|█████████▊| 391277/400000 [00:46<00:01, 8591.44it/s] 98%|█████████▊| 392149/400000 [00:46<00:00, 8627.82it/s] 98%|█████████▊| 393017/400000 [00:46<00:00, 8642.79it/s] 98%|█████████▊| 393882/400000 [00:46<00:00, 8425.86it/s] 99%|█████████▊| 394727/400000 [00:47<00:00, 8068.84it/s] 99%|█████████▉| 395591/400000 [00:47<00:00, 8230.65it/s] 99%|█████████▉| 396466/400000 [00:47<00:00, 8378.38it/s] 99%|█████████▉| 397338/400000 [00:47<00:00, 8477.45it/s]100%|█████████▉| 398189/400000 [00:47<00:00, 8397.20it/s]100%|█████████▉| 399038/400000 [00:47<00:00, 8422.00it/s]100%|█████████▉| 399912/400000 [00:47<00:00, 8514.65it/s]100%|█████████▉| 399999/400000 [00:47<00:00, 8390.13it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f3b55e1d198>, <torchtext.data.dataset.TabularDataset object at 0x7f3b55e1d2e8>, <torchtext.vocab.Vocab object at 0x7f3b55e1d208>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.43 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.43 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  9.43 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.14 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.14 file/s]Dl Completed...: 100%|██████████| 4/4 [00:01<00:00,  3.77 file/s]Dl Completed...: 100%|██████████| 4/4 [00:01<00:00,  3.77 file/s]Dl Completed...: 100%|██████████| 4/4 [00:01<00:00,  3.63 file/s]2020-06-14 18:18:11.557176: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-14 18:18:11.560956: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-06-14 18:18:11.561154: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c2e42157e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-14 18:18:11.561172: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['train', 'cifar10', 'mnist_dataset_small.npy', 'test', 'fashion-mnist_small.npy', 'mnist2'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 41%|████▏     | 4112384/9912422 [00:00<00:00, 40900376.70it/s]9920512it [00:00, 35026898.10it/s]                             
0it [00:00, ?it/s]32768it [00:00, 594464.28it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 477968.73it/s]1654784it [00:00, 12088321.87it/s]                         
0it [00:00, ?it/s]8192it [00:00, 202970.97it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
