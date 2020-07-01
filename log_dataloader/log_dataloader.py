
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/4eb1f9b9d4d68e7db555190ca5254c5fe537ba39', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': '4eb1f9b9d4d68e7db555190ca5254c5fe537ba39', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/4eb1f9b9d4d68e7db555190ca5254c5fe537ba39

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/4eb1f9b9d4d68e7db555190ca5254c5fe537ba39

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/4eb1f9b9d4d68e7db555190ca5254c5fe537ba39

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f4d3e0fcea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f4d3e0fcea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f4da93bc1e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f4da93bc1e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f4dc7704e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f4dc7704e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f4d566e3488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f4d566e3488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f4d566e3488> , (data_info, **args) 

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
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 416, in load
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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:17, 127205.29it/s] 75%|███████▌  | 7471104/9912422 [00:00<00:13, 181589.04it/s]9920512it [00:00, 39928149.02it/s]                           
0it [00:00, ?it/s]32768it [00:00, 529242.38it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 157070.58it/s]1654784it [00:00, 11008841.36it/s]                         
0it [00:00, ?it/s]8192it [00:00, 184012.52it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4d3d891550>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4d3d7306a0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f4d566e30d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f4d566e30d0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f4d566e30d0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:22,  1.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:22,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:22,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:21,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:21,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:20,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<00:57,  2.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:57,  2.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:56,  2.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:56,  2.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:56,  2.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 10/162 [00:00<00:40,  3.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:40,  3.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:40,  3.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:39,  3.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:39,  3.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▊         | 14/162 [00:00<00:28,  5.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:28,  5.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:28,  5.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:28,  5.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:28,  5.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:20,  6.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:20,  6.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<00:20,  6.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:01<00:20,  6.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:01<00:20,  6.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  14%|█▎        | 22/162 [00:01<00:15,  9.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:01<00:15,  9.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:15,  9.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:15,  9.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:15,  9.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:01<00:11, 11.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:11, 11.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:11, 11.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:11, 11.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:11, 11.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:01<00:09, 14.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:09, 14.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:09, 14.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:09, 14.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:08, 14.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:01<00:07, 17.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:07, 17.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:07, 17.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:07, 17.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:07, 17.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:01<00:06, 19.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:06, 19.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:06, 19.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:06, 19.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:06, 19.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:01<00:05, 22.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:05, 22.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:05, 22.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:05, 22.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:05, 22.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 46/162 [00:01<00:04, 23.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:04, 23.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:04, 23.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:04, 23.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:04, 23.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███       | 50/162 [00:02<00:04, 24.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:02<00:04, 24.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:02<00:04, 24.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:02<00:04, 24.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:02<00:04, 25.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:02<00:04, 25.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:02<00:04, 25.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:02<00:04, 25.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:02<00:04, 25.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:02<00:04, 25.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:02<00:04, 25.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:02<00:04, 25.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:02<00:03, 25.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:02<00:03, 25.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:02<00:03, 25.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:02<00:03, 25.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:02<00:03, 25.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:02<00:03, 25.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:02<00:03, 25.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:02<00:03, 25.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:02<00:03, 25.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:02<00:03, 25.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:02<00:03, 25.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:02<00:03, 25.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:02<00:03, 25.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:02<00:03, 25.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:02<00:03, 25.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:02<00:03, 25.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:02<00:03, 25.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:02<00:03, 25.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:02<00:03, 25.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:02<00:03, 25.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:02<00:03, 25.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:02<00:03, 25.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:02<00:03, 25.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:03<00:03, 25.05 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:03<00:03, 25.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:03<00:03, 25.01 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:03<00:03, 25.01 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:03<00:03, 25.01 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:03<00:03, 24.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:03<00:03, 24.47 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:03<00:03, 24.47 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:03<00:03, 24.47 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:03<00:03, 23.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:03<00:03, 23.86 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:03<00:03, 23.86 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:03<00:03, 23.86 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:03<00:03, 23.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:03<00:03, 23.20 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:03<00:03, 23.20 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:03<00:03, 23.20 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:03<00:03, 22.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:03<00:03, 22.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:03<00:03, 22.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:03<00:03, 22.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  57%|█████▋    | 92/162 [00:03<00:03, 22.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:03<00:03, 22.32 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:03<00:03, 22.32 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:03<00:03, 22.32 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:03<00:03, 22.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:03<00:03, 22.07 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:03<00:02, 22.07 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:03<00:02, 22.07 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:04<00:02, 21.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:04<00:02, 21.84 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:04<00:02, 21.84 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:04<00:02, 21.84 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:04<00:02, 21.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:04<00:02, 21.72 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:04<00:02, 21.72 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:04<00:02, 21.72 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:04<00:02, 21.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:04<00:02, 21.64 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:04<00:02, 21.64 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:04<00:02, 21.64 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:04<00:02, 21.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:04<00:02, 21.67 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:04<00:02, 21.67 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:04<00:02, 21.67 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  68%|██████▊   | 110/162 [00:04<00:02, 21.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:04<00:02, 21.57 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:04<00:02, 21.57 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:04<00:02, 21.57 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:04<00:02, 21.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:04<00:02, 21.24 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:04<00:02, 21.24 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:04<00:02, 21.24 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:04<00:02, 20.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:04<00:02, 20.99 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:04<00:02, 20.99 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:04<00:02, 20.99 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:05<00:02, 20.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:05<00:02, 20.78 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:05<00:02, 20.78 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:05<00:01, 20.78 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:05<00:01, 20.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:05<00:01, 20.87 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:05<00:01, 20.87 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:05<00:01, 20.87 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:05<00:01, 22.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:05<00:01, 22.38 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:05<00:01, 22.38 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:05<00:01, 22.38 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:05<00:01, 23.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:05<00:01, 23.58 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:05<00:01, 23.58 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:05<00:01, 23.58 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:05<00:01, 24.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:05<00:01, 24.14 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:05<00:01, 24.14 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:05<00:01, 24.14 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:05<00:01, 23.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:05<00:01, 23.50 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:05<00:01, 23.50 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:05<00:01, 23.50 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:05<00:01, 22.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:05<00:01, 22.62 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:05<00:01, 22.62 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:05<00:01, 22.62 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:05<00:01, 21.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:05<00:01, 21.92 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:05<00:00, 21.92 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:06<00:00, 21.92 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:06<00:00, 21.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:06<00:00, 21.23 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:06<00:00, 21.23 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:06<00:00, 21.23 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:06<00:00, 20.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:06<00:00, 20.78 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:06<00:00, 20.78 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:06<00:00, 20.78 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:06<00:00, 20.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:06<00:00, 20.14 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:06<00:00, 20.14 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:06<00:00, 20.14 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:06<00:00, 19.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:06<00:00, 19.61 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:06<00:00, 19.61 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:06<00:00, 19.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:06<00:00, 19.19 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:06<00:00, 19.19 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:06<00:00, 18.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:06<00:00, 18.58 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:06<00:00, 18.58 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:06<00:00, 18.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:06<00:00, 18.17 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:06<00:00, 18.17 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:06<00:00, 18.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:07<00:00, 18.56 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:07<00:00, 18.56 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:07<00:00, 18.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:07<00:00, 18.13 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:07<00:00,  7.12s/ url]Dl Completed...: 100%|██████████| 1/1 [00:07<00:00,  7.12s/ url]
Dl Size...: 100%|██████████| 162/162 [00:07<00:00, 18.13 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:07<00:00,  7.12s/ url]
Dl Size...: 100%|██████████| 162/162 [00:07<00:00, 18.13 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:07<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:08<00:00,  9.00s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:09<00:00,  7.12s/ url]
Dl Size...: 100%|██████████| 162/162 [00:08<00:00, 18.13 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:08<00:00,  9.00s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:08<00:00,  9.00s/ file]
Dl Size...: 100%|██████████| 162/162 [00:09<00:00, 18.00 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:09<00:00,  9.00s/ url]
0 examples [00:00, ? examples/s]2020-07-01 15:53:13.177691: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-01 15:53:13.190564: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-07-01 15:53:13.190921: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ddb4d4f220 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-01 15:53:13.190938: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
20 examples [00:00, 198.92 examples/s]139 examples [00:00, 265.14 examples/s]259 examples [00:00, 345.90 examples/s]380 examples [00:00, 440.06 examples/s]499 examples [00:00, 542.58 examples/s]617 examples [00:00, 647.49 examples/s]736 examples [00:00, 748.99 examples/s]856 examples [00:00, 843.00 examples/s]977 examples [00:00, 925.58 examples/s]1094 examples [00:01, 986.37 examples/s]1219 examples [00:01, 1051.97 examples/s]1342 examples [00:01, 1099.63 examples/s]1461 examples [00:01, 1120.59 examples/s]1583 examples [00:01, 1147.46 examples/s]1703 examples [00:01, 1155.42 examples/s]1825 examples [00:01, 1171.38 examples/s]1945 examples [00:01, 1177.03 examples/s]2067 examples [00:01, 1188.86 examples/s]2188 examples [00:01, 1195.09 examples/s]2309 examples [00:02, 1184.83 examples/s]2429 examples [00:02, 1181.14 examples/s]2548 examples [00:02, 1179.11 examples/s]2669 examples [00:02, 1187.68 examples/s]2792 examples [00:02, 1198.91 examples/s]2916 examples [00:02, 1209.50 examples/s]3043 examples [00:02, 1224.53 examples/s]3166 examples [00:02, 1201.93 examples/s]3296 examples [00:02, 1228.36 examples/s]3420 examples [00:02, 1222.20 examples/s]3543 examples [00:03, 1219.59 examples/s]3666 examples [00:03, 1204.73 examples/s]3787 examples [00:03, 1182.51 examples/s]3906 examples [00:03, 1164.73 examples/s]4026 examples [00:03, 1172.56 examples/s]4144 examples [00:03, 1173.04 examples/s]4266 examples [00:03, 1182.45 examples/s]4386 examples [00:03, 1187.47 examples/s]4507 examples [00:03, 1194.01 examples/s]4627 examples [00:03, 1153.11 examples/s]4743 examples [00:04, 1154.70 examples/s]4861 examples [00:04, 1161.38 examples/s]4978 examples [00:04, 1160.52 examples/s]5095 examples [00:04, 1141.77 examples/s]5215 examples [00:04, 1157.27 examples/s]5338 examples [00:04, 1177.40 examples/s]5464 examples [00:04, 1200.82 examples/s]5591 examples [00:04, 1217.78 examples/s]5714 examples [00:04, 1220.88 examples/s]5840 examples [00:04, 1231.29 examples/s]5964 examples [00:05, 1218.80 examples/s]6087 examples [00:05, 1221.19 examples/s]6210 examples [00:05, 1203.82 examples/s]6331 examples [00:05, 1199.40 examples/s]6455 examples [00:05, 1209.67 examples/s]6577 examples [00:05, 1203.97 examples/s]6698 examples [00:05, 1202.73 examples/s]6819 examples [00:05, 1192.15 examples/s]6950 examples [00:05, 1224.41 examples/s]7073 examples [00:05, 1217.28 examples/s]7195 examples [00:06, 1213.96 examples/s]7317 examples [00:06, 1208.61 examples/s]7445 examples [00:06, 1227.65 examples/s]7573 examples [00:06, 1240.90 examples/s]7703 examples [00:06, 1256.12 examples/s]7830 examples [00:06, 1259.22 examples/s]7961 examples [00:06, 1272.41 examples/s]8089 examples [00:06, 1274.57 examples/s]8219 examples [00:06, 1280.96 examples/s]8348 examples [00:07, 1273.73 examples/s]8476 examples [00:07, 1254.83 examples/s]8602 examples [00:07, 1227.54 examples/s]8725 examples [00:07, 1202.77 examples/s]8846 examples [00:07, 1200.45 examples/s]8967 examples [00:07, 1193.54 examples/s]9087 examples [00:07, 1178.24 examples/s]9210 examples [00:07, 1191.55 examples/s]9330 examples [00:07, 1186.95 examples/s]9449 examples [00:07, 1165.50 examples/s]9566 examples [00:08, 1158.06 examples/s]9683 examples [00:08, 1159.47 examples/s]9803 examples [00:08, 1170.49 examples/s]9921 examples [00:08, 1134.60 examples/s]10035 examples [00:08, 1067.58 examples/s]10154 examples [00:08, 1101.42 examples/s]10274 examples [00:08, 1127.63 examples/s]10388 examples [00:08, 1123.53 examples/s]10501 examples [00:08, 1093.74 examples/s]10614 examples [00:08, 1103.81 examples/s]10729 examples [00:09, 1116.22 examples/s]10847 examples [00:09, 1132.43 examples/s]10966 examples [00:09, 1146.90 examples/s]11086 examples [00:09, 1160.80 examples/s]11216 examples [00:09, 1198.92 examples/s]11342 examples [00:09, 1216.24 examples/s]11467 examples [00:09, 1223.40 examples/s]11597 examples [00:09, 1244.67 examples/s]11722 examples [00:09, 1217.97 examples/s]11852 examples [00:10, 1240.02 examples/s]11980 examples [00:10, 1248.50 examples/s]12109 examples [00:10, 1258.69 examples/s]12236 examples [00:10, 1228.94 examples/s]12360 examples [00:10, 1212.28 examples/s]12482 examples [00:10, 1204.41 examples/s]12603 examples [00:10, 1195.20 examples/s]12730 examples [00:10, 1213.47 examples/s]12859 examples [00:10, 1234.11 examples/s]12988 examples [00:10, 1249.60 examples/s]13114 examples [00:11, 1247.29 examples/s]13239 examples [00:11, 1247.25 examples/s]13370 examples [00:11, 1262.69 examples/s]13497 examples [00:11, 1244.35 examples/s]13622 examples [00:11, 1229.12 examples/s]13746 examples [00:11, 1225.49 examples/s]13873 examples [00:11, 1236.89 examples/s]13997 examples [00:11, 1209.97 examples/s]14119 examples [00:11, 1199.93 examples/s]14243 examples [00:11, 1209.01 examples/s]14370 examples [00:12, 1226.45 examples/s]14493 examples [00:12, 1218.78 examples/s]14620 examples [00:12, 1231.31 examples/s]14744 examples [00:12, 1206.43 examples/s]14865 examples [00:12, 1187.53 examples/s]14984 examples [00:12, 1184.73 examples/s]15103 examples [00:12, 1141.69 examples/s]15223 examples [00:12, 1156.95 examples/s]15347 examples [00:12, 1178.18 examples/s]15466 examples [00:12, 1137.14 examples/s]15583 examples [00:13, 1145.29 examples/s]15699 examples [00:13, 1148.50 examples/s]15817 examples [00:13, 1155.72 examples/s]15933 examples [00:13, 1150.47 examples/s]16049 examples [00:13, 1125.35 examples/s]16162 examples [00:13, 1126.62 examples/s]16286 examples [00:13, 1155.66 examples/s]16418 examples [00:13, 1199.20 examples/s]16539 examples [00:13, 1188.80 examples/s]16659 examples [00:14, 1171.27 examples/s]16779 examples [00:14, 1179.67 examples/s]16901 examples [00:14, 1189.85 examples/s]17028 examples [00:14, 1212.81 examples/s]17150 examples [00:14, 1204.06 examples/s]17271 examples [00:14, 1186.00 examples/s]17392 examples [00:14, 1191.97 examples/s]17512 examples [00:14, 1183.64 examples/s]17631 examples [00:14, 1170.62 examples/s]17749 examples [00:14, 1148.61 examples/s]17872 examples [00:15, 1171.63 examples/s]17994 examples [00:15, 1185.47 examples/s]18119 examples [00:15, 1202.10 examples/s]18243 examples [00:15, 1212.74 examples/s]18365 examples [00:15, 1205.50 examples/s]18486 examples [00:15, 1198.92 examples/s]18606 examples [00:15, 1182.40 examples/s]18725 examples [00:15, 1173.14 examples/s]18844 examples [00:15, 1176.69 examples/s]18962 examples [00:15, 1165.19 examples/s]19081 examples [00:16, 1169.65 examples/s]19211 examples [00:16, 1203.28 examples/s]19335 examples [00:16, 1211.95 examples/s]19461 examples [00:16, 1223.52 examples/s]19584 examples [00:16, 1224.00 examples/s]19707 examples [00:16, 1217.38 examples/s]19829 examples [00:16, 1208.50 examples/s]19950 examples [00:16, 1193.41 examples/s]20070 examples [00:16, 1138.07 examples/s]20188 examples [00:16, 1150.03 examples/s]20309 examples [00:17, 1165.26 examples/s]20428 examples [00:17, 1170.52 examples/s]20546 examples [00:17, 1168.61 examples/s]20672 examples [00:17, 1193.01 examples/s]20797 examples [00:17, 1207.13 examples/s]20919 examples [00:17, 1208.13 examples/s]21040 examples [00:17, 1204.19 examples/s]21161 examples [00:17, 1196.59 examples/s]21281 examples [00:17, 1192.90 examples/s]21401 examples [00:18, 1143.91 examples/s]21516 examples [00:18, 1127.92 examples/s]21635 examples [00:18, 1143.77 examples/s]21756 examples [00:18, 1160.88 examples/s]21876 examples [00:18, 1170.06 examples/s]21994 examples [00:18, 1168.47 examples/s]22115 examples [00:18, 1180.50 examples/s]22234 examples [00:18, 1116.76 examples/s]22354 examples [00:18, 1139.89 examples/s]22473 examples [00:18, 1152.40 examples/s]22590 examples [00:19, 1156.31 examples/s]22710 examples [00:19, 1167.10 examples/s]22827 examples [00:19, 1167.79 examples/s]22944 examples [00:19, 1137.80 examples/s]23061 examples [00:19, 1144.76 examples/s]23177 examples [00:19, 1149.20 examples/s]23298 examples [00:19, 1166.50 examples/s]23415 examples [00:19, 1146.13 examples/s]23530 examples [00:19, 1144.35 examples/s]23646 examples [00:19, 1146.49 examples/s]23766 examples [00:20, 1160.61 examples/s]23895 examples [00:20, 1194.93 examples/s]24022 examples [00:20, 1215.53 examples/s]24149 examples [00:20, 1229.86 examples/s]24277 examples [00:20, 1243.63 examples/s]24403 examples [00:20, 1248.00 examples/s]24528 examples [00:20, 1218.96 examples/s]24651 examples [00:20, 1194.09 examples/s]24774 examples [00:20, 1201.90 examples/s]24895 examples [00:20, 1181.73 examples/s]25014 examples [00:21, 1176.19 examples/s]25132 examples [00:21, 1173.27 examples/s]25250 examples [00:21, 1170.52 examples/s]25374 examples [00:21, 1188.57 examples/s]25493 examples [00:21, 1185.10 examples/s]25612 examples [00:21, 1184.57 examples/s]25732 examples [00:21, 1186.48 examples/s]25851 examples [00:21, 1182.16 examples/s]25970 examples [00:21, 1177.04 examples/s]26088 examples [00:22, 1154.24 examples/s]26206 examples [00:22, 1159.30 examples/s]26324 examples [00:22, 1164.54 examples/s]26441 examples [00:22, 1129.81 examples/s]26557 examples [00:22, 1137.39 examples/s]26676 examples [00:22, 1151.32 examples/s]26793 examples [00:22, 1155.79 examples/s]26911 examples [00:22, 1161.74 examples/s]27030 examples [00:22, 1168.57 examples/s]27147 examples [00:22, 1156.97 examples/s]27263 examples [00:23, 1153.21 examples/s]27387 examples [00:23, 1177.10 examples/s]27515 examples [00:23, 1205.74 examples/s]27641 examples [00:23, 1221.05 examples/s]27765 examples [00:23, 1225.24 examples/s]27894 examples [00:23, 1243.35 examples/s]28024 examples [00:23, 1256.73 examples/s]28150 examples [00:23, 1256.02 examples/s]28276 examples [00:23, 1227.55 examples/s]28399 examples [00:23, 1204.64 examples/s]28520 examples [00:24, 1171.60 examples/s]28638 examples [00:24, 1166.20 examples/s]28757 examples [00:24, 1172.75 examples/s]28875 examples [00:24, 1173.66 examples/s]28995 examples [00:24, 1179.32 examples/s]29114 examples [00:24, 1180.57 examples/s]29235 examples [00:24, 1188.28 examples/s]29354 examples [00:24, 1172.79 examples/s]29473 examples [00:24, 1175.50 examples/s]29591 examples [00:24, 1158.14 examples/s]29708 examples [00:25, 1160.40 examples/s]29825 examples [00:25, 1112.10 examples/s]29937 examples [00:25, 1104.40 examples/s]30048 examples [00:25, 1076.29 examples/s]30171 examples [00:25, 1117.09 examples/s]30285 examples [00:25, 1122.66 examples/s]30411 examples [00:25, 1159.28 examples/s]30538 examples [00:25, 1189.39 examples/s]30664 examples [00:25, 1207.21 examples/s]30786 examples [00:26, 1198.89 examples/s]30907 examples [00:26, 1189.60 examples/s]31027 examples [00:26, 1176.42 examples/s]31145 examples [00:26, 1168.81 examples/s]31263 examples [00:26, 1152.87 examples/s]31379 examples [00:26, 1152.42 examples/s]31501 examples [00:26, 1171.86 examples/s]31625 examples [00:26, 1190.45 examples/s]31753 examples [00:26, 1213.26 examples/s]31878 examples [00:26, 1223.07 examples/s]32001 examples [00:27, 1212.21 examples/s]32123 examples [00:27, 1190.31 examples/s]32243 examples [00:27, 1174.89 examples/s]32361 examples [00:27, 1168.91 examples/s]32480 examples [00:27, 1173.22 examples/s]32600 examples [00:27, 1181.13 examples/s]32719 examples [00:27, 1180.71 examples/s]32839 examples [00:27, 1183.94 examples/s]32958 examples [00:27, 1138.09 examples/s]33076 examples [00:27, 1148.27 examples/s]33192 examples [00:28, 1149.96 examples/s]33308 examples [00:28, 1125.75 examples/s]33427 examples [00:28, 1142.32 examples/s]33542 examples [00:28, 1139.69 examples/s]33660 examples [00:28, 1148.92 examples/s]33780 examples [00:28, 1162.19 examples/s]33897 examples [00:28, 1101.35 examples/s]34025 examples [00:28, 1147.79 examples/s]34149 examples [00:28, 1173.59 examples/s]34274 examples [00:28, 1194.35 examples/s]34395 examples [00:29, 1175.14 examples/s]34514 examples [00:29, 1173.10 examples/s]34632 examples [00:29, 1174.64 examples/s]34752 examples [00:29, 1181.53 examples/s]34871 examples [00:29, 1175.06 examples/s]34991 examples [00:29, 1182.27 examples/s]35110 examples [00:29, 1180.04 examples/s]35229 examples [00:29, 1176.85 examples/s]35347 examples [00:29, 1172.67 examples/s]35467 examples [00:30, 1178.43 examples/s]35588 examples [00:30, 1186.52 examples/s]35710 examples [00:30, 1192.21 examples/s]35830 examples [00:30, 1183.51 examples/s]35949 examples [00:30, 1168.49 examples/s]36066 examples [00:30, 1165.96 examples/s]36183 examples [00:30, 1160.90 examples/s]36301 examples [00:30, 1164.16 examples/s]36422 examples [00:30, 1176.09 examples/s]36540 examples [00:30, 1169.19 examples/s]36658 examples [00:31, 1169.33 examples/s]36778 examples [00:31, 1177.97 examples/s]36900 examples [00:31, 1190.02 examples/s]37020 examples [00:31, 1188.51 examples/s]37139 examples [00:31, 1181.14 examples/s]37258 examples [00:31, 1174.68 examples/s]37376 examples [00:31, 1159.87 examples/s]37493 examples [00:31, 1142.01 examples/s]37620 examples [00:31, 1177.42 examples/s]37739 examples [00:31, 1164.05 examples/s]37860 examples [00:32, 1176.83 examples/s]37983 examples [00:32, 1191.39 examples/s]38104 examples [00:32, 1196.65 examples/s]38224 examples [00:32, 1160.09 examples/s]38341 examples [00:32, 1144.31 examples/s]38458 examples [00:32, 1151.16 examples/s]38574 examples [00:32, 1152.09 examples/s]38692 examples [00:32, 1158.92 examples/s]38813 examples [00:32, 1172.35 examples/s]38931 examples [00:32, 1164.76 examples/s]39054 examples [00:33, 1181.99 examples/s]39180 examples [00:33, 1201.57 examples/s]39304 examples [00:33, 1210.66 examples/s]39426 examples [00:33, 1202.16 examples/s]39547 examples [00:33, 1170.20 examples/s]39665 examples [00:33, 1168.49 examples/s]39786 examples [00:33, 1177.94 examples/s]39914 examples [00:33, 1205.97 examples/s]40035 examples [00:33, 1163.72 examples/s]40159 examples [00:33, 1183.44 examples/s]40279 examples [00:34, 1186.75 examples/s]40398 examples [00:34, 1167.99 examples/s]40516 examples [00:34, 1147.04 examples/s]40631 examples [00:34, 1121.55 examples/s]40749 examples [00:34, 1136.11 examples/s]40869 examples [00:34, 1154.23 examples/s]40985 examples [00:34, 1125.27 examples/s]41104 examples [00:34, 1143.84 examples/s]41223 examples [00:34, 1156.95 examples/s]41339 examples [00:35, 1156.49 examples/s]41458 examples [00:35, 1165.36 examples/s]41578 examples [00:35, 1174.67 examples/s]41699 examples [00:35, 1183.19 examples/s]41818 examples [00:35, 1184.53 examples/s]41937 examples [00:35, 1181.52 examples/s]42056 examples [00:35, 1179.06 examples/s]42174 examples [00:35, 1177.63 examples/s]42294 examples [00:35, 1182.87 examples/s]42413 examples [00:35, 1184.10 examples/s]42532 examples [00:36, 1183.72 examples/s]42651 examples [00:36, 1176.94 examples/s]42769 examples [00:36, 1166.93 examples/s]42896 examples [00:36, 1195.11 examples/s]43021 examples [00:36, 1209.65 examples/s]43143 examples [00:36, 1195.39 examples/s]43263 examples [00:36, 1194.68 examples/s]43383 examples [00:36, 1190.57 examples/s]43503 examples [00:36, 1188.15 examples/s]43622 examples [00:36, 1181.69 examples/s]43741 examples [00:37, 1172.56 examples/s]43860 examples [00:37, 1176.21 examples/s]43978 examples [00:37, 1171.27 examples/s]44096 examples [00:37, 1172.89 examples/s]44216 examples [00:37, 1178.44 examples/s]44334 examples [00:37, 1173.44 examples/s]44452 examples [00:37, 1168.25 examples/s]44569 examples [00:37, 1127.04 examples/s]44691 examples [00:37, 1150.75 examples/s]44807 examples [00:37, 1144.46 examples/s]44923 examples [00:38, 1146.79 examples/s]45041 examples [00:38, 1155.19 examples/s]45162 examples [00:38, 1170.72 examples/s]45283 examples [00:38, 1179.67 examples/s]45402 examples [00:38, 1174.03 examples/s]45520 examples [00:38, 1175.55 examples/s]45642 examples [00:38, 1186.04 examples/s]45762 examples [00:38, 1188.70 examples/s]45881 examples [00:38, 1188.73 examples/s]46000 examples [00:38, 1173.71 examples/s]46118 examples [00:39, 1174.36 examples/s]46237 examples [00:39, 1176.36 examples/s]46358 examples [00:39, 1183.77 examples/s]46477 examples [00:39, 1136.37 examples/s]46592 examples [00:39, 1131.94 examples/s]46714 examples [00:39, 1156.89 examples/s]46833 examples [00:39, 1164.65 examples/s]46952 examples [00:39, 1170.63 examples/s]47072 examples [00:39, 1178.27 examples/s]47193 examples [00:40, 1185.60 examples/s]47312 examples [00:40, 1180.12 examples/s]47431 examples [00:40, 1180.40 examples/s]47550 examples [00:40, 1179.58 examples/s]47668 examples [00:40, 1168.60 examples/s]47785 examples [00:40, 1167.11 examples/s]47902 examples [00:40, 1160.34 examples/s]48019 examples [00:40, 1160.83 examples/s]48136 examples [00:40, 1129.72 examples/s]48255 examples [00:40, 1146.63 examples/s]48374 examples [00:41, 1156.70 examples/s]48491 examples [00:41, 1160.37 examples/s]48611 examples [00:41, 1170.53 examples/s]48736 examples [00:41, 1190.36 examples/s]48858 examples [00:41, 1196.44 examples/s]48981 examples [00:41, 1206.29 examples/s]49107 examples [00:41, 1220.34 examples/s]49232 examples [00:41, 1228.65 examples/s]49355 examples [00:41, 1220.09 examples/s]49478 examples [00:41, 1213.91 examples/s]49600 examples [00:42, 1204.36 examples/s]49721 examples [00:42, 1197.39 examples/s]49841 examples [00:42, 1196.27 examples/s]49961 examples [00:42, 1193.75 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 18%|█▊        | 9018/50000 [00:00<00:00, 90176.22 examples/s] 49%|████▊     | 24284/50000 [00:00<00:00, 102798.23 examples/s] 77%|███████▋  | 38645/50000 [00:00<00:00, 112378.50 examples/s]                                                                0 examples [00:00, ? examples/s]98 examples [00:00, 971.52 examples/s]218 examples [00:00, 1029.69 examples/s]339 examples [00:00, 1076.39 examples/s]458 examples [00:00, 1106.05 examples/s]578 examples [00:00, 1131.15 examples/s]700 examples [00:00, 1154.09 examples/s]815 examples [00:00, 1152.03 examples/s]937 examples [00:00, 1169.75 examples/s]1053 examples [00:00, 1165.21 examples/s]1175 examples [00:01, 1179.24 examples/s]1291 examples [00:01, 1158.30 examples/s]1410 examples [00:01, 1167.60 examples/s]1532 examples [00:01, 1180.31 examples/s]1650 examples [00:01, 1134.29 examples/s]1768 examples [00:01, 1145.61 examples/s]1883 examples [00:01, 1146.38 examples/s]1998 examples [00:01, 1117.40 examples/s]2110 examples [00:01, 1062.51 examples/s]2230 examples [00:01, 1100.32 examples/s]2361 examples [00:02, 1155.02 examples/s]2486 examples [00:02, 1181.95 examples/s]2607 examples [00:02, 1188.29 examples/s]2731 examples [00:02, 1203.09 examples/s]2852 examples [00:02, 1179.02 examples/s]2973 examples [00:02, 1186.24 examples/s]3096 examples [00:02, 1196.53 examples/s]3216 examples [00:02, 1185.95 examples/s]3335 examples [00:02, 1187.08 examples/s]3454 examples [00:02, 1177.52 examples/s]3576 examples [00:03, 1188.37 examples/s]3695 examples [00:03, 1181.17 examples/s]3815 examples [00:03, 1185.36 examples/s]3935 examples [00:03, 1188.31 examples/s]4054 examples [00:03, 1185.56 examples/s]4176 examples [00:03, 1195.45 examples/s]4296 examples [00:03, 1180.75 examples/s]4416 examples [00:03, 1184.63 examples/s]4535 examples [00:03, 1175.30 examples/s]4653 examples [00:03, 1171.10 examples/s]4775 examples [00:04, 1182.37 examples/s]4894 examples [00:04, 1127.96 examples/s]5014 examples [00:04, 1148.06 examples/s]5134 examples [00:04, 1162.54 examples/s]5251 examples [00:04, 1153.24 examples/s]5371 examples [00:04, 1166.02 examples/s]5490 examples [00:04, 1171.19 examples/s]5613 examples [00:04, 1185.86 examples/s]5732 examples [00:04, 1186.66 examples/s]5851 examples [00:05, 1182.77 examples/s]5974 examples [00:05, 1196.22 examples/s]6094 examples [00:05, 1182.79 examples/s]6213 examples [00:05, 1183.32 examples/s]6335 examples [00:05, 1191.76 examples/s]6455 examples [00:05, 1136.47 examples/s]6570 examples [00:05, 1134.28 examples/s]6692 examples [00:05, 1158.19 examples/s]6809 examples [00:05, 1144.27 examples/s]6929 examples [00:05, 1159.11 examples/s]7047 examples [00:06, 1164.59 examples/s]7168 examples [00:06, 1177.34 examples/s]7298 examples [00:06, 1211.40 examples/s]7422 examples [00:06, 1219.60 examples/s]7546 examples [00:06, 1224.81 examples/s]7669 examples [00:06, 1209.72 examples/s]7791 examples [00:06, 1160.88 examples/s]7910 examples [00:06, 1167.44 examples/s]8029 examples [00:06, 1172.42 examples/s]8147 examples [00:06, 1135.19 examples/s]8263 examples [00:07, 1140.41 examples/s]8379 examples [00:07, 1144.85 examples/s]8504 examples [00:07, 1173.17 examples/s]8627 examples [00:07, 1186.76 examples/s]8749 examples [00:07, 1194.60 examples/s]8869 examples [00:07, 1184.17 examples/s]8994 examples [00:07, 1202.95 examples/s]9118 examples [00:07, 1213.06 examples/s]9240 examples [00:07, 1199.29 examples/s]9363 examples [00:07, 1207.34 examples/s]9484 examples [00:08, 1193.16 examples/s]9605 examples [00:08, 1196.87 examples/s]9734 examples [00:08, 1221.90 examples/s]9861 examples [00:08, 1234.82 examples/s]9989 examples [00:08, 1247.40 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteGSR6BZ/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteGSR6BZ/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f4d566e3488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f4d566e3488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f4d566e3488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4d3d891550>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4cf097f748>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f4d566e3488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f4d566e3488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f4d566e3488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4d3d7300f0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4d3d730470>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f4cd802e268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f4cd802e268> 

  function with postional parmater data_info <function split_train_valid at 0x7f4cd802e268> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f4cd802e378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f4cd802e378> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f4cd802e378> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.47.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.10)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=8ac231e753b8a6c1c79d9ab737c3ade87e0edaed59f2e2d2d0c7168d9251a084
  Stored in directory: /tmp/pip-ephem-wheel-cache-ysun29k9/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.3.0
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:08:33, 11.9kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:19:50, 16.7kB/s].vector_cache/glove.6B.zip:   0%|          | 213k/862M [00:00<10:05:15, 23.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 877k/862M [00:01<7:04:12, 33.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.53M/862M [00:01<4:56:15, 48.3kB/s].vector_cache/glove.6B.zip:   1%|          | 9.81M/862M [00:01<3:25:57, 69.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.7M/862M [00:01<2:23:14, 98.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.7M/862M [00:01<1:40:03, 140kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 24.4M/862M [00:01<1:09:38, 200kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.5M/862M [00:01<48:43, 286kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 32.9M/862M [00:01<33:57, 407kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.1M/862M [00:01<23:49, 578kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.8M/862M [00:02<16:38, 822kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.9M/862M [00:02<11:44, 1.16MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.3M/862M [00:02<08:14, 1.64MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.2M/862M [00:02<06:51, 1.97MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:04<06:41, 2.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:04<08:18, 1.62MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:05<06:37, 2.03MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.0M/862M [00:05<04:50, 2.77MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:06<07:32, 1.77MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:06<06:39, 2.00MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:07<04:56, 2.70MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:08<06:33, 2.03MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<07:17, 1.82MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.7M/862M [00:09<05:39, 2.34MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:09<04:07, 3.21MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:10<10:38, 1.24MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:10<08:47, 1.50MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.7M/862M [00:10<06:25, 2.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:12<07:35, 1.73MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<07:58, 1.65MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:12<06:09, 2.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:13<04:59, 2.63MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:14<7:45:02, 28.2kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<5:25:35, 40.2kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:16<3:49:21, 56.8kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:16<2:43:55, 79.5kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<1:55:26, 113kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 83.9M/862M [00:16<1:20:43, 161kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:18<1:04:32, 201kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:18<46:34, 278kB/s]  .vector_cache/glove.6B.zip:  10%|█         | 86.5M/862M [00:18<32:50, 394kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:20<25:47, 500kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:20<20:40, 623kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<15:00, 858kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.4M/862M [00:20<10:37, 1.21MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:22<18:55, 677kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:22<14:34, 879kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.9M/862M [00:22<10:30, 1.22MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:24<10:18, 1.24MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:24<09:49, 1.30MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:24<07:31, 1.69MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<07:18, 1.74MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:25, 1.97MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<04:48, 2.63MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<06:18, 2.00MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<06:58, 1.81MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:31, 2.28MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<05:53, 2.13MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:39, 1.88MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:12, 2.41MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<03:49, 3.27MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<07:48, 1.60MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:43, 1.85MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<04:58, 2.50MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:24, 1.94MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:44, 2.16MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<04:19, 2.87MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<05:56, 2.08MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<06:41, 1.85MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:18, 2.32MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<03:50, 3.19MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<11:49:26, 17.3kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<8:17:37, 24.6kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<5:47:55, 35.2kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<4:05:39, 49.7kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<2:53:07, 70.5kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<2:01:14, 100kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<1:27:28, 139kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<1:02:25, 194kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<43:54, 276kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<33:31, 360kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<25:50, 467kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<18:37, 647kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<13:06, 916kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<27:20, 439kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<20:23, 588kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<14:30, 825kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<12:55, 923kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<11:27, 1.04MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<08:32, 1.39MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<06:05, 1.95MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<18:34, 638kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<14:01, 845kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<10:05, 1.17MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<07:12, 1.63MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<1:15:17, 157kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<53:52, 219kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:51<37:56, 310kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<27:16, 430kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<7:54:47, 24.7kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<5:32:13, 35.3kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<3:53:44, 49.9kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<2:45:50, 70.3kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<1:56:29, 100kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<1:23:07, 140kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<59:24, 195kB/s]  .vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:57<41:46, 277kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<31:40, 364kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<24:30, 470kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<17:43, 649kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [00:59<12:30, 917kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<1:30:40, 126kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<1:04:35, 177kB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:01<45:24, 252kB/s]  .vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<34:21, 332kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<26:21, 432kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<18:54, 601kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:03<13:19, 850kB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<16:51, 671kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<12:57, 873kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<09:20, 1.21MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<09:09, 1.23MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<07:33, 1.49MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<05:33, 2.02MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<06:30, 1.72MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:41, 1.97MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<04:12, 2.65MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<05:36, 1.98MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<05:02, 2.20MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<03:48, 2.92MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<05:16, 2.10MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<05:55, 1.87MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<04:38, 2.38MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:13<03:27, 3.18MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<09:15, 1.19MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<08:26, 1.30MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<06:23, 1.71MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<06:18, 1.73MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<08:14, 1.32MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<06:44, 1.62MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<04:55, 2.21MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<06:16, 1.73MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<06:20, 1.71MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<04:50, 2.23MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<03:33, 3.03MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<06:59, 1.54MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<06:50, 1.57MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:11, 2.07MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:21<03:44, 2.86MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<16:06, 664kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<13:07, 815kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<09:35, 1.11MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<06:48, 1.56MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<19:48, 537kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<17:49, 596kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<13:25, 791kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<09:34, 1.11MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<09:26, 1.12MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 228M/862M [01:26<08:29, 1.24MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<06:24, 1.65MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<06:14, 1.68MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<08:00, 1.31MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<06:33, 1.60MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<04:46, 2.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<06:01, 1.73MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<06:05, 1.71MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<04:43, 2.20MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<05:03, 2.05MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<07:08, 1.45MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<05:46, 1.79MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<04:15, 2.42MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<05:11, 1.98MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<05:28, 1.88MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<04:17, 2.40MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<03:31, 2.90MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<6:15:01, 27.3kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<4:22:35, 38.9kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<3:04:33, 55.1kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<2:12:59, 76.4kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<1:33:56, 108kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<1:05:45, 154kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<48:29, 208kB/s]  .vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<35:41, 283kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<25:24, 397kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<19:23, 517kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<17:18, 579kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<12:53, 777kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<09:15, 1.08MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<09:00, 1.10MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<08:02, 1.24MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<05:59, 1.66MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<04:17, 2.30MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<11:09, 885kB/s] .vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<09:35, 1.03MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<07:07, 1.39MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<06:37, 1.48MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<08:02, 1.22MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<06:32, 1.50MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<04:44, 2.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<05:50, 1.67MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<05:46, 1.69MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:25, 2.20MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:50<03:10, 3.04MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<34:03, 284kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<27:27, 352kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<20:03, 482kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<14:13, 678kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<12:26, 773kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<10:26, 920kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<07:43, 1.24MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<06:59, 1.36MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<06:36, 1.44MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<05:02, 1.89MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<05:07, 1.85MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<05:17, 1.79MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:06, 2.30MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<04:27, 2.11MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<04:48, 1.95MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<03:46, 2.48MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<04:12, 2.21MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<06:11, 1.51MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<05:09, 1.81MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<03:46, 2.46MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<05:04, 1.82MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<05:13, 1.77MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:04, 2.27MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:23, 2.09MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:43, 1.95MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<03:42, 2.48MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:07, 2.21MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<04:28, 2.04MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<03:32, 2.58MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<03:59, 2.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<06:11, 1.46MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<05:07, 1.77MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<03:46, 2.39MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:59, 1.80MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<05:07, 1.76MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<03:54, 2.29MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<02:49, 3.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<12:33, 710kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<11:53, 750kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<09:01, 988kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<06:29, 1.37MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<06:50, 1.30MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<06:22, 1.39MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<04:47, 1.84MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<03:25, 2.56MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<1:07:24, 130kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<50:13, 175kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<35:55, 244kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<25:15, 346kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<19:53, 438kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<15:29, 562kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<11:13, 775kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<09:15, 933kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<08:02, 1.08MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<06:04, 1.42MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<04:41, 1.84MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<03:45, 2.29MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<03:09, 2.73MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<04:40, 1.83MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<3:27:56, 41.3kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<2:26:44, 58.5kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<1:43:03, 83.2kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<1:12:29, 118kB/s] .vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:23<51:06, 167kB/s]  .vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<36:09, 236kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<25:42, 331kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<1:03:37, 134kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<49:08, 173kB/s]  .vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<35:33, 239kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<25:18, 336kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<18:06, 469kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<13:04, 648kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<09:33, 885kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<11:39, 725kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<12:44, 663kB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:27<09:49, 860kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<07:18, 1.15MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<05:38, 1.49MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:27<04:16, 1.97MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:27<03:22, 2.49MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<02:53, 2.90MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<07:39, 1.09MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<09:54, 846kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<07:59, 1.05MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<06:01, 1.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<04:32, 1.84MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<03:29, 2.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:29<03:04, 2.71MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<02:29, 3.34MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<1:27:17, 95.2kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<1:05:38, 127kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<47:00, 177kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<33:17, 249kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<23:39, 350kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<16:54, 489kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<12:12, 676kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<15:04, 547kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<14:33, 566kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<11:05, 742kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<08:11, 1.00MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<06:09, 1.33MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<04:44, 1.73MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:33<03:38, 2.25MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<02:57, 2.77MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<11:00, 742kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<12:10, 671kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<09:33, 854kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<07:02, 1.16MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<05:16, 1.55MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<04:12, 1.93MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:35<03:19, 2.44MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<02:39, 3.05MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<57:33, 141kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<44:13, 183kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<31:59, 253kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<22:47, 355kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<16:18, 495kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<11:42, 688kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:37<08:33, 939kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<12:21, 651kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<12:59, 619kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<10:07, 793kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<07:30, 1.07MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<05:36, 1.43MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<04:17, 1.86MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:39<03:21, 2.37MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<09:36, 829kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<10:35, 752kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<08:23, 948kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<06:18, 1.26MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<04:46, 1.66MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<03:41, 2.15MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:41<02:55, 2.70MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<10:21, 763kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<11:06, 711kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<08:44, 903kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<06:26, 1.22MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:50, 1.62MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<03:42, 2.11MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:43<02:55, 2.68MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:45<28:20, 276kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:45<23:20, 335kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<17:11, 455kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<12:23, 630kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<08:55, 873kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:45<06:32, 1.19MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<08:01, 966kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<09:02, 858kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<07:08, 1.08MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<05:21, 1.44MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<03:59, 1.93MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<03:04, 2.50MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<06:32, 1.18MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<07:41, 1.00MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<06:07, 1.25MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:31, 1.69MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<03:23, 2.26MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<02:47, 2.74MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<05:22, 1.42MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<06:24, 1.19MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<05:09, 1.47MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:53, 1.96MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<02:55, 2.58MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<04:38, 1.62MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<08:28, 890kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<07:13, 1.05MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<05:23, 1.40MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:59, 1.88MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:53<03:00, 2.50MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<14:00, 534kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<12:04, 620kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<08:55, 837kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<06:28, 1.15MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<04:40, 1.59MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<03:35, 2.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<3:10:06, 39.0kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<2:17:04, 54.1kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<1:36:59, 76.4kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<1:08:00, 109kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<47:39, 155kB/s]  .vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<33:26, 220kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<38:42, 190kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<30:39, 240kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<22:24, 328kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<15:52, 461kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<11:14, 649kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<10:58, 663kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<10:54, 668kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<08:19, 874kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<06:03, 1.20MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<04:25, 1.64MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<05:22, 1.34MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<06:58, 1.03MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<05:38, 1.28MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<04:10, 1.72MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<03:01, 2.36MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<07:55, 901kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<08:25, 848kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<06:29, 1.10MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<04:44, 1.50MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:05<03:25, 2.06MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<36:55, 192kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<28:20, 250kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<20:27, 345kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<14:27, 487kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<11:38, 601kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<10:35, 661kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<07:54, 885kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<05:38, 1.24MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<04:04, 1.70MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<09:40, 716kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<08:45, 792kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<06:38, 1.04MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<04:46, 1.44MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<05:25, 1.27MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<05:44, 1.20MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<04:29, 1.52MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<03:16, 2.08MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<03:11, 2.13MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<3:41:00, 30.8kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<2:34:56, 43.9kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<1:48:03, 62.6kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<1:17:37, 86.8kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<58:26, 115kB/s]   .vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<41:47, 161kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<29:21, 229kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:16<20:33, 325kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<18:30, 360kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<14:45, 452kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<10:45, 619kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<07:36, 870kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<07:54, 834kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<06:59, 944kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<05:12, 1.27MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<03:45, 1.75MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<04:27, 1.46MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<04:47, 1.36MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:41, 1.76MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<02:47, 2.33MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:15, 1.98MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:32, 1.83MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<02:47, 2.31MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:24<02:01, 3.15MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<08:36, 743kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<07:34, 844kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<05:41, 1.12MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<04:03, 1.56MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<06:18, 1.00MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<05:47, 1.09MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<04:21, 1.45MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<03:11, 1.97MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<03:43, 1.68MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<04:07, 1.52MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:10, 1.96MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<02:19, 2.68MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:37, 1.71MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<04:01, 1.53MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:10, 1.94MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:32<02:18, 2.66MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<05:41, 1.07MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<05:28, 1.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<04:10, 1.46MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:34<02:59, 2.02MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<06:05, 993kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<05:14, 1.15MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:51, 1.56MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:36<02:45, 2.17MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<07:57, 751kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<06:50, 874kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<05:05, 1.17MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:38<03:36, 1.64MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<16:11, 365kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<12:34, 470kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<09:04, 649kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:40<06:22, 915kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<41:32, 141kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<30:17, 193kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<21:24, 272kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<14:57, 387kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<14:12, 406kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<11:09, 517kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<08:05, 712kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:44<05:39, 1.01MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<31:48, 179kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<23:27, 243kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<16:40, 341kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:46<11:38, 484kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<40:21, 140kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<29:22, 192kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<20:46, 270kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<14:31, 384kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<12:51, 433kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<09:45, 570kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<07:00, 791kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<05:54, 931kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<06:10, 889kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<04:44, 1.16MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<03:24, 1.60MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<03:42, 1.46MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<03:33, 1.52MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:41, 2.00MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<01:56, 2.77MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<05:53, 908kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<06:07, 874kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<04:46, 1.12MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:27, 1.54MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<03:42, 1.43MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<03:34, 1.48MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:44, 1.92MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:46, 1.88MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:55, 1.79MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:16, 2.28MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:26, 2.11MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:40, 1.92MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:06, 2.44MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:18, 2.20MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<02:33, 1.98MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<02:00, 2.53MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:13, 2.25MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<03:25, 1.46MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<02:51, 1.75MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:05, 2.38MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<01:51, 2.67MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<2:57:33, 27.9kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<2:04:09, 39.8kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<1:26:46, 56.3kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<1:02:32, 78.1kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<44:09, 110kB/s]   .vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:09<30:50, 157kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<22:37, 213kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<16:43, 288kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<11:53, 403kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<09:01, 525kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<08:05, 587kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<06:02, 783kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<04:19, 1.09MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<04:09, 1.12MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<03:45, 1.24MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:48, 1.66MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:15<02:00, 2.30MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<04:14, 1.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<03:48, 1.21MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:50, 1.62MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:17<02:01, 2.25MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<05:02, 901kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<05:13, 867kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<04:00, 1.13MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<02:53, 1.56MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:02, 1.47MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:57, 1.51MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:16, 1.96MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:18, 1.90MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:26, 1.80MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<01:54, 2.30MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:02, 2.12MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:14, 1.94MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:45, 2.46MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<01:55, 2.21MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:09, 1.98MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<01:40, 2.54MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:27<01:12, 3.48MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<04:15, 986kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<04:33, 920kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<03:32, 1.18MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<02:33, 1.63MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:48, 1.47MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:44, 1.50MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:06, 1.95MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:07, 1.90MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:00, 1.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:24, 1.68MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<01:46, 2.26MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:13, 1.80MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:16, 1.76MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:46, 2.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<01:52, 2.09MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<02:46, 1.41MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:14, 1.74MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:39, 2.34MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:06, 1.83MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<02:11, 1.76MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:40, 2.29MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<01:13, 3.12MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:44, 1.38MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<03:19, 1.14MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<02:37, 1.44MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:55, 1.95MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:03, 1.80MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:06, 1.76MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:38, 2.25MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:44, 2.09MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:53, 1.92MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:29, 2.43MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:37, 2.20MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<02:28, 1.44MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<02:01, 1.76MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:29, 2.38MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:54, 1.84MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:59, 1.76MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:32, 2.26MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:38, 2.09MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<02:25, 1.41MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<02:01, 1.70MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:28, 2.31MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:50, 1.83MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:54, 1.76MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:29, 2.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:34, 2.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:20, 1.40MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:56, 1.69MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:25, 2.29MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<01:47, 1.80MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:50, 1.75MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:24, 2.28MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<01:00, 3.12MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:54, 1.08MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<02:37, 1.20MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<01:57, 1.61MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:23, 2.24MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<04:14, 729kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<04:07, 751kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<03:09, 975kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<02:15, 1.35MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:46, 1.70MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<1:58:28, 25.6kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<1:22:40, 36.5kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<57:19, 51.7kB/s]  .vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<41:12, 71.8kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<29:02, 102kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<20:12, 145kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<14:42, 197kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<10:49, 267kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<07:40, 375kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<05:44, 491kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<04:32, 620kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<03:16, 856kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:08<02:18, 1.20MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:57, 931kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<03:06, 885kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:23, 1.15MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<01:42, 1.59MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:47, 1.49MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:44, 1.54MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:20, 2.00MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:21, 1.93MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:26, 1.82MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:07, 2.32MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:11, 2.13MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:47, 1.42MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:26, 1.76MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:16<01:02, 2.40MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:18, 1.88MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:22, 1.81MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:03, 2.31MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:08, 2.12MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:15, 1.93MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<00:58, 2.44MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:03, 2.20MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:09, 2.01MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<00:55, 2.53MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:00, 2.25MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:33, 1.46MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:15, 1.79MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<00:55, 2.41MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:11, 1.86MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:14, 1.77MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<00:57, 2.27MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:01, 2.10MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:06, 1.91MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<00:51, 2.47MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:28<00:37, 3.37MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:37, 1.27MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:54, 1.08MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:31, 1.35MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:06, 1.84MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<01:15, 1.59MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:14, 1.61MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:57, 2.08MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:34<00:58, 1.98MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:24, 1.38MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:07, 1.70MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<00:49, 2.30MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<01:01, 1.81MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:04, 1.74MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:49, 2.23MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:51, 2.08MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:16, 1.41MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:01, 1.74MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:44, 2.37MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:55, 1.87MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:57, 1.78MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:44, 2.30MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:40<00:31, 3.17MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<02:01, 817kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<02:01, 813kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:34, 1.05MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<01:06, 1.45MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:09, 1.37MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:05, 1.45MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:49, 1.91MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<00:34, 2.64MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<01:32, 985kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<01:38, 921kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:17, 1.17MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:55, 1.61MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:59, 1.47MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:57, 1.51MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:43, 1.96MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:43, 1.91MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:45, 1.80MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:34, 2.34MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:24, 3.21MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:17, 1.01MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:23, 938kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:04, 1.21MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:45, 1.68MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:39, 1.93MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:47, 1.55MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:53, 1.39MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:53<00:41, 1.78MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:29, 2.45MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:43, 1.60MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:48, 1.44MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:38, 1.82MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:26, 2.49MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:51, 1.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:53, 1.24MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:40, 1.59MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:28, 2.20MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:52, 1.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:51, 1.19MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:39, 1.54MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:27, 2.13MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:31, 1.82MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<38:09, 25.4kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<26:27, 36.2kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<17:43, 51.6kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<12:42, 70.8kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<09:19, 96.4kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<06:34, 136kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<04:30, 193kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<03:10, 261kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<02:24, 343kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<01:42, 479kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:05<01:08, 678kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<01:14, 610kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<01:02, 731kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:45, 987kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:07<00:30, 1.38MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:49, 845kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:43, 953kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:31, 1.28MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:09<00:21, 1.78MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:41, 892kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:38, 973kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:28, 1.28MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:11<00:19, 1.79MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:32, 1.03MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:29, 1.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:21, 1.49MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:13<00:14, 2.06MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:34, 850kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:29, 967kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:21, 1.29MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:15<00:14, 1.79MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:37, 661kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:31, 775kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:23, 1.04MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:17<00:14, 1.45MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:21, 949kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:19, 1.03MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:14, 1.37MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:09, 1.91MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:13, 1.22MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:13, 1.24MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:09, 1.62MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:06, 2.19MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:06, 1.80MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:07, 1.62MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:05, 2.04MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<00:03, 2.79MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:07, 1.14MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:06, 1.22MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.61MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:25<00:02, 2.22MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:05, 696kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:04, 828kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:02, 1.13MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 1.57MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 1.10MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 566/400000 [00:00<01:10, 5654.45it/s]  0%|          | 1503/400000 [00:00<01:02, 6417.56it/s]  1%|          | 2469/400000 [00:00<00:55, 7135.63it/s]  1%|          | 3485/400000 [00:00<00:50, 7834.47it/s]  1%|          | 4516/400000 [00:00<00:46, 8440.67it/s]  1%|▏         | 5464/400000 [00:00<00:45, 8727.30it/s]  2%|▏         | 6448/400000 [00:00<00:43, 9031.88it/s]  2%|▏         | 7449/400000 [00:00<00:42, 9302.91it/s]  2%|▏         | 8455/400000 [00:00<00:41, 9516.07it/s]  2%|▏         | 9415/400000 [00:01<00:40, 9540.91it/s]  3%|▎         | 10366/400000 [00:01<00:41, 9437.16it/s]  3%|▎         | 11308/400000 [00:01<00:41, 9417.65it/s]  3%|▎         | 12249/400000 [00:01<00:41, 9412.61it/s]  3%|▎         | 13254/400000 [00:01<00:40, 9593.85it/s]  4%|▎         | 14326/400000 [00:01<00:38, 9904.88it/s]  4%|▍         | 15320/400000 [00:01<00:39, 9799.96it/s]  4%|▍         | 16303/400000 [00:01<00:39, 9731.07it/s]  4%|▍         | 17278/400000 [00:01<00:39, 9677.73it/s]  5%|▍         | 18247/400000 [00:01<00:39, 9672.56it/s]  5%|▍         | 19216/400000 [00:02<00:40, 9366.41it/s]  5%|▌         | 20183/400000 [00:02<00:40, 9453.85it/s]  5%|▌         | 21131/400000 [00:02<00:40, 9416.53it/s]  6%|▌         | 22094/400000 [00:02<00:39, 9477.07it/s]  6%|▌         | 23043/400000 [00:02<00:40, 9396.05it/s]  6%|▌         | 23984/400000 [00:02<00:40, 9277.60it/s]  6%|▌         | 24953/400000 [00:02<00:39, 9396.36it/s]  6%|▋         | 25901/400000 [00:02<00:39, 9419.17it/s]  7%|▋         | 26866/400000 [00:02<00:39, 9485.91it/s]  7%|▋         | 27816/400000 [00:02<00:39, 9345.14it/s]  7%|▋         | 28790/400000 [00:03<00:39, 9458.28it/s]  7%|▋         | 29755/400000 [00:03<00:38, 9512.70it/s]  8%|▊         | 30708/400000 [00:03<00:39, 9395.25it/s]  8%|▊         | 31700/400000 [00:03<00:38, 9545.60it/s]  8%|▊         | 32687/400000 [00:03<00:38, 9639.83it/s]  8%|▊         | 33745/400000 [00:03<00:36, 9901.32it/s]  9%|▊         | 34793/400000 [00:03<00:36, 10068.10it/s]  9%|▉         | 35811/400000 [00:03<00:36, 10099.73it/s]  9%|▉         | 36823/400000 [00:03<00:36, 9989.31it/s]   9%|▉         | 37824/400000 [00:03<00:36, 9963.22it/s] 10%|▉         | 38822/400000 [00:04<00:36, 9907.43it/s] 10%|▉         | 39814/400000 [00:04<00:36, 9774.27it/s] 10%|█         | 40793/400000 [00:04<00:37, 9670.83it/s] 10%|█         | 41774/400000 [00:04<00:36, 9710.00it/s] 11%|█         | 42746/400000 [00:04<00:36, 9660.49it/s] 11%|█         | 43726/400000 [00:04<00:36, 9701.85it/s] 11%|█         | 44697/400000 [00:04<00:37, 9602.05it/s] 11%|█▏        | 45658/400000 [00:04<00:37, 9490.27it/s] 12%|█▏        | 46608/400000 [00:04<00:37, 9449.09it/s] 12%|█▏        | 47579/400000 [00:04<00:36, 9525.12it/s] 12%|█▏        | 48555/400000 [00:05<00:36, 9593.13it/s] 12%|█▏        | 49515/400000 [00:05<00:36, 9589.73it/s] 13%|█▎        | 50475/400000 [00:05<00:36, 9539.72it/s] 13%|█▎        | 51437/400000 [00:05<00:36, 9563.66it/s] 13%|█▎        | 52394/400000 [00:05<00:36, 9509.60it/s] 13%|█▎        | 53360/400000 [00:05<00:36, 9552.40it/s] 14%|█▎        | 54317/400000 [00:05<00:36, 9557.03it/s] 14%|█▍        | 55273/400000 [00:05<00:36, 9404.87it/s] 14%|█▍        | 56294/400000 [00:05<00:35, 9631.84it/s] 14%|█▍        | 57259/400000 [00:05<00:35, 9620.18it/s] 15%|█▍        | 58223/400000 [00:06<00:35, 9597.75it/s] 15%|█▍        | 59192/400000 [00:06<00:35, 9623.08it/s] 15%|█▌        | 60155/400000 [00:06<00:35, 9580.41it/s] 15%|█▌        | 61126/400000 [00:06<00:35, 9617.44it/s] 16%|█▌        | 62089/400000 [00:06<00:35, 9485.43it/s] 16%|█▌        | 63039/400000 [00:06<00:35, 9441.16it/s] 16%|█▌        | 63995/400000 [00:06<00:35, 9474.39it/s] 16%|█▋        | 65014/400000 [00:06<00:34, 9678.29it/s] 17%|█▋        | 66059/400000 [00:06<00:33, 9895.43it/s] 17%|█▋        | 67051/400000 [00:06<00:34, 9692.40it/s] 17%|█▋        | 68028/400000 [00:07<00:34, 9712.73it/s] 17%|█▋        | 69001/400000 [00:07<00:34, 9686.99it/s] 17%|█▋        | 69971/400000 [00:07<00:34, 9598.58it/s] 18%|█▊        | 70932/400000 [00:07<00:34, 9568.94it/s] 18%|█▊        | 71890/400000 [00:07<00:34, 9470.99it/s] 18%|█▊        | 72914/400000 [00:07<00:33, 9688.85it/s] 18%|█▊        | 73974/400000 [00:07<00:32, 9944.57it/s] 19%|█▊        | 74999/400000 [00:07<00:32, 10033.88it/s] 19%|█▉        | 76005/400000 [00:07<00:32, 10024.68it/s] 19%|█▉        | 77010/400000 [00:08<00:32, 9943.29it/s]  20%|█▉        | 78006/400000 [00:08<00:32, 9798.17it/s] 20%|█▉        | 78988/400000 [00:08<00:32, 9735.30it/s] 20%|█▉        | 79963/400000 [00:08<00:33, 9483.15it/s] 20%|██        | 80926/400000 [00:08<00:33, 9526.60it/s] 20%|██        | 81920/400000 [00:08<00:32, 9645.74it/s] 21%|██        | 82980/400000 [00:08<00:31, 9912.24it/s] 21%|██        | 83975/400000 [00:08<00:32, 9859.87it/s] 21%|██        | 84964/400000 [00:08<00:32, 9794.62it/s] 22%|██▏       | 86026/400000 [00:08<00:31, 10027.42it/s] 22%|██▏       | 87050/400000 [00:09<00:31, 10090.11it/s] 22%|██▏       | 88061/400000 [00:09<00:31, 9937.12it/s]  22%|██▏       | 89057/400000 [00:09<00:31, 9821.83it/s] 23%|██▎       | 90041/400000 [00:09<00:31, 9787.65it/s] 23%|██▎       | 91021/400000 [00:09<00:32, 9562.45it/s] 23%|██▎       | 92017/400000 [00:09<00:31, 9677.65it/s] 23%|██▎       | 93061/400000 [00:09<00:31, 9892.88it/s] 24%|██▎       | 94100/400000 [00:09<00:30, 10034.86it/s] 24%|██▍       | 95106/400000 [00:09<00:31, 9637.57it/s]  24%|██▍       | 96075/400000 [00:09<00:32, 9497.38it/s] 24%|██▍       | 97029/400000 [00:10<00:31, 9504.11it/s] 24%|██▍       | 97983/400000 [00:10<00:31, 9508.52it/s] 25%|██▍       | 98936/400000 [00:10<00:32, 9353.21it/s] 25%|██▍       | 99874/400000 [00:10<00:32, 9280.71it/s] 25%|██▌       | 100804/400000 [00:10<00:32, 9190.38it/s] 25%|██▌       | 101725/400000 [00:10<00:32, 9061.23it/s] 26%|██▌       | 102633/400000 [00:10<00:33, 8994.74it/s] 26%|██▌       | 103551/400000 [00:10<00:32, 9049.48it/s] 26%|██▌       | 104466/400000 [00:10<00:32, 9078.65it/s] 26%|██▋       | 105378/400000 [00:10<00:32, 9090.07it/s] 27%|██▋       | 106316/400000 [00:11<00:32, 9173.36it/s] 27%|██▋       | 107288/400000 [00:11<00:31, 9329.92it/s] 27%|██▋       | 108303/400000 [00:11<00:30, 9560.72it/s] 27%|██▋       | 109262/400000 [00:11<00:31, 9376.10it/s] 28%|██▊       | 110249/400000 [00:11<00:30, 9517.09it/s] 28%|██▊       | 111249/400000 [00:11<00:29, 9655.69it/s] 28%|██▊       | 112217/400000 [00:11<00:30, 9590.41it/s] 28%|██▊       | 113192/400000 [00:11<00:29, 9634.83it/s] 29%|██▊       | 114157/400000 [00:11<00:29, 9602.38it/s] 29%|██▉       | 115119/400000 [00:12<00:30, 9416.64it/s] 29%|██▉       | 116150/400000 [00:12<00:29, 9667.03it/s] 29%|██▉       | 117172/400000 [00:12<00:28, 9824.08it/s] 30%|██▉       | 118157/400000 [00:12<00:28, 9783.19it/s] 30%|██▉       | 119138/400000 [00:12<00:29, 9615.41it/s] 30%|███       | 120108/400000 [00:12<00:29, 9640.04it/s] 30%|███       | 121074/400000 [00:12<00:29, 9583.17it/s] 31%|███       | 122045/400000 [00:12<00:28, 9620.25it/s] 31%|███       | 123008/400000 [00:12<00:28, 9580.74it/s] 31%|███       | 123967/400000 [00:12<00:29, 9466.12it/s] 31%|███       | 124915/400000 [00:13<00:29, 9459.89it/s] 31%|███▏      | 125887/400000 [00:13<00:28, 9535.99it/s] 32%|███▏      | 126842/400000 [00:13<00:28, 9486.77it/s] 32%|███▏      | 127798/400000 [00:13<00:28, 9507.54it/s] 32%|███▏      | 128750/400000 [00:13<00:28, 9405.75it/s] 32%|███▏      | 129692/400000 [00:13<00:29, 9314.78it/s] 33%|███▎      | 130624/400000 [00:13<00:29, 9288.81it/s] 33%|███▎      | 131554/400000 [00:13<00:29, 9143.56it/s] 33%|███▎      | 132470/400000 [00:13<00:29, 9142.80it/s] 33%|███▎      | 133385/400000 [00:13<00:29, 9141.72it/s] 34%|███▎      | 134339/400000 [00:14<00:28, 9257.19it/s] 34%|███▍      | 135301/400000 [00:14<00:28, 9362.08it/s] 34%|███▍      | 136254/400000 [00:14<00:28, 9409.31it/s] 34%|███▍      | 137197/400000 [00:14<00:27, 9413.28it/s] 35%|███▍      | 138139/400000 [00:14<00:28, 9135.64it/s] 35%|███▍      | 139073/400000 [00:14<00:28, 9195.72it/s] 35%|███▍      | 139998/400000 [00:14<00:28, 9211.43it/s] 35%|███▌      | 140921/400000 [00:14<00:28, 9165.11it/s] 35%|███▌      | 141848/400000 [00:14<00:28, 9194.10it/s] 36%|███▌      | 142768/400000 [00:14<00:28, 9143.14it/s] 36%|███▌      | 143684/400000 [00:15<00:28, 9147.97it/s] 36%|███▌      | 144600/400000 [00:15<00:27, 9145.72it/s] 36%|███▋      | 145523/400000 [00:15<00:27, 9168.88it/s] 37%|███▋      | 146482/400000 [00:15<00:27, 9290.57it/s] 37%|███▋      | 147412/400000 [00:15<00:27, 9248.16it/s] 37%|███▋      | 148338/400000 [00:15<00:27, 9207.04it/s] 37%|███▋      | 149288/400000 [00:15<00:26, 9290.66it/s] 38%|███▊      | 150236/400000 [00:15<00:26, 9345.34it/s] 38%|███▊      | 151210/400000 [00:15<00:26, 9459.90it/s] 38%|███▊      | 152157/400000 [00:15<00:26, 9440.39it/s] 38%|███▊      | 153102/400000 [00:16<00:26, 9426.55it/s] 39%|███▊      | 154093/400000 [00:16<00:25, 9564.60it/s] 39%|███▉      | 155070/400000 [00:16<00:25, 9624.60it/s] 39%|███▉      | 156065/400000 [00:16<00:25, 9719.45it/s] 39%|███▉      | 157038/400000 [00:16<00:25, 9462.70it/s] 39%|███▉      | 157987/400000 [00:16<00:26, 9276.91it/s] 40%|███▉      | 158955/400000 [00:16<00:25, 9392.90it/s] 40%|███▉      | 159925/400000 [00:16<00:25, 9481.43it/s] 40%|████      | 160903/400000 [00:16<00:24, 9569.06it/s] 40%|████      | 161893/400000 [00:16<00:24, 9664.54it/s] 41%|████      | 162899/400000 [00:17<00:24, 9777.74it/s] 41%|████      | 163950/400000 [00:17<00:23, 9985.22it/s] 41%|████▏     | 165009/400000 [00:17<00:23, 10157.90it/s] 42%|████▏     | 166027/400000 [00:17<00:23, 9822.02it/s]  42%|████▏     | 167014/400000 [00:17<00:24, 9670.18it/s] 42%|████▏     | 167990/400000 [00:17<00:23, 9696.00it/s] 42%|████▏     | 169006/400000 [00:17<00:23, 9828.53it/s] 42%|████▏     | 169991/400000 [00:17<00:23, 9789.33it/s] 43%|████▎     | 170972/400000 [00:17<00:23, 9728.15it/s] 43%|████▎     | 171946/400000 [00:17<00:23, 9689.05it/s] 43%|████▎     | 172916/400000 [00:18<00:23, 9649.73it/s] 43%|████▎     | 173894/400000 [00:18<00:23, 9687.27it/s] 44%|████▎     | 174864/400000 [00:18<00:23, 9652.03it/s] 44%|████▍     | 175830/400000 [00:18<00:23, 9575.07it/s] 44%|████▍     | 176832/400000 [00:18<00:23, 9702.92it/s] 44%|████▍     | 177849/400000 [00:18<00:22, 9836.45it/s] 45%|████▍     | 178834/400000 [00:18<00:22, 9684.07it/s] 45%|████▍     | 179831/400000 [00:18<00:22, 9767.18it/s] 45%|████▌     | 180809/400000 [00:18<00:22, 9759.62it/s] 45%|████▌     | 181786/400000 [00:19<00:22, 9741.10it/s] 46%|████▌     | 182857/400000 [00:19<00:21, 10010.33it/s] 46%|████▌     | 183887/400000 [00:19<00:21, 10094.27it/s] 46%|████▌     | 184960/400000 [00:19<00:20, 10275.44it/s] 46%|████▋     | 185990/400000 [00:19<00:20, 10267.02it/s] 47%|████▋     | 187019/400000 [00:19<00:21, 9995.35it/s]  47%|████▋     | 188045/400000 [00:19<00:21, 10071.63it/s] 47%|████▋     | 189121/400000 [00:19<00:20, 10265.77it/s] 48%|████▊     | 190150/400000 [00:19<00:20, 10214.31it/s] 48%|████▊     | 191174/400000 [00:19<00:20, 10110.86it/s] 48%|████▊     | 192187/400000 [00:20<00:21, 9836.32it/s]  48%|████▊     | 193174/400000 [00:20<00:21, 9760.40it/s] 49%|████▊     | 194153/400000 [00:20<00:21, 9766.31it/s] 49%|████▉     | 195152/400000 [00:20<00:20, 9831.89it/s] 49%|████▉     | 196137/400000 [00:20<00:20, 9811.28it/s] 49%|████▉     | 197119/400000 [00:20<00:20, 9731.19it/s] 50%|████▉     | 198113/400000 [00:20<00:20, 9791.53it/s] 50%|████▉     | 199140/400000 [00:20<00:20, 9929.20it/s] 50%|█████     | 200134/400000 [00:20<00:20, 9559.35it/s] 50%|█████     | 201094/400000 [00:20<00:20, 9535.46it/s] 51%|█████     | 202053/400000 [00:21<00:20, 9551.38it/s] 51%|█████     | 203028/400000 [00:21<00:20, 9607.84it/s] 51%|█████     | 203995/400000 [00:21<00:20, 9625.42it/s] 51%|█████     | 204969/400000 [00:21<00:20, 9658.79it/s] 51%|█████▏    | 205936/400000 [00:21<00:20, 9570.99it/s] 52%|█████▏    | 206894/400000 [00:21<00:20, 9527.02it/s] 52%|█████▏    | 207877/400000 [00:21<00:19, 9615.10it/s] 52%|█████▏    | 208855/400000 [00:21<00:19, 9661.78it/s] 52%|█████▏    | 209822/400000 [00:21<00:19, 9651.39it/s] 53%|█████▎    | 210788/400000 [00:21<00:19, 9615.66it/s] 53%|█████▎    | 211750/400000 [00:22<00:19, 9478.83it/s] 53%|█████▎    | 212732/400000 [00:22<00:19, 9578.36it/s] 53%|█████▎    | 213691/400000 [00:22<00:19, 9547.82it/s] 54%|█████▎    | 214672/400000 [00:22<00:19, 9621.80it/s] 54%|█████▍    | 215639/400000 [00:22<00:19, 9633.52it/s] 54%|█████▍    | 216603/400000 [00:22<00:19, 9631.09it/s] 54%|█████▍    | 217593/400000 [00:22<00:18, 9709.21it/s] 55%|█████▍    | 218576/400000 [00:22<00:18, 9744.48it/s] 55%|█████▍    | 219554/400000 [00:22<00:18, 9754.47it/s] 55%|█████▌    | 220530/400000 [00:22<00:18, 9748.80it/s] 55%|█████▌    | 221506/400000 [00:23<00:18, 9749.45it/s] 56%|█████▌    | 222482/400000 [00:23<00:18, 9697.32it/s] 56%|█████▌    | 223454/400000 [00:23<00:18, 9702.16it/s] 56%|█████▌    | 224429/400000 [00:23<00:18, 9715.77it/s] 56%|█████▋    | 225401/400000 [00:23<00:18, 9590.72it/s] 57%|█████▋    | 226361/400000 [00:23<00:18, 9378.09it/s] 57%|█████▋    | 227301/400000 [00:23<00:18, 9284.84it/s] 57%|█████▋    | 228231/400000 [00:23<00:18, 9222.99it/s] 57%|█████▋    | 229158/400000 [00:23<00:18, 9236.85it/s] 58%|█████▊    | 230117/400000 [00:23<00:18, 9338.67it/s] 58%|█████▊    | 231069/400000 [00:24<00:17, 9391.71it/s] 58%|█████▊    | 232048/400000 [00:24<00:17, 9506.39it/s] 58%|█████▊    | 233021/400000 [00:24<00:17, 9570.70it/s] 58%|█████▊    | 233989/400000 [00:24<00:17, 9603.02it/s] 59%|█████▊    | 234950/400000 [00:24<00:17, 9403.94it/s] 59%|█████▉    | 235933/400000 [00:24<00:17, 9527.59it/s] 59%|█████▉    | 236996/400000 [00:24<00:16, 9831.91it/s] 60%|█████▉    | 238043/400000 [00:24<00:16, 10013.34it/s] 60%|█████▉    | 239049/400000 [00:24<00:16, 10026.82it/s] 60%|██████    | 240054/400000 [00:25<00:16, 9852.68it/s]  60%|██████    | 241063/400000 [00:25<00:16, 9922.44it/s] 61%|██████    | 242086/400000 [00:25<00:15, 10011.68it/s] 61%|██████    | 243089/400000 [00:25<00:16, 9788.12it/s]  61%|██████    | 244070/400000 [00:25<00:16, 9684.60it/s] 61%|██████▏   | 245041/400000 [00:25<00:16, 9598.47it/s] 62%|██████▏   | 246034/400000 [00:25<00:15, 9693.20it/s] 62%|██████▏   | 247072/400000 [00:25<00:15, 9888.53it/s] 62%|██████▏   | 248118/400000 [00:25<00:15, 10051.87it/s] 62%|██████▏   | 249193/400000 [00:25<00:14, 10250.32it/s] 63%|██████▎   | 250236/400000 [00:26<00:14, 10301.65it/s] 63%|██████▎   | 251268/400000 [00:26<00:14, 10275.28it/s] 63%|██████▎   | 252297/400000 [00:26<00:14, 10257.87it/s] 63%|██████▎   | 253324/400000 [00:26<00:14, 10039.46it/s] 64%|██████▎   | 254330/400000 [00:26<00:14, 10025.10it/s] 64%|██████▍   | 255376/400000 [00:26<00:14, 10151.36it/s] 64%|██████▍   | 256393/400000 [00:26<00:14, 10047.42it/s] 64%|██████▍   | 257417/400000 [00:26<00:14, 10102.58it/s] 65%|██████▍   | 258453/400000 [00:26<00:13, 10177.04it/s] 65%|██████▍   | 259496/400000 [00:26<00:13, 10249.75it/s] 65%|██████▌   | 260522/400000 [00:27<00:14, 9694.96it/s]  65%|██████▌   | 261565/400000 [00:27<00:13, 9902.07it/s] 66%|██████▌   | 262606/400000 [00:27<00:13, 10049.03it/s] 66%|██████▌   | 263616/400000 [00:27<00:13, 10049.06it/s] 66%|██████▌   | 264626/400000 [00:27<00:13, 10063.78it/s] 66%|██████▋   | 265652/400000 [00:27<00:13, 10120.13it/s] 67%|██████▋   | 266666/400000 [00:27<00:13, 9942.89it/s]  67%|██████▋   | 267663/400000 [00:27<00:13, 9940.80it/s] 67%|██████▋   | 268659/400000 [00:27<00:13, 9930.05it/s] 67%|██████▋   | 269653/400000 [00:27<00:13, 9917.65it/s] 68%|██████▊   | 270650/400000 [00:28<00:13, 9931.96it/s] 68%|██████▊   | 271644/400000 [00:28<00:13, 9789.26it/s] 68%|██████▊   | 272642/400000 [00:28<00:12, 9845.26it/s] 68%|██████▊   | 273628/400000 [00:28<00:12, 9838.82it/s] 69%|██████▊   | 274613/400000 [00:28<00:12, 9695.09it/s] 69%|██████▉   | 275584/400000 [00:28<00:13, 9514.20it/s] 69%|██████▉   | 276537/400000 [00:28<00:13, 9309.00it/s] 69%|██████▉   | 277470/400000 [00:28<00:13, 9263.53it/s] 70%|██████▉   | 278404/400000 [00:28<00:13, 9284.17it/s] 70%|██████▉   | 279361/400000 [00:28<00:12, 9365.12it/s] 70%|███████   | 280392/400000 [00:29<00:12, 9629.65it/s] 70%|███████   | 281382/400000 [00:29<00:12, 9707.61it/s] 71%|███████   | 282454/400000 [00:29<00:11, 9988.63it/s] 71%|███████   | 283504/400000 [00:29<00:11, 10134.79it/s] 71%|███████   | 284521/400000 [00:29<00:11, 10013.29it/s] 71%|███████▏  | 285525/400000 [00:29<00:11, 10018.42it/s] 72%|███████▏  | 286529/400000 [00:29<00:11, 9959.62it/s]  72%|███████▏  | 287527/400000 [00:29<00:11, 9920.40it/s] 72%|███████▏  | 288520/400000 [00:29<00:11, 9903.15it/s] 72%|███████▏  | 289511/400000 [00:29<00:11, 9809.09it/s] 73%|███████▎  | 290493/400000 [00:30<00:11, 9717.36it/s] 73%|███████▎  | 291466/400000 [00:30<00:11, 9653.37it/s] 73%|███████▎  | 292432/400000 [00:30<00:11, 9608.89it/s] 73%|███████▎  | 293402/400000 [00:30<00:11, 9633.45it/s] 74%|███████▎  | 294366/400000 [00:30<00:11, 9527.96it/s] 74%|███████▍  | 295328/400000 [00:30<00:10, 9553.72it/s] 74%|███████▍  | 296284/400000 [00:30<00:10, 9447.70it/s] 74%|███████▍  | 297247/400000 [00:30<00:10, 9500.03it/s] 75%|███████▍  | 298241/400000 [00:30<00:10, 9625.77it/s] 75%|███████▍  | 299205/400000 [00:31<00:10, 9606.03it/s] 75%|███████▌  | 300263/400000 [00:31<00:10, 9878.08it/s] 75%|███████▌  | 301254/400000 [00:31<00:10, 9867.75it/s] 76%|███████▌  | 302248/400000 [00:31<00:09, 9887.37it/s] 76%|███████▌  | 303291/400000 [00:31<00:09, 10042.76it/s] 76%|███████▌  | 304351/400000 [00:31<00:09, 10200.72it/s] 76%|███████▋  | 305373/400000 [00:31<00:09, 10106.61it/s] 77%|███████▋  | 306385/400000 [00:31<00:09, 9952.11it/s]  77%|███████▋  | 307382/400000 [00:31<00:09, 9878.97it/s] 77%|███████▋  | 308372/400000 [00:31<00:09, 9753.14it/s] 77%|███████▋  | 309349/400000 [00:32<00:09, 9752.02it/s] 78%|███████▊  | 310326/400000 [00:32<00:09, 9745.45it/s] 78%|███████▊  | 311302/400000 [00:32<00:09, 9634.71it/s] 78%|███████▊  | 312287/400000 [00:32<00:09, 9697.46it/s] 78%|███████▊  | 313327/400000 [00:32<00:08, 9895.90it/s] 79%|███████▊  | 314363/400000 [00:32<00:08, 10030.19it/s] 79%|███████▉  | 315368/400000 [00:32<00:08, 9850.36it/s]  79%|███████▉  | 316355/400000 [00:32<00:08, 9560.64it/s] 79%|███████▉  | 317316/400000 [00:32<00:08, 9573.84it/s] 80%|███████▉  | 318276/400000 [00:32<00:08, 9523.32it/s] 80%|███████▉  | 319241/400000 [00:33<00:08, 9560.03it/s] 80%|████████  | 320199/400000 [00:33<00:08, 9483.48it/s] 80%|████████  | 321155/400000 [00:33<00:08, 9504.05it/s] 81%|████████  | 322127/400000 [00:33<00:08, 9565.96it/s] 81%|████████  | 323085/400000 [00:33<00:08, 9467.03it/s] 81%|████████  | 324033/400000 [00:33<00:08, 9442.45it/s] 81%|████████▏ | 325081/400000 [00:33<00:07, 9730.78it/s] 82%|████████▏ | 326057/400000 [00:33<00:07, 9631.45it/s] 82%|████████▏ | 327028/400000 [00:33<00:07, 9653.25it/s] 82%|████████▏ | 328002/400000 [00:33<00:07, 9677.57it/s] 82%|████████▏ | 328971/400000 [00:34<00:07, 9408.51it/s] 82%|████████▏ | 329947/400000 [00:34<00:07, 9509.23it/s] 83%|████████▎ | 330931/400000 [00:34<00:07, 9605.53it/s] 83%|████████▎ | 331906/400000 [00:34<00:07, 9646.61it/s] 83%|████████▎ | 332872/400000 [00:34<00:07, 9585.72it/s] 83%|████████▎ | 333832/400000 [00:34<00:06, 9586.15it/s] 84%|████████▎ | 334799/400000 [00:34<00:06, 9608.17it/s] 84%|████████▍ | 335761/400000 [00:34<00:06, 9593.08it/s] 84%|████████▍ | 336721/400000 [00:34<00:06, 9531.38it/s] 84%|████████▍ | 337675/400000 [00:34<00:06, 9514.81it/s] 85%|████████▍ | 338648/400000 [00:35<00:06, 9576.03it/s] 85%|████████▍ | 339627/400000 [00:35<00:06, 9636.76it/s] 85%|████████▌ | 340594/400000 [00:35<00:06, 9644.07it/s] 85%|████████▌ | 341572/400000 [00:35<00:06, 9684.02it/s] 86%|████████▌ | 342541/400000 [00:35<00:06, 9571.53it/s] 86%|████████▌ | 343525/400000 [00:35<00:05, 9648.27it/s] 86%|████████▌ | 344523/400000 [00:35<00:05, 9744.57it/s] 86%|████████▋ | 345536/400000 [00:35<00:05, 9856.27it/s] 87%|████████▋ | 346556/400000 [00:35<00:05, 9956.78it/s] 87%|████████▋ | 347553/400000 [00:35<00:05, 9865.45it/s] 87%|████████▋ | 348541/400000 [00:36<00:05, 9815.03it/s] 87%|████████▋ | 349524/400000 [00:36<00:05, 9695.82it/s] 88%|████████▊ | 350495/400000 [00:36<00:05, 9610.67it/s] 88%|████████▊ | 351457/400000 [00:36<00:05, 9591.32it/s] 88%|████████▊ | 352418/400000 [00:36<00:04, 9596.26it/s] 88%|████████▊ | 353399/400000 [00:36<00:04, 9657.32it/s] 89%|████████▊ | 354366/400000 [00:36<00:04, 9550.53it/s] 89%|████████▉ | 355322/400000 [00:36<00:04, 9444.93it/s] 89%|████████▉ | 356294/400000 [00:36<00:04, 9522.89it/s] 89%|████████▉ | 357247/400000 [00:37<00:04, 9468.97it/s] 90%|████████▉ | 358195/400000 [00:37<00:04, 9447.13it/s] 90%|████████▉ | 359155/400000 [00:37<00:04, 9491.20it/s] 90%|█████████ | 360118/400000 [00:37<00:04, 9531.07it/s] 90%|█████████ | 361094/400000 [00:37<00:04, 9596.72it/s] 91%|█████████ | 362061/400000 [00:37<00:03, 9617.33it/s] 91%|█████████ | 363023/400000 [00:37<00:03, 9610.17it/s] 91%|█████████ | 363985/400000 [00:37<00:03, 9592.61it/s] 91%|█████████ | 364945/400000 [00:37<00:03, 9587.51it/s] 91%|█████████▏| 365933/400000 [00:37<00:03, 9672.90it/s] 92%|█████████▏| 366901/400000 [00:38<00:03, 9572.82it/s] 92%|█████████▏| 367894/400000 [00:38<00:03, 9674.80it/s] 92%|█████████▏| 368863/400000 [00:38<00:03, 9647.82it/s] 92%|█████████▏| 369852/400000 [00:38<00:03, 9716.97it/s] 93%|█████████▎| 370886/400000 [00:38<00:02, 9894.94it/s] 93%|█████████▎| 371884/400000 [00:38<00:02, 9917.29it/s] 93%|█████████▎| 372937/400000 [00:38<00:02, 10091.36it/s] 93%|█████████▎| 373948/400000 [00:38<00:02, 9889.68it/s]  94%|█████████▎| 374966/400000 [00:38<00:02, 9974.59it/s] 94%|█████████▍| 376023/400000 [00:38<00:02, 10144.82it/s] 94%|█████████▍| 377062/400000 [00:39<00:02, 10216.76it/s] 95%|█████████▍| 378085/400000 [00:39<00:02, 10120.91it/s] 95%|█████████▍| 379099/400000 [00:39<00:02, 9969.48it/s]  95%|█████████▌| 380098/400000 [00:39<00:02, 9769.68it/s] 95%|█████████▌| 381077/400000 [00:39<00:02, 9398.90it/s] 96%|█████████▌| 382028/400000 [00:39<00:01, 9429.12it/s] 96%|█████████▌| 382974/400000 [00:39<00:01, 9373.11it/s] 96%|█████████▌| 383914/400000 [00:39<00:01, 9238.90it/s] 96%|█████████▌| 384840/400000 [00:39<00:01, 8912.99it/s] 96%|█████████▋| 385761/400000 [00:39<00:01, 8997.94it/s] 97%|█████████▋| 386739/400000 [00:40<00:01, 9216.75it/s] 97%|█████████▋| 387699/400000 [00:40<00:01, 9327.22it/s] 97%|█████████▋| 388635/400000 [00:40<00:01, 9326.27it/s] 97%|█████████▋| 389588/400000 [00:40<00:01, 9386.08it/s] 98%|█████████▊| 390529/400000 [00:40<00:01, 9067.99it/s] 98%|█████████▊| 391447/400000 [00:40<00:00, 9099.05it/s] 98%|█████████▊| 392379/400000 [00:40<00:00, 9162.29it/s] 98%|█████████▊| 393336/400000 [00:40<00:00, 9280.73it/s] 99%|█████████▊| 394283/400000 [00:40<00:00, 9334.73it/s] 99%|█████████▉| 395252/400000 [00:40<00:00, 9437.78it/s] 99%|█████████▉| 396205/400000 [00:41<00:00, 9462.02it/s] 99%|█████████▉| 397178/400000 [00:41<00:00, 9538.84it/s]100%|█████████▉| 398133/400000 [00:41<00:00, 9490.38it/s]100%|█████████▉| 399083/400000 [00:41<00:00, 9409.44it/s]100%|█████████▉| 399999/400000 [00:41<00:00, 9641.30it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f4cd804f9e8>, <torchtext.data.dataset.TabularDataset object at 0x7f4cd804fb38>, <torchtext.vocab.Vocab object at 0x7f4cd804fa58>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 27.43 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 48.48 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 27.06 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 27.06 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.67 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.67 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  8.68 file/s]2020-07-01 16:02:06.640675: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-01 16:02:06.644746: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-07-01 16:02:06.644954: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5600cfc6c580 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-01 16:02:06.644971: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['mnist_dataset_small.npy', 'cifar10', 'test', 'train', 'mnist2', 'fashion-mnist_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:03, 154777.43it/s] 69%|██████▉   | 6815744/9912422 [00:00<00:14, 220891.06it/s]9920512it [00:00, 42844507.56it/s]                           
0it [00:00, ?it/s]32768it [00:00, 527066.65it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 457208.32it/s]1654784it [00:00, 11298460.94it/s]                         
0it [00:00, ?it/s]8192it [00:00, 206635.34it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
