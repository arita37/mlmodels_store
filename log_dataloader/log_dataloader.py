
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/0c184f4696ef141004a44b14ce9a441a018b9b67', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': '0c184f4696ef141004a44b14ce9a441a018b9b67', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/0c184f4696ef141004a44b14ce9a441a018b9b67

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/0c184f4696ef141004a44b14ce9a441a018b9b67

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/0c184f4696ef141004a44b14ce9a441a018b9b67

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7ff504270ae8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7ff504270ae8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7ff56eee2488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7ff56eee2488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7ff58e10eea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7ff58e10eea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff51c2899d8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff51c2899d8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff51c2899d8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:25, 115385.16it/s] 75%|███████▍  | 7430144/9912422 [00:00<00:15, 164725.12it/s]9920512it [00:00, 37455323.42it/s]                           
0it [00:00, ?it/s]32768it [00:00, 583790.82it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 156533.18it/s]1654784it [00:00, 11165845.37it/s]                         
0it [00:00, ?it/s]8192it [00:00, 130115.53it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff4facae7b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff4facbca20>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7ff51c289620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7ff51c289620> 

  function with postional parmater data_info <function tf_dataset_download at 0x7ff51c289620> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:57,  2.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:57,  2.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:56,  2.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:56,  2.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:56,  2.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:55,  2.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:55,  2.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:55,  2.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:38,  3.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:38,  3.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:38,  3.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:38,  3.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:38,  3.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:37,  3.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:37,  3.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:37,  3.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:37,  3.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:36,  3.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:36,  3.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:25,  5.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:25,  5.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:25,  5.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:25,  5.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:25,  5.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:25,  5.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:25,  5.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:24,  5.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:24,  5.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:24,  5.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:24,  5.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:17,  7.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:17,  7.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:17,  7.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:17,  7.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:16,  7.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:16,  7.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:16,  7.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:16,  7.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:16,  7.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:16,  7.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:16,  7.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:00<00:11, 10.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:11, 10.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:11, 10.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:11, 10.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:11, 10.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:11, 10.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:11, 10.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:11, 10.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:10, 10.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:10, 10.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:10, 10.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:00<00:07, 14.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:07, 14.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:07, 14.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:07, 14.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:07, 14.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:07, 14.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:00<00:07, 14.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:00<00:07, 14.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:00<00:07, 14.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:00<00:07, 14.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:00<00:07, 14.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:00<00:05, 19.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:00<00:05, 19.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 19.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 19.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 19.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 19.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 19.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 19.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 19.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 19.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 19.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 25.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 25.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 25.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 25.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 25.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 25.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 25.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 25.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 25.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 25.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 25.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 32.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 32.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 32.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 32.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 32.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 32.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 32.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 32.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 32.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 32.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 32.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 40.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 40.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 40.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 40.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 40.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 40.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 40.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 40.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 40.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 40.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 48.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 48.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 48.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 48.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 48.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 48.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 48.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 48.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 48.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 48.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 48.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 56.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 56.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 56.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 56.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 56.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 56.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 56.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 56.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 56.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 56.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 56.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 64.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 64.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 64.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 64.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 64.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 64.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 64.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 64.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 64.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 64.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 64.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 70.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 70.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 70.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 70.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 70.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 70.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 70.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 70.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 70.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 70.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 70.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 76.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 76.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 76.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 76.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 76.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 76.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 76.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 76.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 76.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 76.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:01<00:00, 76.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:01<00:00, 80.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:01<00:00, 80.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:01<00:00, 80.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:01<00:00, 80.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:01<00:00, 80.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:01<00:00, 80.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 80.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 80.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 80.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 80.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 80.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 84.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 84.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 84.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 84.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 84.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 84.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 84.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.11s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.11s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 84.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.11s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 84.32 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.95s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.11s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 84.32 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.95s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.95s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 32.70 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.96s/ url]
0 examples [00:00, ? examples/s]2020-07-24 17:41:20.818574: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-24 17:41:20.822930: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-07-24 17:41:20.823070: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5612ce695370 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-24 17:41:20.823085: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
78 examples [00:00, 773.66 examples/s]187 examples [00:00, 847.28 examples/s]287 examples [00:00, 887.22 examples/s]397 examples [00:00, 939.90 examples/s]503 examples [00:00, 971.92 examples/s]602 examples [00:00, 976.78 examples/s]712 examples [00:00, 1010.54 examples/s]818 examples [00:00, 1022.39 examples/s]929 examples [00:00, 1044.77 examples/s]1038 examples [00:01, 1057.75 examples/s]1147 examples [00:01, 1066.83 examples/s]1256 examples [00:01, 1072.90 examples/s]1364 examples [00:01, 1073.93 examples/s]1471 examples [00:01, 1067.56 examples/s]1580 examples [00:01, 1073.17 examples/s]1691 examples [00:01, 1081.20 examples/s]1800 examples [00:01, 1083.62 examples/s]1909 examples [00:01, 1080.07 examples/s]2018 examples [00:01, 1081.70 examples/s]2128 examples [00:02, 1084.55 examples/s]2238 examples [00:02, 1088.29 examples/s]2349 examples [00:02, 1091.91 examples/s]2459 examples [00:02, 1082.95 examples/s]2568 examples [00:02, 1069.31 examples/s]2679 examples [00:02, 1076.93 examples/s]2787 examples [00:02, 1065.43 examples/s]2896 examples [00:02, 1070.12 examples/s]3005 examples [00:02, 1074.22 examples/s]3117 examples [00:02, 1084.85 examples/s]3226 examples [00:03, 1080.72 examples/s]3338 examples [00:03, 1089.35 examples/s]3450 examples [00:03, 1096.24 examples/s]3560 examples [00:03, 1076.98 examples/s]3673 examples [00:03, 1090.80 examples/s]3785 examples [00:03, 1098.93 examples/s]3899 examples [00:03, 1108.46 examples/s]4012 examples [00:03, 1114.82 examples/s]4124 examples [00:03, 1112.58 examples/s]4238 examples [00:03, 1118.40 examples/s]4350 examples [00:04, 1117.56 examples/s]4463 examples [00:04, 1120.75 examples/s]4576 examples [00:04, 1117.81 examples/s]4688 examples [00:04, 1116.32 examples/s]4800 examples [00:04, 1110.08 examples/s]4913 examples [00:04, 1115.49 examples/s]5025 examples [00:04, 1115.67 examples/s]5137 examples [00:04, 1086.18 examples/s]5246 examples [00:04, 1084.28 examples/s]5355 examples [00:04, 1085.04 examples/s]5464 examples [00:05, 1068.65 examples/s]5577 examples [00:05, 1084.83 examples/s]5690 examples [00:05, 1096.17 examples/s]5801 examples [00:05, 1099.98 examples/s]5914 examples [00:05, 1106.27 examples/s]6026 examples [00:05, 1107.95 examples/s]6138 examples [00:05, 1110.09 examples/s]6251 examples [00:05, 1113.65 examples/s]6363 examples [00:05, 1105.25 examples/s]6477 examples [00:05, 1112.69 examples/s]6589 examples [00:06, 1104.88 examples/s]6700 examples [00:06, 1097.78 examples/s]6812 examples [00:06, 1101.44 examples/s]6923 examples [00:06, 1039.33 examples/s]7028 examples [00:06, 1038.59 examples/s]7133 examples [00:06, 1041.65 examples/s]7246 examples [00:06, 1064.68 examples/s]7354 examples [00:06, 1068.84 examples/s]7462 examples [00:06, 1051.80 examples/s]7572 examples [00:07, 1065.56 examples/s]7684 examples [00:07, 1079.08 examples/s]7795 examples [00:07, 1087.65 examples/s]7907 examples [00:07, 1095.58 examples/s]8017 examples [00:07, 1073.04 examples/s]8127 examples [00:07, 1079.82 examples/s]8236 examples [00:07, 1071.81 examples/s]8344 examples [00:07, 1037.68 examples/s]8449 examples [00:07, 1025.04 examples/s]8558 examples [00:07, 1043.45 examples/s]8671 examples [00:08, 1066.46 examples/s]8784 examples [00:08, 1083.06 examples/s]8896 examples [00:08, 1093.69 examples/s]9009 examples [00:08, 1102.47 examples/s]9120 examples [00:08, 1098.73 examples/s]9233 examples [00:08, 1105.05 examples/s]9346 examples [00:08, 1110.39 examples/s]9459 examples [00:08, 1114.48 examples/s]9571 examples [00:08, 1115.15 examples/s]9683 examples [00:08, 1103.40 examples/s]9794 examples [00:09, 1077.03 examples/s]9906 examples [00:09, 1087.00 examples/s]10015 examples [00:09, 1019.33 examples/s]10125 examples [00:09, 1040.51 examples/s]10235 examples [00:09, 1056.92 examples/s]10347 examples [00:09, 1074.93 examples/s]10455 examples [00:09, 1069.94 examples/s]10565 examples [00:09, 1078.20 examples/s]10678 examples [00:09, 1092.13 examples/s]10789 examples [00:09, 1097.20 examples/s]10902 examples [00:10, 1104.44 examples/s]11015 examples [00:10, 1110.55 examples/s]11128 examples [00:10, 1114.96 examples/s]11242 examples [00:10, 1120.00 examples/s]11355 examples [00:10, 1116.29 examples/s]11467 examples [00:10, 1117.23 examples/s]11579 examples [00:10, 1105.55 examples/s]11690 examples [00:10, 1104.15 examples/s]11801 examples [00:10, 1102.68 examples/s]11912 examples [00:10, 1088.67 examples/s]12022 examples [00:11, 1091.56 examples/s]12132 examples [00:11, 1068.89 examples/s]12245 examples [00:11, 1084.04 examples/s]12356 examples [00:11, 1089.30 examples/s]12466 examples [00:11, 1090.15 examples/s]12579 examples [00:11, 1099.86 examples/s]12692 examples [00:11, 1108.03 examples/s]12805 examples [00:11, 1112.19 examples/s]12917 examples [00:11, 1112.34 examples/s]13029 examples [00:12, 1112.83 examples/s]13141 examples [00:12, 1113.28 examples/s]13254 examples [00:12, 1116.78 examples/s]13366 examples [00:12, 1104.84 examples/s]13477 examples [00:12, 1104.73 examples/s]13588 examples [00:12, 1100.68 examples/s]13701 examples [00:12, 1106.64 examples/s]13813 examples [00:12, 1108.30 examples/s]13924 examples [00:12, 1069.45 examples/s]14036 examples [00:12, 1083.09 examples/s]14146 examples [00:13, 1086.38 examples/s]14255 examples [00:13, 1083.99 examples/s]14364 examples [00:13, 1080.27 examples/s]14473 examples [00:13, 1082.62 examples/s]14583 examples [00:13, 1085.40 examples/s]14692 examples [00:13, 1084.93 examples/s]14801 examples [00:13, 1079.16 examples/s]14909 examples [00:13, 1078.59 examples/s]15021 examples [00:13, 1089.14 examples/s]15132 examples [00:13, 1094.67 examples/s]15242 examples [00:14, 1088.92 examples/s]15354 examples [00:14, 1096.28 examples/s]15465 examples [00:14, 1100.28 examples/s]15577 examples [00:14, 1103.30 examples/s]15688 examples [00:14, 1104.75 examples/s]15799 examples [00:14, 1092.10 examples/s]15910 examples [00:14, 1095.60 examples/s]16020 examples [00:14, 1087.14 examples/s]16132 examples [00:14, 1096.18 examples/s]16244 examples [00:14, 1103.15 examples/s]16355 examples [00:15, 1096.97 examples/s]16468 examples [00:15, 1104.31 examples/s]16580 examples [00:15, 1108.25 examples/s]16691 examples [00:15, 1068.10 examples/s]16802 examples [00:15, 1077.71 examples/s]16911 examples [00:15, 1071.57 examples/s]17023 examples [00:15, 1084.64 examples/s]17134 examples [00:15, 1091.93 examples/s]17245 examples [00:15, 1096.74 examples/s]17356 examples [00:15, 1099.69 examples/s]17467 examples [00:16, 1096.37 examples/s]17580 examples [00:16, 1104.64 examples/s]17693 examples [00:16, 1109.61 examples/s]17805 examples [00:16, 1109.42 examples/s]17918 examples [00:16, 1114.50 examples/s]18030 examples [00:16, 1107.59 examples/s]18142 examples [00:16, 1110.69 examples/s]18254 examples [00:16, 1108.77 examples/s]18367 examples [00:16, 1113.42 examples/s]18479 examples [00:16, 1114.59 examples/s]18591 examples [00:17, 1109.91 examples/s]18703 examples [00:17, 1112.10 examples/s]18815 examples [00:17, 1114.04 examples/s]18928 examples [00:17, 1116.48 examples/s]19040 examples [00:17, 1111.97 examples/s]19152 examples [00:17, 1105.00 examples/s]19265 examples [00:17, 1111.36 examples/s]19377 examples [00:17, 1113.49 examples/s]19490 examples [00:17, 1116.82 examples/s]19604 examples [00:17, 1122.13 examples/s]19717 examples [00:18, 1113.17 examples/s]19830 examples [00:18, 1117.46 examples/s]19942 examples [00:18, 1117.17 examples/s]20054 examples [00:18, 1065.51 examples/s]20166 examples [00:18, 1080.78 examples/s]20277 examples [00:18, 1086.50 examples/s]20390 examples [00:18, 1097.88 examples/s]20503 examples [00:18, 1104.76 examples/s]20615 examples [00:18, 1105.77 examples/s]20726 examples [00:19, 1101.40 examples/s]20838 examples [00:19, 1103.54 examples/s]20951 examples [00:19, 1109.01 examples/s]21064 examples [00:19, 1113.98 examples/s]21177 examples [00:19, 1116.39 examples/s]21289 examples [00:19, 1072.11 examples/s]21400 examples [00:19, 1082.37 examples/s]21509 examples [00:19, 1073.76 examples/s]21622 examples [00:19, 1090.02 examples/s]21735 examples [00:19, 1098.88 examples/s]21846 examples [00:20, 1100.98 examples/s]21957 examples [00:20, 1090.16 examples/s]22070 examples [00:20, 1097.90 examples/s]22183 examples [00:20, 1105.06 examples/s]22295 examples [00:20, 1108.01 examples/s]22407 examples [00:20, 1110.35 examples/s]22519 examples [00:20, 1107.56 examples/s]22632 examples [00:20, 1112.56 examples/s]22745 examples [00:20, 1116.03 examples/s]22857 examples [00:20, 1113.98 examples/s]22970 examples [00:21, 1115.83 examples/s]23082 examples [00:21, 1115.63 examples/s]23194 examples [00:21, 1114.91 examples/s]23307 examples [00:21, 1116.68 examples/s]23420 examples [00:21, 1118.45 examples/s]23532 examples [00:21, 1118.57 examples/s]23644 examples [00:21, 1094.27 examples/s]23757 examples [00:21, 1104.08 examples/s]23870 examples [00:21, 1108.99 examples/s]23983 examples [00:21, 1114.48 examples/s]24095 examples [00:22, 1115.80 examples/s]24207 examples [00:22, 1109.93 examples/s]24321 examples [00:22, 1116.56 examples/s]24433 examples [00:22, 1115.93 examples/s]24545 examples [00:22, 1115.31 examples/s]24657 examples [00:22, 1115.25 examples/s]24769 examples [00:22, 1099.60 examples/s]24880 examples [00:22, 1098.41 examples/s]24991 examples [00:22, 1101.54 examples/s]25102 examples [00:22, 1100.62 examples/s]25213 examples [00:23, 1101.10 examples/s]25324 examples [00:23, 1094.40 examples/s]25436 examples [00:23, 1099.30 examples/s]25547 examples [00:23, 1100.03 examples/s]25658 examples [00:23, 1087.15 examples/s]25769 examples [00:23, 1092.70 examples/s]25879 examples [00:23, 1063.48 examples/s]25990 examples [00:23, 1076.34 examples/s]26101 examples [00:23, 1084.86 examples/s]26211 examples [00:23, 1089.27 examples/s]26324 examples [00:24, 1098.96 examples/s]26434 examples [00:24, 1092.10 examples/s]26546 examples [00:24, 1099.87 examples/s]26657 examples [00:24, 1098.13 examples/s]26768 examples [00:24, 1100.71 examples/s]26880 examples [00:24, 1105.06 examples/s]26991 examples [00:24, 1091.93 examples/s]27102 examples [00:24, 1095.72 examples/s]27214 examples [00:24, 1101.48 examples/s]27325 examples [00:25, 1091.74 examples/s]27437 examples [00:25, 1097.98 examples/s]27547 examples [00:25, 1066.39 examples/s]27660 examples [00:25, 1083.22 examples/s]27772 examples [00:25, 1091.46 examples/s]27885 examples [00:25, 1100.07 examples/s]27998 examples [00:25, 1107.51 examples/s]28109 examples [00:25, 1099.41 examples/s]28220 examples [00:25, 1054.62 examples/s]28330 examples [00:25, 1065.68 examples/s]28440 examples [00:26, 1075.45 examples/s]28551 examples [00:26, 1084.79 examples/s]28660 examples [00:26, 1080.56 examples/s]28769 examples [00:26, 1082.75 examples/s]28880 examples [00:26, 1087.93 examples/s]28990 examples [00:26, 1090.42 examples/s]29100 examples [00:26, 1092.34 examples/s]29210 examples [00:26, 1073.11 examples/s]29321 examples [00:26, 1083.31 examples/s]29433 examples [00:26, 1091.45 examples/s]29544 examples [00:27, 1094.79 examples/s]29654 examples [00:27, 1092.85 examples/s]29764 examples [00:27, 1088.15 examples/s]29875 examples [00:27, 1091.92 examples/s]29986 examples [00:27, 1095.49 examples/s]30096 examples [00:27, 1041.07 examples/s]30204 examples [00:27, 1051.26 examples/s]30314 examples [00:27, 1065.02 examples/s]30425 examples [00:27, 1077.16 examples/s]30533 examples [00:27, 1070.90 examples/s]30643 examples [00:28, 1078.22 examples/s]30751 examples [00:28, 1078.57 examples/s]30859 examples [00:28, 1077.74 examples/s]30967 examples [00:28, 1074.71 examples/s]31077 examples [00:28, 1079.42 examples/s]31188 examples [00:28, 1087.14 examples/s]31299 examples [00:28, 1091.41 examples/s]31409 examples [00:28, 1091.97 examples/s]31520 examples [00:28, 1095.69 examples/s]31630 examples [00:28, 1084.96 examples/s]31740 examples [00:29, 1089.06 examples/s]31850 examples [00:29, 1091.76 examples/s]31960 examples [00:29, 1090.51 examples/s]32070 examples [00:29, 1089.43 examples/s]32181 examples [00:29, 1094.73 examples/s]32291 examples [00:29, 1092.84 examples/s]32401 examples [00:29, 1093.50 examples/s]32511 examples [00:29, 1095.06 examples/s]32623 examples [00:29, 1100.36 examples/s]32734 examples [00:29, 1101.84 examples/s]32845 examples [00:30, 1059.94 examples/s]32955 examples [00:30, 1070.92 examples/s]33065 examples [00:30, 1077.18 examples/s]33176 examples [00:30, 1085.38 examples/s]33285 examples [00:30, 1085.41 examples/s]33398 examples [00:30, 1097.56 examples/s]33510 examples [00:30, 1102.22 examples/s]33621 examples [00:30, 1101.40 examples/s]33735 examples [00:30, 1109.86 examples/s]33847 examples [00:31, 1112.10 examples/s]33959 examples [00:31, 1111.65 examples/s]34071 examples [00:31, 1109.60 examples/s]34183 examples [00:31, 1109.80 examples/s]34296 examples [00:31, 1114.30 examples/s]34408 examples [00:31, 1115.80 examples/s]34521 examples [00:31, 1117.37 examples/s]34633 examples [00:31, 1113.46 examples/s]34745 examples [00:31, 1097.01 examples/s]34857 examples [00:31, 1102.04 examples/s]34970 examples [00:32, 1109.96 examples/s]35082 examples [00:32, 1104.02 examples/s]35194 examples [00:32, 1107.28 examples/s]35305 examples [00:32, 1095.63 examples/s]35415 examples [00:32, 1095.52 examples/s]35526 examples [00:32, 1097.00 examples/s]35637 examples [00:32, 1098.64 examples/s]35747 examples [00:32, 1094.36 examples/s]35857 examples [00:32, 1092.08 examples/s]35968 examples [00:32, 1095.22 examples/s]36080 examples [00:33, 1099.68 examples/s]36191 examples [00:33, 1102.39 examples/s]36302 examples [00:33, 1095.23 examples/s]36412 examples [00:33, 1091.45 examples/s]36523 examples [00:33, 1094.25 examples/s]36633 examples [00:33, 1092.89 examples/s]36745 examples [00:33, 1098.86 examples/s]36855 examples [00:33, 1096.54 examples/s]36965 examples [00:33, 1092.69 examples/s]37076 examples [00:33, 1096.72 examples/s]37186 examples [00:34, 1090.56 examples/s]37298 examples [00:34, 1096.96 examples/s]37408 examples [00:34, 1054.49 examples/s]37518 examples [00:34, 1065.17 examples/s]37630 examples [00:34, 1080.89 examples/s]37743 examples [00:34, 1093.68 examples/s]37856 examples [00:34, 1104.01 examples/s]37968 examples [00:34, 1107.33 examples/s]38079 examples [00:34, 1105.85 examples/s]38192 examples [00:34, 1110.45 examples/s]38305 examples [00:35, 1113.71 examples/s]38417 examples [00:35, 1112.72 examples/s]38529 examples [00:35, 1111.05 examples/s]38641 examples [00:35, 1107.83 examples/s]38753 examples [00:35, 1109.37 examples/s]38866 examples [00:35, 1113.72 examples/s]38978 examples [00:35, 1112.38 examples/s]39090 examples [00:35, 1112.74 examples/s]39202 examples [00:35, 1111.15 examples/s]39315 examples [00:35, 1113.94 examples/s]39428 examples [00:36, 1116.89 examples/s]39541 examples [00:36, 1119.00 examples/s]39653 examples [00:36, 1116.96 examples/s]39765 examples [00:36, 1079.60 examples/s]39875 examples [00:36, 1084.90 examples/s]39986 examples [00:36, 1091.46 examples/s]40096 examples [00:36, 1037.50 examples/s]40207 examples [00:36, 1056.18 examples/s]40318 examples [00:36, 1071.76 examples/s]40430 examples [00:37, 1084.05 examples/s]40541 examples [00:37, 1090.29 examples/s]40653 examples [00:37, 1098.80 examples/s]40765 examples [00:37, 1103.40 examples/s]40876 examples [00:37, 1103.52 examples/s]40989 examples [00:37, 1108.46 examples/s]41102 examples [00:37, 1112.44 examples/s]41215 examples [00:37, 1115.36 examples/s]41327 examples [00:37, 1095.84 examples/s]41437 examples [00:37, 1091.77 examples/s]41547 examples [00:38, 1086.05 examples/s]41658 examples [00:38, 1092.05 examples/s]41769 examples [00:38, 1096.79 examples/s]41879 examples [00:38, 1096.28 examples/s]41989 examples [00:38, 1071.95 examples/s]42097 examples [00:38, 1071.33 examples/s]42207 examples [00:38, 1078.45 examples/s]42317 examples [00:38, 1084.72 examples/s]42428 examples [00:38, 1089.78 examples/s]42538 examples [00:38, 1083.30 examples/s]42648 examples [00:39, 1088.15 examples/s]42759 examples [00:39, 1092.28 examples/s]42870 examples [00:39, 1095.94 examples/s]42981 examples [00:39, 1098.25 examples/s]43091 examples [00:39, 1092.80 examples/s]43202 examples [00:39, 1097.16 examples/s]43314 examples [00:39, 1103.46 examples/s]43425 examples [00:39, 1104.50 examples/s]43536 examples [00:39, 1105.20 examples/s]43647 examples [00:39, 1095.17 examples/s]43758 examples [00:40, 1097.85 examples/s]43869 examples [00:40, 1099.63 examples/s]43980 examples [00:40, 1102.34 examples/s]44091 examples [00:40, 1103.31 examples/s]44202 examples [00:40, 1092.63 examples/s]44312 examples [00:40, 1063.54 examples/s]44422 examples [00:40, 1073.60 examples/s]44535 examples [00:40, 1087.29 examples/s]44646 examples [00:40, 1092.55 examples/s]44756 examples [00:40, 1089.19 examples/s]44867 examples [00:41, 1093.98 examples/s]44978 examples [00:41, 1097.64 examples/s]45088 examples [00:41, 1088.45 examples/s]45197 examples [00:41, 1087.60 examples/s]45306 examples [00:41, 1085.78 examples/s]45415 examples [00:41, 1086.48 examples/s]45526 examples [00:41, 1091.29 examples/s]45638 examples [00:41, 1097.40 examples/s]45749 examples [00:41, 1100.27 examples/s]45860 examples [00:41, 1095.80 examples/s]45971 examples [00:42, 1097.37 examples/s]46082 examples [00:42, 1100.33 examples/s]46193 examples [00:42, 1099.77 examples/s]46304 examples [00:42, 1099.95 examples/s]46414 examples [00:42, 1095.20 examples/s]46524 examples [00:42, 1095.17 examples/s]46634 examples [00:42, 1078.42 examples/s]46743 examples [00:42, 1080.56 examples/s]46852 examples [00:42, 1081.48 examples/s]46961 examples [00:42, 1070.79 examples/s]47071 examples [00:43, 1077.53 examples/s]47179 examples [00:43, 1077.02 examples/s]47289 examples [00:43, 1081.64 examples/s]47399 examples [00:43, 1084.99 examples/s]47508 examples [00:43, 1080.54 examples/s]47619 examples [00:43, 1088.81 examples/s]47731 examples [00:43, 1095.82 examples/s]47842 examples [00:43, 1099.85 examples/s]47953 examples [00:43, 1084.80 examples/s]48062 examples [00:44, 1082.94 examples/s]48171 examples [00:44, 1068.42 examples/s]48281 examples [00:44, 1077.50 examples/s]48391 examples [00:44, 1083.22 examples/s]48501 examples [00:44, 1086.64 examples/s]48610 examples [00:44, 1085.29 examples/s]48719 examples [00:44, 1086.02 examples/s]48828 examples [00:44, 1076.61 examples/s]48936 examples [00:44, 1042.07 examples/s]49047 examples [00:44, 1059.01 examples/s]49154 examples [00:45, 1041.86 examples/s]49264 examples [00:45, 1057.21 examples/s]49376 examples [00:45, 1074.23 examples/s]49485 examples [00:45, 1076.40 examples/s]49593 examples [00:45, 1076.39 examples/s]49702 examples [00:45, 1079.29 examples/s]49812 examples [00:45, 1083.69 examples/s]49922 examples [00:45, 1088.44 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 7201/50000 [00:00<00:00, 72008.87 examples/s] 40%|███▉      | 19872/50000 [00:00<00:00, 82721.64 examples/s] 66%|██████▌   | 33071/50000 [00:00<00:00, 93151.39 examples/s] 93%|█████████▎| 46335/50000 [00:00<00:00, 102208.98 examples/s]                                                                0 examples [00:00, ? examples/s]90 examples [00:00, 899.82 examples/s]200 examples [00:00, 951.11 examples/s]306 examples [00:00, 981.25 examples/s]417 examples [00:00, 1015.25 examples/s]529 examples [00:00, 1043.03 examples/s]640 examples [00:00, 1062.16 examples/s]751 examples [00:00, 1075.13 examples/s]861 examples [00:00, 1081.54 examples/s]972 examples [00:00, 1087.96 examples/s]1083 examples [00:01, 1093.40 examples/s]1194 examples [00:01, 1094.73 examples/s]1304 examples [00:01, 1095.42 examples/s]1414 examples [00:01, 1094.32 examples/s]1525 examples [00:01, 1096.57 examples/s]1637 examples [00:01, 1102.11 examples/s]1748 examples [00:01, 1102.75 examples/s]1859 examples [00:01, 1102.91 examples/s]1970 examples [00:01, 1076.11 examples/s]2078 examples [00:01, 1076.67 examples/s]2188 examples [00:02, 1082.14 examples/s]2299 examples [00:02, 1089.24 examples/s]2408 examples [00:02, 1087.19 examples/s]2517 examples [00:02, 1077.75 examples/s]2627 examples [00:02, 1082.61 examples/s]2737 examples [00:02, 1086.36 examples/s]2846 examples [00:02, 1086.83 examples/s]2956 examples [00:02, 1088.75 examples/s]3065 examples [00:02, 1069.10 examples/s]3175 examples [00:02, 1075.99 examples/s]3287 examples [00:03, 1086.46 examples/s]3398 examples [00:03, 1091.56 examples/s]3508 examples [00:03, 1093.76 examples/s]3618 examples [00:03, 1094.32 examples/s]3729 examples [00:03, 1098.50 examples/s]3839 examples [00:03, 1097.40 examples/s]3949 examples [00:03, 1088.33 examples/s]4058 examples [00:03, 1087.45 examples/s]4169 examples [00:03, 1092.88 examples/s]4282 examples [00:03, 1103.13 examples/s]4393 examples [00:04, 1102.06 examples/s]4506 examples [00:04, 1108.71 examples/s]4618 examples [00:04, 1110.13 examples/s]4730 examples [00:04, 1109.12 examples/s]4843 examples [00:04, 1113.82 examples/s]4955 examples [00:04, 1114.64 examples/s]5067 examples [00:04, 1081.33 examples/s]5180 examples [00:04, 1092.74 examples/s]5291 examples [00:04, 1094.83 examples/s]5404 examples [00:04, 1102.65 examples/s]5516 examples [00:05, 1105.95 examples/s]5628 examples [00:05, 1109.42 examples/s]5740 examples [00:05, 1112.56 examples/s]5852 examples [00:05, 1112.13 examples/s]5965 examples [00:05, 1116.92 examples/s]6078 examples [00:05, 1119.01 examples/s]6191 examples [00:05, 1121.62 examples/s]6304 examples [00:05, 1116.21 examples/s]6416 examples [00:05, 1105.26 examples/s]6527 examples [00:05, 1105.87 examples/s]6638 examples [00:06, 1104.93 examples/s]6749 examples [00:06, 1106.34 examples/s]6860 examples [00:06, 1103.08 examples/s]6971 examples [00:06, 1094.63 examples/s]7081 examples [00:06, 1092.54 examples/s]7191 examples [00:06, 1093.21 examples/s]7301 examples [00:06, 1087.69 examples/s]7413 examples [00:06, 1094.79 examples/s]7523 examples [00:06, 1091.06 examples/s]7634 examples [00:06, 1096.22 examples/s]7745 examples [00:07, 1097.85 examples/s]7855 examples [00:07, 1081.94 examples/s]7965 examples [00:07, 1085.29 examples/s]8076 examples [00:07, 1089.98 examples/s]8188 examples [00:07, 1096.69 examples/s]8300 examples [00:07, 1103.32 examples/s]8414 examples [00:07, 1111.50 examples/s]8526 examples [00:07, 1110.69 examples/s]8638 examples [00:07, 1112.30 examples/s]8750 examples [00:07, 1110.51 examples/s]8862 examples [00:08, 1105.16 examples/s]8975 examples [00:08, 1111.82 examples/s]9087 examples [00:08, 1106.14 examples/s]9199 examples [00:08, 1107.60 examples/s]9312 examples [00:08, 1111.70 examples/s]9424 examples [00:08, 1113.33 examples/s]9536 examples [00:08, 1102.91 examples/s]9647 examples [00:08, 1051.32 examples/s]9757 examples [00:08, 1064.59 examples/s]9868 examples [00:09, 1077.49 examples/s]9979 examples [00:09, 1086.21 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteM47Z2C/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteM47Z2C/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff51c2899d8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff51c2899d8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff51c2899d8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff4facae7b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff51a2ecc88>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff51c2899d8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff51c2899d8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff51c2899d8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff4facbc400>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff4facbc2b0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7ff494a7a6a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7ff494a7a6a8> 

  function with postional parmater data_info <function split_train_valid at 0x7ff494a7a6a8> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7ff494a7a7b8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7ff494a7a7b8> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7ff494a7a7b8> , (data_info, **args) 
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<18:14:29, 13.1kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<13:00:02, 18.4kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:09:14, 26.2kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:25:00, 37.3kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<4:28:51, 53.2kB/s].vector_cache/glove.6B.zip:   1%|          | 9.93M/862M [00:01<3:06:54, 76.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.4M/862M [00:01<2:10:04, 108kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 18.5M/862M [00:01<1:30:51, 155kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.2M/862M [00:01<1:03:16, 221kB/s].vector_cache/glove.6B.zip:   3%|▎         | 30.0M/862M [00:01<44:05, 315kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 35.7M/862M [00:01<30:44, 448kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.4M/862M [00:02<21:28, 637kB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.1M/862M [00:02<15:01, 904kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.5M/862M [00:02<10:38, 1.27MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.6M/862M [00:04<09:19, 1.44MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.7M/862M [00:04<10:22, 1.30MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:04<08:10, 1.64MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.6M/862M [00:04<05:56, 2.26MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.7M/862M [00:06<09:53, 1.35MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:06<08:30, 1.57MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.4M/862M [00:06<06:16, 2.13MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.9M/862M [00:08<07:12, 1.85MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:08<07:44, 1.72MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<05:59, 2.22MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.4M/862M [00:08<04:20, 3.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:10<14:13, 930kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:10<11:05, 1.19MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:10<08:08, 1.62MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.0M/862M [00:10<05:52, 2.24MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.1M/862M [00:12<1:22:30, 160kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:12<1:00:23, 218kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<42:54, 307kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 76.2M/862M [00:14<32:02, 409kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:14<23:44, 551kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:14<16:55, 772kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:16<14:52, 876kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:16<13:02, 998kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<09:41, 1.34MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.0M/862M [00:16<07:00, 1.85MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:18<09:02, 1.43MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:18<07:40, 1.69MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.4M/862M [00:18<05:41, 2.27MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.6M/862M [00:20<07:00, 1.84MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:20<06:12, 2.08MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.5M/862M [00:20<04:35, 2.80MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.7M/862M [00:22<06:17, 2.04MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:22<07:00, 1.83MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<05:33, 2.31MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.8M/862M [00:23<05:56, 2.15MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:24<05:28, 2.33MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.7M/862M [00:24<04:05, 3.11MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<05:50, 2.17MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<06:40, 1.90MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:18, 2.39MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<05:44, 2.19MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<06:41, 1.89MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:14, 2.41MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<03:49, 3.29MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<09:11, 1.36MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<07:31, 1.67MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:30, 2.27MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:30<04:01, 3.10MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<53:03, 235kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<39:39, 315kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<28:21, 440kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<21:47, 570kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<17:45, 699kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<13:03, 949kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<09:15, 1.33MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<1:47:29, 115kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<1:16:27, 161kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<53:43, 229kB/s]  .vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<40:21, 304kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<30:37, 401kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<21:58, 558kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 129M/862M [00:38<15:27, 790kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<32:39, 374kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<24:06, 506kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:39<17:09, 710kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<14:49, 819kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<12:50, 945kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<09:31, 1.27MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:41<06:46, 1.78MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<15:23, 784kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<12:01, 1.00MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:43<08:42, 1.38MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<08:53, 1.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<07:25, 1.61MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:45<05:28, 2.19MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<06:37, 1.80MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<06:57, 1.72MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<05:28, 2.18MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<05:44, 2.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<06:31, 1.82MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<05:10, 2.29MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:49<03:44, 3.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<1:18:28, 150kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<56:08, 210kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<39:30, 298kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<30:20, 387kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<23:36, 497kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<17:05, 685kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<13:48, 845kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<10:50, 1.08MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<07:51, 1.48MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<08:12, 1.41MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<08:05, 1.43MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<06:15, 1.85MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:57<04:29, 2.57MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<11:06:42, 17.3kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<7:47:37, 24.6kB/s] .vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<5:26:54, 35.1kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<3:50:46, 49.6kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<2:43:48, 69.9kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<1:55:07, 99.4kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<1:22:04, 139kB/s] .vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<59:54, 190kB/s]  .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<42:28, 268kB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:03<29:44, 381kB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<1:41:27, 112kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<1:12:09, 157kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<50:40, 223kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<38:00, 296kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<28:53, 389kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<20:46, 540kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<16:17, 686kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<12:20, 906kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<08:53, 1.25MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:09<06:22, 1.74MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<51:44, 215kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<38:27, 289kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<27:22, 405kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:11<19:16, 574kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<18:27, 598kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<13:50, 797kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<09:57, 1.11MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<09:31, 1.15MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<08:55, 1.23MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<06:48, 1.61MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<06:30, 1.68MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<05:41, 1.92MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<04:12, 2.58MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<05:29, 1.97MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<06:02, 1.79MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<04:46, 2.27MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<05:04, 2.12MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<04:38, 2.32MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<03:30, 3.06MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<04:58, 2.15MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<05:39, 1.89MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<04:28, 2.39MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<03:15, 3.27MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<46:06, 231kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<33:20, 319kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<23:33, 450kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<18:54, 559kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<14:18, 737kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<10:15, 1.03MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<09:37, 1.09MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<08:52, 1.18MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<06:39, 1.57MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<04:46, 2.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<11:08, 936kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<08:51, 1.18MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:30<06:24, 1.62MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<06:54, 1.50MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<05:53, 1.76MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:32<04:22, 2.36MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<05:29, 1.87MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<04:54, 2.10MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<03:39, 2.81MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<04:58, 2.05MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<05:33, 1.84MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<04:19, 2.36MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:36<03:13, 3.16MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<05:19, 1.91MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<04:46, 2.13MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<03:35, 2.82MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<04:51, 2.08MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<05:26, 1.85MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<04:14, 2.38MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:40<03:05, 3.25MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<07:19, 1.37MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<06:08, 1.63MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<04:32, 2.20MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<05:29, 1.81MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<05:52, 1.69MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:36, 2.15MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:44<03:19, 2.97MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<9:31:17, 17.3kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<6:40:40, 24.6kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 272M/862M [01:46<4:39:58, 35.2kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<3:17:34, 49.6kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<2:20:14, 69.9kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<1:38:32, 99.4kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<1:10:11, 139kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<49:56, 195kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<35:06, 276kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<26:46, 361kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<20:41, 467kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<14:53, 648kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:52<10:30, 915kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<12:33, 764kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<09:45, 983kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<07:03, 1.35MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<07:09, 1.33MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<07:01, 1.36MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<05:20, 1.78MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:56<03:50, 2.47MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<09:06, 1.04MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<07:22, 1.28MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<05:21, 1.76MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<05:52, 1.60MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<05:05, 1.84MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<03:48, 2.46MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:48, 1.94MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<05:20, 1.75MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:09, 2.24MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:02<03:01, 3.07MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<07:10, 1.29MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<05:59, 1.54MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<04:25, 2.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<05:12, 1.76MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<05:34, 1.65MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:19, 2.12MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<03:09, 2.89MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<05:41, 1.60MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<04:56, 1.84MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<03:41, 2.46MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<04:36, 1.97MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:10, 2.17MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<03:06, 2.90MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:15, 2.11MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:12<04:47, 1.87MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<03:45, 2.39MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<02:43, 3.27MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<07:11, 1.24MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<05:56, 1.50MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<04:21, 2.04MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<05:05, 1.74MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<05:25, 1.63MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<04:10, 2.11MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<03:03, 2.88MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<05:35, 1.57MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<04:49, 1.82MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<03:33, 2.46MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<04:30, 1.94MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<04:59, 1.74MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 341M/862M [02:20<03:56, 2.20MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:20<02:51, 3.02MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<28:01, 308kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<20:30, 421kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<14:30, 594kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<12:05, 709kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<10:16, 834kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<07:38, 1.12MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:24<05:24, 1.57MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<24:15, 351kB/s] .vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<17:50, 476kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<12:40, 668kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<10:46, 782kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<08:24, 1.00MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:27<06:05, 1.38MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<06:10, 1.36MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<05:12, 1.61MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<03:49, 2.18MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:34, 1.81MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:57, 1.67MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:49, 2.16MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:31<02:47, 2.96MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<05:35, 1.47MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<04:46, 1.72MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<03:30, 2.33MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:19, 1.89MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:53, 2.10MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<02:55, 2.77MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<03:54, 2.08MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<03:34, 2.26MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<02:40, 3.01MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:43, 2.16MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:17, 1.87MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<03:25, 2.34MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:39<02:29, 3.20MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<27:01, 294kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<19:43, 403kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<13:58, 567kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<11:33, 683kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<09:44, 809kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<07:10, 1.10MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:43<05:05, 1.54MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<08:45, 893kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<06:55, 1.13MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<05:02, 1.54MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<05:18, 1.46MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<05:20, 1.45MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:08, 1.87MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:47<02:59, 2.58MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<25:13, 304kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<18:27, 416kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<13:04, 585kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<10:51, 701kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<09:12, 827kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<06:46, 1.12MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<04:49, 1.56MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 410M/862M [02:53<06:48, 1.11MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<05:32, 1.36MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<04:03, 1.85MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<04:33, 1.64MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<04:46, 1.57MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:39, 2.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<02:39, 2.79MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<05:17, 1.40MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<04:27, 1.66MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<03:16, 2.25MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<03:58, 1.85MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<04:20, 1.69MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:24, 2.14MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<02:28, 2.94MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<23:38, 308kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<17:17, 420kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<12:14, 592kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<10:11, 707kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<08:38, 833kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<06:24, 1.12MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:03<04:32, 1.57MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<54:11, 132kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<38:39, 184kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<27:08, 262kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<20:32, 344kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<15:51, 445kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<11:23, 619kB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:07<08:01, 874kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<10:12, 686kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<07:54, 883kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<05:42, 1.22MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<05:33, 1.25MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<05:21, 1.29MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<04:02, 1.71MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:55, 2.35MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<04:43, 1.45MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<04:01, 1.70MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:12<02:58, 2.29MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:39, 1.86MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:59, 1.70MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<03:05, 2.19MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<02:14, 3.01MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<05:50, 1.15MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<04:46, 1.41MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:16<03:30, 1.91MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:58, 1.67MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:28, 1.92MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<02:34, 2.58MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:18, 1.99MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<02:59, 2.19MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<02:15, 2.90MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:05, 2.11MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:32, 1.84MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<02:45, 2.35MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:22<02:03, 3.16MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:22, 1.91MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<03:01, 2.13MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<02:16, 2.82MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<03:03, 2.09MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:29, 1.83MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<02:45, 2.31MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:26<01:59, 3.16MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<46:37, 135kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<33:15, 190kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<23:21, 269kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<17:41, 353kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<13:41, 456kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<09:53, 630kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:30<06:56, 889kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<49:03, 126kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<34:57, 176kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<24:31, 250kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<18:28, 330kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<14:13, 429kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<10:12, 597kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:34<07:11, 843kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<07:54, 764kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<06:09, 978kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<04:27, 1.35MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<04:28, 1.33MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<04:23, 1.36MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:21, 1.78MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:38<02:23, 2.47MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<07:34, 779kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<05:54, 998kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<04:16, 1.37MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<04:18, 1.35MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<03:37, 1.61MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:40, 2.17MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<03:12, 1.80MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:29, 1.65MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:44<02:41, 2.14MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<01:59, 2.87MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:58, 1.91MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:41, 2.11MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:01, 2.80MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:40, 2.10MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:04, 1.83MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:26, 2.30MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:48<01:45, 3.15MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<41:00, 135kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<29:16, 190kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<20:30, 269kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<15:32, 353kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<12:01, 456kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<08:41, 630kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:52<06:05, 890kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<17:16, 314kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<12:38, 428kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<08:56, 602kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<07:27, 718kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<06:21, 842kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<04:40, 1.14MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<03:20, 1.59MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<04:08, 1.27MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<03:27, 1.52MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:33, 2.06MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:56, 1.77MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [03:59<03:07, 1.67MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:27, 2.12MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<01:46, 2.91MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<43:06, 119kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<30:40, 168kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<21:29, 238kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<16:06, 315kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<12:20, 411kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<08:52, 570kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:04<06:12, 806kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<16:25, 305kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<12:00, 417kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:05<08:29, 586kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<07:02, 702kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<05:58, 827kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<04:23, 1.12MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<03:07, 1.57MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<04:19, 1.12MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<03:31, 1.38MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:09<02:33, 1.89MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:54, 1.65MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<03:02, 1.57MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<02:20, 2.05MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:11<01:42, 2.79MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:50, 1.66MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:28, 1.90MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<01:50, 2.54MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<02:22, 1.97MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:38, 1.76MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:03, 2.26MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:15<01:29, 3.10MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<03:34, 1.28MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:59, 1.54MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<02:11, 2.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:19<02:34, 1.76MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:45, 1.65MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:07, 2.13MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:19<01:31, 2.92MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<03:15, 1.37MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:44, 1.63MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:21<02:00, 2.21MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:24, 1.83MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:36, 1.68MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<02:01, 2.17MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<01:29, 2.90MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:13, 1.95MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:59, 2.16MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<01:29, 2.88MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:01, 2.11MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<01:51, 2.30MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<01:22, 3.06MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<01:55, 2.17MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:13, 1.87MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:46, 2.35MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:29<01:16, 3.22MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<30:18, 136kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<21:37, 190kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<15:07, 270kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<11:25, 354kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<08:50, 457kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<06:21, 635kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<04:27, 896kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<04:57, 804kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<03:48, 1.04MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:35<02:44, 1.44MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:35<01:57, 1.99MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<15:55, 245kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<11:32, 338kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<08:07, 477kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<06:31, 589kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<05:22, 715kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<03:56, 969kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:39<02:46, 1.36MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<20:09, 187kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<14:29, 260kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<10:09, 368kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<07:54, 469kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<06:18, 587kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<04:34, 808kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<03:12, 1.14MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<04:51, 748kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<03:43, 976kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<02:40, 1.35MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:41, 1.33MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<02:38, 1.35MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:00, 1.77MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:47<01:25, 2.45MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<03:24, 1.03MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<02:44, 1.28MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:59, 1.74MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<02:10, 1.58MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<02:12, 1.55MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:43, 1.99MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<01:13, 2.74MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<24:55, 135kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<17:45, 189kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<12:24, 268kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<09:21, 352kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<07:14, 455kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<05:12, 629kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<03:38, 888kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<25:36, 126kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<18:13, 177kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:56<12:44, 251kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<09:32, 331kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<07:20, 430kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<05:17, 594kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [04:59<03:40, 840kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<11:51, 260kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<08:36, 358kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:00<06:03, 505kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<04:53, 618kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<03:43, 809kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<02:39, 1.13MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:31, 1.17MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:23, 1.23MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:49, 1.61MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:04<01:17, 2.24MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<14:44, 195kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<10:36, 271kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<07:25, 384kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<05:47, 486kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<04:38, 606kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<03:22, 828kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:08<02:21, 1.17MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<08:15, 332kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<06:00, 456kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<04:14, 640kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<03:32, 755kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<03:02, 879kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<02:14, 1.19MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:12<01:34, 1.66MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:28, 1.06MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:59, 1.31MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<01:26, 1.78MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:35, 1.60MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:38, 1.54MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:15, 2.00MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<00:54, 2.74MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:45, 1.41MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:29, 1.65MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<01:05, 2.22MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:18, 1.84MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:25, 1.69MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:05, 2.18MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<00:47, 3.00MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:58, 1.18MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:37, 1.43MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<01:10, 1.95MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:20, 1.69MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:24, 1.60MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:04, 2.08MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<00:46, 2.85MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:34, 1.39MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:19, 1.65MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<00:58, 2.22MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:10, 1.82MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:15, 1.68MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<00:58, 2.17MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:28<00:42, 2.95MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:14, 1.67MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:04, 1.90MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:47, 2.54MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:00, 1.97MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:07, 1.77MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:32<00:52, 2.27MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:32<00:37, 3.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<01:20, 1.43MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:08, 1.68MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<00:50, 2.26MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<01:00, 1.85MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:05, 1.70MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:50, 2.18MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<00:35, 3.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<01:38, 1.09MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:19, 1.34MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:57, 1.83MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<01:03, 1.62MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:05, 1.58MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:49, 2.05MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:35, 2.82MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<01:11, 1.38MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<01:00, 1.63MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:44, 2.20MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:52, 1.82MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:56, 1.68MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:43, 2.17MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:31, 2.95MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:51, 1.75MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:45, 1.97MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:33, 2.65MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:42, 2.02MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:48, 1.79MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<00:38, 2.25MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<00:26, 3.08MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<10:05, 136kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<07:10, 191kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:49<04:58, 270kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<03:41, 355kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<02:50, 458kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<02:02, 635kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:51<01:23, 897kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<01:47, 692kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<01:22, 896kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<00:58, 1.24MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:55, 1.25MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:53, 1.30MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:40, 1.69MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:55<00:28, 2.34MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<03:49, 288kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<02:46, 394kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<01:55, 555kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<01:32, 670kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<01:17, 796kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:56, 1.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [05:59<00:38, 1.51MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:53, 1.07MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:43, 1.32MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:01<00:31, 1.80MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:03<00:33, 1.61MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:34, 1.55MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:26, 2.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:03<00:18, 2.77MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:34, 1.44MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:29, 1.69MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:20, 2.29MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:24, 1.86MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:26, 1.71MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:20, 2.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:07<00:14, 3.01MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:31, 1.31MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:25, 1.61MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<00:18, 2.17MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:20, 1.80MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:22, 1.67MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:17, 2.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:11<00:11, 2.91MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<01:47, 307kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<01:18, 420kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:52, 591kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:41, 706kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:34, 831kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:25, 1.12MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:15<00:16, 1.56MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<01:26, 289kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<01:02, 396kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:41, 559kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:30, 674kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:25, 805kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:18, 1.09MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:19<00:11, 1.53MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:17, 959kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:13, 1.20MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:08, 1.65MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:08, 1.52MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:08, 1.49MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:05, 1.95MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<00:03, 2.68MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:06, 1.34MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:05, 1.60MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:03, 2.15MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:02, 1.79MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.69MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.14MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:27<00:00, 2.95MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:01, 135kB/s] .vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<70:25:05,  1.58it/s]  0%|          | 827/400000 [00:00<49:11:42,  2.25it/s]  0%|          | 1690/400000 [00:00<34:21:57,  3.22it/s]  1%|          | 2551/400000 [00:00<24:00:28,  4.60it/s]  1%|          | 3413/400000 [00:01<16:46:22,  6.57it/s]  1%|          | 4272/400000 [00:01<11:43:10,  9.38it/s]  1%|▏         | 5117/400000 [00:01<8:11:24, 13.39it/s]   1%|▏         | 5972/400000 [00:01<5:43:28, 19.12it/s]  2%|▏         | 6809/400000 [00:01<4:00:09, 27.29it/s]  2%|▏         | 7671/400000 [00:01<2:47:57, 38.93it/s]  2%|▏         | 8526/400000 [00:01<1:57:32, 55.51it/s]  2%|▏         | 9385/400000 [00:01<1:22:19, 79.07it/s]  3%|▎         | 10223/400000 [00:01<57:45, 112.49it/s]  3%|▎         | 11092/400000 [00:01<40:33, 159.81it/s]  3%|▎         | 11956/400000 [00:02<28:33, 226.50it/s]  3%|▎         | 12803/400000 [00:02<20:10, 319.89it/s]  3%|▎         | 13664/400000 [00:02<14:18, 449.82it/s]  4%|▎         | 14514/400000 [00:02<10:13, 627.99it/s]  4%|▍         | 15372/400000 [00:02<07:22, 869.83it/s]  4%|▍         | 16233/400000 [00:02<05:22, 1191.00it/s]  4%|▍         | 17096/400000 [00:02<03:58, 1606.34it/s]  4%|▍         | 17961/400000 [00:02<02:59, 2125.48it/s]  5%|▍         | 18819/400000 [00:02<02:18, 2744.31it/s]  5%|▍         | 19677/400000 [00:02<01:50, 3447.46it/s]  5%|▌         | 20538/400000 [00:03<01:30, 4203.38it/s]  5%|▌         | 21397/400000 [00:03<01:16, 4962.05it/s]  6%|▌         | 22255/400000 [00:03<01:07, 5636.29it/s]  6%|▌         | 23104/400000 [00:03<01:00, 6263.14it/s]  6%|▌         | 23963/400000 [00:03<00:55, 6816.89it/s]  6%|▌         | 24829/400000 [00:03<00:51, 7281.73it/s]  6%|▋         | 25696/400000 [00:03<00:48, 7648.53it/s]  7%|▋         | 26555/400000 [00:03<00:47, 7908.47it/s]  7%|▋         | 27414/400000 [00:03<00:46, 8014.39it/s]  7%|▋         | 28275/400000 [00:03<00:45, 8183.80it/s]  7%|▋         | 29139/400000 [00:04<00:44, 8314.92it/s]  8%|▊         | 30006/400000 [00:04<00:43, 8415.86it/s]  8%|▊         | 30870/400000 [00:04<00:43, 8480.63it/s]  8%|▊         | 31731/400000 [00:04<00:43, 8406.63it/s]  8%|▊         | 32587/400000 [00:04<00:43, 8451.83it/s]  8%|▊         | 33439/400000 [00:04<00:43, 8451.51it/s]  9%|▊         | 34302/400000 [00:04<00:43, 8503.58it/s]  9%|▉         | 35162/400000 [00:04<00:42, 8530.14it/s]  9%|▉         | 36018/400000 [00:04<00:42, 8494.69it/s]  9%|▉         | 36880/400000 [00:04<00:42, 8529.17it/s]  9%|▉         | 37739/400000 [00:05<00:42, 8545.42it/s] 10%|▉         | 38604/400000 [00:05<00:42, 8575.41it/s] 10%|▉         | 39463/400000 [00:05<00:42, 8562.04it/s] 10%|█         | 40320/400000 [00:05<00:42, 8529.44it/s] 10%|█         | 41181/400000 [00:05<00:41, 8550.61it/s] 11%|█         | 42037/400000 [00:05<00:42, 8512.04it/s] 11%|█         | 42889/400000 [00:05<00:42, 8434.12it/s] 11%|█         | 43733/400000 [00:05<00:42, 8409.20it/s] 11%|█         | 44575/400000 [00:05<00:42, 8328.17it/s] 11%|█▏        | 45409/400000 [00:05<00:42, 8325.88it/s] 12%|█▏        | 46270/400000 [00:06<00:42, 8406.34it/s] 12%|█▏        | 47111/400000 [00:06<00:42, 8388.54it/s] 12%|█▏        | 47972/400000 [00:06<00:41, 8453.48it/s] 12%|█▏        | 48818/400000 [00:06<00:41, 8420.64it/s] 12%|█▏        | 49685/400000 [00:06<00:41, 8493.74it/s] 13%|█▎        | 50541/400000 [00:06<00:41, 8511.19it/s] 13%|█▎        | 51409/400000 [00:06<00:40, 8561.00it/s] 13%|█▎        | 52266/400000 [00:06<00:40, 8556.57it/s] 13%|█▎        | 53122/400000 [00:06<00:40, 8539.22it/s] 13%|█▎        | 53985/400000 [00:06<00:40, 8564.88it/s] 14%|█▎        | 54847/400000 [00:07<00:40, 8580.70it/s] 14%|█▍        | 55716/400000 [00:07<00:39, 8612.56it/s] 14%|█▍        | 56578/400000 [00:07<00:40, 8532.47it/s] 14%|█▍        | 57432/400000 [00:07<00:40, 8388.67it/s] 15%|█▍        | 58300/400000 [00:07<00:40, 8471.65it/s] 15%|█▍        | 59156/400000 [00:07<00:40, 8497.80it/s] 15%|█▌        | 60020/400000 [00:07<00:39, 8539.93it/s] 15%|█▌        | 60875/400000 [00:07<00:39, 8483.94it/s] 15%|█▌        | 61724/400000 [00:07<00:39, 8480.19it/s] 16%|█▌        | 62592/400000 [00:07<00:39, 8538.06it/s] 16%|█▌        | 63457/400000 [00:08<00:39, 8570.40it/s] 16%|█▌        | 64315/400000 [00:08<00:39, 8561.77it/s] 16%|█▋        | 65183/400000 [00:08<00:38, 8595.92it/s] 17%|█▋        | 66043/400000 [00:08<00:40, 8325.51it/s] 17%|█▋        | 66903/400000 [00:08<00:39, 8405.33it/s] 17%|█▋        | 67771/400000 [00:08<00:39, 8484.80it/s] 17%|█▋        | 68638/400000 [00:08<00:38, 8538.05it/s] 17%|█▋        | 69504/400000 [00:08<00:38, 8572.62it/s] 18%|█▊        | 70362/400000 [00:08<00:38, 8556.27it/s] 18%|█▊        | 71219/400000 [00:08<00:38, 8529.93it/s] 18%|█▊        | 72077/400000 [00:09<00:38, 8542.38it/s] 18%|█▊        | 72942/400000 [00:09<00:38, 8571.95it/s] 18%|█▊        | 73800/400000 [00:09<00:38, 8559.69it/s] 19%|█▊        | 74657/400000 [00:09<00:38, 8490.13it/s] 19%|█▉        | 75518/400000 [00:09<00:38, 8523.67it/s] 19%|█▉        | 76383/400000 [00:09<00:37, 8561.00it/s] 19%|█▉        | 77251/400000 [00:09<00:37, 8593.27it/s] 20%|█▉        | 78112/400000 [00:09<00:37, 8597.28it/s] 20%|█▉        | 78972/400000 [00:09<00:37, 8462.16it/s] 20%|█▉        | 79829/400000 [00:10<00:37, 8493.86it/s] 20%|██        | 80686/400000 [00:10<00:37, 8515.32it/s] 20%|██        | 81551/400000 [00:10<00:37, 8552.71it/s] 21%|██        | 82411/400000 [00:10<00:37, 8566.01it/s] 21%|██        | 83268/400000 [00:10<00:37, 8541.43it/s] 21%|██        | 84132/400000 [00:10<00:36, 8570.43it/s] 21%|██        | 84990/400000 [00:10<00:37, 8507.37it/s] 21%|██▏       | 85849/400000 [00:10<00:36, 8531.88it/s] 22%|██▏       | 86714/400000 [00:10<00:36, 8565.36it/s] 22%|██▏       | 87572/400000 [00:10<00:36, 8569.39it/s] 22%|██▏       | 88435/400000 [00:11<00:36, 8587.01it/s] 22%|██▏       | 89306/400000 [00:11<00:36, 8620.93it/s] 23%|██▎       | 90169/400000 [00:11<00:35, 8621.92it/s] 23%|██▎       | 91032/400000 [00:11<00:35, 8613.34it/s] 23%|██▎       | 91894/400000 [00:11<00:35, 8600.98it/s] 23%|██▎       | 92757/400000 [00:11<00:35, 8607.87it/s] 23%|██▎       | 93625/400000 [00:11<00:35, 8627.04it/s] 24%|██▎       | 94494/400000 [00:11<00:35, 8643.48it/s] 24%|██▍       | 95360/400000 [00:11<00:35, 8648.33it/s] 24%|██▍       | 96225/400000 [00:11<00:35, 8631.86it/s] 24%|██▍       | 97095/400000 [00:12<00:35, 8650.47it/s] 24%|██▍       | 97965/400000 [00:12<00:34, 8665.19it/s] 25%|██▍       | 98834/400000 [00:12<00:34, 8671.69it/s] 25%|██▍       | 99702/400000 [00:12<00:34, 8668.98it/s] 25%|██▌       | 100569/400000 [00:12<00:34, 8630.72it/s] 25%|██▌       | 101438/400000 [00:12<00:34, 8646.76it/s] 26%|██▌       | 102306/400000 [00:12<00:34, 8656.69it/s] 26%|██▌       | 103172/400000 [00:12<00:34, 8598.56it/s] 26%|██▌       | 104032/400000 [00:12<00:34, 8598.98it/s] 26%|██▌       | 104894/400000 [00:12<00:34, 8598.30it/s] 26%|██▋       | 105759/400000 [00:13<00:34, 8613.65it/s] 27%|██▋       | 106628/400000 [00:13<00:33, 8633.65it/s] 27%|██▋       | 107495/400000 [00:13<00:33, 8642.22it/s] 27%|██▋       | 108360/400000 [00:13<00:34, 8556.08it/s] 27%|██▋       | 109216/400000 [00:13<00:34, 8523.77it/s] 28%|██▊       | 110080/400000 [00:13<00:33, 8556.75it/s] 28%|██▊       | 110953/400000 [00:13<00:33, 8605.46it/s] 28%|██▊       | 111824/400000 [00:13<00:33, 8634.30it/s] 28%|██▊       | 112688/400000 [00:13<00:33, 8633.26it/s] 28%|██▊       | 113554/400000 [00:13<00:33, 8640.02it/s] 29%|██▊       | 114419/400000 [00:14<00:33, 8606.61it/s] 29%|██▉       | 115280/400000 [00:14<00:33, 8569.05it/s] 29%|██▉       | 116142/400000 [00:14<00:33, 8582.18it/s] 29%|██▉       | 117008/400000 [00:14<00:32, 8603.69it/s] 29%|██▉       | 117869/400000 [00:14<00:32, 8592.30it/s] 30%|██▉       | 118729/400000 [00:14<00:32, 8574.73it/s] 30%|██▉       | 119596/400000 [00:14<00:32, 8600.29it/s] 30%|███       | 120460/400000 [00:14<00:32, 8609.81it/s] 30%|███       | 121327/400000 [00:14<00:32, 8626.77it/s] 31%|███       | 122194/400000 [00:14<00:32, 8637.83it/s] 31%|███       | 123058/400000 [00:15<00:32, 8635.60it/s] 31%|███       | 123927/400000 [00:15<00:31, 8649.86it/s] 31%|███       | 124793/400000 [00:15<00:31, 8641.95it/s] 31%|███▏      | 125664/400000 [00:15<00:31, 8661.33it/s] 32%|███▏      | 126531/400000 [00:15<00:31, 8651.54it/s] 32%|███▏      | 127397/400000 [00:15<00:31, 8634.06it/s] 32%|███▏      | 128268/400000 [00:15<00:31, 8654.52it/s] 32%|███▏      | 129134/400000 [00:15<00:31, 8648.61it/s] 32%|███▎      | 130000/400000 [00:15<00:31, 8650.15it/s] 33%|███▎      | 130866/400000 [00:15<00:31, 8647.19it/s] 33%|███▎      | 131731/400000 [00:16<00:31, 8634.08it/s] 33%|███▎      | 132595/400000 [00:16<00:31, 8501.67it/s] 33%|███▎      | 133460/400000 [00:16<00:31, 8543.97it/s] 34%|███▎      | 134322/400000 [00:16<00:31, 8565.21it/s] 34%|███▍      | 135185/400000 [00:16<00:30, 8584.54it/s] 34%|███▍      | 136044/400000 [00:16<00:30, 8554.76it/s] 34%|███▍      | 136911/400000 [00:16<00:30, 8587.11it/s] 34%|███▍      | 137777/400000 [00:16<00:30, 8607.62it/s] 35%|███▍      | 138645/400000 [00:16<00:30, 8627.09it/s] 35%|███▍      | 139510/400000 [00:16<00:30, 8631.23it/s] 35%|███▌      | 140374/400000 [00:17<00:30, 8590.95it/s] 35%|███▌      | 141242/400000 [00:17<00:30, 8615.01it/s] 36%|███▌      | 142104/400000 [00:17<00:29, 8614.79it/s] 36%|███▌      | 142967/400000 [00:17<00:29, 8618.38it/s] 36%|███▌      | 143830/400000 [00:17<00:29, 8619.84it/s] 36%|███▌      | 144693/400000 [00:17<00:29, 8607.94it/s] 36%|███▋      | 145559/400000 [00:17<00:29, 8620.83it/s] 37%|███▋      | 146422/400000 [00:17<00:29, 8610.67it/s] 37%|███▋      | 147290/400000 [00:17<00:29, 8628.79it/s] 37%|███▋      | 148155/400000 [00:17<00:29, 8634.20it/s] 37%|███▋      | 149019/400000 [00:18<00:29, 8622.89it/s] 37%|███▋      | 149887/400000 [00:18<00:28, 8638.91it/s] 38%|███▊      | 150751/400000 [00:18<00:28, 8632.36it/s] 38%|███▊      | 151618/400000 [00:18<00:28, 8641.24it/s] 38%|███▊      | 152483/400000 [00:18<00:28, 8603.38it/s] 38%|███▊      | 153344/400000 [00:18<00:28, 8566.31it/s] 39%|███▊      | 154215/400000 [00:18<00:28, 8608.77it/s] 39%|███▉      | 155086/400000 [00:18<00:28, 8636.82it/s] 39%|███▉      | 155953/400000 [00:18<00:28, 8645.47it/s] 39%|███▉      | 156821/400000 [00:18<00:28, 8653.24it/s] 39%|███▉      | 157687/400000 [00:19<00:28, 8616.88it/s] 40%|███▉      | 158559/400000 [00:19<00:27, 8646.18it/s] 40%|███▉      | 159427/400000 [00:19<00:27, 8655.79it/s] 40%|████      | 160293/400000 [00:19<00:27, 8650.49it/s] 40%|████      | 161159/400000 [00:19<00:27, 8646.84it/s] 41%|████      | 162024/400000 [00:19<00:27, 8613.65it/s] 41%|████      | 162893/400000 [00:19<00:27, 8635.31it/s] 41%|████      | 163761/400000 [00:19<00:27, 8647.09it/s] 41%|████      | 164626/400000 [00:19<00:27, 8630.64it/s] 41%|████▏     | 165490/400000 [00:19<00:27, 8631.20it/s] 42%|████▏     | 166354/400000 [00:20<00:27, 8589.55it/s] 42%|████▏     | 167220/400000 [00:20<00:27, 8609.64it/s] 42%|████▏     | 168091/400000 [00:20<00:26, 8636.80it/s] 42%|████▏     | 168958/400000 [00:20<00:26, 8645.74it/s] 42%|████▏     | 169826/400000 [00:20<00:26, 8654.64it/s] 43%|████▎     | 170692/400000 [00:20<00:26, 8612.56it/s] 43%|████▎     | 171555/400000 [00:20<00:26, 8617.02it/s] 43%|████▎     | 172425/400000 [00:20<00:26, 8640.98it/s] 43%|████▎     | 173294/400000 [00:20<00:26, 8655.39it/s] 44%|████▎     | 174164/400000 [00:20<00:26, 8666.05it/s] 44%|████▍     | 175031/400000 [00:21<00:26, 8611.84it/s] 44%|████▍     | 175898/400000 [00:21<00:25, 8628.46it/s] 44%|████▍     | 176767/400000 [00:21<00:25, 8644.84it/s] 44%|████▍     | 177632/400000 [00:21<00:25, 8617.74it/s] 45%|████▍     | 178494/400000 [00:21<00:25, 8548.96it/s] 45%|████▍     | 179350/400000 [00:21<00:25, 8506.82it/s] 45%|████▌     | 180222/400000 [00:21<00:25, 8567.40it/s] 45%|████▌     | 181089/400000 [00:21<00:25, 8596.07it/s] 45%|████▌     | 181949/400000 [00:21<00:25, 8589.80it/s] 46%|████▌     | 182816/400000 [00:21<00:25, 8611.36it/s] 46%|████▌     | 183678/400000 [00:22<00:25, 8503.44it/s] 46%|████▌     | 184546/400000 [00:22<00:25, 8555.57it/s] 46%|████▋     | 185414/400000 [00:22<00:24, 8591.66it/s] 47%|████▋     | 186274/400000 [00:22<00:24, 8587.72it/s] 47%|████▋     | 187137/400000 [00:22<00:24, 8599.00it/s] 47%|████▋     | 187998/400000 [00:22<00:24, 8553.99it/s] 47%|████▋     | 188863/400000 [00:22<00:24, 8580.08it/s] 47%|████▋     | 189727/400000 [00:22<00:24, 8597.16it/s] 48%|████▊     | 190587/400000 [00:22<00:24, 8592.72it/s] 48%|████▊     | 191453/400000 [00:22<00:24, 8610.84it/s] 48%|████▊     | 192315/400000 [00:23<00:24, 8578.40it/s] 48%|████▊     | 193182/400000 [00:23<00:24, 8605.38it/s] 49%|████▊     | 194045/400000 [00:23<00:23, 8611.10it/s] 49%|████▊     | 194907/400000 [00:23<00:23, 8610.52it/s] 49%|████▉     | 195769/400000 [00:23<00:23, 8599.26it/s] 49%|████▉     | 196629/400000 [00:23<00:23, 8552.75it/s] 49%|████▉     | 197485/400000 [00:23<00:23, 8529.08it/s] 50%|████▉     | 198338/400000 [00:23<00:23, 8501.12it/s] 50%|████▉     | 199189/400000 [00:23<00:23, 8414.90it/s] 50%|█████     | 200057/400000 [00:23<00:23, 8490.89it/s] 50%|█████     | 200917/400000 [00:24<00:23, 8521.92it/s] 50%|█████     | 201780/400000 [00:24<00:23, 8553.55it/s] 51%|█████     | 202649/400000 [00:24<00:22, 8592.57it/s] 51%|█████     | 203518/400000 [00:24<00:22, 8619.47it/s] 51%|█████     | 204383/400000 [00:24<00:22, 8626.19it/s] 51%|█████▏    | 205248/400000 [00:24<00:22, 8630.88it/s] 52%|█████▏    | 206112/400000 [00:24<00:22, 8577.49it/s] 52%|█████▏    | 206974/400000 [00:24<00:22, 8588.61it/s] 52%|█████▏    | 207836/400000 [00:24<00:22, 8597.68it/s] 52%|█████▏    | 208705/400000 [00:24<00:22, 8623.91it/s] 52%|█████▏    | 209575/400000 [00:25<00:22, 8644.97it/s] 53%|█████▎    | 210440/400000 [00:25<00:22, 8601.69it/s] 53%|█████▎    | 211308/400000 [00:25<00:21, 8622.28it/s] 53%|█████▎    | 212175/400000 [00:25<00:21, 8636.06it/s] 53%|█████▎    | 213039/400000 [00:25<00:21, 8635.71it/s] 53%|█████▎    | 213904/400000 [00:25<00:21, 8637.36it/s] 54%|█████▎    | 214768/400000 [00:25<00:21, 8580.31it/s] 54%|█████▍    | 215630/400000 [00:25<00:21, 8590.65it/s] 54%|█████▍    | 216491/400000 [00:25<00:21, 8595.16it/s] 54%|█████▍    | 217351/400000 [00:25<00:21, 8587.51it/s] 55%|█████▍    | 218220/400000 [00:26<00:21, 8616.19it/s] 55%|█████▍    | 219082/400000 [00:26<00:21, 8347.90it/s] 55%|█████▍    | 219950/400000 [00:26<00:21, 8443.85it/s] 55%|█████▌    | 220806/400000 [00:26<00:21, 8475.61it/s] 55%|█████▌    | 221657/400000 [00:26<00:21, 8483.55it/s] 56%|█████▌    | 222520/400000 [00:26<00:20, 8526.61it/s] 56%|█████▌    | 223376/400000 [00:26<00:20, 8536.03it/s] 56%|█████▌    | 224244/400000 [00:26<00:20, 8576.62it/s] 56%|█████▋    | 225113/400000 [00:26<00:20, 8607.63it/s] 56%|█████▋    | 225975/400000 [00:26<00:20, 8609.47it/s] 57%|█████▋    | 226846/400000 [00:27<00:20, 8638.02it/s] 57%|█████▋    | 227710/400000 [00:27<00:19, 8634.02it/s] 57%|█████▋    | 228576/400000 [00:27<00:19, 8640.59it/s] 57%|█████▋    | 229441/400000 [00:27<00:19, 8632.26it/s] 58%|█████▊    | 230305/400000 [00:27<00:19, 8624.41it/s] 58%|█████▊    | 231174/400000 [00:27<00:19, 8641.98it/s] 58%|█████▊    | 232039/400000 [00:27<00:19, 8611.61it/s] 58%|█████▊    | 232901/400000 [00:27<00:19, 8601.31it/s] 58%|█████▊    | 233762/400000 [00:27<00:19, 8547.16it/s] 59%|█████▊    | 234625/400000 [00:28<00:19, 8571.70it/s] 59%|█████▉    | 235497/400000 [00:28<00:19, 8614.38it/s] 59%|█████▉    | 236361/400000 [00:28<00:18, 8619.20it/s] 59%|█████▉    | 237231/400000 [00:28<00:18, 8641.08it/s] 60%|█████▉    | 238097/400000 [00:28<00:18, 8645.61it/s] 60%|█████▉    | 238962/400000 [00:28<00:18, 8559.17it/s] 60%|█████▉    | 239829/400000 [00:28<00:18, 8589.98it/s] 60%|██████    | 240689/400000 [00:28<00:18, 8589.26it/s] 60%|██████    | 241549/400000 [00:28<00:18, 8580.41it/s] 61%|██████    | 242416/400000 [00:28<00:18, 8607.00it/s] 61%|██████    | 243279/400000 [00:29<00:18, 8613.16it/s] 61%|██████    | 244149/400000 [00:29<00:18, 8637.26it/s] 61%|██████▏   | 245013/400000 [00:29<00:17, 8632.74it/s] 61%|██████▏   | 245877/400000 [00:29<00:17, 8565.06it/s] 62%|██████▏   | 246735/400000 [00:29<00:17, 8569.51it/s] 62%|██████▏   | 247593/400000 [00:29<00:17, 8570.59it/s] 62%|██████▏   | 248457/400000 [00:29<00:17, 8589.40it/s] 62%|██████▏   | 249317/400000 [00:29<00:17, 8590.66it/s] 63%|██████▎   | 250183/400000 [00:29<00:17, 8609.62it/s] 63%|██████▎   | 251045/400000 [00:29<00:17, 8598.37it/s] 63%|██████▎   | 251905/400000 [00:30<00:17, 8553.33it/s] 63%|██████▎   | 252769/400000 [00:30<00:17, 8577.53it/s] 63%|██████▎   | 253627/400000 [00:30<00:17, 8546.61it/s] 64%|██████▎   | 254495/400000 [00:30<00:16, 8583.09it/s] 64%|██████▍   | 255360/400000 [00:30<00:16, 8602.88it/s] 64%|██████▍   | 256223/400000 [00:30<00:16, 8608.80it/s] 64%|██████▍   | 257092/400000 [00:30<00:16, 8631.46it/s] 64%|██████▍   | 257956/400000 [00:30<00:16, 8610.22it/s] 65%|██████▍   | 258818/400000 [00:30<00:16, 8589.23it/s] 65%|██████▍   | 259684/400000 [00:30<00:16, 8608.98it/s] 65%|██████▌   | 260553/400000 [00:31<00:16, 8632.66it/s] 65%|██████▌   | 261417/400000 [00:31<00:16, 8625.12it/s] 66%|██████▌   | 262280/400000 [00:31<00:16, 8604.59it/s] 66%|██████▌   | 263141/400000 [00:31<00:15, 8588.87it/s] 66%|██████▌   | 264009/400000 [00:31<00:15, 8615.36it/s] 66%|██████▌   | 264871/400000 [00:31<00:15, 8614.35it/s] 66%|██████▋   | 265733/400000 [00:31<00:15, 8615.89it/s] 67%|██████▋   | 266595/400000 [00:31<00:15, 8600.49it/s] 67%|██████▋   | 267456/400000 [00:31<00:15, 8583.11it/s] 67%|██████▋   | 268323/400000 [00:31<00:15, 8606.32it/s] 67%|██████▋   | 269191/400000 [00:32<00:15, 8625.54it/s] 68%|██████▊   | 270058/400000 [00:32<00:15, 8636.83it/s] 68%|██████▊   | 270922/400000 [00:32<00:14, 8637.05it/s] 68%|██████▊   | 271786/400000 [00:32<00:14, 8556.87it/s] 68%|██████▊   | 272651/400000 [00:32<00:14, 8582.19it/s] 68%|██████▊   | 273517/400000 [00:32<00:14, 8604.20it/s] 69%|██████▊   | 274378/400000 [00:32<00:14, 8575.60it/s] 69%|██████▉   | 275236/400000 [00:32<00:14, 8484.96it/s] 69%|██████▉   | 276085/400000 [00:32<00:14, 8452.78it/s] 69%|██████▉   | 276952/400000 [00:32<00:14, 8515.15it/s] 69%|██████▉   | 277819/400000 [00:33<00:14, 8559.10it/s] 70%|██████▉   | 278687/400000 [00:33<00:14, 8593.10it/s] 70%|██████▉   | 279547/400000 [00:33<00:14, 8587.88it/s] 70%|███████   | 280406/400000 [00:33<00:14, 8527.56it/s] 70%|███████   | 281273/400000 [00:33<00:13, 8567.71it/s] 71%|███████   | 282133/400000 [00:33<00:13, 8575.13it/s] 71%|███████   | 282997/400000 [00:33<00:13, 8586.31it/s] 71%|███████   | 283863/400000 [00:33<00:13, 8606.08it/s] 71%|███████   | 284724/400000 [00:33<00:13, 8594.69it/s] 71%|███████▏  | 285584/400000 [00:33<00:13, 8595.05it/s] 72%|███████▏  | 286449/400000 [00:34<00:13, 8610.45it/s] 72%|███████▏  | 287311/400000 [00:34<00:13, 8592.57it/s] 72%|███████▏  | 288179/400000 [00:34<00:12, 8617.67it/s] 72%|███████▏  | 289041/400000 [00:34<00:12, 8582.08it/s] 72%|███████▏  | 289904/400000 [00:34<00:12, 8593.60it/s] 73%|███████▎  | 290767/400000 [00:34<00:12, 8604.14it/s] 73%|███████▎  | 291632/400000 [00:34<00:12, 8616.21it/s] 73%|███████▎  | 292497/400000 [00:34<00:12, 8624.39it/s] 73%|███████▎  | 293360/400000 [00:34<00:12, 8469.25it/s] 74%|███████▎  | 294228/400000 [00:34<00:12, 8529.66it/s] 74%|███████▍  | 295084/400000 [00:35<00:12, 8536.83it/s] 74%|███████▍  | 295952/400000 [00:35<00:12, 8577.77it/s] 74%|███████▍  | 296818/400000 [00:35<00:11, 8599.44it/s] 74%|███████▍  | 297679/400000 [00:35<00:11, 8536.14it/s] 75%|███████▍  | 298545/400000 [00:35<00:11, 8570.31it/s] 75%|███████▍  | 299410/400000 [00:35<00:11, 8592.81it/s] 75%|███████▌  | 300279/400000 [00:35<00:11, 8620.75it/s] 75%|███████▌  | 301142/400000 [00:35<00:11, 8622.58it/s] 76%|███████▌  | 302005/400000 [00:35<00:11, 8582.05it/s] 76%|███████▌  | 302872/400000 [00:35<00:11, 8606.86it/s] 76%|███████▌  | 303733/400000 [00:36<00:11, 8509.83it/s] 76%|███████▌  | 304585/400000 [00:36<00:11, 8370.43it/s] 76%|███████▋  | 305446/400000 [00:36<00:11, 8440.65it/s] 77%|███████▋  | 306296/400000 [00:36<00:11, 8458.16it/s] 77%|███████▋  | 307162/400000 [00:36<00:10, 8515.28it/s] 77%|███████▋  | 308025/400000 [00:36<00:10, 8547.96it/s] 77%|███████▋  | 308894/400000 [00:36<00:10, 8588.61it/s] 77%|███████▋  | 309757/400000 [00:36<00:10, 8599.59it/s] 78%|███████▊  | 310618/400000 [00:36<00:10, 8530.65it/s] 78%|███████▊  | 311478/400000 [00:36<00:10, 8548.45it/s] 78%|███████▊  | 312344/400000 [00:37<00:10, 8579.69it/s] 78%|███████▊  | 313203/400000 [00:37<00:10, 8573.47it/s] 79%|███████▊  | 314070/400000 [00:37<00:09, 8600.64it/s] 79%|███████▊  | 314931/400000 [00:37<00:09, 8573.76it/s] 79%|███████▉  | 315793/400000 [00:37<00:09, 8585.88it/s] 79%|███████▉  | 316652/400000 [00:37<00:09, 8509.02it/s] 79%|███████▉  | 317513/400000 [00:37<00:09, 8536.23it/s] 80%|███████▉  | 318367/400000 [00:37<00:09, 8506.43it/s] 80%|███████▉  | 319218/400000 [00:37<00:09, 8413.89it/s] 80%|████████  | 320077/400000 [00:37<00:09, 8464.85it/s] 80%|████████  | 320924/400000 [00:38<00:09, 8392.24it/s] 80%|████████  | 321774/400000 [00:38<00:09, 8423.74it/s] 81%|████████  | 322636/400000 [00:38<00:09, 8481.57it/s] 81%|████████  | 323485/400000 [00:38<00:09, 8451.96it/s] 81%|████████  | 324350/400000 [00:38<00:08, 8510.19it/s] 81%|████████▏ | 325213/400000 [00:38<00:08, 8544.02it/s] 82%|████████▏ | 326079/400000 [00:38<00:08, 8576.93it/s] 82%|████████▏ | 326942/400000 [00:38<00:08, 8591.12it/s] 82%|████████▏ | 327802/400000 [00:38<00:08, 8564.75it/s] 82%|████████▏ | 328666/400000 [00:38<00:08, 8586.20it/s] 82%|████████▏ | 329532/400000 [00:39<00:08, 8606.09it/s] 83%|████████▎ | 330396/400000 [00:39<00:08, 8615.49it/s] 83%|████████▎ | 331258/400000 [00:39<00:07, 8610.93it/s] 83%|████████▎ | 332120/400000 [00:39<00:07, 8559.18it/s] 83%|████████▎ | 332977/400000 [00:39<00:07, 8558.18it/s] 83%|████████▎ | 333837/400000 [00:39<00:07, 8569.60it/s] 84%|████████▎ | 334700/400000 [00:39<00:07, 8586.22it/s] 84%|████████▍ | 335560/400000 [00:39<00:07, 8589.83it/s] 84%|████████▍ | 336420/400000 [00:39<00:07, 8525.02it/s] 84%|████████▍ | 337285/400000 [00:39<00:07, 8559.68it/s] 85%|████████▍ | 338142/400000 [00:40<00:07, 8514.85it/s] 85%|████████▍ | 338994/400000 [00:40<00:07, 8509.20it/s] 85%|████████▍ | 339846/400000 [00:40<00:07, 8433.79it/s] 85%|████████▌ | 340690/400000 [00:40<00:07, 8434.84it/s] 85%|████████▌ | 341543/400000 [00:40<00:06, 8461.76it/s] 86%|████████▌ | 342390/400000 [00:40<00:06, 8449.96it/s] 86%|████████▌ | 343257/400000 [00:40<00:06, 8512.63it/s] 86%|████████▌ | 344109/400000 [00:40<00:06, 8498.18it/s] 86%|████████▌ | 344959/400000 [00:40<00:06, 8480.92it/s] 86%|████████▋ | 345816/400000 [00:40<00:06, 8506.21it/s] 87%|████████▋ | 346675/400000 [00:41<00:06, 8530.66it/s] 87%|████████▋ | 347535/400000 [00:41<00:06, 8548.74it/s] 87%|████████▋ | 348392/400000 [00:41<00:06, 8552.61it/s] 87%|████████▋ | 349248/400000 [00:41<00:05, 8486.33it/s] 88%|████████▊ | 350108/400000 [00:41<00:05, 8518.23it/s] 88%|████████▊ | 350961/400000 [00:41<00:05, 8520.43it/s] 88%|████████▊ | 351822/400000 [00:41<00:05, 8547.05it/s] 88%|████████▊ | 352677/400000 [00:41<00:05, 8515.16it/s] 88%|████████▊ | 353531/400000 [00:41<00:05, 8522.33it/s] 89%|████████▊ | 354395/400000 [00:41<00:05, 8554.95it/s] 89%|████████▉ | 355256/400000 [00:42<00:05, 8570.82it/s] 89%|████████▉ | 356119/400000 [00:42<00:05, 8588.35it/s] 89%|████████▉ | 356978/400000 [00:42<00:05, 8588.59it/s] 89%|████████▉ | 357837/400000 [00:42<00:04, 8584.80it/s] 90%|████████▉ | 358698/400000 [00:42<00:04, 8591.51it/s] 90%|████████▉ | 359562/400000 [00:42<00:04, 8604.75it/s] 90%|█████████ | 360428/400000 [00:42<00:04, 8621.24it/s] 90%|█████████ | 361291/400000 [00:42<00:04, 8610.69it/s] 91%|█████████ | 362153/400000 [00:42<00:04, 8583.25it/s] 91%|█████████ | 363014/400000 [00:42<00:04, 8588.92it/s] 91%|█████████ | 363879/400000 [00:43<00:04, 8604.96it/s] 91%|█████████ | 364741/400000 [00:43<00:04, 8609.35it/s] 91%|█████████▏| 365602/400000 [00:43<00:04, 8585.08it/s] 92%|█████████▏| 366461/400000 [00:43<00:03, 8443.94it/s] 92%|█████████▏| 367321/400000 [00:43<00:03, 8489.88it/s] 92%|█████████▏| 368186/400000 [00:43<00:03, 8536.59it/s] 92%|█████████▏| 369050/400000 [00:43<00:03, 8565.45it/s] 92%|█████████▏| 369913/400000 [00:43<00:03, 8583.62it/s] 93%|█████████▎| 370778/400000 [00:43<00:03, 8602.18it/s] 93%|█████████▎| 371639/400000 [00:44<00:03, 8564.99it/s] 93%|█████████▎| 372496/400000 [00:44<00:03, 8565.61it/s] 93%|█████████▎| 373362/400000 [00:44<00:03, 8592.21it/s] 94%|█████████▎| 374226/400000 [00:44<00:02, 8604.03it/s] 94%|█████████▍| 375089/400000 [00:44<00:02, 8609.85it/s] 94%|█████████▍| 375951/400000 [00:44<00:02, 8570.95it/s] 94%|█████████▍| 376817/400000 [00:44<00:02, 8594.66it/s] 94%|█████████▍| 377682/400000 [00:44<00:02, 8608.83it/s] 95%|█████████▍| 378543/400000 [00:44<00:02, 8596.62it/s] 95%|█████████▍| 379404/400000 [00:44<00:02, 8600.26it/s] 95%|█████████▌| 380265/400000 [00:45<00:02, 8509.05it/s] 95%|█████████▌| 381127/400000 [00:45<00:02, 8539.31it/s] 95%|█████████▌| 381987/400000 [00:45<00:02, 8555.88it/s] 96%|█████████▌| 382851/400000 [00:45<00:01, 8579.21it/s] 96%|█████████▌| 383715/400000 [00:45<00:01, 8596.43it/s] 96%|█████████▌| 384575/400000 [00:45<00:01, 8579.10it/s] 96%|█████████▋| 385436/400000 [00:45<00:01, 8587.98it/s] 97%|█████████▋| 386295/400000 [00:45<00:01, 8552.41it/s] 97%|█████████▋| 387158/400000 [00:45<00:01, 8575.17it/s] 97%|█████████▋| 388021/400000 [00:45<00:01, 8589.17it/s] 97%|█████████▋| 388880/400000 [00:46<00:01, 8446.05it/s] 97%|█████████▋| 389742/400000 [00:46<00:01, 8495.32it/s] 98%|█████████▊| 390607/400000 [00:46<00:01, 8538.61it/s] 98%|█████████▊| 391468/400000 [00:46<00:00, 8557.78it/s] 98%|█████████▊| 392326/400000 [00:46<00:00, 8561.75it/s] 98%|█████████▊| 393183/400000 [00:46<00:00, 8542.50it/s] 99%|█████████▊| 394038/400000 [00:46<00:00, 8544.63it/s] 99%|█████████▊| 394904/400000 [00:46<00:00, 8576.79it/s] 99%|█████████▉| 395762/400000 [00:46<00:00, 8569.39it/s] 99%|█████████▉| 396626/400000 [00:46<00:00, 8588.95it/s] 99%|█████████▉| 397485/400000 [00:47<00:00, 8584.39it/s]100%|█████████▉| 398349/400000 [00:47<00:00, 8599.26it/s]100%|█████████▉| 399211/400000 [00:47<00:00, 8602.93it/s]100%|█████████▉| 399999/400000 [00:47<00:00, 8454.32it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7ff494a91518>, <torchtext.data.dataset.TabularDataset object at 0x7ff494a913c8>, <torchtext.vocab.Vocab object at 0x7ff494a914a8>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 28.77 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 47.97 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 24.41 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 24.41 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.20 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.20 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.05 file/s]2020-07-24 17:50:21.000015: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-24 17:50:21.004071: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-07-24 17:50:21.004557: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a04787f4d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-24 17:50:21.004866: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['cifar10', 'test', 'fashion-mnist_small.npy', 'mnist2', 'mnist_dataset_small.npy', 'train'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:25, 115333.45it/s] 76%|███████▋  | 7577600/9912422 [00:00<00:14, 164654.29it/s]9920512it [00:00, 37820209.46it/s]                           
0it [00:00, ?it/s]32768it [00:00, 393606.01it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 467296.12it/s]1654784it [00:00, 11794849.09it/s]                         
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 74027.55it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
