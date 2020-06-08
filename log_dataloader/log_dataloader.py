
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/a52f98f683706b2028e624558aa050dda1a05090', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': 'a52f98f683706b2028e624558aa050dda1a05090', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/a52f98f683706b2028e624558aa050dda1a05090

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/a52f98f683706b2028e624558aa050dda1a05090

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/a52f98f683706b2028e624558aa050dda1a05090

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f7cc6d07620> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f7cc6d07620> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f7d322ce0d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f7d322ce0d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f7d4c01fea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f7d4c01fea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7cdfb48950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7cdfb48950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f7cdfb48950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:01, 159938.08it/s] 66%|██████▋   | 6569984/9912422 [00:00<00:14, 228243.61it/s]9920512it [00:00, 42727611.11it/s]                           
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 277042.16it/s]           
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 465949.51it/s]1654784it [00:00, 11960439.82it/s]                         
0it [00:00, ?it/s]8192it [00:00, 161918.43it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7cdcf68d30>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7cdcf619e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f7cdfb48598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f7cdfb48598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f7cdfb48598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:54,  2.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:54,  2.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:53,  2.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:53,  2.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:53,  2.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:52,  2.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:52,  2.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:52,  2.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:37,  4.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:37,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:36,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:36,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:36,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:36,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:35,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:35,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:35,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:35,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:24,  5.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:23,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:23,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:23,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:23,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:00<00:16,  8.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:16,  8.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:16,  8.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:16,  8.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:16,  8.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:16,  8.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:16,  8.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:15,  8.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:15,  8.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:15,  8.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:15,  8.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:00<00:11, 11.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:11, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:11, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:11, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:10, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:10, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:10, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:10, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:10, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:10, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:10, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:00<00:07, 15.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:07, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:07, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:07, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:07, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:07, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:07, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:00<00:07, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:00<00:07, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:00<00:07, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:00<00:06, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:00<00:05, 20.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:00<00:05, 20.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:00<00:05, 20.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:00<00:05, 20.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 20.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 20.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 20.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 20.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 20.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 20.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 26.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 26.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 26.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 26.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 26.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 26.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 26.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 26.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 26.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 26.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 33.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 33.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 33.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 33.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 33.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 33.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 33.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 33.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 33.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 33.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 40.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 40.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 40.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 40.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 40.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
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

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 48.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 48.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 48.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 48.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 48.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 48.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 48.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 48.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 48.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 48.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 48.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 56.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 56.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 56.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 56.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 56.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 56.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 56.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 56.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 56.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 56.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 56.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 64.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 64.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 64.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 64.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 64.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 64.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 64.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 64.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 64.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 64.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 64.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 71.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 71.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 71.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 71.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 71.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 71.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 71.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 71.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 71.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 71.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 71.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 77.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 77.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 77.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 77.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 77.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 77.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 77.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 77.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 77.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 77.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 77.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 81.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 81.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 81.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:01<00:00, 81.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:01<00:00, 81.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:01<00:00, 81.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:01<00:00, 81.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:01<00:00, 81.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:01<00:00, 81.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 81.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 81.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 84.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 84.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 84.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 84.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 84.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 84.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 84.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 84.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 84.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 84.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.12s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.12s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 84.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.12s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 84.01 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.19s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.12s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 84.01 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.19s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.19s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 38.68 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.19s/ url]
0 examples [00:00, ? examples/s]2020-06-08 05:53:20.469288: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-08 05:53:20.486431: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-08 05:53:20.487199: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555d13b61eb0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-08 05:53:20.487221: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
8 examples [00:00, 79.88 examples/s]108 examples [00:00, 110.31 examples/s]208 examples [00:00, 150.25 examples/s]296 examples [00:00, 200.00 examples/s]403 examples [00:00, 264.41 examples/s]512 examples [00:00, 342.09 examples/s]616 examples [00:00, 427.99 examples/s]722 examples [00:00, 520.63 examples/s]831 examples [00:00, 616.99 examples/s]935 examples [00:01, 701.87 examples/s]1037 examples [00:01, 773.50 examples/s]1145 examples [00:01, 843.85 examples/s]1248 examples [00:01, 889.33 examples/s]1351 examples [00:01, 915.03 examples/s]1453 examples [00:01, 919.77 examples/s]1557 examples [00:01, 951.74 examples/s]1658 examples [00:01, 967.83 examples/s]1759 examples [00:01, 972.53 examples/s]1859 examples [00:01, 967.30 examples/s]1962 examples [00:02, 981.99 examples/s]2062 examples [00:02, 986.01 examples/s]2167 examples [00:02, 1003.98 examples/s]2273 examples [00:02, 1019.93 examples/s]2376 examples [00:02, 1018.30 examples/s]2481 examples [00:02, 1024.99 examples/s]2584 examples [00:02, 1009.26 examples/s]2686 examples [00:02, 1010.02 examples/s]2788 examples [00:02, 989.26 examples/s] 2894 examples [00:02, 1006.96 examples/s]3000 examples [00:03, 1022.17 examples/s]3110 examples [00:03, 1043.16 examples/s]3215 examples [00:03, 1040.36 examples/s]3320 examples [00:03, 1035.92 examples/s]3424 examples [00:03, 1007.49 examples/s]3526 examples [00:03, 1006.63 examples/s]3627 examples [00:03, 995.72 examples/s] 3727 examples [00:03, 971.16 examples/s]3834 examples [00:03, 996.66 examples/s]3941 examples [00:03, 1014.39 examples/s]4043 examples [00:04, 1003.94 examples/s]4144 examples [00:04, 990.06 examples/s] 4249 examples [00:04, 1004.77 examples/s]4351 examples [00:04, 1008.57 examples/s]4453 examples [00:04, 996.18 examples/s] 4553 examples [00:04, 992.33 examples/s]4663 examples [00:04, 1021.95 examples/s]4767 examples [00:04, 1025.98 examples/s]4870 examples [00:04, 1008.28 examples/s]4972 examples [00:05, 1001.60 examples/s]5076 examples [00:05, 1012.11 examples/s]5178 examples [00:05, 1012.07 examples/s]5281 examples [00:05, 1015.13 examples/s]5387 examples [00:05, 1025.28 examples/s]5490 examples [00:05, 985.57 examples/s] 5589 examples [00:05, 958.11 examples/s]5686 examples [00:05, 955.52 examples/s]5786 examples [00:05, 966.55 examples/s]5887 examples [00:05, 977.38 examples/s]5985 examples [00:06, 970.54 examples/s]6085 examples [00:06, 977.52 examples/s]6184 examples [00:06, 978.85 examples/s]6285 examples [00:06, 986.28 examples/s]6388 examples [00:06, 998.03 examples/s]6488 examples [00:06, 992.32 examples/s]6588 examples [00:06, 989.78 examples/s]6688 examples [00:06, 964.45 examples/s]6802 examples [00:06, 1009.58 examples/s]6909 examples [00:06, 1024.63 examples/s]7012 examples [00:07, 1002.35 examples/s]7120 examples [00:07, 1022.34 examples/s]7229 examples [00:07, 1038.98 examples/s]7341 examples [00:07, 1061.23 examples/s]7453 examples [00:07, 1076.89 examples/s]7562 examples [00:07, 1059.82 examples/s]7677 examples [00:07, 1085.20 examples/s]7788 examples [00:07, 1090.47 examples/s]7904 examples [00:07, 1108.41 examples/s]8021 examples [00:07, 1124.05 examples/s]8134 examples [00:08, 1105.25 examples/s]8250 examples [00:08, 1120.29 examples/s]8366 examples [00:08, 1129.73 examples/s]8480 examples [00:08, 1124.83 examples/s]8593 examples [00:08, 1090.74 examples/s]8703 examples [00:08, 1059.97 examples/s]8810 examples [00:08, 1013.77 examples/s]8913 examples [00:08, 999.98 examples/s] 9015 examples [00:08, 1003.48 examples/s]9116 examples [00:09, 1003.58 examples/s]9217 examples [00:09, 985.21 examples/s] 9316 examples [00:09, 983.46 examples/s]9424 examples [00:09, 1008.23 examples/s]9530 examples [00:09, 1020.56 examples/s]9633 examples [00:09, 1022.80 examples/s]9736 examples [00:09, 1008.37 examples/s]9841 examples [00:09, 1018.92 examples/s]9944 examples [00:09, 1009.49 examples/s]10046 examples [00:09, 940.12 examples/s]10142 examples [00:10, 940.41 examples/s]10237 examples [00:10, 936.13 examples/s]10338 examples [00:10, 957.10 examples/s]10435 examples [00:10, 955.49 examples/s]10531 examples [00:10, 955.24 examples/s]10629 examples [00:10, 961.82 examples/s]10726 examples [00:10, 950.07 examples/s]10831 examples [00:10, 976.31 examples/s]10934 examples [00:10, 989.99 examples/s]11037 examples [00:10, 999.79 examples/s]11138 examples [00:11, 1002.54 examples/s]11239 examples [00:11, 996.76 examples/s] 11339 examples [00:11, 996.30 examples/s]11439 examples [00:11, 994.89 examples/s]11539 examples [00:11, 996.10 examples/s]11639 examples [00:11, 985.64 examples/s]11738 examples [00:11, 984.91 examples/s]11841 examples [00:11, 996.23 examples/s]11943 examples [00:11, 1002.64 examples/s]12045 examples [00:12, 1006.33 examples/s]12146 examples [00:12, 982.53 examples/s] 12245 examples [00:12, 970.95 examples/s]12343 examples [00:12, 973.20 examples/s]12443 examples [00:12, 981.03 examples/s]12549 examples [00:12, 1002.23 examples/s]12655 examples [00:12, 1017.68 examples/s]12760 examples [00:12, 1024.48 examples/s]12863 examples [00:12, 1007.72 examples/s]12970 examples [00:12, 1023.93 examples/s]13073 examples [00:13, 1019.78 examples/s]13177 examples [00:13, 1023.91 examples/s]13283 examples [00:13, 1034.28 examples/s]13390 examples [00:13, 1043.67 examples/s]13496 examples [00:13, 1048.51 examples/s]13610 examples [00:13, 1073.00 examples/s]13718 examples [00:13, 1070.97 examples/s]13826 examples [00:13, 1059.28 examples/s]13937 examples [00:13, 1072.93 examples/s]14056 examples [00:13, 1104.90 examples/s]14173 examples [00:14, 1122.83 examples/s]14286 examples [00:14, 1103.80 examples/s]14397 examples [00:14, 1088.21 examples/s]14515 examples [00:14, 1113.36 examples/s]14628 examples [00:14, 1118.25 examples/s]14741 examples [00:14, 1095.93 examples/s]14851 examples [00:14, 1095.20 examples/s]14961 examples [00:14, 1080.01 examples/s]15073 examples [00:14, 1088.97 examples/s]15183 examples [00:14, 1062.77 examples/s]15297 examples [00:15, 1083.49 examples/s]15406 examples [00:15, 1071.84 examples/s]15520 examples [00:15, 1091.08 examples/s]15633 examples [00:15, 1099.61 examples/s]15744 examples [00:15, 1083.65 examples/s]15853 examples [00:15, 1077.95 examples/s]15961 examples [00:15, 1063.70 examples/s]16068 examples [00:15, 1047.90 examples/s]16179 examples [00:15, 1063.52 examples/s]16289 examples [00:15, 1071.84 examples/s]16402 examples [00:16, 1087.49 examples/s]16511 examples [00:16, 1080.77 examples/s]16622 examples [00:16, 1088.72 examples/s]16731 examples [00:16, 1067.63 examples/s]16845 examples [00:16, 1086.34 examples/s]16962 examples [00:16, 1109.35 examples/s]17076 examples [00:16, 1117.36 examples/s]17188 examples [00:16, 1114.65 examples/s]17303 examples [00:16, 1122.40 examples/s]17416 examples [00:17, 1122.06 examples/s]17529 examples [00:17, 1101.29 examples/s]17640 examples [00:17, 1080.06 examples/s]17749 examples [00:17, 1076.15 examples/s]17857 examples [00:17, 1075.94 examples/s]17970 examples [00:17, 1091.01 examples/s]18080 examples [00:17, 1080.58 examples/s]18189 examples [00:17, 1079.77 examples/s]18298 examples [00:17, 1054.67 examples/s]18414 examples [00:17, 1082.44 examples/s]18528 examples [00:18, 1097.89 examples/s]18645 examples [00:18, 1117.65 examples/s]18758 examples [00:18, 1110.22 examples/s]18873 examples [00:18, 1120.72 examples/s]18986 examples [00:18, 1099.32 examples/s]19098 examples [00:18, 1104.76 examples/s]19215 examples [00:18, 1122.87 examples/s]19328 examples [00:18, 1068.04 examples/s]19436 examples [00:18, 1059.94 examples/s]19551 examples [00:18, 1083.56 examples/s]19666 examples [00:19, 1101.54 examples/s]19780 examples [00:19, 1111.82 examples/s]19892 examples [00:19, 1103.02 examples/s]20003 examples [00:19, 1064.23 examples/s]20118 examples [00:19, 1088.03 examples/s]20233 examples [00:19, 1105.61 examples/s]20348 examples [00:19, 1116.36 examples/s]20461 examples [00:19, 1118.95 examples/s]20574 examples [00:19, 1089.60 examples/s]20684 examples [00:20, 1088.69 examples/s]20800 examples [00:20, 1107.98 examples/s]20914 examples [00:20, 1114.96 examples/s]21026 examples [00:20, 1112.69 examples/s]21138 examples [00:20, 1109.59 examples/s]21250 examples [00:20, 1088.59 examples/s]21360 examples [00:20, 1053.90 examples/s]21469 examples [00:20, 1060.69 examples/s]21576 examples [00:20, 1061.84 examples/s]21689 examples [00:20, 1081.21 examples/s]21798 examples [00:21, 1068.71 examples/s]21916 examples [00:21, 1097.40 examples/s]22027 examples [00:21, 1096.88 examples/s]22139 examples [00:21, 1101.17 examples/s]22250 examples [00:21, 1102.59 examples/s]22366 examples [00:21, 1116.57 examples/s]22480 examples [00:21, 1123.13 examples/s]22593 examples [00:21, 1106.56 examples/s]22704 examples [00:21, 1095.51 examples/s]22818 examples [00:21, 1105.82 examples/s]22933 examples [00:22, 1115.88 examples/s]23045 examples [00:22, 1104.95 examples/s]23156 examples [00:22, 1084.34 examples/s]23267 examples [00:22, 1089.43 examples/s]23381 examples [00:22, 1103.95 examples/s]23492 examples [00:22, 1102.69 examples/s]23603 examples [00:22, 1102.65 examples/s]23714 examples [00:22, 1100.75 examples/s]23825 examples [00:22, 1099.06 examples/s]23935 examples [00:22, 1076.28 examples/s]24050 examples [00:23, 1096.75 examples/s]24164 examples [00:23, 1109.34 examples/s]24278 examples [00:23, 1115.34 examples/s]24390 examples [00:23, 1114.22 examples/s]24505 examples [00:23, 1123.77 examples/s]24618 examples [00:23, 1100.53 examples/s]24729 examples [00:23, 1072.06 examples/s]24837 examples [00:23, 1063.82 examples/s]24949 examples [00:23, 1079.04 examples/s]25067 examples [00:23, 1104.89 examples/s]25179 examples [00:24, 1107.63 examples/s]25296 examples [00:24, 1123.19 examples/s]25409 examples [00:24, 1100.98 examples/s]25521 examples [00:24, 1105.82 examples/s]25636 examples [00:24, 1115.90 examples/s]25753 examples [00:24, 1129.53 examples/s]25869 examples [00:24, 1137.67 examples/s]25983 examples [00:24, 1130.22 examples/s]26097 examples [00:24, 1131.93 examples/s]26211 examples [00:25, 1112.23 examples/s]26323 examples [00:25, 1108.14 examples/s]26435 examples [00:25, 1109.64 examples/s]26547 examples [00:25, 1095.32 examples/s]26657 examples [00:25, 1094.24 examples/s]26767 examples [00:25, 1082.99 examples/s]26881 examples [00:25, 1098.69 examples/s]26998 examples [00:25, 1117.14 examples/s]27110 examples [00:25, 1103.82 examples/s]27221 examples [00:25, 1097.98 examples/s]27331 examples [00:26, 1060.51 examples/s]27444 examples [00:26, 1080.17 examples/s]27557 examples [00:26, 1091.37 examples/s]27667 examples [00:26, 1069.92 examples/s]27775 examples [00:26, 1051.59 examples/s]27881 examples [00:26, 1046.11 examples/s]27989 examples [00:26, 1053.95 examples/s]28095 examples [00:26, 1054.46 examples/s]28201 examples [00:26, 1029.86 examples/s]28307 examples [00:26, 1037.16 examples/s]28411 examples [00:27, 1033.86 examples/s]28515 examples [00:27, 1019.48 examples/s]28624 examples [00:27, 1038.91 examples/s]28730 examples [00:27, 1043.88 examples/s]28840 examples [00:27, 1059.26 examples/s]28954 examples [00:27, 1080.58 examples/s]29068 examples [00:27, 1095.49 examples/s]29178 examples [00:27, 1095.29 examples/s]29288 examples [00:27, 1086.68 examples/s]29397 examples [00:27, 1076.77 examples/s]29506 examples [00:28, 1079.03 examples/s]29616 examples [00:28, 1085.19 examples/s]29728 examples [00:28, 1093.93 examples/s]29838 examples [00:28, 1070.37 examples/s]29951 examples [00:28, 1086.37 examples/s]30060 examples [00:29, 390.35 examples/s] 30167 examples [00:29, 481.95 examples/s]30278 examples [00:29, 579.88 examples/s]30392 examples [00:29, 679.26 examples/s]30509 examples [00:29, 776.24 examples/s]30624 examples [00:29, 859.94 examples/s]30739 examples [00:29, 928.63 examples/s]30849 examples [00:29, 948.81 examples/s]30957 examples [00:30, 984.39 examples/s]31066 examples [00:30, 1012.25 examples/s]31174 examples [00:30, 1003.42 examples/s]31280 examples [00:30, 1017.24 examples/s]31385 examples [00:30, 1012.47 examples/s]31489 examples [00:30, 990.08 examples/s] 31592 examples [00:30, 1000.64 examples/s]31702 examples [00:30, 1025.95 examples/s]31809 examples [00:30, 1036.40 examples/s]31914 examples [00:30, 1017.00 examples/s]32017 examples [00:31, 1010.64 examples/s]32125 examples [00:31, 1029.34 examples/s]32239 examples [00:31, 1058.02 examples/s]32350 examples [00:31, 1071.65 examples/s]32458 examples [00:31, 1064.41 examples/s]32573 examples [00:31, 1088.05 examples/s]32683 examples [00:31, 1080.91 examples/s]32793 examples [00:31, 1086.29 examples/s]32908 examples [00:31, 1103.49 examples/s]33019 examples [00:31, 1092.00 examples/s]33134 examples [00:32, 1107.87 examples/s]33245 examples [00:32, 1070.83 examples/s]33353 examples [00:32, 1070.03 examples/s]33466 examples [00:32, 1082.68 examples/s]33575 examples [00:32, 1073.66 examples/s]33687 examples [00:32, 1086.45 examples/s]33806 examples [00:32, 1114.79 examples/s]33922 examples [00:32, 1127.87 examples/s]34036 examples [00:32, 1128.14 examples/s]34149 examples [00:32, 1098.96 examples/s]34260 examples [00:33, 1063.60 examples/s]34367 examples [00:33, 1061.86 examples/s]34474 examples [00:33, 1039.27 examples/s]34579 examples [00:33, 1020.89 examples/s]34682 examples [00:33, 994.85 examples/s] 34783 examples [00:33, 996.95 examples/s]34890 examples [00:33, 1017.73 examples/s]34995 examples [00:33, 1024.59 examples/s]35098 examples [00:33, 1023.32 examples/s]35201 examples [00:34, 1011.38 examples/s]35306 examples [00:34, 1021.77 examples/s]35410 examples [00:34, 1025.82 examples/s]35519 examples [00:34, 1043.03 examples/s]35627 examples [00:34, 1052.82 examples/s]35733 examples [00:34, 1012.09 examples/s]35845 examples [00:34, 1040.41 examples/s]35959 examples [00:34, 1066.09 examples/s]36072 examples [00:34, 1081.93 examples/s]36187 examples [00:34, 1100.90 examples/s]36301 examples [00:35, 1111.56 examples/s]36413 examples [00:35, 1106.70 examples/s]36526 examples [00:35, 1112.85 examples/s]36643 examples [00:35, 1126.77 examples/s]36756 examples [00:35, 1109.21 examples/s]36868 examples [00:35, 1085.54 examples/s]36977 examples [00:35, 1059.58 examples/s]37085 examples [00:35, 1063.28 examples/s]37196 examples [00:35, 1076.13 examples/s]37304 examples [00:35, 1069.64 examples/s]37414 examples [00:36, 1076.50 examples/s]37527 examples [00:36, 1089.95 examples/s]37638 examples [00:36, 1093.98 examples/s]37748 examples [00:36, 1085.97 examples/s]37857 examples [00:36, 1080.12 examples/s]37966 examples [00:36, 1080.96 examples/s]38078 examples [00:36, 1090.56 examples/s]38190 examples [00:36, 1097.42 examples/s]38300 examples [00:36, 1071.47 examples/s]38411 examples [00:37, 1082.24 examples/s]38520 examples [00:37, 1076.68 examples/s]38631 examples [00:37, 1084.66 examples/s]38740 examples [00:37, 1073.08 examples/s]38850 examples [00:37, 1080.97 examples/s]38959 examples [00:37, 1068.19 examples/s]39074 examples [00:37, 1089.40 examples/s]39188 examples [00:37, 1103.43 examples/s]39304 examples [00:37, 1119.41 examples/s]39421 examples [00:37, 1131.94 examples/s]39536 examples [00:38, 1134.58 examples/s]39650 examples [00:38, 1108.05 examples/s]39763 examples [00:38, 1112.41 examples/s]39875 examples [00:38, 1109.59 examples/s]39987 examples [00:38, 1094.51 examples/s]40097 examples [00:38, 1015.31 examples/s]40214 examples [00:38, 1055.49 examples/s]40328 examples [00:38, 1077.96 examples/s]40440 examples [00:38, 1089.55 examples/s]40550 examples [00:38, 1088.68 examples/s]40660 examples [00:39, 1064.04 examples/s]40773 examples [00:39, 1081.00 examples/s]40882 examples [00:39, 1077.39 examples/s]40991 examples [00:39, 1078.92 examples/s]41104 examples [00:39, 1092.66 examples/s]41214 examples [00:39, 1089.90 examples/s]41329 examples [00:39, 1106.92 examples/s]41445 examples [00:39, 1121.36 examples/s]41559 examples [00:39, 1126.52 examples/s]41676 examples [00:39, 1138.01 examples/s]41790 examples [00:40, 1115.15 examples/s]41902 examples [00:40, 1080.05 examples/s]42011 examples [00:40, 1076.66 examples/s]42125 examples [00:40, 1093.38 examples/s]42237 examples [00:40, 1100.34 examples/s]42348 examples [00:40, 1038.99 examples/s]42453 examples [00:40, 1042.25 examples/s]42561 examples [00:40, 1051.28 examples/s]42673 examples [00:40, 1068.00 examples/s]42785 examples [00:41, 1081.81 examples/s]42894 examples [00:41, 1066.81 examples/s]43007 examples [00:41, 1084.16 examples/s]43121 examples [00:41, 1097.21 examples/s]43240 examples [00:41, 1123.07 examples/s]43353 examples [00:41, 1094.01 examples/s]43463 examples [00:41, 1057.58 examples/s]43577 examples [00:41, 1080.85 examples/s]43693 examples [00:41, 1100.81 examples/s]43807 examples [00:41, 1111.08 examples/s]43920 examples [00:42, 1115.25 examples/s]44032 examples [00:42, 1095.60 examples/s]44142 examples [00:42, 1090.52 examples/s]44252 examples [00:42, 1073.67 examples/s]44360 examples [00:42, 1052.70 examples/s]44468 examples [00:42, 1058.29 examples/s]44574 examples [00:42, 1049.56 examples/s]44680 examples [00:42, 1035.85 examples/s]44784 examples [00:42, 1036.94 examples/s]44891 examples [00:42, 1046.17 examples/s]44996 examples [00:43, 1045.16 examples/s]45101 examples [00:43, 1042.74 examples/s]45210 examples [00:43, 1054.81 examples/s]45323 examples [00:43, 1076.21 examples/s]45431 examples [00:43, 1045.01 examples/s]45536 examples [00:43, 1019.41 examples/s]45644 examples [00:43, 1035.87 examples/s]45750 examples [00:43, 1041.02 examples/s]45855 examples [00:43, 1037.21 examples/s]45965 examples [00:44, 1054.26 examples/s]46071 examples [00:44, 1038.66 examples/s]46176 examples [00:44, 1007.93 examples/s]46283 examples [00:44, 1024.34 examples/s]46402 examples [00:44, 1068.84 examples/s]46513 examples [00:44, 1080.75 examples/s]46622 examples [00:44, 1039.95 examples/s]46728 examples [00:44, 1045.44 examples/s]46834 examples [00:44, 1046.63 examples/s]46940 examples [00:44, 1032.69 examples/s]47046 examples [00:45, 1039.79 examples/s]47152 examples [00:45, 1044.80 examples/s]47257 examples [00:45, 1040.29 examples/s]47362 examples [00:45, 1042.55 examples/s]47476 examples [00:45, 1069.00 examples/s]47584 examples [00:45, 1066.99 examples/s]47691 examples [00:45, 1053.53 examples/s]47797 examples [00:45, 1027.83 examples/s]47905 examples [00:45, 1042.19 examples/s]48014 examples [00:45, 1054.42 examples/s]48127 examples [00:46, 1075.83 examples/s]48238 examples [00:46, 1083.06 examples/s]48349 examples [00:46, 1088.68 examples/s]48460 examples [00:46, 1094.30 examples/s]48570 examples [00:46, 1090.63 examples/s]48680 examples [00:46, 1082.07 examples/s]48789 examples [00:46, 1078.79 examples/s]48897 examples [00:46, 1075.09 examples/s]49005 examples [00:46, 1073.18 examples/s]49115 examples [00:46, 1077.82 examples/s]49223 examples [00:47, 1060.01 examples/s]49330 examples [00:47, 1055.71 examples/s]49437 examples [00:47, 1059.92 examples/s]49547 examples [00:47, 1069.73 examples/s]49655 examples [00:47, 1049.00 examples/s]49761 examples [00:47, 1043.41 examples/s]49866 examples [00:47, 1037.54 examples/s]49977 examples [00:47, 1057.95 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 18%|█▊        | 9166/50000 [00:00<00:00, 91658.12 examples/s] 45%|████▌     | 22553/50000 [00:00<00:00, 101233.64 examples/s] 71%|███████   | 35383/50000 [00:00<00:00, 108072.62 examples/s] 95%|█████████▍| 47335/50000 [00:00<00:00, 111267.88 examples/s]                                                                0 examples [00:00, ? examples/s]83 examples [00:00, 824.24 examples/s]189 examples [00:00, 882.11 examples/s]294 examples [00:00, 925.67 examples/s]398 examples [00:00, 956.94 examples/s]500 examples [00:00, 972.88 examples/s]603 examples [00:00, 988.78 examples/s]711 examples [00:00, 1012.04 examples/s]817 examples [00:00, 1025.22 examples/s]918 examples [00:00, 1018.27 examples/s]1017 examples [00:01, 1001.17 examples/s]1126 examples [00:01, 1024.24 examples/s]1234 examples [00:01, 1039.52 examples/s]1342 examples [00:01, 1049.42 examples/s]1447 examples [00:01, 852.34 examples/s] 1553 examples [00:01, 904.71 examples/s]1661 examples [00:01, 950.66 examples/s]1770 examples [00:01, 987.22 examples/s]1881 examples [00:01, 1020.17 examples/s]1986 examples [00:01, 1022.98 examples/s]2101 examples [00:02, 1056.53 examples/s]2217 examples [00:02, 1084.87 examples/s]2331 examples [00:02, 1099.70 examples/s]2444 examples [00:02, 1107.90 examples/s]2556 examples [00:02, 1053.21 examples/s]2664 examples [00:02, 1061.02 examples/s]2771 examples [00:02, 1058.32 examples/s]2884 examples [00:02, 1075.74 examples/s]2995 examples [00:02, 1085.51 examples/s]3104 examples [00:03, 1058.44 examples/s]3220 examples [00:03, 1084.32 examples/s]3330 examples [00:03, 1088.38 examples/s]3440 examples [00:03, 1088.77 examples/s]3550 examples [00:03, 1078.49 examples/s]3659 examples [00:03, 1056.08 examples/s]3765 examples [00:03, 991.51 examples/s] 3866 examples [00:03, 940.92 examples/s]3964 examples [00:03, 949.73 examples/s]4061 examples [00:03, 953.36 examples/s]4159 examples [00:04, 961.19 examples/s]4259 examples [00:04, 970.00 examples/s]4357 examples [00:04, 963.34 examples/s]4454 examples [00:04, 927.91 examples/s]4555 examples [00:04, 950.18 examples/s]4664 examples [00:04, 986.47 examples/s]4775 examples [00:04, 1020.11 examples/s]4879 examples [00:04, 1023.98 examples/s]4986 examples [00:04, 1035.51 examples/s]5091 examples [00:05, 1035.86 examples/s]5198 examples [00:05, 1044.27 examples/s]5304 examples [00:05, 1046.92 examples/s]5409 examples [00:05, 1047.83 examples/s]5515 examples [00:05, 1051.00 examples/s]5621 examples [00:05, 1051.26 examples/s]5727 examples [00:05, 1006.09 examples/s]5837 examples [00:05, 1032.44 examples/s]5945 examples [00:05, 1044.29 examples/s]6050 examples [00:05, 1040.45 examples/s]6155 examples [00:06, 1034.76 examples/s]6259 examples [00:06, 1023.61 examples/s]6367 examples [00:06, 1037.68 examples/s]6473 examples [00:06, 1044.02 examples/s]6578 examples [00:06, 1045.13 examples/s]6683 examples [00:06, 1036.91 examples/s]6787 examples [00:06, 1033.69 examples/s]6892 examples [00:06, 1038.34 examples/s]7000 examples [00:06, 1047.84 examples/s]7105 examples [00:06, 1047.96 examples/s]7214 examples [00:07, 1057.51 examples/s]7320 examples [00:07, 1038.37 examples/s]7430 examples [00:07, 1055.34 examples/s]7538 examples [00:07, 1060.48 examples/s]7645 examples [00:07, 1062.41 examples/s]7752 examples [00:07, 1053.50 examples/s]7858 examples [00:07, 1012.63 examples/s]7960 examples [00:07, 1011.65 examples/s]8066 examples [00:07, 1024.23 examples/s]8169 examples [00:07, 1006.21 examples/s]8270 examples [00:08, 1006.46 examples/s]8371 examples [00:08, 999.05 examples/s] 8472 examples [00:08, 995.01 examples/s]8574 examples [00:08, 997.89 examples/s]8674 examples [00:08, 998.07 examples/s]8774 examples [00:08, 985.29 examples/s]8873 examples [00:08, 982.59 examples/s]8978 examples [00:08, 999.80 examples/s]9082 examples [00:08, 1010.84 examples/s]9193 examples [00:08, 1037.31 examples/s]9305 examples [00:09, 1060.24 examples/s]9412 examples [00:09, 1048.67 examples/s]9518 examples [00:09, 1014.82 examples/s]9620 examples [00:09, 1005.53 examples/s]9725 examples [00:09, 1017.82 examples/s]9828 examples [00:09, 1020.89 examples/s]9931 examples [00:09, 1013.75 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteKZJYQJ/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteKZJYQJ/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7cdfb48950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7cdfb48950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f7cdfb48950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7d2b536400>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7c68eed1d0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7cdfb48950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7cdfb48950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f7cdfb48950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7c68f35860>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7cdd09c0f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f7c630922f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f7c630922f0> 

  function with postional parmater data_info <function split_train_valid at 0x7f7c630922f0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f7c63092400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f7c63092400> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f7c63092400> , (data_info, **args) 
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:05<153:29:21, 1.56kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:05<107:40:26, 2.22kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:05<75:24:46, 3.17kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:05<52:45:41, 4.53kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:05<36:49:08, 6.48kB/s].vector_cache/glove.6B.zip:   1%|          | 8.45M/862M [00:05<25:37:50, 9.25kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.9M/862M [00:05<17:50:55, 13.2kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.1M/862M [00:06<12:26:01, 18.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.7M/862M [00:06<8:42:50, 26.9kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 23.7M/862M [00:06<6:03:25, 38.5kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.0M/862M [00:06<4:13:32, 54.9kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.7M/862M [00:06<2:56:33, 78.4kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.3M/862M [00:06<2:03:10, 112kB/s] .vector_cache/glove.6B.zip:   5%|▍         | 39.9M/862M [00:06<1:25:50, 160kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.8M/862M [00:06<59:54, 228kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 48.3M/862M [00:06<41:47, 325kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.3M/862M [00:07<29:45, 454kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:09<22:36, 594kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:09<18:06, 741kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.6M/862M [00:09<13:14, 1.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:09<09:22, 1.43MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:11<3:22:59, 65.8kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:11<2:23:44, 92.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.2M/862M [00:11<1:40:53, 132kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:11<1:10:34, 188kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:13<1:32:54, 143kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:13<1:06:38, 199kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.4M/862M [00:13<46:59, 282kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:15<35:28, 373kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:15<26:17, 503kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.7M/862M [00:15<18:44, 704kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:17<15:55, 826kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:17<14:05, 933kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:17<10:29, 1.25MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.6M/862M [00:17<07:33, 1.73MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:19<09:17, 1.41MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:19<07:49, 1.67MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:19<05:48, 2.25MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:21<07:07, 1.83MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:21<07:40, 1.69MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.3M/862M [00:21<05:57, 2.18MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:21<04:19, 3.00MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:23<10:39, 1.22MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:23<08:34, 1.51MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.4M/862M [00:23<06:14, 2.07MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:23<04:32, 2.83MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:25<55:04, 234kB/s] .vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:25<39:51, 323kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.5M/862M [00:25<28:10, 456kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:27<22:41, 564kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:27<17:11, 745kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.6M/862M [00:27<12:20, 1.03MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:29<11:37, 1.10MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:29<09:26, 1.35MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.8M/862M [00:29<06:55, 1.84MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:31<07:50, 1.62MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:31<08:07, 1.56MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:31<06:19, 2.00MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:31<04:32, 2.77MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:33<12:09:46, 17.3kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:33<8:31:53, 24.6kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:33<5:57:54, 35.1kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:35<4:12:45, 49.6kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:35<2:59:34, 69.8kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:35<2:06:14, 99.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:35<1:28:18, 141kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:37<1:07:23, 185kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:37<57:44, 216kB/s]  .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:39<41:19, 300kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:39<30:37, 405kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:39<21:45, 569kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:39<15:20, 804kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:41<25:47, 478kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:41<19:25, 634kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:41<14:30, 848kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:41<10:18, 1.19MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:42<11:32, 1.06MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:43<09:44, 1.26MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:43<07:10, 1.71MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:44<07:20, 1.66MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:45<08:30, 1.43MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:45<06:43, 1.81MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:45<04:54, 2.47MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:46<08:30, 1.42MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:47<07:37, 1.59MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:47<05:44, 2.11MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:48<06:18, 1.91MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:49<06:04, 1.98MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:49<04:38, 2.59MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:51<06:58, 1.72MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:52<14:42, 815kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:52<13:06, 914kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:52<09:44, 1.23MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:52<07:01, 1.70MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:53<08:16, 1.44MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:54<07:24, 1.61MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:54<05:35, 2.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:55<06:09, 1.92MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:56<05:56, 1.99MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:56<04:33, 2.59MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:57<05:25, 2.17MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:57<05:23, 2.18MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:58<04:07, 2.84MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:58<03:02, 3.86MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:59<18:56, 618kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:59<14:50, 789kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [01:00<10:41, 1.09MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:00<07:36, 1.53MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:01<32:07, 362kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:01<25:39, 453kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [01:02<18:38, 624kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:02<13:10, 880kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:03<13:27, 859kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:03<10:59, 1.05MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:04<08:04, 1.43MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:05<07:50, 1.47MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:05<08:36, 1.34MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:05<06:42, 1.71MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:06<04:53, 2.34MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:07<06:42, 1.71MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:07<07:46, 1.47MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:08<06:14, 1.83MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:08<04:31, 2.51MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:09<08:04, 1.41MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:09<07:12, 1.58MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:10<05:22, 2.11MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:11<05:55, 1.91MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:11<05:40, 1.99MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:12<04:17, 2.63MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:13<05:09, 2.17MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:13<06:39, 1.68MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:14<05:40, 1.97MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:14<04:08, 2.69MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:15<07:55, 1.41MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:15<07:04, 1.58MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:16<05:19, 2.09MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:17<05:49, 1.90MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:17<05:35, 1.98MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:17<04:17, 2.58MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:19<05:04, 2.17MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:19<05:05, 2.16MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:19<03:56, 2.79MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:21<04:49, 2.27MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:21<04:53, 2.23MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:21<03:45, 2.91MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:23<04:41, 2.32MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:23<04:48, 2.26MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:23<03:44, 2.90MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:25<04:39, 2.32MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:25<04:46, 2.26MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:25<03:42, 2.90MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:27<04:37, 2.32MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:27<04:44, 2.27MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:27<03:37, 2.95MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:29<04:34, 2.33MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:29<04:41, 2.28MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:29<03:34, 2.97MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:29<02:38, 4.01MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:31<11:19, 936kB/s] .vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:31<09:23, 1.13MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:31<06:53, 1.54MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:33<06:49, 1.54MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:33<06:14, 1.69MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:33<04:40, 2.25MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:35<05:15, 1.99MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:35<05:07, 2.04MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:35<03:53, 2.69MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:35<02:49, 3.67MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:37<34:46, 299kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:37<26:59, 385kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:37<19:26, 534kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:37<13:44, 753kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:39<13:00, 793kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:39<10:30, 982kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:39<07:41, 1.34MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:41<07:19, 1.40MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:41<06:31, 1.57MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:41<04:51, 2.11MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<03:30, 2.90MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:43<24:08, 422kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:43<18:08, 561kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:43<12:59, 782kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:45<11:05, 912kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:45<09:10, 1.10MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:45<06:45, 1.49MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:47<06:37, 1.52MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:47<06:01, 1.66MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:47<04:33, 2.20MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:49<05:04, 1.97MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:49<04:56, 2.02MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:49<03:47, 2.62MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:51<04:31, 2.19MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:51<06:00, 1.65MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:51<04:48, 2.06MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:51<03:29, 2.83MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:53<06:29, 1.52MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:53<05:30, 1.78MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:53<04:13, 2.32MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:53<03:06, 3.15MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:55<08:57, 1.09MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:55<09:05, 1.08MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:55<07:01, 1.39MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:55<05:03, 1.92MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:57<07:34, 1.28MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:57<06:38, 1.46MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:57<04:56, 1.96MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:57<03:33, 2.71MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:59<18:56, 508kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:59<14:35, 659kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:59<10:29, 916kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:59<07:27, 1.28MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:01<11:45, 813kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [02:01<09:21, 1.02MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [02:01<06:46, 1.41MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:01<04:51, 1.96MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:03<25:15, 376kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:03<18:46, 505kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:03<13:20, 709kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:05<11:18, 833kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:05<10:06, 932kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:05<07:37, 1.23MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:05<05:26, 1.72MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:07<11:41, 800kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:07<09:17, 1.01MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:07<06:45, 1.38MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:09<06:38, 1.40MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:09<05:43, 1.62MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:09<04:15, 2.17MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:11<04:55, 1.87MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:11<05:57, 1.54MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:11<04:43, 1.95MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:11<03:25, 2.68MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:13<05:46, 1.58MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:13<05:18, 1.72MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:13<03:59, 2.29MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:13<02:53, 3.14MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:15<17:56, 506kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:15<13:27, 674kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:15<09:38, 939kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:15<06:51, 1.31MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:17<11:51, 760kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:17<09:35, 938kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:17<06:57, 1.29MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:19<06:33, 1.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:19<05:48, 1.54MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:19<04:18, 2.07MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:21<04:43, 1.88MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:21<05:44, 1.55MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:21<04:37, 1.92MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:21<03:22, 2.61MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:23<06:09, 1.43MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:23<05:20, 1.65MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:23<03:56, 2.22MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:25<04:36, 1.90MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:25<04:23, 1.98MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:25<03:20, 2.61MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:26<03:59, 2.17MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:27<03:57, 2.18MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:27<03:04, 2.82MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:28<03:45, 2.28MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:29<03:50, 2.24MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:29<02:58, 2.88MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:30<03:41, 2.30MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:31<03:46, 2.26MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:31<02:53, 2.95MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:31<02:07, 3.97MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:32<08:19, 1.02MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:33<07:19, 1.15MB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:33<05:39, 1.49MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:33<04:08, 2.03MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:33<03:01, 2.78MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:34<50:05, 167kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:35<36:12, 231kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:35<25:32, 327kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:35<17:53, 465kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:36<27:34, 302kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:37<20:26, 407kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:37<14:31, 571kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:38<11:41, 705kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:38<09:17, 887kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:39<06:44, 1.22MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:39<04:47, 1.71MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:40<30:25, 269kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:40<22:24, 365kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:41<15:55, 512kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:42<12:38, 642kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:42<09:47, 828kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:43<07:07, 1.14MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:43<05:03, 1.59MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:44<13:09, 611kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:44<09:58, 805kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:44<07:13, 1.11MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:45<05:08, 1.55MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:46<09:16, 859kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:46<08:39, 921kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:47<06:36, 1.21MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:47<04:43, 1.68MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:48<06:43, 1.18MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:48<05:47, 1.37MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:49<04:16, 1.84MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:49<03:03, 2.56MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:50<2:37:38, 49.7kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:50<1:51:20, 70.3kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:50<1:18:01, 100kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:52<55:46, 139kB/s]  .vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:52<40:04, 194kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:52<28:12, 274kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:54<21:06, 364kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:54<15:48, 486kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:54<11:16, 680kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:56<09:18, 819kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:56<07:33, 1.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:56<05:32, 1.37MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:57<03:56, 1.92MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:58<7:31:52, 16.7kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:58<5:16:48, 23.8kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:58<3:41:27, 34.0kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:58<2:34:25, 48.6kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [03:00<1:53:15, 66.1kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [03:00<1:21:17, 92.1kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [03:00<57:20, 130kB/s]   .vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:01<40:03, 186kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:02<31:10, 238kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:02<22:40, 327kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:02<16:00, 461kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:04<12:42, 578kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:04<10:52, 675kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:04<08:06, 905kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:04<05:45, 1.27MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:06<07:03, 1.03MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:06<05:57, 1.22MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:06<04:22, 1.66MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:06<03:08, 2.30MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:08<16:03, 449kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:08<12:04, 597kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:08<08:36, 834kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:10<07:29, 953kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:10<07:10, 994kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:10<05:26, 1.31MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:10<03:53, 1.82MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:12<05:27, 1.29MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:12<04:47, 1.48MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:12<03:33, 1.98MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:14<03:49, 1.83MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:14<04:18, 1.63MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:14<03:24, 2.05MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:14<02:28, 2.81MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:16<07:31, 922kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:16<06:11, 1.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:16<04:32, 1.52MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:18<04:28, 1.53MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:18<04:04, 1.69MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:18<03:02, 2.25MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:20<03:25, 1.99MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:20<03:19, 2.04MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:20<02:31, 2.68MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:20<01:51, 3.63MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:22<06:17, 1.07MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:22<05:11, 1.29MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:22<03:49, 1.75MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:24<04:03, 1.64MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:24<04:44, 1.40MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:24<03:42, 1.79MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:24<02:42, 2.45MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:26<03:42, 1.77MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:26<03:31, 1.87MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:26<02:39, 2.48MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:26<01:56, 3.35MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:28<05:35, 1.17MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:28<04:49, 1.35MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:28<03:33, 1.83MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:28<02:34, 2.51MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:30<06:43, 958kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:30<05:42, 1.13MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:30<04:16, 1.50MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:30<03:05, 2.07MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:32<04:24, 1.45MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:32<04:14, 1.50MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:32<03:10, 2.00MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:34<03:22, 1.87MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:34<03:09, 1.99MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:34<02:43, 2.31MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:34<02:08, 2.93MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:34<01:35, 3.94MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:36<05:17, 1.18MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:36<04:32, 1.37MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:36<03:19, 1.86MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:36<02:25, 2.55MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:38<06:46, 912kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:38<05:36, 1.10MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:38<04:05, 1.50MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:38<02:57, 2.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:40<04:59, 1.22MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:40<04:24, 1.38MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:40<03:14, 1.87MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:40<02:31, 2.41MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:42<03:11, 1.89MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:42<03:15, 1.85MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:42<02:32, 2.36MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:44<02:51, 2.09MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:44<02:50, 2.10MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:44<02:09, 2.76MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:44<01:34, 3.75MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:46<08:00, 736kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:46<06:25, 917kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:46<04:40, 1.25MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:48<04:21, 1.34MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:48<03:52, 1.50MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:48<02:52, 2.02MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:48<02:03, 2.79MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:50<19:31, 295kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:50<14:27, 398kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:50<10:16, 558kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:52<08:13, 691kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:52<07:18, 779kB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:52<05:29, 1.03MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:52<03:54, 1.44MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:54<05:03, 1.11MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:54<04:19, 1.30MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:54<03:12, 1.74MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:56<03:17, 1.69MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:56<03:05, 1.79MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:56<02:20, 2.36MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:58<02:40, 2.05MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:58<02:36, 2.10MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:58<02:00, 2.72MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [04:00<02:25, 2.23MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [04:00<03:11, 1.70MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [04:00<02:36, 2.08MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:00<01:53, 2.84MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:01<03:36, 1.48MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:02<03:16, 1.63MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:02<02:26, 2.18MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:03<02:42, 1.95MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:04<02:36, 2.02MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:04<01:58, 2.65MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:05<02:22, 2.19MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:06<03:04, 1.69MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:06<02:31, 2.06MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:06<01:50, 2.80MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:07<03:25, 1.50MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:08<02:58, 1.72MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:08<02:23, 2.15MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:08<01:43, 2.94MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:09<03:22, 1.50MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:10<03:06, 1.63MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:10<02:29, 2.02MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:10<01:49, 2.76MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:11<02:50, 1.76MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:12<02:26, 2.04MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:12<02:02, 2.44MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:12<01:28, 3.34MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:13<03:36, 1.37MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:13<03:55, 1.25MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:14<03:16, 1.51MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:14<02:28, 1.99MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:14<01:46, 2.74MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:15<04:39, 1.04MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:15<04:13, 1.15MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:16<03:19, 1.46MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:16<02:25, 1.99MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:17<02:45, 1.74MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:17<02:38, 1.81MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:18<02:08, 2.23MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:18<01:34, 3.00MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:19<02:23, 1.97MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:19<02:23, 1.97MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:20<01:55, 2.44MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:20<01:25, 3.30MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:21<02:27, 1.89MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:21<03:00, 1.55MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:22<02:25, 1.92MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:22<01:45, 2.61MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:23<03:12, 1.43MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:23<02:52, 1.59MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:24<02:09, 2.11MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:25<02:21, 1.91MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:25<02:16, 1.98MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:26<01:44, 2.58MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:27<02:02, 2.17MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:27<02:39, 1.67MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:28<02:07, 2.08MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:28<01:33, 2.83MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:29<02:23, 1.83MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:29<02:17, 1.91MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:29<01:43, 2.53MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:30<01:14, 3.46MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:31<07:49, 551kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:31<06:03, 709kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:31<04:22, 979kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:33<03:50, 1.10MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:33<03:11, 1.33MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:33<02:20, 1.80MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:35<02:29, 1.67MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:35<02:18, 1.80MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:35<01:45, 2.37MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:37<02:00, 2.05MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:37<01:57, 2.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:37<01:29, 2.72MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:39<01:48, 2.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:39<02:21, 1.71MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:39<01:55, 2.09MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:40<01:23, 2.85MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:41<02:40, 1.48MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:41<02:25, 1.63MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:41<01:49, 2.16MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:43<02:00, 1.94MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:43<01:56, 2.00MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:43<01:28, 2.63MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:43<01:03, 3.60MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:45<10:12, 374kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:45<07:30, 508kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:45<05:26, 700kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:45<03:48, 988kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:47<05:22, 697kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:47<04:51, 772kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:47<03:38, 1.03MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:47<02:35, 1.43MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:49<03:23, 1.08MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:49<02:52, 1.28MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:49<02:06, 1.73MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:51<02:09, 1.68MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:51<02:00, 1.80MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:51<01:31, 2.36MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:53<01:43, 2.05MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:53<01:36, 2.19MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:53<01:14, 2.84MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:53<00:55, 3.79MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:55<02:03, 1.68MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:55<02:27, 1.41MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:55<01:55, 1.80MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:55<01:23, 2.46MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:57<02:22, 1.43MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:57<02:08, 1.59MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:57<01:35, 2.13MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:59<01:44, 1.92MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:59<01:40, 1.98MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:59<01:16, 2.61MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:01<01:30, 2.18MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [05:01<01:30, 2.17MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:01<01:09, 2.80MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:03<01:24, 2.27MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:03<01:25, 2.23MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:03<01:05, 2.91MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:05<01:21, 2.31MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:05<01:50, 1.70MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:05<01:28, 2.12MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:05<01:04, 2.87MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:07<01:34, 1.95MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:07<01:31, 2.01MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:07<01:09, 2.61MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:09<01:22, 2.18MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:09<01:22, 2.17MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:09<01:03, 2.81MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:11<01:16, 2.28MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:11<01:18, 2.23MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:11<00:59, 2.91MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:11<00:43, 3.97MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:13<14:39, 194kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:13<10:38, 267kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:13<07:29, 377kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:15<05:41, 489kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:15<04:21, 637kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:15<03:06, 886kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:15<02:11, 1.24MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:17<04:34, 592kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:17<03:49, 707kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:17<02:49, 953kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:17<01:59, 1.34MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:19<03:34, 738kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:19<02:52, 919kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:19<02:04, 1.26MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:21<01:55, 1.34MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:21<01:41, 1.51MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:21<01:16, 2.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:23<01:20, 1.86MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:23<01:17, 1.94MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:23<00:58, 2.55MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:23<00:42, 3.48MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:25<03:07, 779kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:25<02:31, 964kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:25<01:49, 1.32MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:27<01:42, 1.39MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:27<01:50, 1.29MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:27<01:25, 1.65MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:27<01:00, 2.29MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:29<01:34, 1.46MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:29<01:25, 1.61MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:29<01:03, 2.15MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:31<01:09, 1.93MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:31<01:07, 1.99MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:31<00:50, 2.63MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:31<00:36, 3.55MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:33<01:47, 1.20MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:33<01:33, 1.38MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:33<01:08, 1.86MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:35<01:11, 1.76MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:35<01:04, 1.94MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:35<00:47, 2.60MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:37<00:58, 2.08MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:37<01:13, 1.65MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:37<00:58, 2.05MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:37<00:42, 2.80MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:38<01:07, 1.73MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:39<01:02, 1.85MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:39<00:47, 2.44MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:39<00:33, 3.35MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:40<12:46, 147kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:41<09:10, 205kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:41<06:25, 290kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:42<04:43, 383kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:43<03:33, 509kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:43<02:31, 709kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:44<02:03, 849kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:45<01:40, 1.04MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:45<01:12, 1.42MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:46<01:08, 1.46MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:47<01:01, 1.63MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:47<00:45, 2.17MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:48<00:49, 1.94MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:49<00:47, 2.02MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:49<00:36, 2.62MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:50<00:42, 2.18MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:50<00:42, 2.18MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:51<00:32, 2.82MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:52<00:38, 2.28MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:52<00:38, 2.26MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:53<00:29, 2.90MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:54<00:36, 2.32MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:54<00:36, 2.29MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:55<00:28, 2.94MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:56<00:34, 2.33MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:56<00:45, 1.75MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:57<00:36, 2.18MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:57<00:26, 2.93MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:58<00:37, 2.01MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:58<00:36, 2.06MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:59<00:27, 2.69MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:00<00:32, 2.21MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:00<00:41, 1.70MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:00<00:33, 2.08MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:01<00:24, 2.83MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:02<00:46, 1.46MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:02<00:41, 1.62MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:02<00:30, 2.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:03<00:21, 2.99MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:04<01:46, 595kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:04<01:31, 689kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:04<01:07, 931kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:05<00:47, 1.29MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:06<00:46, 1.27MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:06<00:40, 1.45MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:06<00:29, 1.94MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:08<00:30, 1.81MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:08<00:28, 1.92MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:08<00:21, 2.50MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:10<00:23, 2.13MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:10<00:30, 1.65MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:10<00:24, 2.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:11<00:17, 2.82MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:12<00:25, 1.85MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:12<00:19, 2.30MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:12<00:14, 3.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:14<00:20, 2.10MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:14<00:25, 1.66MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:14<00:20, 2.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:14<00:14, 2.76MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:16<00:18, 2.08MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:16<00:18, 2.10MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:16<00:13, 2.75MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:16<00:09, 3.73MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:18<00:44, 757kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:18<00:35, 939kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:18<00:25, 1.29MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:18<00:17, 1.79MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:20<00:33, 891kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:20<00:27, 1.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:20<00:19, 1.47MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:20<00:13, 2.04MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:22<00:24, 1.05MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:22<00:20, 1.24MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:22<00:14, 1.69MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:22<00:09, 2.32MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:24<00:19, 1.09MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:24<00:16, 1.28MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:24<00:11, 1.73MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:24<00:07, 2.38MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:26<00:15, 1.17MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:26<00:12, 1.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:26<00:08, 1.81MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:28<00:07, 1.73MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:28<00:07, 1.83MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:28<00:04, 2.40MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:30<00:04, 2.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:30<00:04, 2.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:30<00:02, 2.73MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:32<00:02, 2.24MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:32<00:02, 2.34MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:32<00:01, 2.98MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:32<00:00, 3.97MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:34<00:00, 1.71MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:34<00:00, 1.43MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:34<00:00, 1.79MB/s].vector_cache/glove.6B.zip: 862MB [06:34, 2.18MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<17:59:37,  6.17it/s]  0%|          | 891/400000 [00:00<12:34:16,  8.82it/s]  0%|          | 1796/400000 [00:00<8:47:01, 12.59it/s]  1%|          | 2709/400000 [00:00<6:08:17, 17.98it/s]  1%|          | 3585/400000 [00:00<4:17:27, 25.66it/s]  1%|          | 4416/400000 [00:00<3:00:04, 36.61it/s]  1%|▏         | 5360/400000 [00:00<2:05:57, 52.22it/s]  2%|▏         | 6283/400000 [00:00<1:28:10, 74.41it/s]  2%|▏         | 7225/400000 [00:00<1:01:47, 105.95it/s]  2%|▏         | 8125/400000 [00:01<43:22, 150.59it/s]    2%|▏         | 9009/400000 [00:01<30:30, 213.57it/s]  2%|▏         | 9956/400000 [00:01<21:30, 302.18it/s]  3%|▎         | 10894/400000 [00:01<15:13, 425.80it/s]  3%|▎         | 11808/400000 [00:01<10:50, 596.37it/s]  3%|▎         | 12719/400000 [00:01<07:47, 828.47it/s]  3%|▎         | 13628/400000 [00:01<05:39, 1137.88it/s]  4%|▎         | 14570/400000 [00:01<04:09, 1545.48it/s]  4%|▍         | 15484/400000 [00:01<03:08, 2038.93it/s]  4%|▍         | 16385/400000 [00:01<02:24, 2655.12it/s]  4%|▍         | 17308/400000 [00:02<01:53, 3376.43it/s]  5%|▍         | 18205/400000 [00:02<01:32, 4146.67it/s]  5%|▍         | 19166/400000 [00:02<01:16, 4999.08it/s]  5%|▌         | 20087/400000 [00:02<01:05, 5793.13it/s]  5%|▌         | 21012/400000 [00:02<00:58, 6523.59it/s]  5%|▌         | 21976/400000 [00:02<00:52, 7223.98it/s]  6%|▌         | 22908/400000 [00:02<00:49, 7600.22it/s]  6%|▌         | 23869/400000 [00:02<00:46, 8107.56it/s]  6%|▌         | 24807/400000 [00:02<00:44, 8450.24it/s]  6%|▋         | 25739/400000 [00:02<00:43, 8692.12it/s]  7%|▋         | 26701/400000 [00:03<00:41, 8950.86it/s]  7%|▋         | 27641/400000 [00:03<00:41, 8912.32it/s]  7%|▋         | 28567/400000 [00:03<00:41, 9012.90it/s]  7%|▋         | 29491/400000 [00:03<00:41, 9027.14it/s]  8%|▊         | 30410/400000 [00:03<00:41, 8939.24it/s]  8%|▊         | 31315/400000 [00:03<00:41, 8866.57it/s]  8%|▊         | 32210/400000 [00:03<00:41, 8779.73it/s]  8%|▊         | 33163/400000 [00:03<00:40, 8990.17it/s]  9%|▊         | 34098/400000 [00:03<00:40, 9093.74it/s]  9%|▉         | 35036/400000 [00:04<00:39, 9177.17it/s]  9%|▉         | 35977/400000 [00:04<00:39, 9245.20it/s]  9%|▉         | 36904/400000 [00:04<00:39, 9200.78it/s]  9%|▉         | 37864/400000 [00:04<00:38, 9316.02it/s] 10%|▉         | 38798/400000 [00:04<00:38, 9313.57it/s] 10%|▉         | 39731/400000 [00:04<00:39, 9236.33it/s] 10%|█         | 40687/400000 [00:04<00:38, 9329.59it/s] 10%|█         | 41621/400000 [00:04<00:38, 9271.90it/s] 11%|█         | 42549/400000 [00:04<00:38, 9240.29it/s] 11%|█         | 43495/400000 [00:04<00:38, 9303.56it/s] 11%|█         | 44443/400000 [00:05<00:38, 9355.24it/s] 11%|█▏        | 45379/400000 [00:05<00:38, 9320.65it/s] 12%|█▏        | 46312/400000 [00:05<00:39, 9037.15it/s] 12%|█▏        | 47225/400000 [00:05<00:38, 9063.29it/s] 12%|█▏        | 48138/400000 [00:05<00:38, 9082.27it/s] 12%|█▏        | 49064/400000 [00:05<00:38, 9134.82it/s] 13%|█▎        | 50024/400000 [00:05<00:37, 9268.47it/s] 13%|█▎        | 50952/400000 [00:05<00:38, 9178.46it/s] 13%|█▎        | 51871/400000 [00:05<00:39, 8922.93it/s] 13%|█▎        | 52766/400000 [00:05<00:41, 8457.47it/s] 13%|█▎        | 53619/400000 [00:06<00:42, 8245.99it/s] 14%|█▎        | 54512/400000 [00:06<00:40, 8439.49it/s] 14%|█▍        | 55373/400000 [00:06<00:40, 8488.55it/s] 14%|█▍        | 56311/400000 [00:06<00:39, 8735.36it/s] 14%|█▍        | 57240/400000 [00:06<00:38, 8894.03it/s] 15%|█▍        | 58193/400000 [00:06<00:37, 9073.48it/s] 15%|█▍        | 59124/400000 [00:06<00:37, 9142.71it/s] 15%|█▌        | 60041/400000 [00:06<00:37, 8993.71it/s] 15%|█▌        | 60943/400000 [00:06<00:38, 8873.28it/s] 15%|█▌        | 61869/400000 [00:06<00:37, 8983.60it/s] 16%|█▌        | 62795/400000 [00:07<00:37, 9062.83it/s] 16%|█▌        | 63742/400000 [00:07<00:36, 9180.00it/s] 16%|█▌        | 64662/400000 [00:07<00:37, 8951.78it/s] 16%|█▋        | 65583/400000 [00:07<00:37, 9027.11it/s] 17%|█▋        | 66518/400000 [00:07<00:36, 9121.61it/s] 17%|█▋        | 67480/400000 [00:07<00:35, 9265.31it/s] 17%|█▋        | 68427/400000 [00:07<00:35, 9323.75it/s] 17%|█▋        | 69361/400000 [00:07<00:35, 9200.18it/s] 18%|█▊        | 70308/400000 [00:07<00:35, 9278.18it/s] 18%|█▊        | 71273/400000 [00:07<00:35, 9384.48it/s] 18%|█▊        | 72231/400000 [00:08<00:34, 9440.47it/s] 18%|█▊        | 73176/400000 [00:08<00:34, 9350.46it/s] 19%|█▊        | 74112/400000 [00:08<00:35, 9224.52it/s] 19%|█▉        | 75071/400000 [00:08<00:34, 9330.15it/s] 19%|█▉        | 76005/400000 [00:08<00:35, 9126.14it/s] 19%|█▉        | 76920/400000 [00:08<00:36, 8882.14it/s] 19%|█▉        | 77833/400000 [00:08<00:35, 8954.08it/s] 20%|█▉        | 78731/400000 [00:08<00:35, 8945.78it/s] 20%|█▉        | 79676/400000 [00:08<00:35, 9090.99it/s] 20%|██        | 80587/400000 [00:09<00:35, 8986.99it/s] 20%|██        | 81488/400000 [00:09<00:35, 8945.00it/s] 21%|██        | 82436/400000 [00:09<00:34, 9097.14it/s] 21%|██        | 83348/400000 [00:09<00:35, 8911.46it/s] 21%|██        | 84258/400000 [00:09<00:35, 8965.13it/s] 21%|██▏       | 85196/400000 [00:09<00:34, 9085.66it/s] 22%|██▏       | 86111/400000 [00:09<00:34, 9103.57it/s] 22%|██▏       | 87053/400000 [00:09<00:34, 9192.72it/s] 22%|██▏       | 87974/400000 [00:09<00:34, 9035.30it/s] 22%|██▏       | 88926/400000 [00:09<00:33, 9173.42it/s] 22%|██▏       | 89887/400000 [00:10<00:33, 9299.77it/s] 23%|██▎       | 90819/400000 [00:10<00:33, 9150.68it/s] 23%|██▎       | 91736/400000 [00:10<00:34, 8872.32it/s] 23%|██▎       | 92651/400000 [00:10<00:34, 8953.55it/s] 23%|██▎       | 93584/400000 [00:10<00:33, 9061.46it/s] 24%|██▎       | 94501/400000 [00:10<00:33, 9093.00it/s] 24%|██▍       | 95426/400000 [00:10<00:33, 9138.35it/s] 24%|██▍       | 96343/400000 [00:10<00:33, 9146.56it/s] 24%|██▍       | 97259/400000 [00:10<00:33, 8951.88it/s] 25%|██▍       | 98221/400000 [00:10<00:33, 9139.78it/s] 25%|██▍       | 99137/400000 [00:11<00:33, 9075.18it/s] 25%|██▌       | 100060/400000 [00:11<00:32, 9119.19it/s] 25%|██▌       | 101027/400000 [00:11<00:32, 9276.59it/s] 25%|██▌       | 101957/400000 [00:11<00:32, 9191.42it/s] 26%|██▌       | 102905/400000 [00:11<00:32, 9229.24it/s] 26%|██▌       | 103829/400000 [00:11<00:32, 9219.46it/s] 26%|██▌       | 104752/400000 [00:11<00:32, 9195.72it/s] 26%|██▋       | 105715/400000 [00:11<00:31, 9321.29it/s] 27%|██▋       | 106648/400000 [00:11<00:32, 8899.78it/s] 27%|██▋       | 107543/400000 [00:11<00:32, 8910.23it/s] 27%|██▋       | 108479/400000 [00:12<00:32, 9038.43it/s] 27%|██▋       | 109406/400000 [00:12<00:31, 9106.63it/s] 28%|██▊       | 110338/400000 [00:12<00:31, 9169.19it/s] 28%|██▊       | 111257/400000 [00:12<00:31, 9147.84it/s] 28%|██▊       | 112181/400000 [00:12<00:31, 9172.97it/s] 28%|██▊       | 113131/400000 [00:12<00:30, 9268.52it/s] 29%|██▊       | 114059/400000 [00:12<00:31, 9204.57it/s] 29%|██▊       | 114986/400000 [00:12<00:30, 9222.30it/s] 29%|██▉       | 115909/400000 [00:12<00:31, 9113.58it/s] 29%|██▉       | 116845/400000 [00:13<00:30, 9183.60it/s] 29%|██▉       | 117771/400000 [00:13<00:30, 9205.91it/s] 30%|██▉       | 118700/400000 [00:13<00:30, 9229.41it/s] 30%|██▉       | 119648/400000 [00:13<00:30, 9300.44it/s] 30%|███       | 120579/400000 [00:13<00:30, 9122.06it/s] 30%|███       | 121493/400000 [00:13<00:31, 8957.06it/s] 31%|███       | 122391/400000 [00:13<00:31, 8879.25it/s] 31%|███       | 123302/400000 [00:13<00:30, 8946.97it/s] 31%|███       | 124257/400000 [00:13<00:30, 9119.59it/s] 31%|███▏      | 125171/400000 [00:13<00:30, 8947.84it/s] 32%|███▏      | 126082/400000 [00:14<00:30, 8995.60it/s] 32%|███▏      | 127056/400000 [00:14<00:29, 9205.86it/s] 32%|███▏      | 127985/400000 [00:14<00:29, 9228.42it/s] 32%|███▏      | 128943/400000 [00:14<00:29, 9328.23it/s] 32%|███▏      | 129878/400000 [00:14<00:29, 9152.09it/s] 33%|███▎      | 130795/400000 [00:14<00:29, 9084.43it/s] 33%|███▎      | 131752/400000 [00:14<00:29, 9222.52it/s] 33%|███▎      | 132724/400000 [00:14<00:28, 9365.63it/s] 33%|███▎      | 133663/400000 [00:14<00:28, 9306.19it/s] 34%|███▎      | 134595/400000 [00:14<00:29, 9091.47it/s] 34%|███▍      | 135559/400000 [00:15<00:28, 9246.85it/s] 34%|███▍      | 136486/400000 [00:15<00:28, 9224.41it/s] 34%|███▍      | 137410/400000 [00:15<00:28, 9149.22it/s] 35%|███▍      | 138339/400000 [00:15<00:28, 9188.30it/s] 35%|███▍      | 139259/400000 [00:15<00:28, 9108.49it/s] 35%|███▌      | 140218/400000 [00:15<00:28, 9246.81it/s] 35%|███▌      | 141182/400000 [00:15<00:27, 9361.30it/s] 36%|███▌      | 142125/400000 [00:15<00:27, 9381.39it/s] 36%|███▌      | 143064/400000 [00:15<00:27, 9377.47it/s] 36%|███▌      | 144003/400000 [00:15<00:27, 9236.24it/s] 36%|███▌      | 144928/400000 [00:16<00:28, 9044.97it/s] 36%|███▋      | 145834/400000 [00:16<00:28, 8910.89it/s] 37%|███▋      | 146733/400000 [00:16<00:28, 8932.32it/s] 37%|███▋      | 147674/400000 [00:16<00:27, 9070.01it/s] 37%|███▋      | 148583/400000 [00:16<00:27, 8990.84it/s] 37%|███▋      | 149523/400000 [00:16<00:27, 9109.79it/s] 38%|███▊      | 150436/400000 [00:16<00:27, 9110.10it/s] 38%|███▊      | 151352/400000 [00:16<00:27, 9123.12it/s] 38%|███▊      | 152265/400000 [00:16<00:28, 8783.99it/s] 38%|███▊      | 153147/400000 [00:16<00:28, 8705.39it/s] 39%|███▊      | 154085/400000 [00:17<00:27, 8895.80it/s] 39%|███▊      | 154988/400000 [00:17<00:27, 8935.43it/s] 39%|███▉      | 155944/400000 [00:17<00:26, 9112.91it/s] 39%|███▉      | 156887/400000 [00:17<00:26, 9205.61it/s] 39%|███▉      | 157810/400000 [00:17<00:26, 9157.82it/s] 40%|███▉      | 158767/400000 [00:17<00:26, 9277.43it/s] 40%|███▉      | 159697/400000 [00:17<00:26, 9221.83it/s] 40%|████      | 160669/400000 [00:17<00:25, 9363.32it/s] 40%|████      | 161626/400000 [00:17<00:25, 9423.95it/s] 41%|████      | 162570/400000 [00:18<00:25, 9245.18it/s] 41%|████      | 163496/400000 [00:18<00:25, 9226.22it/s] 41%|████      | 164453/400000 [00:18<00:25, 9324.44it/s] 41%|████▏     | 165390/400000 [00:18<00:25, 9335.35it/s] 42%|████▏     | 166355/400000 [00:18<00:24, 9427.56it/s] 42%|████▏     | 167299/400000 [00:18<00:25, 8959.86it/s] 42%|████▏     | 168201/400000 [00:18<00:25, 8916.03it/s] 42%|████▏     | 169166/400000 [00:18<00:25, 9120.08it/s] 43%|████▎     | 170112/400000 [00:18<00:24, 9217.36it/s] 43%|████▎     | 171037/400000 [00:18<00:24, 9172.08it/s] 43%|████▎     | 171957/400000 [00:19<00:25, 8990.27it/s] 43%|████▎     | 172902/400000 [00:19<00:24, 9121.12it/s] 43%|████▎     | 173863/400000 [00:19<00:24, 9261.15it/s] 44%|████▎     | 174829/400000 [00:19<00:24, 9374.31it/s] 44%|████▍     | 175782/400000 [00:19<00:23, 9418.50it/s] 44%|████▍     | 176726/400000 [00:19<00:24, 9090.93it/s] 44%|████▍     | 177683/400000 [00:19<00:24, 9228.97it/s] 45%|████▍     | 178650/400000 [00:19<00:23, 9355.20it/s] 45%|████▍     | 179608/400000 [00:19<00:23, 9420.96it/s] 45%|████▌     | 180552/400000 [00:19<00:23, 9290.68it/s] 45%|████▌     | 181483/400000 [00:20<00:23, 9175.24it/s] 46%|████▌     | 182403/400000 [00:20<00:23, 9130.48it/s] 46%|████▌     | 183318/400000 [00:20<00:23, 9057.00it/s] 46%|████▌     | 184225/400000 [00:20<00:23, 9047.24it/s] 46%|████▋     | 185149/400000 [00:20<00:23, 9103.81it/s] 47%|████▋     | 186060/400000 [00:20<00:23, 8969.45it/s] 47%|████▋     | 187022/400000 [00:20<00:23, 9151.16it/s] 47%|████▋     | 187989/400000 [00:20<00:22, 9299.06it/s] 47%|████▋     | 188921/400000 [00:20<00:22, 9287.08it/s] 47%|████▋     | 189856/400000 [00:20<00:22, 9303.37it/s] 48%|████▊     | 190788/400000 [00:21<00:22, 9108.46it/s] 48%|████▊     | 191734/400000 [00:21<00:22, 9210.98it/s] 48%|████▊     | 192691/400000 [00:21<00:22, 9315.27it/s] 48%|████▊     | 193624/400000 [00:21<00:22, 9260.27it/s] 49%|████▊     | 194569/400000 [00:21<00:22, 9316.23it/s] 49%|████▉     | 195502/400000 [00:21<00:22, 9174.73it/s] 49%|████▉     | 196438/400000 [00:21<00:22, 9227.40it/s] 49%|████▉     | 197362/400000 [00:21<00:22, 9181.62it/s] 50%|████▉     | 198281/400000 [00:21<00:22, 9023.75it/s] 50%|████▉     | 199214/400000 [00:21<00:22, 9111.99it/s] 50%|█████     | 200127/400000 [00:22<00:22, 9016.05it/s] 50%|█████     | 201075/400000 [00:22<00:21, 9148.99it/s] 50%|█████     | 201991/400000 [00:22<00:21, 9123.33it/s] 51%|█████     | 202948/400000 [00:22<00:21, 9252.62it/s] 51%|█████     | 203884/400000 [00:22<00:21, 9282.85it/s] 51%|█████     | 204813/400000 [00:22<00:21, 9255.45it/s] 51%|█████▏    | 205749/400000 [00:22<00:20, 9286.17it/s] 52%|█████▏    | 206679/400000 [00:22<00:20, 9266.90it/s] 52%|█████▏    | 207635/400000 [00:22<00:20, 9351.12it/s] 52%|█████▏    | 208586/400000 [00:22<00:20, 9397.71it/s] 52%|█████▏    | 209527/400000 [00:23<00:20, 9241.49it/s] 53%|█████▎    | 210467/400000 [00:23<00:20, 9286.30it/s] 53%|█████▎    | 211436/400000 [00:23<00:20, 9402.22it/s] 53%|█████▎    | 212380/400000 [00:23<00:19, 9413.42it/s] 53%|█████▎    | 213322/400000 [00:23<00:20, 9274.54it/s] 54%|█████▎    | 214251/400000 [00:23<00:20, 9154.63it/s] 54%|█████▍    | 215172/400000 [00:23<00:20, 9170.46it/s] 54%|█████▍    | 216111/400000 [00:23<00:19, 9233.66it/s] 54%|█████▍    | 217075/400000 [00:23<00:19, 9349.80it/s] 55%|█████▍    | 218017/400000 [00:24<00:19, 9369.89it/s] 55%|█████▍    | 218955/400000 [00:24<00:19, 9111.75it/s] 55%|█████▍    | 219893/400000 [00:24<00:19, 9190.06it/s] 55%|█████▌    | 220855/400000 [00:24<00:19, 9314.31it/s] 55%|█████▌    | 221818/400000 [00:24<00:18, 9400.87it/s] 56%|█████▌    | 222760/400000 [00:24<00:18, 9364.21it/s] 56%|█████▌    | 223698/400000 [00:24<00:19, 9152.77it/s] 56%|█████▌    | 224662/400000 [00:24<00:18, 9291.14it/s] 56%|█████▋    | 225633/400000 [00:24<00:18, 9410.40it/s] 57%|█████▋    | 226576/400000 [00:24<00:18, 9396.36it/s] 57%|█████▋    | 227517/400000 [00:25<00:18, 9270.70it/s] 57%|█████▋    | 228446/400000 [00:25<00:18, 9151.98it/s] 57%|█████▋    | 229370/400000 [00:25<00:18, 9176.31it/s] 58%|█████▊    | 230327/400000 [00:25<00:18, 9289.51it/s] 58%|█████▊    | 231292/400000 [00:25<00:17, 9392.48it/s] 58%|█████▊    | 232233/400000 [00:25<00:17, 9330.79it/s] 58%|█████▊    | 233167/400000 [00:25<00:18, 9254.29it/s] 59%|█████▊    | 234135/400000 [00:25<00:17, 9376.70it/s] 59%|█████▉    | 235081/400000 [00:25<00:17, 9399.58it/s] 59%|█████▉    | 236022/400000 [00:25<00:17, 9379.51it/s] 59%|█████▉    | 236961/400000 [00:26<00:17, 9300.68it/s] 59%|█████▉    | 237892/400000 [00:26<00:17, 9241.30it/s] 60%|█████▉    | 238840/400000 [00:26<00:17, 9310.65it/s] 60%|█████▉    | 239788/400000 [00:26<00:17, 9358.25it/s] 60%|██████    | 240725/400000 [00:26<00:17, 9314.21it/s] 60%|██████    | 241657/400000 [00:26<00:17, 9248.40it/s] 61%|██████    | 242583/400000 [00:26<00:17, 9135.72it/s] 61%|██████    | 243511/400000 [00:26<00:17, 9177.09it/s] 61%|██████    | 244430/400000 [00:26<00:17, 9021.73it/s] 61%|██████▏   | 245334/400000 [00:26<00:17, 8987.70it/s] 62%|██████▏   | 246248/400000 [00:27<00:17, 9030.58it/s] 62%|██████▏   | 247155/400000 [00:27<00:16, 9040.97it/s] 62%|██████▏   | 248109/400000 [00:27<00:16, 9184.49it/s] 62%|██████▏   | 249048/400000 [00:27<00:16, 9242.73it/s] 63%|██████▎   | 250014/400000 [00:27<00:16, 9361.82it/s] 63%|██████▎   | 250960/400000 [00:27<00:15, 9389.59it/s] 63%|██████▎   | 251900/400000 [00:27<00:16, 9246.52it/s] 63%|██████▎   | 252863/400000 [00:27<00:15, 9355.83it/s] 63%|██████▎   | 253820/400000 [00:27<00:15, 9416.42it/s] 64%|██████▎   | 254763/400000 [00:27<00:15, 9418.77it/s] 64%|██████▍   | 255706/400000 [00:28<00:15, 9320.73it/s] 64%|██████▍   | 256639/400000 [00:28<00:15, 9251.93it/s] 64%|██████▍   | 257565/400000 [00:28<00:15, 9090.96it/s] 65%|██████▍   | 258516/400000 [00:28<00:15, 9210.34it/s] 65%|██████▍   | 259439/400000 [00:28<00:15, 9145.78it/s] 65%|██████▌   | 260355/400000 [00:28<00:15, 8755.05it/s] 65%|██████▌   | 261278/400000 [00:28<00:15, 8890.36it/s] 66%|██████▌   | 262200/400000 [00:28<00:15, 8985.69it/s] 66%|██████▌   | 263142/400000 [00:28<00:15, 9110.09it/s] 66%|██████▌   | 264056/400000 [00:29<00:15, 8933.78it/s] 66%|██████▌   | 264966/400000 [00:29<00:15, 8982.20it/s] 66%|██████▋   | 265866/400000 [00:29<00:14, 8958.93it/s] 67%|██████▋   | 266785/400000 [00:29<00:14, 9024.59it/s] 67%|██████▋   | 267727/400000 [00:29<00:14, 9138.63it/s] 67%|██████▋   | 268675/400000 [00:29<00:14, 9235.85it/s] 67%|██████▋   | 269600/400000 [00:29<00:14, 9165.98it/s] 68%|██████▊   | 270518/400000 [00:29<00:14, 9065.13it/s] 68%|██████▊   | 271426/400000 [00:29<00:14, 9065.13it/s] 68%|██████▊   | 272334/400000 [00:29<00:14, 8919.48it/s] 68%|██████▊   | 273263/400000 [00:30<00:14, 9027.20it/s] 69%|██████▊   | 274167/400000 [00:30<00:14, 8792.85it/s] 69%|██████▉   | 275049/400000 [00:30<00:14, 8737.28it/s] 69%|██████▉   | 275970/400000 [00:30<00:13, 8873.89it/s] 69%|██████▉   | 276928/400000 [00:30<00:13, 9072.42it/s] 69%|██████▉   | 277861/400000 [00:30<00:13, 9146.96it/s] 70%|██████▉   | 278778/400000 [00:30<00:13, 9002.37it/s] 70%|██████▉   | 279718/400000 [00:30<00:13, 9115.46it/s] 70%|███████   | 280685/400000 [00:30<00:12, 9273.61it/s] 70%|███████   | 281641/400000 [00:30<00:12, 9356.14it/s] 71%|███████   | 282582/400000 [00:31<00:12, 9370.72it/s] 71%|███████   | 283521/400000 [00:31<00:12, 9157.15it/s] 71%|███████   | 284443/400000 [00:31<00:12, 9173.84it/s] 71%|███████▏  | 285397/400000 [00:31<00:12, 9279.26it/s] 72%|███████▏  | 286356/400000 [00:31<00:12, 9370.01it/s] 72%|███████▏  | 287295/400000 [00:31<00:12, 9354.65it/s] 72%|███████▏  | 288232/400000 [00:31<00:12, 9255.55it/s] 72%|███████▏  | 289159/400000 [00:31<00:12, 9221.65it/s] 73%|███████▎  | 290082/400000 [00:31<00:12, 8854.68it/s] 73%|███████▎  | 290997/400000 [00:31<00:12, 8940.01it/s] 73%|███████▎  | 291894/400000 [00:32<00:12, 8839.44it/s] 73%|███████▎  | 292800/400000 [00:32<00:12, 8903.95it/s] 73%|███████▎  | 293731/400000 [00:32<00:11, 9021.18it/s] 74%|███████▎  | 294641/400000 [00:32<00:11, 9042.61it/s] 74%|███████▍  | 295592/400000 [00:32<00:11, 9176.13it/s] 74%|███████▍  | 296530/400000 [00:32<00:11, 9235.47it/s] 74%|███████▍  | 297455/400000 [00:32<00:11, 9088.81it/s] 75%|███████▍  | 298391/400000 [00:32<00:11, 9167.57it/s] 75%|███████▍  | 299352/400000 [00:32<00:10, 9295.42it/s] 75%|███████▌  | 300283/400000 [00:32<00:10, 9274.20it/s] 75%|███████▌  | 301234/400000 [00:33<00:10, 9343.25it/s] 76%|███████▌  | 302170/400000 [00:33<00:10, 9252.35it/s] 76%|███████▌  | 303102/400000 [00:33<00:10, 9272.35it/s] 76%|███████▌  | 304030/400000 [00:33<00:10, 9204.58it/s] 76%|███████▌  | 304951/400000 [00:33<00:10, 9056.76it/s] 76%|███████▋  | 305872/400000 [00:33<00:10, 9101.33it/s] 77%|███████▋  | 306783/400000 [00:33<00:10, 9101.57it/s] 77%|███████▋  | 307708/400000 [00:33<00:10, 9144.05it/s] 77%|███████▋  | 308660/400000 [00:33<00:09, 9251.14it/s] 77%|███████▋  | 309588/400000 [00:33<00:09, 9259.53it/s] 78%|███████▊  | 310529/400000 [00:34<00:09, 9303.96it/s] 78%|███████▊  | 311491/400000 [00:34<00:09, 9395.26it/s] 78%|███████▊  | 312431/400000 [00:34<00:09, 9087.39it/s] 78%|███████▊  | 313343/400000 [00:34<00:09, 9094.92it/s] 79%|███████▊  | 314255/400000 [00:34<00:09, 8838.43it/s] 79%|███████▉  | 315142/400000 [00:34<00:09, 8819.09it/s] 79%|███████▉  | 316026/400000 [00:34<00:09, 8732.80it/s] 79%|███████▉  | 316928/400000 [00:34<00:09, 8815.95it/s] 79%|███████▉  | 317854/400000 [00:34<00:09, 8944.10it/s] 80%|███████▉  | 318765/400000 [00:35<00:09, 8991.64it/s] 80%|███████▉  | 319666/400000 [00:35<00:09, 8844.02it/s] 80%|████████  | 320552/400000 [00:35<00:09, 8673.48it/s] 80%|████████  | 321436/400000 [00:35<00:09, 8722.07it/s] 81%|████████  | 322372/400000 [00:35<00:08, 8903.58it/s] 81%|████████  | 323326/400000 [00:35<00:08, 9083.22it/s] 81%|████████  | 324254/400000 [00:35<00:08, 9138.21it/s] 81%|████████▏ | 325170/400000 [00:35<00:08, 8923.63it/s] 82%|████████▏ | 326065/400000 [00:35<00:08, 8735.89it/s] 82%|████████▏ | 326944/400000 [00:35<00:08, 8750.70it/s] 82%|████████▏ | 327860/400000 [00:36<00:08, 8868.72it/s] 82%|████████▏ | 328798/400000 [00:36<00:07, 9014.15it/s] 82%|████████▏ | 329702/400000 [00:36<00:07, 8970.04it/s] 83%|████████▎ | 330601/400000 [00:36<00:07, 8934.74it/s] 83%|████████▎ | 331532/400000 [00:36<00:07, 9041.11it/s] 83%|████████▎ | 332464/400000 [00:36<00:07, 9120.49it/s] 83%|████████▎ | 333407/400000 [00:36<00:07, 9209.57it/s] 84%|████████▎ | 334329/400000 [00:36<00:07, 9057.91it/s] 84%|████████▍ | 335236/400000 [00:36<00:07, 8758.89it/s] 84%|████████▍ | 336115/400000 [00:36<00:07, 8688.77it/s] 84%|████████▍ | 337038/400000 [00:37<00:07, 8843.77it/s] 84%|████████▍ | 337973/400000 [00:37<00:06, 8987.37it/s] 85%|████████▍ | 338874/400000 [00:37<00:07, 8719.06it/s] 85%|████████▍ | 339811/400000 [00:37<00:06, 8903.42it/s] 85%|████████▌ | 340777/400000 [00:37<00:06, 9115.37it/s] 85%|████████▌ | 341736/400000 [00:37<00:06, 9252.44it/s] 86%|████████▌ | 342665/400000 [00:37<00:06, 9191.73it/s] 86%|████████▌ | 343587/400000 [00:37<00:06, 9076.38it/s] 86%|████████▌ | 344545/400000 [00:37<00:06, 9220.40it/s] 86%|████████▋ | 345481/400000 [00:37<00:05, 9259.14it/s] 87%|████████▋ | 346439/400000 [00:38<00:05, 9352.32it/s] 87%|████████▋ | 347376/400000 [00:38<00:05, 9245.88it/s] 87%|████████▋ | 348302/400000 [00:38<00:05, 9055.50it/s] 87%|████████▋ | 349214/400000 [00:38<00:05, 9072.76it/s] 88%|████████▊ | 350123/400000 [00:38<00:05, 8917.31it/s] 88%|████████▊ | 351017/400000 [00:38<00:05, 8772.15it/s] 88%|████████▊ | 351896/400000 [00:38<00:05, 8625.05it/s] 88%|████████▊ | 352812/400000 [00:38<00:05, 8778.35it/s] 88%|████████▊ | 353754/400000 [00:38<00:05, 8960.13it/s] 89%|████████▊ | 354681/400000 [00:39<00:05, 9048.37it/s] 89%|████████▉ | 355588/400000 [00:39<00:05, 8773.82it/s] 89%|████████▉ | 356469/400000 [00:39<00:04, 8723.75it/s] 89%|████████▉ | 357393/400000 [00:39<00:04, 8872.23it/s] 90%|████████▉ | 358322/400000 [00:39<00:04, 8992.98it/s] 90%|████████▉ | 359224/400000 [00:39<00:04, 8999.42it/s] 90%|█████████ | 360126/400000 [00:39<00:04, 8946.78it/s] 90%|█████████ | 361022/400000 [00:39<00:04, 8793.86it/s] 90%|█████████ | 361939/400000 [00:39<00:04, 8902.67it/s] 91%|█████████ | 362866/400000 [00:39<00:04, 9009.63it/s] 91%|█████████ | 363771/400000 [00:40<00:04, 9020.88it/s] 91%|█████████ | 364674/400000 [00:40<00:03, 8948.58it/s] 91%|█████████▏| 365570/400000 [00:40<00:03, 8770.48it/s] 92%|█████████▏| 366449/400000 [00:40<00:03, 8750.99it/s] 92%|█████████▏| 367347/400000 [00:40<00:03, 8816.17it/s] 92%|█████████▏| 368270/400000 [00:40<00:03, 8935.25it/s] 92%|█████████▏| 369165/400000 [00:40<00:03, 8916.07it/s] 93%|█████████▎| 370058/400000 [00:40<00:03, 8909.37it/s] 93%|█████████▎| 371008/400000 [00:40<00:03, 9076.49it/s] 93%|█████████▎| 371958/400000 [00:40<00:03, 9198.82it/s] 93%|█████████▎| 372883/400000 [00:41<00:02, 9213.40it/s] 93%|█████████▎| 373818/400000 [00:41<00:02, 9252.08it/s] 94%|█████████▎| 374744/400000 [00:41<00:02, 9001.69it/s] 94%|█████████▍| 375675/400000 [00:41<00:02, 9091.62it/s] 94%|█████████▍| 376616/400000 [00:41<00:02, 9184.80it/s] 94%|█████████▍| 377536/400000 [00:41<00:02, 9178.98it/s] 95%|█████████▍| 378470/400000 [00:41<00:02, 9223.72it/s] 95%|█████████▍| 379394/400000 [00:41<00:02, 9086.69it/s] 95%|█████████▌| 380304/400000 [00:41<00:02, 9080.35it/s] 95%|█████████▌| 381213/400000 [00:41<00:02, 8979.37it/s] 96%|█████████▌| 382144/400000 [00:42<00:01, 9074.80it/s] 96%|█████████▌| 383079/400000 [00:42<00:01, 9155.21it/s] 96%|█████████▌| 383996/400000 [00:42<00:01, 9113.48it/s] 96%|█████████▌| 384928/400000 [00:42<00:01, 9173.85it/s] 96%|█████████▋| 385856/400000 [00:42<00:01, 9202.81it/s] 97%|█████████▋| 386779/400000 [00:42<00:01, 9209.40it/s] 97%|█████████▋| 387707/400000 [00:42<00:01, 9229.34it/s] 97%|█████████▋| 388631/400000 [00:42<00:01, 9108.73it/s] 97%|█████████▋| 389548/400000 [00:42<00:01, 9125.29it/s] 98%|█████████▊| 390488/400000 [00:42<00:01, 9204.69it/s] 98%|█████████▊| 391429/400000 [00:43<00:00, 9262.85it/s] 98%|█████████▊| 392373/400000 [00:43<00:00, 9313.73it/s] 98%|█████████▊| 393305/400000 [00:43<00:00, 9182.47it/s] 99%|█████████▊| 394224/400000 [00:43<00:00, 9064.54it/s] 99%|█████████▉| 395148/400000 [00:43<00:00, 9116.40it/s] 99%|█████████▉| 396061/400000 [00:43<00:00, 8895.96it/s] 99%|█████████▉| 396953/400000 [00:43<00:00, 8666.63it/s] 99%|█████████▉| 397823/400000 [00:43<00:00, 8536.56it/s]100%|█████████▉| 398679/400000 [00:43<00:00, 8511.47it/s]100%|█████████▉| 399532/400000 [00:44<00:00, 8345.09it/s]100%|█████████▉| 399999/400000 [00:44<00:00, 9074.55it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f7c5739f7f0>, <torchtext.data.dataset.TabularDataset object at 0x7f7c5739f940>, <torchtext.vocab.Vocab object at 0x7f7c5739f860>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.22 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.22 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  6.22 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.27 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.27 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.91 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.91 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.83 file/s]2020-06-08 06:02:20.341331: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-08 06:02:20.345074: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-08 06:02:20.345244: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561221a10000 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-08 06:02:20.345260: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:06, 147917.74it/s] 79%|███████▊  | 7790592/9912422 [00:00<00:10, 211138.40it/s]9920512it [00:00, 43621922.45it/s]                           
0it [00:00, ?it/s]32768it [00:00, 780140.73it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 162641.95it/s]1654784it [00:00, 11093264.21it/s]                         
0it [00:00, ?it/s]8192it [00:00, 173536.66it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
