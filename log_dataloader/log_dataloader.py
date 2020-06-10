
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '2675f1e090030e6958e45c46c6313291532e6ed8', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/2675f1e090030e6958e45c46c6313291532e6ed8

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f1c509246a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f1c509246a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f1cbbeec0d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f1cbbeec0d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f1cd5c19ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f1cd5c19ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1c69767950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1c69767950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1c69767950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:03, 155478.50it/s] 71%|███████▏  | 7077888/9912422 [00:00<00:12, 221902.61it/s]9920512it [00:00, 43117416.09it/s]                           
0it [00:00, ?it/s]32768it [00:00, 462433.56it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 10392118.25it/s]         
0it [00:00, ?it/s]8192it [00:00, 162685.07it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1c507ffb00>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1c507f5d68>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f1c69767598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f1c69767598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f1c69767598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:16,  2.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:16,  2.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:16,  2.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:15,  2.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:15,  2.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:14,  2.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:14,  2.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:52,  2.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:52,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:52,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:51,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:51,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:51,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:50,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:50,  2.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▊         | 14/162 [00:00<00:35,  4.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:35,  4.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:35,  4.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:35,  4.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:35,  4.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:34,  4.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:34,  4.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:34,  4.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:00<00:24,  5.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:24,  5.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:24,  5.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:24,  5.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:23,  5.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:23,  5.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:23,  5.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:23,  5.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:23,  5.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:00<00:16,  7.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:16,  7.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:16,  7.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:16,  7.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:16,  7.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:16,  7.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:16,  7.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:15,  7.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  22%|██▏       | 36/162 [00:01<00:11, 10.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:11, 10.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:11, 10.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:11, 10.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:11, 10.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:11, 10.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:11, 10.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:11, 10.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:10, 10.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:01<00:08, 14.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:08, 14.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:08, 14.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:07, 14.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:07, 14.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:07, 14.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:07, 14.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:07, 14.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:07, 14.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 19.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 19.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 19.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 19.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 19.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 19.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 19.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 19.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 19.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 24.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 24.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 24.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 24.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 24.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 24.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 24.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 24.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 24.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 30.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 30.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 30.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 30.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 30.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 30.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 30.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 30.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 30.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 37.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 37.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 37.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 37.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 37.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 37.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 37.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 37.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 37.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 43.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 43.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 43.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 43.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 43.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 43.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 43.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 43.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 43.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 48.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 48.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 48.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 48.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 48.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 48.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 48.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 48.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 48.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 54.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 54.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 54.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 54.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 54.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 54.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 54.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 54.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 54.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 58.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 58.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 58.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 58.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 58.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 58.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 58.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 58.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 58.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 61.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 61.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 61.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 61.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 61.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 61.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 61.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 61.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 61.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 64.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 64.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 64.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 64.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 64.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 64.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 64.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 64.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 64.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 66.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 66.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 66.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 66.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 66.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 66.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 66.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 66.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 66.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 67.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 67.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 67.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 67.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 67.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 67.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 67.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 67.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 67.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 66.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 66.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 66.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 66.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 66.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 66.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 66.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 66.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 66.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 67.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 67.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 67.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 67.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 67.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 67.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 67.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 67.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.77s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.77s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 67.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.77s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 67.82 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.03s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.77s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 67.82 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.03s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.03s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 32.19 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.03s/ url]
0 examples [00:00, ? examples/s]2020-06-10 00:09:46.392032: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-10 00:09:46.407990: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-10 00:09:46.408187: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55df943614f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-10 00:09:46.408206: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.14 examples/s]91 examples [00:00,  3.06 examples/s]184 examples [00:00,  4.36 examples/s]273 examples [00:00,  6.21 examples/s]353 examples [00:00,  8.85 examples/s]446 examples [00:00, 12.59 examples/s]542 examples [00:01, 17.88 examples/s]639 examples [00:01, 25.34 examples/s]733 examples [00:01, 35.79 examples/s]822 examples [00:01, 50.26 examples/s]912 examples [00:01, 70.11 examples/s]1002 examples [00:01, 96.91 examples/s]1094 examples [00:01, 132.45 examples/s]1184 examples [00:01, 177.79 examples/s]1277 examples [00:01, 234.61 examples/s]1372 examples [00:01, 302.98 examples/s]1464 examples [00:02, 379.26 examples/s]1556 examples [00:02, 457.50 examples/s]1652 examples [00:02, 541.76 examples/s]1744 examples [00:02, 616.26 examples/s]1836 examples [00:02, 666.91 examples/s]1926 examples [00:02, 721.27 examples/s]2015 examples [00:02, 764.08 examples/s]2111 examples [00:02, 811.75 examples/s]2204 examples [00:02, 843.15 examples/s]2299 examples [00:02, 870.29 examples/s]2392 examples [00:03, 879.76 examples/s]2485 examples [00:03, 892.79 examples/s]2577 examples [00:03, 895.82 examples/s]2672 examples [00:03, 910.45 examples/s]2767 examples [00:03, 919.47 examples/s]2860 examples [00:03, 913.92 examples/s]2954 examples [00:03, 921.01 examples/s]3048 examples [00:03, 924.74 examples/s]3141 examples [00:03, 918.35 examples/s]3238 examples [00:04, 931.09 examples/s]3332 examples [00:04, 933.12 examples/s]3426 examples [00:04, 873.00 examples/s]3515 examples [00:04, 867.52 examples/s]3604 examples [00:04, 873.99 examples/s]3693 examples [00:04, 877.80 examples/s]3788 examples [00:04, 895.85 examples/s]3883 examples [00:04, 909.84 examples/s]3975 examples [00:04, 887.65 examples/s]4070 examples [00:04, 902.63 examples/s]4164 examples [00:05, 913.44 examples/s]4256 examples [00:05, 910.46 examples/s]4348 examples [00:05, 879.25 examples/s]4437 examples [00:05, 880.35 examples/s]4526 examples [00:05, 871.60 examples/s]4614 examples [00:05, 872.57 examples/s]4702 examples [00:05, 838.98 examples/s]4791 examples [00:05, 851.76 examples/s]4877 examples [00:05, 844.21 examples/s]4962 examples [00:06, 817.43 examples/s]5045 examples [00:06, 820.78 examples/s]5137 examples [00:06, 846.59 examples/s]5229 examples [00:06, 866.15 examples/s]5317 examples [00:06, 868.12 examples/s]5406 examples [00:06, 872.14 examples/s]5497 examples [00:06, 880.54 examples/s]5586 examples [00:06, 867.44 examples/s]5676 examples [00:06, 875.83 examples/s]5766 examples [00:06, 880.63 examples/s]5857 examples [00:07, 887.06 examples/s]5951 examples [00:07, 901.53 examples/s]6042 examples [00:07, 867.54 examples/s]6130 examples [00:07, 861.32 examples/s]6217 examples [00:07, 831.47 examples/s]6301 examples [00:07, 818.25 examples/s]6391 examples [00:07, 839.32 examples/s]6479 examples [00:07, 849.64 examples/s]6578 examples [00:07, 885.14 examples/s]6668 examples [00:07, 888.82 examples/s]6758 examples [00:08, 866.04 examples/s]6851 examples [00:08, 882.40 examples/s]6941 examples [00:08, 885.22 examples/s]7031 examples [00:08, 887.10 examples/s]7121 examples [00:08, 890.12 examples/s]7211 examples [00:08, 887.71 examples/s]7300 examples [00:08, 873.78 examples/s]7397 examples [00:08, 899.92 examples/s]7493 examples [00:08, 914.28 examples/s]7586 examples [00:08, 916.79 examples/s]7682 examples [00:09, 927.88 examples/s]7775 examples [00:09, 928.44 examples/s]7868 examples [00:09, 917.08 examples/s]7960 examples [00:09, 898.00 examples/s]8050 examples [00:09, 885.84 examples/s]8143 examples [00:09, 897.93 examples/s]8237 examples [00:09, 907.51 examples/s]8332 examples [00:09, 917.30 examples/s]8424 examples [00:09, 917.96 examples/s]8517 examples [00:10, 919.34 examples/s]8611 examples [00:10, 924.25 examples/s]8704 examples [00:10, 920.79 examples/s]8801 examples [00:10, 933.94 examples/s]8896 examples [00:10, 936.96 examples/s]8990 examples [00:10, 921.03 examples/s]9084 examples [00:10, 925.14 examples/s]9184 examples [00:10, 944.87 examples/s]9280 examples [00:10, 948.88 examples/s]9375 examples [00:10, 947.05 examples/s]9470 examples [00:11, 938.58 examples/s]9564 examples [00:11, 936.72 examples/s]9658 examples [00:11, 933.37 examples/s]9757 examples [00:11, 947.71 examples/s]9852 examples [00:11, 938.46 examples/s]9946 examples [00:11, 909.60 examples/s]10038 examples [00:11, 834.88 examples/s]10135 examples [00:11, 871.18 examples/s]10230 examples [00:11, 892.03 examples/s]10321 examples [00:11, 882.17 examples/s]10416 examples [00:12, 899.92 examples/s]10512 examples [00:12, 915.40 examples/s]10608 examples [00:12, 927.37 examples/s]10702 examples [00:12, 928.38 examples/s]10796 examples [00:12, 930.95 examples/s]10893 examples [00:12, 940.64 examples/s]10988 examples [00:12, 940.99 examples/s]11083 examples [00:12, 939.04 examples/s]11180 examples [00:12, 945.42 examples/s]11275 examples [00:12, 936.71 examples/s]11369 examples [00:13, 934.29 examples/s]11465 examples [00:13, 940.18 examples/s]11562 examples [00:13, 946.16 examples/s]11657 examples [00:13, 937.86 examples/s]11751 examples [00:13, 933.48 examples/s]11850 examples [00:13, 949.47 examples/s]11947 examples [00:13, 952.95 examples/s]12046 examples [00:13, 961.79 examples/s]12143 examples [00:13, 960.98 examples/s]12240 examples [00:14, 943.94 examples/s]12335 examples [00:14, 932.54 examples/s]12429 examples [00:14, 933.82 examples/s]12525 examples [00:14, 940.52 examples/s]12624 examples [00:14, 952.92 examples/s]12720 examples [00:14, 942.55 examples/s]12815 examples [00:14, 928.69 examples/s]12909 examples [00:14, 929.89 examples/s]13006 examples [00:14, 939.40 examples/s]13101 examples [00:14, 925.42 examples/s]13194 examples [00:15, 912.67 examples/s]13289 examples [00:15, 923.26 examples/s]13384 examples [00:15, 929.51 examples/s]13479 examples [00:15, 934.94 examples/s]13576 examples [00:15, 944.00 examples/s]13671 examples [00:15, 932.20 examples/s]13765 examples [00:15, 923.25 examples/s]13863 examples [00:15, 936.89 examples/s]13960 examples [00:15, 946.32 examples/s]14059 examples [00:15, 956.76 examples/s]14155 examples [00:16, 952.32 examples/s]14251 examples [00:16, 953.81 examples/s]14347 examples [00:16, 952.82 examples/s]14446 examples [00:16, 961.82 examples/s]14543 examples [00:16, 957.44 examples/s]14639 examples [00:16, 934.34 examples/s]14737 examples [00:16, 945.87 examples/s]14834 examples [00:16, 952.91 examples/s]14934 examples [00:16, 963.68 examples/s]15031 examples [00:16, 959.67 examples/s]15128 examples [00:17, 961.59 examples/s]15225 examples [00:17, 963.42 examples/s]15324 examples [00:17, 969.34 examples/s]15421 examples [00:17, 954.49 examples/s]15519 examples [00:17, 959.41 examples/s]15616 examples [00:17, 956.68 examples/s]15716 examples [00:17, 968.59 examples/s]15813 examples [00:17, 968.90 examples/s]15910 examples [00:17, 961.77 examples/s]16007 examples [00:17, 945.06 examples/s]16102 examples [00:18, 920.37 examples/s]16195 examples [00:18, 913.89 examples/s]16292 examples [00:18, 928.41 examples/s]16388 examples [00:18, 937.22 examples/s]16482 examples [00:18, 936.11 examples/s]16576 examples [00:18, 936.69 examples/s]16672 examples [00:18, 943.55 examples/s]16768 examples [00:18, 947.49 examples/s]16863 examples [00:18, 947.92 examples/s]16958 examples [00:18, 939.38 examples/s]17053 examples [00:19, 941.94 examples/s]17148 examples [00:19, 943.51 examples/s]17244 examples [00:19, 947.81 examples/s]17339 examples [00:19, 946.93 examples/s]17434 examples [00:19, 916.09 examples/s]17526 examples [00:19, 902.30 examples/s]17622 examples [00:19, 917.87 examples/s]17715 examples [00:19, 920.70 examples/s]17808 examples [00:19, 920.72 examples/s]17901 examples [00:20, 923.27 examples/s]17994 examples [00:20, 913.21 examples/s]18086 examples [00:20, 913.27 examples/s]18178 examples [00:20, 911.03 examples/s]18275 examples [00:20, 925.43 examples/s]18369 examples [00:20, 928.49 examples/s]18462 examples [00:20, 914.30 examples/s]18559 examples [00:20, 930.28 examples/s]18653 examples [00:20, 921.14 examples/s]18748 examples [00:20, 929.06 examples/s]18842 examples [00:21, 894.96 examples/s]18932 examples [00:21, 893.88 examples/s]19026 examples [00:21, 905.83 examples/s]19122 examples [00:21, 919.95 examples/s]19215 examples [00:21, 919.85 examples/s]19310 examples [00:21, 928.46 examples/s]19405 examples [00:21, 932.54 examples/s]19500 examples [00:21, 937.26 examples/s]19596 examples [00:21, 942.62 examples/s]19691 examples [00:21, 942.67 examples/s]19786 examples [00:22, 944.55 examples/s]19881 examples [00:22, 929.87 examples/s]19979 examples [00:22, 942.28 examples/s]20074 examples [00:22, 880.87 examples/s]20171 examples [00:22, 904.71 examples/s]20266 examples [00:22, 916.19 examples/s]20360 examples [00:22, 920.58 examples/s]20457 examples [00:22, 934.51 examples/s]20551 examples [00:22, 911.28 examples/s]20647 examples [00:22, 922.85 examples/s]20740 examples [00:23, 921.89 examples/s]20833 examples [00:23, 918.59 examples/s]20926 examples [00:23, 904.39 examples/s]21017 examples [00:23, 900.05 examples/s]21108 examples [00:23, 897.45 examples/s]21198 examples [00:23, 880.43 examples/s]21287 examples [00:23, 868.73 examples/s]21377 examples [00:23, 877.38 examples/s]21467 examples [00:23, 881.64 examples/s]21556 examples [00:24, 874.54 examples/s]21644 examples [00:24, 862.48 examples/s]21732 examples [00:24, 865.94 examples/s]21828 examples [00:24, 891.90 examples/s]21924 examples [00:24, 910.87 examples/s]22018 examples [00:24, 918.17 examples/s]22111 examples [00:24, 906.17 examples/s]22204 examples [00:24, 911.38 examples/s]22299 examples [00:24, 920.08 examples/s]22392 examples [00:24, 909.43 examples/s]22484 examples [00:25, 897.02 examples/s]22574 examples [00:25, 890.56 examples/s]22664 examples [00:25, 886.29 examples/s]22753 examples [00:25, 851.65 examples/s]22843 examples [00:25, 863.74 examples/s]22936 examples [00:25, 881.16 examples/s]23027 examples [00:25, 887.06 examples/s]23121 examples [00:25, 901.57 examples/s]23216 examples [00:25, 912.91 examples/s]23312 examples [00:25, 925.36 examples/s]23407 examples [00:26, 932.41 examples/s]23501 examples [00:26, 923.70 examples/s]23594 examples [00:26, 901.02 examples/s]23686 examples [00:26, 906.01 examples/s]23781 examples [00:26, 917.88 examples/s]23873 examples [00:26, 911.38 examples/s]23967 examples [00:26, 917.84 examples/s]24062 examples [00:26, 925.59 examples/s]24157 examples [00:26, 930.50 examples/s]24252 examples [00:26, 935.89 examples/s]24348 examples [00:27, 941.70 examples/s]24443 examples [00:27, 915.70 examples/s]24539 examples [00:27, 926.09 examples/s]24636 examples [00:27, 936.60 examples/s]24734 examples [00:27, 947.05 examples/s]24830 examples [00:27, 947.95 examples/s]24926 examples [00:27, 950.13 examples/s]25024 examples [00:27, 958.67 examples/s]25120 examples [00:27, 951.84 examples/s]25216 examples [00:28, 944.01 examples/s]25312 examples [00:28, 945.79 examples/s]25407 examples [00:28, 946.28 examples/s]25502 examples [00:28, 945.77 examples/s]25597 examples [00:28, 936.84 examples/s]25691 examples [00:28, 919.79 examples/s]25784 examples [00:28, 921.12 examples/s]25877 examples [00:28, 917.69 examples/s]25969 examples [00:28, 902.03 examples/s]26060 examples [00:28, 900.19 examples/s]26151 examples [00:29, 886.85 examples/s]26240 examples [00:29, 886.82 examples/s]26334 examples [00:29, 899.50 examples/s]26427 examples [00:29, 905.73 examples/s]26520 examples [00:29, 912.10 examples/s]26616 examples [00:29, 924.53 examples/s]26709 examples [00:29, 923.52 examples/s]26803 examples [00:29, 925.60 examples/s]26896 examples [00:29, 924.96 examples/s]26992 examples [00:29, 935.17 examples/s]27087 examples [00:30, 936.80 examples/s]27181 examples [00:30, 912.31 examples/s]27273 examples [00:30, 876.52 examples/s]27364 examples [00:30, 884.32 examples/s]27458 examples [00:30, 898.78 examples/s]27555 examples [00:30, 916.72 examples/s]27653 examples [00:30, 933.96 examples/s]27747 examples [00:30, 934.62 examples/s]27844 examples [00:30, 944.18 examples/s]27940 examples [00:30, 948.29 examples/s]28035 examples [00:31, 942.47 examples/s]28130 examples [00:31, 943.60 examples/s]28225 examples [00:31, 919.77 examples/s]28319 examples [00:31, 923.15 examples/s]28414 examples [00:31, 928.54 examples/s]28512 examples [00:31, 941.44 examples/s]28610 examples [00:31, 951.49 examples/s]28706 examples [00:31, 946.53 examples/s]28801 examples [00:31, 947.29 examples/s]28896 examples [00:32, 941.30 examples/s]28991 examples [00:32, 907.66 examples/s]29083 examples [00:32, 899.43 examples/s]29174 examples [00:32, 901.30 examples/s]29270 examples [00:32, 916.82 examples/s]29365 examples [00:32, 925.73 examples/s]29462 examples [00:32, 935.88 examples/s]29557 examples [00:32, 938.88 examples/s]29651 examples [00:32, 938.35 examples/s]29745 examples [00:32, 932.83 examples/s]29839 examples [00:33, 932.95 examples/s]29934 examples [00:33, 937.84 examples/s]30028 examples [00:33, 860.43 examples/s]30116 examples [00:33, 863.84 examples/s]30208 examples [00:33, 878.44 examples/s]30303 examples [00:33, 896.75 examples/s]30400 examples [00:33, 915.22 examples/s]30495 examples [00:33, 922.66 examples/s]30588 examples [00:33, 922.73 examples/s]30682 examples [00:33, 924.95 examples/s]30775 examples [00:34, 912.97 examples/s]30871 examples [00:34, 924.69 examples/s]30965 examples [00:34, 926.35 examples/s]31058 examples [00:34, 924.82 examples/s]31151 examples [00:34, 912.40 examples/s]31243 examples [00:34, 910.32 examples/s]31335 examples [00:34, 907.01 examples/s]31433 examples [00:34, 927.54 examples/s]31526 examples [00:34, 925.90 examples/s]31619 examples [00:34, 921.91 examples/s]31720 examples [00:35, 944.44 examples/s]31817 examples [00:35, 950.53 examples/s]31913 examples [00:35, 953.26 examples/s]32009 examples [00:35, 943.94 examples/s]32104 examples [00:35, 941.94 examples/s]32199 examples [00:35, 941.91 examples/s]32294 examples [00:35, 936.93 examples/s]32389 examples [00:35, 938.20 examples/s]32484 examples [00:35, 941.26 examples/s]32579 examples [00:36, 925.27 examples/s]32676 examples [00:36, 935.97 examples/s]32771 examples [00:36, 939.84 examples/s]32866 examples [00:36, 929.22 examples/s]32963 examples [00:36, 939.44 examples/s]33059 examples [00:36, 943.09 examples/s]33155 examples [00:36, 947.96 examples/s]33251 examples [00:36, 950.51 examples/s]33348 examples [00:36, 954.98 examples/s]33444 examples [00:36, 937.85 examples/s]33538 examples [00:37, 899.12 examples/s]33634 examples [00:37, 916.28 examples/s]33731 examples [00:37, 930.04 examples/s]33828 examples [00:37, 939.77 examples/s]33923 examples [00:37, 939.08 examples/s]34018 examples [00:37, 941.19 examples/s]34113 examples [00:37, 942.72 examples/s]34209 examples [00:37, 947.36 examples/s]34304 examples [00:37, 945.84 examples/s]34399 examples [00:37, 938.10 examples/s]34493 examples [00:38, 915.43 examples/s]34588 examples [00:38, 923.53 examples/s]34684 examples [00:38, 933.75 examples/s]34778 examples [00:38, 916.16 examples/s]34870 examples [00:38, 905.71 examples/s]34963 examples [00:38, 909.81 examples/s]35057 examples [00:38, 916.14 examples/s]35151 examples [00:38, 922.96 examples/s]35247 examples [00:38, 930.90 examples/s]35341 examples [00:38, 914.49 examples/s]35435 examples [00:39, 918.80 examples/s]35531 examples [00:39, 930.11 examples/s]35625 examples [00:39, 931.96 examples/s]35719 examples [00:39, 898.48 examples/s]35810 examples [00:39, 901.34 examples/s]35901 examples [00:39, 902.89 examples/s]35992 examples [00:39, 877.59 examples/s]36082 examples [00:39, 883.01 examples/s]36176 examples [00:39, 896.64 examples/s]36267 examples [00:39, 899.70 examples/s]36361 examples [00:40, 909.85 examples/s]36454 examples [00:40, 914.17 examples/s]36546 examples [00:40, 914.76 examples/s]36638 examples [00:40, 913.05 examples/s]36730 examples [00:40, 903.07 examples/s]36821 examples [00:40, 850.25 examples/s]36913 examples [00:40, 868.25 examples/s]37004 examples [00:40, 879.61 examples/s]37093 examples [00:40, 879.76 examples/s]37182 examples [00:41, 882.46 examples/s]37271 examples [00:41, 879.14 examples/s]37360 examples [00:41, 879.54 examples/s]37449 examples [00:41, 850.69 examples/s]37535 examples [00:41, 848.57 examples/s]37626 examples [00:41, 865.13 examples/s]37717 examples [00:41, 875.68 examples/s]37812 examples [00:41, 896.25 examples/s]37908 examples [00:41, 912.59 examples/s]38001 examples [00:41, 915.29 examples/s]38094 examples [00:42, 917.82 examples/s]38188 examples [00:42, 923.85 examples/s]38282 examples [00:42, 926.56 examples/s]38375 examples [00:42, 922.70 examples/s]38468 examples [00:42, 879.75 examples/s]38558 examples [00:42, 884.24 examples/s]38651 examples [00:42, 896.25 examples/s]38743 examples [00:42, 899.78 examples/s]38835 examples [00:42, 904.13 examples/s]38926 examples [00:42, 871.04 examples/s]39014 examples [00:43, 843.09 examples/s]39108 examples [00:43, 868.48 examples/s]39203 examples [00:43, 889.85 examples/s]39297 examples [00:43, 901.41 examples/s]39388 examples [00:43, 902.45 examples/s]39479 examples [00:43, 904.12 examples/s]39574 examples [00:43, 915.31 examples/s]39668 examples [00:43, 920.63 examples/s]39763 examples [00:43, 926.39 examples/s]39856 examples [00:44, 918.14 examples/s]39948 examples [00:44, 892.71 examples/s]40038 examples [00:44, 841.56 examples/s]40135 examples [00:44, 875.69 examples/s]40230 examples [00:44, 896.34 examples/s]40322 examples [00:44, 901.61 examples/s]40413 examples [00:44, 897.49 examples/s]40504 examples [00:44, 860.79 examples/s]40599 examples [00:44, 883.50 examples/s]40694 examples [00:44, 900.23 examples/s]40786 examples [00:45, 903.89 examples/s]40881 examples [00:45, 916.63 examples/s]40977 examples [00:45, 926.63 examples/s]41072 examples [00:45, 932.40 examples/s]41166 examples [00:45, 910.87 examples/s]41258 examples [00:45, 909.32 examples/s]41350 examples [00:45, 891.68 examples/s]41440 examples [00:45, 871.50 examples/s]41528 examples [00:45, 870.95 examples/s]41619 examples [00:45, 880.03 examples/s]41708 examples [00:46, 874.12 examples/s]41803 examples [00:46, 895.42 examples/s]41900 examples [00:46, 914.32 examples/s]41992 examples [00:46, 915.65 examples/s]42086 examples [00:46, 921.59 examples/s]42179 examples [00:46, 918.85 examples/s]42271 examples [00:46, 909.21 examples/s]42367 examples [00:46, 923.75 examples/s]42465 examples [00:46, 938.25 examples/s]42559 examples [00:47, 933.26 examples/s]42653 examples [00:47, 931.64 examples/s]42747 examples [00:47, 933.99 examples/s]42843 examples [00:47, 939.54 examples/s]42937 examples [00:47, 920.88 examples/s]43030 examples [00:47, 912.31 examples/s]43122 examples [00:47, 899.27 examples/s]43215 examples [00:47, 905.72 examples/s]43308 examples [00:47, 910.74 examples/s]43400 examples [00:47, 907.12 examples/s]43491 examples [00:48, 897.75 examples/s]43583 examples [00:48, 902.37 examples/s]43676 examples [00:48, 907.80 examples/s]43769 examples [00:48, 912.70 examples/s]43862 examples [00:48, 916.96 examples/s]43954 examples [00:48, 904.45 examples/s]44045 examples [00:48, 900.56 examples/s]44136 examples [00:48, 901.50 examples/s]44227 examples [00:48, 898.27 examples/s]44319 examples [00:48, 902.87 examples/s]44410 examples [00:49, 903.11 examples/s]44501 examples [00:49, 886.82 examples/s]44591 examples [00:49, 890.61 examples/s]44681 examples [00:49, 893.27 examples/s]44772 examples [00:49, 897.39 examples/s]44864 examples [00:49, 902.82 examples/s]44955 examples [00:49, 882.56 examples/s]45044 examples [00:49, 870.57 examples/s]45136 examples [00:49, 883.63 examples/s]45228 examples [00:49, 893.55 examples/s]45318 examples [00:50, 892.51 examples/s]45411 examples [00:50, 900.98 examples/s]45506 examples [00:50, 914.03 examples/s]45602 examples [00:50, 926.49 examples/s]45697 examples [00:50, 932.27 examples/s]45793 examples [00:50, 940.02 examples/s]45888 examples [00:50, 929.70 examples/s]45983 examples [00:50, 933.08 examples/s]46079 examples [00:50, 937.85 examples/s]46173 examples [00:50, 929.01 examples/s]46267 examples [00:51, 930.46 examples/s]46361 examples [00:51, 926.39 examples/s]46454 examples [00:51, 919.55 examples/s]46547 examples [00:51, 921.40 examples/s]46640 examples [00:51, 922.17 examples/s]46733 examples [00:51, 900.27 examples/s]46825 examples [00:51, 903.53 examples/s]46916 examples [00:51, 890.12 examples/s]47009 examples [00:51, 899.75 examples/s]47100 examples [00:52, 902.30 examples/s]47193 examples [00:52, 910.36 examples/s]47285 examples [00:52, 906.21 examples/s]47376 examples [00:52, 905.25 examples/s]47468 examples [00:52, 908.92 examples/s]47561 examples [00:52, 914.13 examples/s]47654 examples [00:52, 917.44 examples/s]47746 examples [00:52, 908.35 examples/s]47838 examples [00:52, 911.52 examples/s]47931 examples [00:52, 914.48 examples/s]48023 examples [00:53, 912.52 examples/s]48115 examples [00:53, 911.77 examples/s]48207 examples [00:53, 906.13 examples/s]48300 examples [00:53, 910.40 examples/s]48392 examples [00:53, 908.60 examples/s]48485 examples [00:53, 912.01 examples/s]48577 examples [00:53, 827.10 examples/s]48662 examples [00:53, 823.71 examples/s]48749 examples [00:53, 836.84 examples/s]48834 examples [00:53, 834.30 examples/s]48922 examples [00:54, 847.14 examples/s]49008 examples [00:54, 838.27 examples/s]49096 examples [00:54, 849.99 examples/s]49186 examples [00:54, 862.55 examples/s]49279 examples [00:54, 880.68 examples/s]49373 examples [00:54, 895.60 examples/s]49463 examples [00:54, 851.12 examples/s]49549 examples [00:54, 844.45 examples/s]49645 examples [00:54, 874.75 examples/s]49738 examples [00:54, 889.83 examples/s]49832 examples [00:55, 902.93 examples/s]49927 examples [00:55, 913.90 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 6982/50000 [00:00<00:00, 69807.92 examples/s] 37%|███▋      | 18744/50000 [00:00<00:00, 79502.92 examples/s] 61%|██████▏   | 30678/50000 [00:00<00:00, 88349.32 examples/s] 85%|████████▌ | 42603/50000 [00:00<00:00, 95795.21 examples/s]                                                               0 examples [00:00, ? examples/s]76 examples [00:00, 750.44 examples/s]168 examples [00:00, 793.06 examples/s]260 examples [00:00, 826.27 examples/s]351 examples [00:00, 849.51 examples/s]437 examples [00:00, 850.94 examples/s]524 examples [00:00, 854.22 examples/s]610 examples [00:00, 854.55 examples/s]704 examples [00:00, 877.83 examples/s]793 examples [00:00, 879.42 examples/s]884 examples [00:01, 885.89 examples/s]976 examples [00:01, 894.92 examples/s]1068 examples [00:01, 900.30 examples/s]1158 examples [00:01, 896.08 examples/s]1251 examples [00:01, 903.67 examples/s]1342 examples [00:01, 904.31 examples/s]1433 examples [00:01, 726.24 examples/s]1529 examples [00:01, 782.80 examples/s]1622 examples [00:01, 820.06 examples/s]1708 examples [00:02, 817.58 examples/s]1797 examples [00:02, 837.67 examples/s]1890 examples [00:02, 861.68 examples/s]1978 examples [00:02, 851.98 examples/s]2070 examples [00:02, 869.78 examples/s]2159 examples [00:02, 873.61 examples/s]2252 examples [00:02, 887.32 examples/s]2344 examples [00:02, 895.06 examples/s]2436 examples [00:02, 902.27 examples/s]2530 examples [00:02, 912.71 examples/s]2622 examples [00:03, 891.27 examples/s]2718 examples [00:03, 908.56 examples/s]2813 examples [00:03, 917.84 examples/s]2905 examples [00:03, 918.40 examples/s]2998 examples [00:03, 921.73 examples/s]3091 examples [00:03, 923.09 examples/s]3185 examples [00:03, 926.24 examples/s]3278 examples [00:03, 901.66 examples/s]3369 examples [00:03, 897.63 examples/s]3459 examples [00:03, 894.18 examples/s]3549 examples [00:04, 862.76 examples/s]3636 examples [00:04, 850.35 examples/s]3728 examples [00:04, 868.10 examples/s]3818 examples [00:04, 876.34 examples/s]3907 examples [00:04, 880.20 examples/s]3996 examples [00:04, 876.76 examples/s]4087 examples [00:04, 885.04 examples/s]4176 examples [00:04, 885.98 examples/s]4266 examples [00:04, 887.53 examples/s]4355 examples [00:04, 862.32 examples/s]4444 examples [00:05, 868.38 examples/s]4539 examples [00:05, 889.33 examples/s]4633 examples [00:05, 902.97 examples/s]4724 examples [00:05, 902.33 examples/s]4816 examples [00:05, 906.91 examples/s]4907 examples [00:05, 874.96 examples/s]4995 examples [00:05, 846.84 examples/s]5081 examples [00:05, 842.29 examples/s]5177 examples [00:05, 873.79 examples/s]5274 examples [00:06, 899.08 examples/s]5368 examples [00:06, 909.07 examples/s]5462 examples [00:06, 916.89 examples/s]5558 examples [00:06, 926.14 examples/s]5653 examples [00:06, 931.89 examples/s]5750 examples [00:06, 941.31 examples/s]5845 examples [00:06, 934.89 examples/s]5941 examples [00:06, 939.20 examples/s]6037 examples [00:06, 943.03 examples/s]6133 examples [00:06, 947.48 examples/s]6228 examples [00:07, 948.08 examples/s]6323 examples [00:07, 942.50 examples/s]6418 examples [00:07, 911.04 examples/s]6510 examples [00:07, 906.56 examples/s]6601 examples [00:07, 897.60 examples/s]6691 examples [00:07, 890.50 examples/s]6781 examples [00:07, 888.53 examples/s]6873 examples [00:07, 896.85 examples/s]6963 examples [00:07, 882.89 examples/s]7054 examples [00:07, 888.75 examples/s]7143 examples [00:08, 860.62 examples/s]7230 examples [00:08, 858.98 examples/s]7327 examples [00:08, 887.05 examples/s]7423 examples [00:08, 905.55 examples/s]7514 examples [00:08, 887.19 examples/s]7610 examples [00:08, 906.53 examples/s]7701 examples [00:08, 904.60 examples/s]7797 examples [00:08, 919.17 examples/s]7891 examples [00:08, 925.28 examples/s]7984 examples [00:08, 925.24 examples/s]8077 examples [00:09, 912.24 examples/s]8169 examples [00:09, 894.73 examples/s]8263 examples [00:09, 906.23 examples/s]8359 examples [00:09, 918.58 examples/s]8454 examples [00:09, 927.40 examples/s]8547 examples [00:09, 917.56 examples/s]8639 examples [00:09, 906.81 examples/s]8730 examples [00:09, 892.03 examples/s]8820 examples [00:09, 877.51 examples/s]8914 examples [00:10, 894.99 examples/s]9004 examples [00:10, 895.28 examples/s]9095 examples [00:10, 897.55 examples/s]9185 examples [00:10, 890.56 examples/s]9276 examples [00:10, 894.36 examples/s]9369 examples [00:10, 902.73 examples/s]9464 examples [00:10, 914.29 examples/s]9556 examples [00:10, 908.34 examples/s]9647 examples [00:10, 899.51 examples/s]9738 examples [00:10, 883.63 examples/s]9830 examples [00:11, 892.93 examples/s]9920 examples [00:11, 867.27 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteACGS4T/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteACGS4T/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1c69767950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1c69767950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1c69767950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1bf2c0ecf8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1bf3cac860>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1c69767950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1c69767950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1c69767950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1c69754c50>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1c66eef0f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f1be11812f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f1be11812f0> 

  function with postional parmater data_info <function split_train_valid at 0x7f1be11812f0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f1be1181400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f1be1181400> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f1be1181400> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=f10cdc4311437f77a1b028a4dffd528f58f6bdcb87c8ccba105ecf9ed3d4bc03
  Stored in directory: /tmp/pip-ephem-wheel-cache-qidb28e2/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:04<138:00:56, 1.74kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:04<96:50:24, 2.47kB/s] .vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:04<67:49:49, 3.53kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:05<47:27:28, 5.04kB/s].vector_cache/glove.6B.zip:   0%|          | 3.58M/862M [00:05<33:07:12, 7.20kB/s].vector_cache/glove.6B.zip:   1%|          | 7.32M/862M [00:05<23:05:06, 10.3kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.7M/862M [00:05<16:03:34, 14.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.0M/862M [00:05<11:10:23, 21.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.3M/862M [00:05<7:47:35, 30.0kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 26.9M/862M [00:05<5:25:12, 42.8kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.6M/862M [00:05<3:46:11, 61.1kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.4M/862M [00:06<2:37:20, 87.3kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.0M/862M [00:06<1:49:45, 125kB/s] .vector_cache/glove.6B.zip:   6%|▌         | 48.1M/862M [00:06<1:16:20, 178kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.0M/862M [00:06<53:23, 253kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:06<38:54, 347kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:08<29:01, 463kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:09<22:34, 595kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.7M/862M [00:09<16:15, 824kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:09<11:30, 1.16MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:10<23:01, 580kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:11<17:50, 748kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.2M/862M [00:11<12:49, 1.04MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:11<09:07, 1.46MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:12<29:49, 446kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:13<22:24, 593kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.5M/862M [00:13<15:59, 829kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:14<13:56, 948kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:15<12:26, 1.06MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.0M/862M [00:15<09:22, 1.41MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:15<06:42, 1.96MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:16<1:07:26, 195kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:16<48:32, 271kB/s]  .vector_cache/glove.6B.zip:   9%|▊         | 75.1M/862M [00:17<34:14, 383kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:18<26:58, 485kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:18<21:32, 607kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:19<15:45, 829kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:19<11:09, 1.17MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:20<1:53:52, 114kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:20<1:21:00, 161kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.3M/862M [00:21<56:55, 228kB/s]  .vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:22<42:45, 303kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:22<32:43, 396kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.4M/862M [00:22<23:27, 551kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:23<16:32, 779kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:24<18:32, 694kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:24<14:19, 898kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.5M/862M [00:24<10:20, 1.24MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:26<10:13, 1.25MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:26<09:52, 1.30MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.6M/862M [00:26<07:34, 1.69MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:27<05:25, 2.35MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:28<1:25:28, 149kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:28<1:01:07, 208kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.7M/862M [00:28<42:59, 296kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:30<32:58, 384kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:30<25:39, 494kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:30<18:35, 681kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:32<15:00, 840kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:32<11:47, 1.07MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:32<08:30, 1.48MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:34<08:53, 1.41MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:34<08:41, 1.44MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:34<06:37, 1.89MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:34<04:46, 2.61MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:36<11:37, 1.07MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:36<09:25, 1.32MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:36<06:54, 1.80MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:37<05:31, 2.24MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:38<9:39:20, 21.4kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:38<6:45:45, 30.5kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:38<4:42:59, 43.6kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:40<3:46:21, 54.5kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:40<2:40:56, 76.6kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:40<1:53:08, 109kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:42<1:20:52, 152kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:42<57:51, 212kB/s]  .vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:42<40:44, 300kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:44<31:15, 390kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:44<24:21, 500kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:44<17:39, 689kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:44<12:26, 974kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:46<11:49:36, 17.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:46<8:17:42, 24.3kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:46<5:47:54, 34.8kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:48<4:05:39, 49.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:48<2:54:22, 69.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:48<2:02:26, 98.3kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:48<1:25:35, 140kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:50<1:06:56, 179kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:50<48:05, 249kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:50<33:53, 353kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:52<26:27, 451kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:52<19:44, 603kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:52<14:05, 843kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:54<12:37, 939kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:54<11:15, 1.05MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:54<08:22, 1.41MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:54<06:01, 1.96MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:56<09:31, 1.24MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:56<07:53, 1.49MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:56<05:49, 2.02MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:57<06:47, 1.72MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:58<07:09, 1.64MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:58<05:30, 2.12MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:58<03:59, 2.92MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:59<08:44, 1.33MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:00<07:18, 1.59MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [01:00<05:21, 2.17MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:01<06:30, 1.78MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:02<05:44, 2.02MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:02<04:15, 2.71MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:03<05:43, 2.01MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:04<05:10, 2.22MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:04<03:54, 2.94MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:05<05:26, 2.10MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:06<04:56, 2.31MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:06<03:41, 3.09MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:07<05:17, 2.15MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:07<04:52, 2.33MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:08<03:41, 3.07MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:09<05:15, 2.15MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:09<04:49, 2.34MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:10<03:39, 3.08MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:11<05:12, 2.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:11<05:55, 1.89MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:12<04:43, 2.38MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:13<05:06, 2.19MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:13<04:44, 2.36MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:13<03:35, 3.10MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:15<05:06, 2.17MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:15<04:42, 2.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:15<03:31, 3.14MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:17<05:02, 2.19MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:17<05:41, 1.93MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:17<04:32, 2.42MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:19<04:57, 2.21MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:19<04:36, 2.38MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:19<03:30, 3.12MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:21<04:57, 2.20MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:21<04:34, 2.38MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:21<03:28, 3.12MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:23<04:57, 2.18MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:23<04:21, 2.48MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:23<03:24, 3.17MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:23<02:29, 4.31MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:25<42:23, 254kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:25<30:45, 349kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:25<21:45, 493kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:27<17:42, 603kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:27<13:28, 792kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:27<09:38, 1.10MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:29<09:15, 1.15MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:29<08:50, 1.20MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:29<06:45, 1.57MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:29<04:51, 2.17MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:31<19:56, 529kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:31<15:05, 698kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:31<10:47, 974kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:33<09:51, 1.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:33<09:07, 1.15MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:33<06:56, 1.51MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:33<04:58, 2.09MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:34<10:32, 988kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:35<7:38:40, 22.7kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:35<5:21:16, 32.4kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:35<3:43:57, 46.2kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:37<2:48:46, 61.2kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:37<2:00:23, 85.8kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:37<1:24:38, 122kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:37<59:11, 174kB/s]  .vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:39<45:56, 223kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:39<33:05, 310kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:39<23:19, 439kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:39<16:25, 621kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:41<22:53, 446kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:41<18:10, 561kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<13:10, 773kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:41<09:19, 1.09MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:43<11:04, 915kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:43<08:50, 1.14MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:43<06:24, 1.58MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:45<06:42, 1.50MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:45<06:55, 1.45MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:45<05:18, 1.89MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:45<03:50, 2.61MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:47<07:46, 1.28MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:47<06:31, 1.53MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:47<04:49, 2.06MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:49<05:33, 1.78MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:49<06:05, 1.63MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:49<04:47, 2.06MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:49<03:28, 2.84MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:51<17:55, 549kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 272M/862M [01:51<13:36, 723kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:51<09:46, 1.00MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:53<08:58, 1.09MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:53<08:22, 1.17MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:53<06:23, 1.53MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:53<04:34, 2.12MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:55<18:28, 526kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:55<13:58, 694kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:55<10:01, 966kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:57<09:07, 1.06MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:57<08:26, 1.14MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:57<06:20, 1.52MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:57<04:33, 2.10MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:59<06:52, 1.39MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:59<05:50, 1.64MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:59<04:18, 2.22MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:01<05:06, 1.86MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:01<05:40, 1.67MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:01<04:29, 2.11MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:01<03:14, 2.91MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:03<16:28, 572kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:03<12:33, 751kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:03<09:01, 1.04MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:05<08:21, 1.12MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:05<07:50, 1.19MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:05<05:54, 1.58MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:05<04:14, 2.19MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:07<07:28, 1.24MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:07<06:14, 1.49MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:07<04:34, 2.03MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:09<05:14, 1.76MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:09<05:37, 1.64MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:09<04:26, 2.07MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 313M/862M [02:09<03:11, 2.86MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:11<13:42, 667kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:11<10:35, 864kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:11<07:36, 1.20MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:13<07:19, 1.24MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:13<07:08, 1.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:13<05:25, 1.67MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:13<03:53, 2.32MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:15<08:05, 1.11MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:15<06:38, 1.36MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:15<04:52, 1.84MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:17<05:22, 1.66MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:17<05:45, 1.55MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:17<04:26, 2.01MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:17<03:12, 2.77MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:19<06:08, 1.44MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:19<05:15, 1.68MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:19<03:53, 2.28MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:21<04:38, 1.90MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:21<05:07, 1.72MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:21<03:59, 2.20MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:21<02:53, 3.03MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:23<07:04, 1.23MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:23<05:53, 1.48MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:23<04:19, 2.01MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:25<04:55, 1.76MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:25<05:22, 1.61MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:25<04:10, 2.08MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:25<03:00, 2.86MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:27<06:50, 1.26MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:27<05:43, 1.50MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:27<04:13, 2.03MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:29<04:51, 1.76MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:29<05:17, 1.61MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:29<04:05, 2.08MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:29<02:58, 2.86MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:31<05:35, 1.51MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:31<04:50, 1.75MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:31<03:36, 2.33MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:33<04:22, 1.92MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:33<05:02, 1.66MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:33<03:58, 2.11MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:33<03:07, 2.68MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:33<02:17, 3.62MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:34<33:00, 252kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:35<24:09, 344kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:35<17:03, 486kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:35<12:03, 685kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:36<18:37, 443kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:37<14:00, 589kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:37<09:57, 825kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:37<07:04, 1.16MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:38<12:30, 654kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:39<09:56, 822kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:39<07:14, 1.13MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:40<06:44, 1.20MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:41<05:49, 1.39MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:41<04:43, 1.71MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:41<03:23, 2.38MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:42<14:12, 566kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:43<10:53, 738kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:43<07:49, 1.02MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:43<05:37, 1.42MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:44<15:30, 514kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:45<12:01, 662kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:45<08:35, 926kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:45<06:06, 1.29MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:46<10:00, 790kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:46<07:51, 1.00MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:47<05:42, 1.38MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:48<05:43, 1.37MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:48<05:40, 1.38MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:49<04:18, 1.82MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:49<03:07, 2.49MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:50<05:02, 1.54MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:50<04:13, 1.83MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:51<03:09, 2.45MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:52<03:56, 1.95MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:52<03:32, 2.17MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:53<02:40, 2.87MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:54<03:38, 2.10MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:54<04:10, 1.83MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:55<03:20, 2.28MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:55<02:24, 3.14MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:56<11:08, 678kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:56<08:34, 880kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:56<06:09, 1.22MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:58<06:01, 1.24MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:58<05:00, 1.49MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:58<03:39, 2.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:00<04:16, 1.74MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:00<03:46, 1.96MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:00<02:49, 2.61MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:02<03:39, 2.01MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:02<04:11, 1.75MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:02<03:16, 2.24MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:02<02:22, 3.07MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:04<05:06, 1.42MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:04<04:22, 1.67MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:04<03:14, 2.24MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:06<03:51, 1.87MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:06<04:14, 1.70MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:06<03:21, 2.14MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:06<02:25, 2.94MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:08<12:57, 551kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:08<09:50, 725kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:08<07:03, 1.01MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:10<06:29, 1.09MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:10<05:58, 1.18MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:10<04:29, 1.57MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:10<03:13, 2.18MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:12<05:19, 1.32MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:12<04:29, 1.56MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:12<03:19, 2.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:14<03:51, 1.80MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:14<04:10, 1.66MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:14<03:17, 2.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:14<02:22, 2.90MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:16<12:00, 572kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:16<09:07, 752kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:16<06:31, 1.05MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:18<06:06, 1.11MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:18<04:59, 1.36MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:18<03:37, 1.86MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:19<02:56, 2.29MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:20<5:01:43, 22.3kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:20<3:31:11, 31.8kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:20<2:26:50, 45.4kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:22<1:50:28, 60.3kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:22<1:18:47, 84.6kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:22<55:21, 120kB/s]   .vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:22<38:39, 171kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:24<29:36, 223kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:24<21:25, 308kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:24<15:07, 434kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:26<11:58, 545kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:26<09:42, 672kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:26<07:07, 914kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:26<05:01, 1.28MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:28<49:37, 130kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:28<35:23, 182kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:28<24:49, 259kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:30<18:42, 341kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:30<14:27, 442kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:30<10:22, 614kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:30<07:19, 865kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:32<07:28, 845kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:32<05:54, 1.07MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:32<04:15, 1.48MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:34<04:23, 1.42MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:34<04:24, 1.42MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:34<03:22, 1.85MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:34<02:24, 2.57MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:36<05:49, 1.06MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:36<04:44, 1.30MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:36<03:26, 1.78MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:38<03:45, 1.63MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:38<03:56, 1.55MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:38<03:01, 2.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:38<02:11, 2.77MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:40<04:25, 1.37MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:40<03:44, 1.61MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:40<02:45, 2.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:42<03:14, 1.84MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:42<03:33, 1.68MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:42<02:47, 2.14MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:42<02:00, 2.93MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:44<43:29, 136kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:44<31:03, 190kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:44<21:47, 270kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:46<16:26, 355kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:46<12:41, 459kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:46<09:07, 637kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:46<06:25, 899kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:48<07:19, 787kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:48<05:45, 1.00MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:48<04:09, 1.38MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:50<04:09, 1.37MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:50<04:07, 1.38MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:50<03:11, 1.78MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:50<02:17, 2.47MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:52<08:38, 652kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:52<06:38, 846kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:52<04:45, 1.17MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:54<04:32, 1.22MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:54<04:21, 1.27MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:54<03:19, 1.66MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:54<02:22, 2.31MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:55<28:03, 196kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:56<20:12, 271kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:56<14:13, 384kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:57<11:06, 488kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:58<08:52, 610kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:58<06:26, 840kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:58<04:33, 1.18MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:59<05:18, 1.01MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:00<04:16, 1.25MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:00<03:07, 1.70MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:01<03:21, 1.57MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:02<02:55, 1.80MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:02<02:10, 2.41MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:03<02:40, 1.95MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [04:04<02:58, 1.75MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:04<02:19, 2.24MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:04<01:41, 3.05MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:05<03:05, 1.66MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:06<02:43, 1.89MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:06<02:02, 2.51MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:07<02:32, 2.00MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:08<02:55, 1.74MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:08<02:15, 2.23MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:08<01:38, 3.05MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:09<03:08, 1.59MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:10<02:44, 1.82MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:10<02:01, 2.45MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:11<02:30, 1.97MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:12<02:51, 1.73MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:12<02:15, 2.17MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:12<01:38, 2.98MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:13<08:47, 554kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:13<06:38, 731kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:14<04:45, 1.02MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:15<04:24, 1.09MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:15<04:09, 1.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:16<03:09, 1.51MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:16<02:14, 2.11MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:17<07:21, 642kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:17<05:38, 836kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:18<04:03, 1.16MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:19<03:53, 1.20MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:19<03:42, 1.25MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:20<02:47, 1.66MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:20<02:01, 2.28MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:21<02:59, 1.53MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:21<02:35, 1.77MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:22<01:55, 2.38MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:23<02:22, 1.90MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:23<02:39, 1.70MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:23<02:06, 2.14MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:24<01:30, 2.95MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:25<06:37, 671kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:25<05:05, 872kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:25<03:38, 1.21MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:27<03:32, 1.24MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:27<02:56, 1.49MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:27<02:09, 2.02MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:29<02:29, 1.73MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:29<02:39, 1.62MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:29<02:05, 2.06MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:30<01:29, 2.85MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:31<09:48, 433kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:31<07:18, 580kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:31<05:10, 814kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:33<04:32, 918kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:33<04:04, 1.03MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:33<03:03, 1.36MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:33<02:10, 1.89MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:35<07:58, 515kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:35<06:00, 682kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:35<04:17, 950kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:37<03:54, 1.03MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:37<03:37, 1.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:37<02:42, 1.48MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:37<01:56, 2.05MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:39<02:56, 1.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:39<02:29, 1.59MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:39<01:49, 2.16MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:41<02:07, 1.84MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:41<02:18, 1.68MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:41<01:47, 2.16MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:41<01:17, 2.98MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:43<03:39, 1.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:43<02:57, 1.29MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:43<02:09, 1.76MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:45<02:21, 1.59MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:45<02:03, 1.82MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:45<01:31, 2.43MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:47<01:52, 1.96MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:47<02:06, 1.75MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:47<01:40, 2.20MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:47<01:12, 3.01MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:49<06:22, 568kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:49<04:50, 747kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:49<03:26, 1.04MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:50<02:36, 1.36MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:51<2:40:50, 22.1kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:51<1:52:23, 31.5kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:51<1:17:36, 45.0kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:53<59:02, 59.1kB/s]  .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:53<42:01, 82.9kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:53<29:28, 118kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:53<20:22, 168kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:55<20:21, 168kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 657M/862M [04:55<14:35, 234kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:55<10:12, 332kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:57<07:49, 428kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:57<06:11, 541kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:57<04:28, 745kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:57<03:08, 1.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:59<04:01, 814kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:59<03:10, 1.03MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:59<02:17, 1.42MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:01<02:17, 1.40MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:01<02:19, 1.38MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:01<01:46, 1.80MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:01<01:15, 2.51MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:03<03:39, 860kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:03<02:53, 1.08MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:03<02:04, 1.49MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:05<02:07, 1.45MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:05<01:47, 1.70MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:05<01:19, 2.29MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:07<01:36, 1.87MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:07<01:46, 1.69MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:07<01:23, 2.13MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:07<01:00, 2.92MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:09<05:10, 566kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:09<03:55, 744kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:09<02:47, 1.04MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:11<02:35, 1.11MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:11<02:25, 1.18MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:11<01:49, 1.56MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:11<01:18, 2.15MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:13<01:47, 1.56MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:13<01:33, 1.79MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:13<01:08, 2.42MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:15<01:23, 1.95MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:15<01:16, 2.14MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:15<00:57, 2.82MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:17<01:14, 2.13MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:17<01:26, 1.85MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:17<01:07, 2.34MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:17<00:48, 3.20MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:19<01:57, 1.32MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:19<01:35, 1.61MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:19<01:10, 2.17MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:21<01:22, 1.83MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:21<01:28, 1.71MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:21<01:09, 2.16MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:21<00:49, 2.99MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:23<05:10, 473kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:23<03:52, 629kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:23<02:44, 880kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:25<02:25, 983kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:25<02:11, 1.08MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:25<01:38, 1.44MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:25<01:09, 2.00MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:27<01:44, 1.32MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:27<01:25, 1.61MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:27<01:03, 2.16MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:29<01:13, 1.84MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:29<01:19, 1.68MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:29<01:01, 2.17MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:29<00:43, 2.98MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:30<01:39, 1.30MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:31<01:23, 1.55MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:31<01:00, 2.11MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:32<01:10, 1.79MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:33<01:16, 1.65MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:33<00:58, 2.13MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:33<00:42, 2.92MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:34<01:23, 1.47MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:35<01:11, 1.70MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:35<00:52, 2.28MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:36<01:02, 1.89MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:37<01:08, 1.72MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:37<00:52, 2.21MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:37<00:38, 3.01MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:38<01:12, 1.56MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:39<01:03, 1.79MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:39<00:46, 2.41MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:40<00:55, 1.95MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:41<01:03, 1.72MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:41<00:50, 2.17MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:41<00:35, 2.97MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:42<03:10, 553kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:43<02:23, 730kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:43<01:41, 1.02MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:44<01:32, 1.09MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:45<01:26, 1.17MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:45<01:04, 1.54MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:45<00:45, 2.15MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:46<02:11, 736kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:47<01:42, 943kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:47<01:12, 1.30MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:48<01:10, 1.32MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:49<01:09, 1.33MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:49<00:53, 1.72MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:49<00:37, 2.37MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:50<02:45, 536kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:50<02:04, 709kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:51<01:27, 991kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:52<01:18, 1.07MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:52<01:13, 1.15MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:53<00:55, 1.51MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:53<00:38, 2.10MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:54<02:32, 525kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:54<01:54, 694kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:55<01:20, 970kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:56<01:12, 1.05MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:56<01:07, 1.12MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:57<00:50, 1.49MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:57<00:35, 2.07MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:58<00:51, 1.39MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:58<00:43, 1.63MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:59<00:31, 2.19MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:00<00:36, 1.85MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:00<00:39, 1.69MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:00<00:30, 2.18MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:01<00:21, 2.96MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:02<00:37, 1.69MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:02<00:33, 1.91MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:02<00:24, 2.57MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:04<00:29, 2.02MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:04<00:32, 1.81MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:04<00:25, 2.33MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:05<00:17, 3.17MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:06<00:38, 1.43MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:06<00:32, 1.67MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:06<00:23, 2.26MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:08<00:27, 1.88MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:08<00:29, 1.71MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:08<00:22, 2.20MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:09<00:15, 3.01MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:10<00:34, 1.36MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:10<00:28, 1.65MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:10<00:20, 2.21MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:12<00:23, 1.86MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:12<00:25, 1.70MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:12<00:19, 2.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:12<00:13, 2.93MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:14<00:20, 1.93MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:14<00:18, 2.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:14<00:13, 2.81MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:16<00:16, 2.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:16<00:18, 1.83MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:16<00:14, 2.30MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:17<00:09, 3.16MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:18<02:31, 200kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:18<01:47, 278kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:18<01:12, 394kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:18<00:47, 558kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:20<01:01, 422kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:20<00:48, 531kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:20<00:34, 733kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:20<00:23, 1.02MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:22<00:20, 1.05MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:22<00:16, 1.28MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:22<00:11, 1.76MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:24<00:11, 1.61MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:24<00:11, 1.52MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:24<00:08, 1.94MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:24<00:05, 2.68MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:26<00:24, 566kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:26<00:17, 745kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:26<00:11, 1.04MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:28<00:08, 1.10MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:28<00:07, 1.18MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:28<00:05, 1.56MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:28<00:03, 2.15MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:30<00:03, 1.53MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:30<00:02, 1.76MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:30<00:01, 2.37MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:32<00:00, 1.94MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:32<00:00, 2.13MB/s].vector_cache/glove.6B.zip: 862MB [06:32, 2.20MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<31:50:03,  3.49it/s]  0%|          | 766/400000 [00:00<22:14:44,  4.99it/s]  0%|          | 1555/400000 [00:00<15:32:43,  7.12it/s]  1%|          | 2355/400000 [00:00<10:51:50, 10.17it/s]  1%|          | 3146/400000 [00:00<7:35:38, 14.52it/s]   1%|          | 3912/400000 [00:00<5:18:35, 20.72it/s]  1%|          | 4677/400000 [00:00<3:42:50, 29.57it/s]  1%|▏         | 5483/400000 [00:00<2:35:54, 42.17it/s]  2%|▏         | 6280/400000 [00:01<1:49:10, 60.11it/s]  2%|▏         | 7079/400000 [00:01<1:16:30, 85.59it/s]  2%|▏         | 7867/400000 [00:01<53:41, 121.71it/s]   2%|▏         | 8636/400000 [00:01<37:46, 172.67it/s]  2%|▏         | 9415/400000 [00:01<26:38, 244.35it/s]  3%|▎         | 10219/400000 [00:01<18:51, 344.57it/s]  3%|▎         | 11006/400000 [00:01<13:25, 483.16it/s]  3%|▎         | 11787/400000 [00:01<09:37, 672.23it/s]  3%|▎         | 12565/400000 [00:01<06:59, 922.73it/s]  3%|▎         | 13373/400000 [00:02<05:07, 1256.63it/s]  4%|▎         | 14145/400000 [00:02<03:51, 1666.99it/s]  4%|▎         | 14896/400000 [00:02<02:59, 2144.92it/s]  4%|▍         | 15695/400000 [00:02<02:19, 2747.97it/s]  4%|▍         | 16454/400000 [00:02<01:52, 3397.82it/s]  4%|▍         | 17226/400000 [00:02<01:33, 4083.72it/s]  5%|▍         | 18025/400000 [00:02<01:19, 4785.28it/s]  5%|▍         | 18838/400000 [00:02<01:09, 5457.95it/s]  5%|▍         | 19636/400000 [00:02<01:03, 6028.49it/s]  5%|▌         | 20423/400000 [00:02<00:58, 6481.10it/s]  5%|▌         | 21210/400000 [00:03<00:55, 6777.14it/s]  5%|▌         | 21995/400000 [00:03<00:53, 7064.56it/s]  6%|▌         | 22804/400000 [00:03<00:51, 7341.74it/s]  6%|▌         | 23596/400000 [00:03<00:50, 7504.17it/s]  6%|▌         | 24386/400000 [00:03<00:49, 7609.99it/s]  6%|▋         | 25175/400000 [00:03<00:49, 7637.02it/s]  6%|▋         | 25958/400000 [00:03<00:48, 7672.00it/s]  7%|▋         | 26742/400000 [00:03<00:48, 7719.33it/s]  7%|▋         | 27524/400000 [00:03<00:48, 7749.13it/s]  7%|▋         | 28306/400000 [00:03<00:48, 7719.83it/s]  7%|▋         | 29100/400000 [00:04<00:47, 7783.56it/s]  7%|▋         | 29882/400000 [00:04<00:48, 7656.94it/s]  8%|▊         | 30662/400000 [00:04<00:47, 7697.97it/s]  8%|▊         | 31460/400000 [00:04<00:47, 7779.89it/s]  8%|▊         | 32240/400000 [00:04<00:47, 7686.02it/s]  8%|▊         | 33027/400000 [00:04<00:47, 7739.13it/s]  8%|▊         | 33802/400000 [00:04<00:47, 7662.71it/s]  9%|▊         | 34570/400000 [00:04<00:47, 7665.58it/s]  9%|▉         | 35376/400000 [00:04<00:46, 7778.65it/s]  9%|▉         | 36155/400000 [00:04<00:46, 7769.38it/s]  9%|▉         | 36952/400000 [00:05<00:46, 7826.95it/s]  9%|▉         | 37752/400000 [00:05<00:45, 7876.72it/s] 10%|▉         | 38541/400000 [00:05<00:46, 7856.61it/s] 10%|▉         | 39354/400000 [00:05<00:45, 7935.37it/s] 10%|█         | 40148/400000 [00:05<00:45, 7832.74it/s] 10%|█         | 40950/400000 [00:05<00:45, 7886.57it/s] 10%|█         | 41762/400000 [00:05<00:45, 7952.50it/s] 11%|█         | 42558/400000 [00:05<00:45, 7905.57it/s] 11%|█         | 43349/400000 [00:05<00:45, 7840.62it/s] 11%|█         | 44134/400000 [00:05<00:45, 7808.09it/s] 11%|█         | 44941/400000 [00:06<00:45, 7884.03it/s] 11%|█▏        | 45759/400000 [00:06<00:44, 7969.50it/s] 12%|█▏        | 46583/400000 [00:06<00:43, 8046.83it/s] 12%|█▏        | 47389/400000 [00:06<00:44, 7914.90it/s] 12%|█▏        | 48182/400000 [00:06<00:45, 7757.37it/s] 12%|█▏        | 48960/400000 [00:06<00:45, 7743.74it/s] 12%|█▏        | 49739/400000 [00:06<00:45, 7756.73it/s] 13%|█▎        | 50537/400000 [00:06<00:44, 7821.24it/s] 13%|█▎        | 51320/400000 [00:06<00:45, 7706.13it/s] 13%|█▎        | 52092/400000 [00:06<00:45, 7674.74it/s] 13%|█▎        | 52892/400000 [00:07<00:44, 7767.00it/s] 13%|█▎        | 53690/400000 [00:07<00:44, 7828.67it/s] 14%|█▎        | 54496/400000 [00:07<00:43, 7894.89it/s] 14%|█▍        | 55287/400000 [00:07<00:43, 7843.69it/s] 14%|█▍        | 56072/400000 [00:07<00:45, 7533.26it/s] 14%|█▍        | 56830/400000 [00:07<00:45, 7545.67it/s] 14%|█▍        | 57622/400000 [00:07<00:44, 7652.06it/s] 15%|█▍        | 58424/400000 [00:07<00:44, 7757.08it/s] 15%|█▍        | 59238/400000 [00:07<00:43, 7865.96it/s] 15%|█▌        | 60027/400000 [00:08<00:44, 7721.69it/s] 15%|█▌        | 60824/400000 [00:08<00:43, 7792.11it/s] 15%|█▌        | 61629/400000 [00:08<00:43, 7866.85it/s] 16%|█▌        | 62426/400000 [00:08<00:42, 7897.32it/s] 16%|█▌        | 63235/400000 [00:08<00:42, 7950.92it/s] 16%|█▌        | 64031/400000 [00:08<00:42, 7864.15it/s] 16%|█▌        | 64819/400000 [00:08<00:42, 7820.55it/s] 16%|█▋        | 65604/400000 [00:08<00:42, 7827.94it/s] 17%|█▋        | 66388/400000 [00:08<00:42, 7821.36it/s] 17%|█▋        | 67171/400000 [00:08<00:43, 7738.56it/s] 17%|█▋        | 67946/400000 [00:09<00:43, 7623.31it/s] 17%|█▋        | 68709/400000 [00:09<00:44, 7519.35it/s] 17%|█▋        | 69462/400000 [00:09<00:45, 7343.87it/s] 18%|█▊        | 70254/400000 [00:09<00:43, 7506.60it/s] 18%|█▊        | 71033/400000 [00:09<00:43, 7588.54it/s] 18%|█▊        | 71794/400000 [00:09<00:43, 7579.06it/s] 18%|█▊        | 72601/400000 [00:09<00:42, 7719.32it/s] 18%|█▊        | 73377/400000 [00:09<00:42, 7729.61it/s] 19%|█▊        | 74192/400000 [00:09<00:41, 7848.05it/s] 19%|█▊        | 74994/400000 [00:09<00:41, 7898.46it/s] 19%|█▉        | 75785/400000 [00:10<00:41, 7864.37it/s] 19%|█▉        | 76603/400000 [00:10<00:40, 7955.59it/s] 19%|█▉        | 77400/400000 [00:10<00:41, 7832.19it/s] 20%|█▉        | 78214/400000 [00:10<00:40, 7919.23it/s] 20%|█▉        | 79028/400000 [00:10<00:40, 7981.99it/s] 20%|█▉        | 79827/400000 [00:10<00:40, 7840.24it/s] 20%|██        | 80618/400000 [00:10<00:40, 7860.18it/s] 20%|██        | 81413/400000 [00:10<00:40, 7885.36it/s] 21%|██        | 82203/400000 [00:10<00:41, 7645.78it/s] 21%|██        | 82970/400000 [00:10<00:41, 7647.70it/s] 21%|██        | 83744/400000 [00:11<00:41, 7673.28it/s] 21%|██        | 84551/400000 [00:11<00:40, 7786.03it/s] 21%|██▏       | 85336/400000 [00:11<00:40, 7804.77it/s] 22%|██▏       | 86121/400000 [00:11<00:40, 7816.10it/s] 22%|██▏       | 86937/400000 [00:11<00:39, 7914.64it/s] 22%|██▏       | 87730/400000 [00:11<00:40, 7770.57it/s] 22%|██▏       | 88509/400000 [00:11<00:40, 7765.70it/s] 22%|██▏       | 89291/400000 [00:11<00:39, 7780.49it/s] 23%|██▎       | 90094/400000 [00:11<00:39, 7853.33it/s] 23%|██▎       | 90880/400000 [00:11<00:39, 7768.60it/s] 23%|██▎       | 91658/400000 [00:12<00:39, 7765.68it/s] 23%|██▎       | 92458/400000 [00:12<00:39, 7832.94it/s] 23%|██▎       | 93253/400000 [00:12<00:38, 7867.24it/s] 24%|██▎       | 94041/400000 [00:12<00:39, 7768.71it/s] 24%|██▎       | 94819/400000 [00:12<00:40, 7625.15it/s] 24%|██▍       | 95583/400000 [00:12<00:40, 7505.01it/s] 24%|██▍       | 96356/400000 [00:12<00:40, 7568.37it/s] 24%|██▍       | 97160/400000 [00:12<00:39, 7702.16it/s] 24%|██▍       | 97932/400000 [00:12<00:39, 7705.77it/s] 25%|██▍       | 98718/400000 [00:12<00:38, 7749.80it/s] 25%|██▍       | 99494/400000 [00:13<00:39, 7633.95it/s] 25%|██▌       | 100263/400000 [00:13<00:39, 7647.36it/s] 25%|██▌       | 101064/400000 [00:13<00:38, 7752.52it/s] 25%|██▌       | 101866/400000 [00:13<00:38, 7829.56it/s] 26%|██▌       | 102650/400000 [00:13<00:38, 7655.87it/s] 26%|██▌       | 103417/400000 [00:13<00:39, 7579.93it/s] 26%|██▌       | 104217/400000 [00:13<00:38, 7698.98it/s] 26%|██▋       | 105023/400000 [00:13<00:37, 7802.25it/s] 26%|██▋       | 105827/400000 [00:13<00:37, 7870.33it/s] 27%|██▋       | 106622/400000 [00:14<00:37, 7893.04it/s] 27%|██▋       | 107413/400000 [00:14<00:37, 7814.10it/s] 27%|██▋       | 108196/400000 [00:14<00:38, 7576.99it/s] 27%|██▋       | 108956/400000 [00:14<00:38, 7504.08it/s] 27%|██▋       | 109761/400000 [00:14<00:37, 7658.49it/s] 28%|██▊       | 110568/400000 [00:14<00:37, 7775.19it/s] 28%|██▊       | 111348/400000 [00:14<00:37, 7666.21it/s] 28%|██▊       | 112121/400000 [00:14<00:37, 7682.59it/s] 28%|██▊       | 112925/400000 [00:14<00:36, 7783.74it/s] 28%|██▊       | 113726/400000 [00:14<00:36, 7848.77it/s] 29%|██▊       | 114523/400000 [00:15<00:36, 7882.23it/s] 29%|██▉       | 115312/400000 [00:15<00:36, 7819.91it/s] 29%|██▉       | 116095/400000 [00:15<00:36, 7820.28it/s] 29%|██▉       | 116878/400000 [00:15<00:36, 7733.41it/s] 29%|██▉       | 117693/400000 [00:15<00:35, 7853.13it/s] 30%|██▉       | 118501/400000 [00:15<00:35, 7918.96it/s] 30%|██▉       | 119294/400000 [00:15<00:35, 7835.91it/s] 30%|███       | 120079/400000 [00:15<00:36, 7748.45it/s] 30%|███       | 120855/400000 [00:15<00:36, 7703.00it/s] 30%|███       | 121629/400000 [00:15<00:36, 7712.03it/s] 31%|███       | 122434/400000 [00:16<00:35, 7809.01it/s] 31%|███       | 123222/400000 [00:16<00:35, 7829.02it/s] 31%|███       | 124049/400000 [00:16<00:34, 7953.73it/s] 31%|███       | 124854/400000 [00:16<00:34, 7981.86it/s] 31%|███▏      | 125677/400000 [00:16<00:34, 8051.88it/s] 32%|███▏      | 126518/400000 [00:16<00:33, 8154.94it/s] 32%|███▏      | 127335/400000 [00:16<00:33, 8131.43it/s] 32%|███▏      | 128171/400000 [00:16<00:33, 8197.18it/s] 32%|███▏      | 128998/400000 [00:16<00:32, 8216.84it/s] 32%|███▏      | 129821/400000 [00:16<00:33, 8161.67it/s] 33%|███▎      | 130666/400000 [00:17<00:32, 8243.60it/s] 33%|███▎      | 131491/400000 [00:17<00:33, 8136.20it/s] 33%|███▎      | 132306/400000 [00:17<00:33, 8097.35it/s] 33%|███▎      | 133157/400000 [00:17<00:32, 8215.15it/s] 33%|███▎      | 133980/400000 [00:17<00:32, 8201.72it/s] 34%|███▎      | 134801/400000 [00:17<00:32, 8116.40it/s] 34%|███▍      | 135614/400000 [00:17<00:33, 7936.48it/s] 34%|███▍      | 136447/400000 [00:17<00:32, 8049.20it/s] 34%|███▍      | 137287/400000 [00:17<00:32, 8149.93it/s] 35%|███▍      | 138104/400000 [00:17<00:32, 8099.69it/s] 35%|███▍      | 138915/400000 [00:18<00:32, 7941.56it/s] 35%|███▍      | 139711/400000 [00:18<00:33, 7866.04it/s] 35%|███▌      | 140520/400000 [00:18<00:32, 7927.68it/s] 35%|███▌      | 141326/400000 [00:18<00:32, 7966.37it/s] 36%|███▌      | 142136/400000 [00:18<00:32, 8004.81it/s] 36%|███▌      | 142965/400000 [00:18<00:31, 8087.41it/s] 36%|███▌      | 143775/400000 [00:18<00:31, 8043.84it/s] 36%|███▌      | 144580/400000 [00:18<00:31, 8031.25it/s] 36%|███▋      | 145391/400000 [00:18<00:31, 8053.55it/s] 37%|███▋      | 146223/400000 [00:18<00:31, 8131.35it/s] 37%|███▋      | 147039/400000 [00:19<00:31, 8137.95it/s] 37%|███▋      | 147854/400000 [00:19<00:31, 8080.09it/s] 37%|███▋      | 148663/400000 [00:19<00:31, 7989.64it/s] 37%|███▋      | 149463/400000 [00:19<00:33, 7587.12it/s] 38%|███▊      | 150266/400000 [00:19<00:32, 7711.84it/s] 38%|███▊      | 151041/400000 [00:19<00:32, 7705.42it/s] 38%|███▊      | 151826/400000 [00:19<00:32, 7745.55it/s] 38%|███▊      | 152672/400000 [00:19<00:31, 7944.68it/s] 38%|███▊      | 153524/400000 [00:19<00:30, 8106.78it/s] 39%|███▊      | 154369/400000 [00:20<00:29, 8203.98it/s] 39%|███▉      | 155192/400000 [00:20<00:30, 8112.98it/s] 39%|███▉      | 156005/400000 [00:20<00:30, 8050.96it/s] 39%|███▉      | 156822/400000 [00:20<00:30, 8085.12it/s] 39%|███▉      | 157632/400000 [00:20<00:30, 8077.26it/s] 40%|███▉      | 158479/400000 [00:20<00:29, 8189.00it/s] 40%|███▉      | 159299/400000 [00:20<00:29, 8190.25it/s] 40%|████      | 160119/400000 [00:20<00:29, 8122.07it/s] 40%|████      | 160932/400000 [00:20<00:29, 8117.14it/s] 40%|████      | 161748/400000 [00:20<00:29, 8129.03it/s] 41%|████      | 162562/400000 [00:21<00:29, 8049.27it/s] 41%|████      | 163375/400000 [00:21<00:29, 8072.07it/s] 41%|████      | 164183/400000 [00:21<00:29, 7861.34it/s] 41%|████      | 164989/400000 [00:21<00:29, 7918.04it/s] 41%|████▏     | 165797/400000 [00:21<00:29, 7963.13it/s] 42%|████▏     | 166595/400000 [00:21<00:29, 7955.75it/s] 42%|████▏     | 167402/400000 [00:21<00:29, 7987.97it/s] 42%|████▏     | 168202/400000 [00:21<00:29, 7753.68it/s] 42%|████▏     | 168980/400000 [00:21<00:29, 7738.99it/s] 42%|████▏     | 169788/400000 [00:21<00:29, 7836.42it/s] 43%|████▎     | 170612/400000 [00:22<00:28, 7950.88it/s] 43%|████▎     | 171439/400000 [00:22<00:28, 8040.42it/s] 43%|████▎     | 172245/400000 [00:22<00:30, 7526.44it/s] 43%|████▎     | 173006/400000 [00:22<00:30, 7478.12it/s] 43%|████▎     | 173789/400000 [00:22<00:29, 7577.86it/s] 44%|████▎     | 174551/400000 [00:22<00:30, 7482.24it/s] 44%|████▍     | 175303/400000 [00:22<00:30, 7312.52it/s] 44%|████▍     | 176066/400000 [00:22<00:30, 7403.39it/s] 44%|████▍     | 176855/400000 [00:22<00:29, 7541.48it/s] 44%|████▍     | 177689/400000 [00:22<00:28, 7764.05it/s] 45%|████▍     | 178469/400000 [00:23<00:28, 7728.30it/s] 45%|████▍     | 179245/400000 [00:23<00:28, 7682.45it/s] 45%|████▌     | 180018/400000 [00:23<00:28, 7695.52it/s] 45%|████▌     | 180830/400000 [00:23<00:28, 7816.88it/s] 45%|████▌     | 181641/400000 [00:23<00:27, 7902.00it/s] 46%|████▌     | 182481/400000 [00:23<00:27, 8043.73it/s] 46%|████▌     | 183302/400000 [00:23<00:26, 8091.12it/s] 46%|████▌     | 184113/400000 [00:23<00:27, 7876.18it/s] 46%|████▌     | 184903/400000 [00:23<00:27, 7712.75it/s] 46%|████▋     | 185690/400000 [00:24<00:27, 7757.07it/s] 47%|████▋     | 186489/400000 [00:24<00:27, 7824.64it/s] 47%|████▋     | 187278/400000 [00:24<00:27, 7842.34it/s] 47%|████▋     | 188064/400000 [00:24<00:28, 7526.37it/s] 47%|████▋     | 188820/400000 [00:24<00:28, 7533.48it/s] 47%|████▋     | 189576/400000 [00:24<00:27, 7527.29it/s] 48%|████▊     | 190332/400000 [00:24<00:27, 7536.58it/s] 48%|████▊     | 191111/400000 [00:24<00:27, 7608.87it/s] 48%|████▊     | 191875/400000 [00:24<00:27, 7615.77it/s] 48%|████▊     | 192645/400000 [00:24<00:27, 7638.93it/s] 48%|████▊     | 193410/400000 [00:25<00:27, 7633.91it/s] 49%|████▊     | 194174/400000 [00:25<00:27, 7542.76it/s] 49%|████▊     | 194953/400000 [00:25<00:26, 7615.17it/s] 49%|████▉     | 195734/400000 [00:25<00:26, 7670.92it/s] 49%|████▉     | 196534/400000 [00:25<00:26, 7764.24it/s] 49%|████▉     | 197340/400000 [00:25<00:25, 7848.08it/s] 50%|████▉     | 198126/400000 [00:25<00:26, 7744.15it/s] 50%|████▉     | 198936/400000 [00:25<00:25, 7845.28it/s] 50%|████▉     | 199737/400000 [00:25<00:25, 7891.83it/s] 50%|█████     | 200541/400000 [00:25<00:25, 7933.15it/s] 50%|█████     | 201335/400000 [00:26<00:25, 7751.83it/s] 51%|█████     | 202112/400000 [00:26<00:26, 7452.61it/s] 51%|█████     | 202861/400000 [00:26<00:26, 7383.09it/s] 51%|█████     | 203602/400000 [00:26<00:27, 7250.16it/s] 51%|█████     | 204349/400000 [00:26<00:26, 7312.72it/s] 51%|█████▏    | 205158/400000 [00:26<00:25, 7527.49it/s] 51%|█████▏    | 205934/400000 [00:26<00:25, 7593.90it/s] 52%|█████▏    | 206696/400000 [00:26<00:25, 7522.67it/s] 52%|█████▏    | 207450/400000 [00:26<00:25, 7499.11it/s] 52%|█████▏    | 208253/400000 [00:26<00:25, 7648.76it/s] 52%|█████▏    | 209042/400000 [00:27<00:24, 7718.99it/s] 52%|█████▏    | 209816/400000 [00:27<00:24, 7634.46it/s] 53%|█████▎    | 210581/400000 [00:27<00:25, 7416.39it/s] 53%|█████▎    | 211325/400000 [00:27<00:25, 7418.64it/s] 53%|█████▎    | 212117/400000 [00:27<00:24, 7559.87it/s] 53%|█████▎    | 212906/400000 [00:27<00:24, 7654.07it/s] 53%|█████▎    | 213673/400000 [00:27<00:24, 7638.14it/s] 54%|█████▎    | 214438/400000 [00:27<00:24, 7641.36it/s] 54%|█████▍    | 215203/400000 [00:27<00:24, 7560.86it/s] 54%|█████▍    | 215990/400000 [00:27<00:24, 7648.82it/s] 54%|█████▍    | 216800/400000 [00:28<00:23, 7776.39it/s] 54%|█████▍    | 217596/400000 [00:28<00:23, 7829.68it/s] 55%|█████▍    | 218395/400000 [00:28<00:23, 7875.29it/s] 55%|█████▍    | 219184/400000 [00:28<00:23, 7694.47it/s] 55%|█████▍    | 219985/400000 [00:28<00:23, 7786.14it/s] 55%|█████▌    | 220793/400000 [00:28<00:22, 7871.41it/s] 55%|█████▌    | 221605/400000 [00:28<00:22, 7942.77it/s] 56%|█████▌    | 222412/400000 [00:28<00:22, 7978.49it/s] 56%|█████▌    | 223211/400000 [00:28<00:22, 7865.37it/s] 56%|█████▌    | 223999/400000 [00:29<00:22, 7774.53it/s] 56%|█████▌    | 224811/400000 [00:29<00:22, 7874.05it/s] 56%|█████▋    | 225616/400000 [00:29<00:22, 7924.33it/s] 57%|█████▋    | 226410/400000 [00:29<00:21, 7900.31it/s] 57%|█████▋    | 227201/400000 [00:29<00:21, 7867.93it/s] 57%|█████▋    | 227989/400000 [00:29<00:22, 7780.10it/s] 57%|█████▋    | 228799/400000 [00:29<00:21, 7872.13it/s] 57%|█████▋    | 229587/400000 [00:29<00:21, 7842.78it/s] 58%|█████▊    | 230372/400000 [00:29<00:21, 7806.45it/s] 58%|█████▊    | 231153/400000 [00:29<00:21, 7787.80it/s] 58%|█████▊    | 231958/400000 [00:30<00:21, 7864.20it/s] 58%|█████▊    | 232747/400000 [00:30<00:21, 7869.63it/s] 58%|█████▊    | 233551/400000 [00:30<00:21, 7916.27it/s] 59%|█████▊    | 234346/400000 [00:30<00:20, 7925.08it/s] 59%|█████▉    | 235139/400000 [00:30<00:20, 7883.42it/s] 59%|█████▉    | 235951/400000 [00:30<00:20, 7951.57it/s] 59%|█████▉    | 236747/400000 [00:30<00:20, 7836.92it/s] 59%|█████▉    | 237544/400000 [00:30<00:20, 7873.96it/s] 60%|█████▉    | 238356/400000 [00:30<00:20, 7944.97it/s] 60%|█████▉    | 239151/400000 [00:30<00:20, 7903.15it/s] 60%|█████▉    | 239949/400000 [00:31<00:20, 7925.89it/s] 60%|██████    | 240742/400000 [00:31<00:20, 7712.51it/s] 60%|██████    | 241535/400000 [00:31<00:20, 7774.37it/s] 61%|██████    | 242320/400000 [00:31<00:20, 7796.53it/s] 61%|██████    | 243101/400000 [00:31<00:20, 7668.73it/s] 61%|██████    | 243887/400000 [00:31<00:20, 7723.76it/s] 61%|██████    | 244676/400000 [00:31<00:19, 7772.23it/s] 61%|██████▏   | 245454/400000 [00:31<00:19, 7754.53it/s] 62%|██████▏   | 246269/400000 [00:31<00:19, 7867.46it/s] 62%|██████▏   | 247073/400000 [00:31<00:19, 7916.10it/s] 62%|██████▏   | 247898/400000 [00:32<00:18, 8012.81it/s] 62%|██████▏   | 248700/400000 [00:32<00:18, 8003.74it/s] 62%|██████▏   | 249501/400000 [00:32<00:18, 7952.18it/s] 63%|██████▎   | 250309/400000 [00:32<00:18, 7987.44it/s] 63%|██████▎   | 251109/400000 [00:32<00:18, 7929.04it/s] 63%|██████▎   | 251914/400000 [00:32<00:18, 7962.82it/s] 63%|██████▎   | 252743/400000 [00:32<00:18, 8057.13it/s] 63%|██████▎   | 253550/400000 [00:32<00:18, 7862.08it/s] 64%|██████▎   | 254344/400000 [00:32<00:18, 7885.03it/s] 64%|██████▍   | 255157/400000 [00:32<00:18, 7955.62it/s] 64%|██████▍   | 255984/400000 [00:33<00:17, 8045.55it/s] 64%|██████▍   | 256790/400000 [00:33<00:17, 8005.14it/s] 64%|██████▍   | 257592/400000 [00:33<00:17, 7947.28it/s] 65%|██████▍   | 258388/400000 [00:33<00:17, 7944.62it/s] 65%|██████▍   | 259183/400000 [00:33<00:18, 7788.99it/s] 65%|██████▍   | 259980/400000 [00:33<00:17, 7841.21it/s] 65%|██████▌   | 260765/400000 [00:33<00:17, 7777.96it/s] 65%|██████▌   | 261555/400000 [00:33<00:17, 7812.26it/s] 66%|██████▌   | 262337/400000 [00:33<00:17, 7709.93it/s] 66%|██████▌   | 263109/400000 [00:33<00:17, 7687.62it/s] 66%|██████▌   | 263879/400000 [00:34<00:17, 7592.11it/s] 66%|██████▌   | 264662/400000 [00:34<00:17, 7659.63it/s] 66%|██████▋   | 265446/400000 [00:34<00:17, 7711.89it/s] 67%|██████▋   | 266218/400000 [00:34<00:17, 7572.71it/s] 67%|██████▋   | 266977/400000 [00:34<00:18, 7208.78it/s] 67%|██████▋   | 267779/400000 [00:34<00:17, 7433.22it/s] 67%|██████▋   | 268528/400000 [00:34<00:18, 7140.58it/s] 67%|██████▋   | 269300/400000 [00:34<00:17, 7304.73it/s] 68%|██████▊   | 270063/400000 [00:34<00:17, 7397.94it/s] 68%|██████▊   | 270807/400000 [00:35<00:17, 7314.73it/s] 68%|██████▊   | 271580/400000 [00:35<00:17, 7428.04it/s] 68%|██████▊   | 272354/400000 [00:35<00:16, 7518.15it/s] 68%|██████▊   | 273159/400000 [00:35<00:16, 7669.07it/s] 68%|██████▊   | 273973/400000 [00:35<00:16, 7803.29it/s] 69%|██████▊   | 274756/400000 [00:35<00:16, 7776.26it/s] 69%|██████▉   | 275573/400000 [00:35<00:15, 7888.04it/s] 69%|██████▉   | 276383/400000 [00:35<00:15, 7947.69it/s] 69%|██████▉   | 277187/400000 [00:35<00:15, 7974.38it/s] 70%|██████▉   | 278001/400000 [00:35<00:15, 8023.18it/s] 70%|██████▉   | 278804/400000 [00:36<00:15, 7938.58it/s] 70%|██████▉   | 279599/400000 [00:36<00:16, 7310.83it/s] 70%|███████   | 280344/400000 [00:36<00:16, 7351.07it/s] 70%|███████   | 281145/400000 [00:36<00:15, 7536.76it/s] 70%|███████   | 281943/400000 [00:36<00:15, 7663.15it/s] 71%|███████   | 282724/400000 [00:36<00:15, 7705.87it/s] 71%|███████   | 283509/400000 [00:36<00:15, 7745.37it/s] 71%|███████   | 284320/400000 [00:36<00:14, 7849.42it/s] 71%|███████▏  | 285159/400000 [00:36<00:14, 8003.41it/s] 72%|███████▏  | 286007/400000 [00:36<00:14, 8140.38it/s] 72%|███████▏  | 286824/400000 [00:37<00:14, 7996.71it/s] 72%|███████▏  | 287626/400000 [00:37<00:14, 7781.75it/s] 72%|███████▏  | 288407/400000 [00:37<00:14, 7783.82it/s] 72%|███████▏  | 289188/400000 [00:37<00:14, 7775.82it/s] 72%|███████▏  | 289967/400000 [00:37<00:14, 7753.01it/s] 73%|███████▎  | 290744/400000 [00:37<00:14, 7625.27it/s] 73%|███████▎  | 291508/400000 [00:37<00:14, 7614.81it/s] 73%|███████▎  | 292271/400000 [00:37<00:14, 7277.54it/s] 73%|███████▎  | 293071/400000 [00:37<00:14, 7479.10it/s] 73%|███████▎  | 293874/400000 [00:38<00:13, 7633.57it/s] 74%|███████▎  | 294664/400000 [00:38<00:13, 7711.58it/s] 74%|███████▍  | 295461/400000 [00:38<00:13, 7785.20it/s] 74%|███████▍  | 296253/400000 [00:38<00:13, 7824.47it/s] 74%|███████▍  | 297063/400000 [00:38<00:13, 7902.69it/s] 74%|███████▍  | 297856/400000 [00:38<00:12, 7910.23it/s] 75%|███████▍  | 298648/400000 [00:38<00:12, 7895.50it/s] 75%|███████▍  | 299461/400000 [00:38<00:12, 7962.94it/s] 75%|███████▌  | 300259/400000 [00:38<00:12, 7966.73it/s] 75%|███████▌  | 301057/400000 [00:38<00:12, 7927.74it/s] 75%|███████▌  | 301851/400000 [00:39<00:12, 7669.63it/s] 76%|███████▌  | 302624/400000 [00:39<00:12, 7687.43it/s] 76%|███████▌  | 303410/400000 [00:39<00:12, 7737.26it/s] 76%|███████▌  | 304200/400000 [00:39<00:12, 7782.09it/s] 76%|███████▌  | 304980/400000 [00:39<00:12, 7654.32it/s] 76%|███████▋  | 305755/400000 [00:39<00:12, 7681.01it/s] 77%|███████▋  | 306526/400000 [00:39<00:12, 7687.72it/s] 77%|███████▋  | 307321/400000 [00:39<00:11, 7763.12it/s] 77%|███████▋  | 308130/400000 [00:39<00:11, 7858.38it/s] 77%|███████▋  | 308987/400000 [00:39<00:11, 8058.46it/s] 77%|███████▋  | 309802/400000 [00:40<00:11, 8082.63it/s] 78%|███████▊  | 310612/400000 [00:40<00:11, 7976.36it/s] 78%|███████▊  | 311411/400000 [00:40<00:11, 7713.98it/s] 78%|███████▊  | 312212/400000 [00:40<00:11, 7800.29it/s] 78%|███████▊  | 312995/400000 [00:40<00:11, 7802.91it/s] 78%|███████▊  | 313778/400000 [00:40<00:11, 7809.13it/s] 79%|███████▊  | 314560/400000 [00:40<00:10, 7776.47it/s] 79%|███████▉  | 315359/400000 [00:40<00:10, 7836.16it/s] 79%|███████▉  | 316164/400000 [00:40<00:10, 7897.61it/s] 79%|███████▉  | 316963/400000 [00:40<00:10, 7921.81it/s] 79%|███████▉  | 317756/400000 [00:41<00:10, 7825.48it/s] 80%|███████▉  | 318540/400000 [00:41<00:10, 7746.05it/s] 80%|███████▉  | 319316/400000 [00:41<00:10, 7652.84it/s] 80%|████████  | 320104/400000 [00:41<00:10, 7717.72it/s] 80%|████████  | 320927/400000 [00:41<00:10, 7864.08it/s] 80%|████████  | 321715/400000 [00:41<00:09, 7846.48it/s] 81%|████████  | 322501/400000 [00:41<00:09, 7825.12it/s] 81%|████████  | 323306/400000 [00:41<00:09, 7891.09it/s] 81%|████████  | 324127/400000 [00:41<00:09, 7981.47it/s] 81%|████████  | 324935/400000 [00:41<00:09, 8010.32it/s] 81%|████████▏ | 325737/400000 [00:42<00:09, 8011.45it/s] 82%|████████▏ | 326539/400000 [00:42<00:09, 7864.93it/s] 82%|████████▏ | 327337/400000 [00:42<00:09, 7897.69it/s] 82%|████████▏ | 328137/400000 [00:42<00:09, 7926.38it/s] 82%|████████▏ | 328935/400000 [00:42<00:08, 7940.31it/s] 82%|████████▏ | 329730/400000 [00:42<00:09, 7793.21it/s] 83%|████████▎ | 330511/400000 [00:42<00:09, 7621.11it/s] 83%|████████▎ | 331275/400000 [00:42<00:09, 7457.91it/s] 83%|████████▎ | 332023/400000 [00:42<00:09, 7382.02it/s] 83%|████████▎ | 332805/400000 [00:42<00:08, 7506.65it/s] 83%|████████▎ | 333626/400000 [00:43<00:08, 7704.31it/s] 84%|████████▎ | 334399/400000 [00:43<00:08, 7642.89it/s] 84%|████████▍ | 335173/400000 [00:43<00:08, 7667.35it/s] 84%|████████▍ | 335987/400000 [00:43<00:08, 7801.43it/s] 84%|████████▍ | 336769/400000 [00:43<00:08, 7800.95it/s] 84%|████████▍ | 337563/400000 [00:43<00:07, 7839.88it/s] 85%|████████▍ | 338351/400000 [00:43<00:07, 7848.26it/s] 85%|████████▍ | 339137/400000 [00:43<00:07, 7818.12it/s] 85%|████████▍ | 339958/400000 [00:43<00:07, 7929.63it/s] 85%|████████▌ | 340762/400000 [00:43<00:07, 7959.79it/s] 85%|████████▌ | 341559/400000 [00:44<00:07, 7939.50it/s] 86%|████████▌ | 342354/400000 [00:44<00:07, 7873.16it/s] 86%|████████▌ | 343142/400000 [00:44<00:07, 7825.62it/s] 86%|████████▌ | 343950/400000 [00:44<00:07, 7897.19it/s] 86%|████████▌ | 344784/400000 [00:44<00:06, 8024.65it/s] 86%|████████▋ | 345588/400000 [00:44<00:06, 8022.64it/s] 87%|████████▋ | 346391/400000 [00:44<00:06, 7945.44it/s] 87%|████████▋ | 347195/400000 [00:44<00:06, 7973.07it/s] 87%|████████▋ | 348012/400000 [00:44<00:06, 8029.02it/s] 87%|████████▋ | 348816/400000 [00:45<00:06, 8008.90it/s] 87%|████████▋ | 349627/400000 [00:45<00:06, 8036.63it/s] 88%|████████▊ | 350431/400000 [00:45<00:06, 7981.00it/s] 88%|████████▊ | 351230/400000 [00:45<00:06, 7937.57it/s] 88%|████████▊ | 352024/400000 [00:45<00:06, 7734.11it/s] 88%|████████▊ | 352820/400000 [00:45<00:06, 7800.13it/s] 88%|████████▊ | 353617/400000 [00:45<00:05, 7846.25it/s] 89%|████████▊ | 354403/400000 [00:45<00:05, 7776.68it/s] 89%|████████▉ | 355226/400000 [00:45<00:05, 7906.59it/s] 89%|████████▉ | 356039/400000 [00:45<00:05, 7971.44it/s] 89%|████████▉ | 356837/400000 [00:46<00:05, 7964.47it/s] 89%|████████▉ | 357635/400000 [00:46<00:05, 7906.29it/s] 90%|████████▉ | 358427/400000 [00:46<00:05, 7850.74it/s] 90%|████████▉ | 359213/400000 [00:46<00:05, 7727.51it/s] 90%|████████▉ | 359987/400000 [00:46<00:05, 7695.35it/s] 90%|█████████ | 360770/400000 [00:46<00:05, 7734.06it/s] 90%|█████████ | 361568/400000 [00:46<00:04, 7806.12it/s] 91%|█████████ | 362350/400000 [00:46<00:04, 7730.77it/s] 91%|█████████ | 363154/400000 [00:46<00:04, 7820.45it/s] 91%|█████████ | 363955/400000 [00:46<00:04, 7873.20it/s] 91%|█████████ | 364752/400000 [00:47<00:04, 7901.04it/s] 91%|█████████▏| 365543/400000 [00:47<00:04, 7796.97it/s] 92%|█████████▏| 366324/400000 [00:47<00:04, 7691.16it/s] 92%|█████████▏| 367094/400000 [00:47<00:04, 7659.32it/s] 92%|█████████▏| 367907/400000 [00:47<00:04, 7792.42it/s] 92%|█████████▏| 368712/400000 [00:47<00:03, 7866.39it/s] 92%|█████████▏| 369530/400000 [00:47<00:03, 7956.64it/s] 93%|█████████▎| 370327/400000 [00:47<00:03, 7843.88it/s] 93%|█████████▎| 371113/400000 [00:47<00:03, 7768.36it/s] 93%|█████████▎| 371898/400000 [00:47<00:03, 7790.05it/s] 93%|█████████▎| 372678/400000 [00:48<00:03, 7709.85it/s] 93%|█████████▎| 373495/400000 [00:48<00:03, 7840.81it/s] 94%|█████████▎| 374281/400000 [00:48<00:03, 7778.29it/s] 94%|█████████▍| 375094/400000 [00:48<00:03, 7879.41it/s] 94%|█████████▍| 375883/400000 [00:48<00:03, 7875.68it/s] 94%|█████████▍| 376672/400000 [00:48<00:02, 7879.55it/s] 94%|█████████▍| 377461/400000 [00:48<00:02, 7806.89it/s] 95%|█████████▍| 378243/400000 [00:48<00:02, 7769.94it/s] 95%|█████████▍| 379050/400000 [00:48<00:02, 7857.29it/s] 95%|█████████▍| 379854/400000 [00:48<00:02, 7910.72it/s] 95%|█████████▌| 380646/400000 [00:49<00:02, 7875.61it/s] 95%|█████████▌| 381434/400000 [00:49<00:02, 7793.41it/s] 96%|█████████▌| 382214/400000 [00:49<00:02, 7724.63it/s] 96%|█████████▌| 383008/400000 [00:49<00:02, 7785.37it/s] 96%|█████████▌| 383787/400000 [00:49<00:02, 7733.16it/s] 96%|█████████▌| 384561/400000 [00:49<00:02, 7706.60it/s] 96%|█████████▋| 385332/400000 [00:49<00:01, 7685.26it/s] 97%|█████████▋| 386101/400000 [00:49<00:01, 7624.90it/s] 97%|█████████▋| 386879/400000 [00:49<00:01, 7669.17it/s] 97%|█████████▋| 387670/400000 [00:49<00:01, 7739.32it/s] 97%|█████████▋| 388476/400000 [00:50<00:01, 7831.46it/s] 97%|█████████▋| 389281/400000 [00:50<00:01, 7895.74it/s] 98%|█████████▊| 390072/400000 [00:50<00:01, 7788.24it/s] 98%|█████████▊| 390862/400000 [00:50<00:01, 7820.29it/s] 98%|█████████▊| 391645/400000 [00:50<00:01, 7743.26it/s] 98%|█████████▊| 392420/400000 [00:50<00:00, 7685.20it/s] 98%|█████████▊| 393209/400000 [00:50<00:00, 7743.37it/s] 98%|█████████▊| 393984/400000 [00:50<00:00, 7594.44it/s] 99%|█████████▊| 394751/400000 [00:50<00:00, 7614.90it/s] 99%|█████████▉| 395569/400000 [00:51<00:00, 7774.11it/s] 99%|█████████▉| 396377/400000 [00:51<00:00, 7863.01it/s] 99%|█████████▉| 397165/400000 [00:51<00:00, 7794.95it/s] 99%|█████████▉| 397946/400000 [00:51<00:00, 7784.11it/s]100%|█████████▉| 398743/400000 [00:51<00:00, 7838.44it/s]100%|█████████▉| 399545/400000 [00:51<00:00, 7890.03it/s]100%|█████████▉| 399999/400000 [00:51<00:00, 7757.61it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f1cd599a0f0>, <torchtext.data.dataset.TabularDataset object at 0x7f1cd599a240>, <torchtext.vocab.Vocab object at 0x7f1cd599a160>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 12.80 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 14.08 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 14.08 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 14.08 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.42 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.42 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.70 file/s]2020-06-10 00:19:11.352726: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-10 00:19:11.357649: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-10 00:19:11.357885: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56180c20b3a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-10 00:19:11.357904: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:05, 150073.33it/s] 82%|████████▏ | 8101888/9912422 [00:00<00:08, 214219.97it/s]9920512it [00:00, 43520679.73it/s]                           
0it [00:00, ?it/s]32768it [00:00, 483686.21it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 157699.92it/s]1654784it [00:00, 11095427.73it/s]                         
0it [00:00, ?it/s]8192it [00:00, 145671.12it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
