
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/bd7dfc233939710e05244f8e6a394a7ce12a3485', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'bd7dfc233939710e05244f8e6a394a7ce12a3485', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/bd7dfc233939710e05244f8e6a394a7ce12a3485

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/bd7dfc233939710e05244f8e6a394a7ce12a3485

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/bd7dfc233939710e05244f8e6a394a7ce12a3485

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fb7be139f28> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fb7be139f28> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fb8296f10d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fb8296f10d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fb84341dea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fb84341dea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb7d6f6c950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb7d6f6c950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb7d6f6c950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:21, 466578.02it/s] 62%|██████▏   | 6176768/9912422 [00:00<00:05, 664343.61it/s]9920512it [00:00, 40643347.58it/s]                           
0it [00:00, ?it/s]32768it [00:00, 577872.79it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 475984.55it/s]1654784it [00:00, 11565118.61it/s]                         
0it [00:00, ?it/s]8192it [00:00, 207513.91it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb7d45b96a0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb7d45af940>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fb7d6f6c598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fb7d6f6c598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fb7d6f6c598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:15,  2.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:15,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:15,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:14,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:14,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:13,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:13,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:12,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:12,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<00:50,  3.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:50,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:50,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:50,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:49,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:49,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:49,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:48,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:48,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:48,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:34,  4.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:34,  4.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:33,  4.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:33,  4.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:33,  4.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:33,  4.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:32,  4.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:32,  4.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:32,  4.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:00<00:23,  5.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:23,  5.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:22,  5.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:22,  5.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:22,  5.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:22,  5.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:22,  5.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:21,  5.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:21,  5.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:21,  5.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:00<00:15,  8.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:15,  8.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:15,  8.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:15,  8.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:15,  8.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:14,  8.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:14,  8.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:14,  8.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:14,  8.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:14,  8.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:00<00:10, 11.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:10, 11.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:10, 11.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:10, 11.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 11.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 11.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 11.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:09, 11.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:09, 11.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:09, 11.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 15.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 15.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:07, 15.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 15.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:06, 15.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 15.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 15.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 15.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 15.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 15.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 20.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 20.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 20.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 20.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 20.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 20.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 20.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 20.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 20.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 20.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 26.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 26.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 26.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 26.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 26.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 26.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 26.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 26.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 26.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 26.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 33.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 33.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 33.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 33.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 33.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 33.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 33.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 33.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 33.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 33.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 40.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 40.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 40.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 40.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 40.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 40.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 40.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 40.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 40.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 40.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 48.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 48.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 48.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 48.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 48.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 48.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 48.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 48.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 48.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 48.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 54.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 54.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 54.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 54.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 54.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 54.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 54.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 54.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 54.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 54.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 54.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 61.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 61.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 61.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 61.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 61.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 61.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 61.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 61.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 61.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 61.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 67.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 67.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 67.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 67.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 67.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 67.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 67.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 67.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 67.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 67.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 71.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 71.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 71.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 71.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 71.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 71.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 71.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 71.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 71.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 71.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 74.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 74.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 74.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 74.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 74.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 74.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 74.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 74.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 74.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 74.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 77.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 77.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 77.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 77.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 77.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 77.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 77.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 77.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 77.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 77.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.39s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.39s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.39s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.26 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.61s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.39s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 79.26 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.61s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.61s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 35.13 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.61s/ url]
0 examples [00:00, ? examples/s]2020-06-17 18:09:13.280949: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-17 18:09:13.298346: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-06-17 18:09:13.298613: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56148e0766d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-17 18:09:13.298644: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
15 examples [00:00, 147.82 examples/s]125 examples [00:00, 199.67 examples/s]237 examples [00:00, 264.90 examples/s]348 examples [00:00, 343.12 examples/s]459 examples [00:00, 432.48 examples/s]571 examples [00:00, 529.80 examples/s]681 examples [00:00, 627.36 examples/s]790 examples [00:00, 717.80 examples/s]899 examples [00:00, 798.61 examples/s]1006 examples [00:01, 864.31 examples/s]1118 examples [00:01, 927.76 examples/s]1225 examples [00:01, 961.91 examples/s]1336 examples [00:01, 999.37 examples/s]1444 examples [00:01, 1017.23 examples/s]1554 examples [00:01, 1039.96 examples/s]1666 examples [00:01, 1060.69 examples/s]1775 examples [00:01, 1049.47 examples/s]1882 examples [00:01, 1027.10 examples/s]1992 examples [00:01, 1045.35 examples/s]2098 examples [00:02, 1030.77 examples/s]2209 examples [00:02, 1051.36 examples/s]2315 examples [00:02, 1045.69 examples/s]2423 examples [00:02, 1054.25 examples/s]2530 examples [00:02, 1056.55 examples/s]2640 examples [00:02, 1067.93 examples/s]2751 examples [00:02, 1079.35 examples/s]2860 examples [00:02, 1077.85 examples/s]2968 examples [00:02, 1067.09 examples/s]3075 examples [00:02, 1065.76 examples/s]3185 examples [00:03, 1074.66 examples/s]3296 examples [00:03, 1082.74 examples/s]3405 examples [00:03, 1076.48 examples/s]3515 examples [00:03, 1080.98 examples/s]3624 examples [00:03, 1077.20 examples/s]3734 examples [00:03, 1081.45 examples/s]3845 examples [00:03, 1089.25 examples/s]3954 examples [00:03, 1087.79 examples/s]4063 examples [00:03, 1081.25 examples/s]4172 examples [00:03, 1067.60 examples/s]4279 examples [00:04, 1064.70 examples/s]4389 examples [00:04, 1074.07 examples/s]4497 examples [00:04, 1068.69 examples/s]4607 examples [00:04, 1075.36 examples/s]4716 examples [00:04, 1078.95 examples/s]4827 examples [00:04, 1085.55 examples/s]4936 examples [00:04, 1085.85 examples/s]5045 examples [00:04, 1071.56 examples/s]5153 examples [00:04, 1056.39 examples/s]5262 examples [00:04, 1064.21 examples/s]5374 examples [00:05, 1079.41 examples/s]5485 examples [00:05, 1087.35 examples/s]5594 examples [00:05, 1083.91 examples/s]5705 examples [00:05, 1091.00 examples/s]5816 examples [00:05, 1094.77 examples/s]5926 examples [00:05, 1080.72 examples/s]6036 examples [00:05, 1084.33 examples/s]6145 examples [00:05, 1081.37 examples/s]6254 examples [00:05, 1082.34 examples/s]6365 examples [00:05, 1089.26 examples/s]6475 examples [00:06, 1090.59 examples/s]6587 examples [00:06, 1096.90 examples/s]6697 examples [00:06, 1084.22 examples/s]6806 examples [00:06, 1047.26 examples/s]6914 examples [00:06, 1056.65 examples/s]7023 examples [00:06, 1064.03 examples/s]7130 examples [00:06, 1062.68 examples/s]7237 examples [00:06, 1048.24 examples/s]7348 examples [00:06, 1065.25 examples/s]7458 examples [00:07, 1073.99 examples/s]7567 examples [00:07, 1076.90 examples/s]7675 examples [00:07, 1068.70 examples/s]7782 examples [00:07, 1066.58 examples/s]7892 examples [00:07, 1075.15 examples/s]8003 examples [00:07, 1083.15 examples/s]8112 examples [00:07, 1078.43 examples/s]8223 examples [00:07, 1085.50 examples/s]8332 examples [00:07, 1043.07 examples/s]8437 examples [00:07, 1044.62 examples/s]8546 examples [00:08, 1057.00 examples/s]8658 examples [00:08, 1073.02 examples/s]8767 examples [00:08, 1076.54 examples/s]8877 examples [00:08, 1082.80 examples/s]8988 examples [00:08, 1090.70 examples/s]9099 examples [00:08, 1094.03 examples/s]9210 examples [00:08, 1096.85 examples/s]9320 examples [00:08, 1096.47 examples/s]9430 examples [00:08, 1070.06 examples/s]9541 examples [00:08, 1079.53 examples/s]9650 examples [00:09, 1036.75 examples/s]9761 examples [00:09, 1056.48 examples/s]9868 examples [00:09, 1034.47 examples/s]9977 examples [00:09, 1048.88 examples/s]10083 examples [00:09, 1013.92 examples/s]10194 examples [00:09, 1039.04 examples/s]10304 examples [00:09, 1055.75 examples/s]10414 examples [00:09, 1068.07 examples/s]10522 examples [00:09, 1050.97 examples/s]10628 examples [00:09, 1050.98 examples/s]10738 examples [00:10, 1064.46 examples/s]10847 examples [00:10, 1069.93 examples/s]10958 examples [00:10, 1078.56 examples/s]11068 examples [00:10, 1082.98 examples/s]11177 examples [00:10, 1082.53 examples/s]11286 examples [00:10, 1070.29 examples/s]11397 examples [00:10, 1081.69 examples/s]11506 examples [00:10, 1082.18 examples/s]11615 examples [00:10, 1051.53 examples/s]11721 examples [00:11, 1053.18 examples/s]11833 examples [00:11, 1071.61 examples/s]11943 examples [00:11, 1078.43 examples/s]12051 examples [00:11, 1076.80 examples/s]12159 examples [00:11, 1077.73 examples/s]12269 examples [00:11, 1082.69 examples/s]12378 examples [00:11, 1080.68 examples/s]12487 examples [00:11, 1066.87 examples/s]12594 examples [00:11, 1058.96 examples/s]12705 examples [00:11, 1071.06 examples/s]12813 examples [00:12, 1070.37 examples/s]12924 examples [00:12, 1079.81 examples/s]13033 examples [00:12, 1078.05 examples/s]13141 examples [00:12, 1065.49 examples/s]13251 examples [00:12, 1073.43 examples/s]13359 examples [00:12, 1071.03 examples/s]13467 examples [00:12, 1071.99 examples/s]13577 examples [00:12, 1080.04 examples/s]13686 examples [00:12, 1060.88 examples/s]13795 examples [00:12, 1067.80 examples/s]13902 examples [00:13, 1066.09 examples/s]14011 examples [00:13, 1072.68 examples/s]14121 examples [00:13, 1079.12 examples/s]14229 examples [00:13, 1072.77 examples/s]14338 examples [00:13, 1076.32 examples/s]14447 examples [00:13, 1079.53 examples/s]14557 examples [00:13, 1085.43 examples/s]14667 examples [00:13, 1086.94 examples/s]14776 examples [00:13, 1066.97 examples/s]14883 examples [00:13, 1045.14 examples/s]14988 examples [00:14, 1016.83 examples/s]15097 examples [00:14, 1037.72 examples/s]15206 examples [00:14, 1052.37 examples/s]15314 examples [00:14, 1057.93 examples/s]15424 examples [00:14, 1069.20 examples/s]15532 examples [00:14, 1068.93 examples/s]15640 examples [00:14, 1056.65 examples/s]15748 examples [00:14, 1061.45 examples/s]15855 examples [00:14, 1047.17 examples/s]15961 examples [00:14, 1048.64 examples/s]16067 examples [00:15, 1049.55 examples/s]16177 examples [00:15, 1063.12 examples/s]16288 examples [00:15, 1074.64 examples/s]16398 examples [00:15, 1080.79 examples/s]16508 examples [00:15, 1085.40 examples/s]16618 examples [00:15, 1087.59 examples/s]16727 examples [00:15, 1087.06 examples/s]16836 examples [00:15, 1082.61 examples/s]16945 examples [00:15, 1084.11 examples/s]17055 examples [00:15, 1087.39 examples/s]17167 examples [00:16, 1094.72 examples/s]17279 examples [00:16, 1100.27 examples/s]17390 examples [00:16, 1097.25 examples/s]17500 examples [00:16, 1084.33 examples/s]17609 examples [00:16, 1075.37 examples/s]17719 examples [00:16, 1080.51 examples/s]17830 examples [00:16, 1088.37 examples/s]17939 examples [00:16, 1088.57 examples/s]18048 examples [00:16, 1070.33 examples/s]18156 examples [00:17, 1046.32 examples/s]18261 examples [00:17, 1046.11 examples/s]18366 examples [00:17, 1046.59 examples/s]18471 examples [00:17, 1039.85 examples/s]18576 examples [00:17, 1040.97 examples/s]18685 examples [00:17, 1054.97 examples/s]18795 examples [00:17, 1067.62 examples/s]18904 examples [00:17, 1073.18 examples/s]19012 examples [00:17, 1071.27 examples/s]19121 examples [00:17, 1075.83 examples/s]19232 examples [00:18, 1085.37 examples/s]19341 examples [00:18, 1080.73 examples/s]19451 examples [00:18, 1085.77 examples/s]19561 examples [00:18, 1088.24 examples/s]19671 examples [00:18, 1090.38 examples/s]19781 examples [00:18, 1085.17 examples/s]19891 examples [00:18, 1088.92 examples/s]20000 examples [00:18, 1075.67 examples/s]20108 examples [00:18, 991.61 examples/s] 20214 examples [00:18, 1010.94 examples/s]20317 examples [00:19, 982.91 examples/s] 20426 examples [00:19, 1010.54 examples/s]20533 examples [00:19, 1025.59 examples/s]20644 examples [00:19, 1049.15 examples/s]20750 examples [00:19, 1048.57 examples/s]20861 examples [00:19, 1065.85 examples/s]20972 examples [00:19, 1075.94 examples/s]21084 examples [00:19, 1086.59 examples/s]21193 examples [00:19, 1086.63 examples/s]21302 examples [00:19, 1081.18 examples/s]21411 examples [00:20, 1049.08 examples/s]21523 examples [00:20, 1067.48 examples/s]21634 examples [00:20, 1077.60 examples/s]21745 examples [00:20, 1084.89 examples/s]21854 examples [00:20, 1078.68 examples/s]21965 examples [00:20, 1085.53 examples/s]22074 examples [00:20, 1070.99 examples/s]22185 examples [00:20, 1081.32 examples/s]22294 examples [00:20, 1073.24 examples/s]22402 examples [00:21, 1072.07 examples/s]22511 examples [00:21, 1075.94 examples/s]22623 examples [00:21, 1087.89 examples/s]22734 examples [00:21, 1093.07 examples/s]22846 examples [00:21, 1100.35 examples/s]22957 examples [00:21, 1085.95 examples/s]23068 examples [00:21, 1091.14 examples/s]23178 examples [00:21, 1086.48 examples/s]23290 examples [00:21, 1093.87 examples/s]23401 examples [00:21, 1097.95 examples/s]23511 examples [00:22, 1090.12 examples/s]23623 examples [00:22, 1097.29 examples/s]23735 examples [00:22, 1103.44 examples/s]23846 examples [00:22, 1092.74 examples/s]23956 examples [00:22, 1094.56 examples/s]24066 examples [00:22, 1085.28 examples/s]24175 examples [00:22, 1071.64 examples/s]24286 examples [00:22, 1080.46 examples/s]24395 examples [00:22, 1075.88 examples/s]24503 examples [00:22, 1076.48 examples/s]24611 examples [00:23, 1070.34 examples/s]24719 examples [00:23, 1040.33 examples/s]24829 examples [00:23, 1057.02 examples/s]24935 examples [00:23, 1057.90 examples/s]25046 examples [00:23, 1071.68 examples/s]25154 examples [00:23, 1067.77 examples/s]25262 examples [00:23, 1070.47 examples/s]25370 examples [00:23, 1071.12 examples/s]25481 examples [00:23, 1080.99 examples/s]25591 examples [00:23, 1085.78 examples/s]25700 examples [00:24, 1075.56 examples/s]25808 examples [00:24, 1056.17 examples/s]25916 examples [00:24, 1060.43 examples/s]26025 examples [00:24, 1068.34 examples/s]26132 examples [00:24, 1068.17 examples/s]26239 examples [00:24, 1059.86 examples/s]26347 examples [00:24, 1065.16 examples/s]26458 examples [00:24, 1075.54 examples/s]26569 examples [00:24, 1083.64 examples/s]26679 examples [00:24, 1085.87 examples/s]26788 examples [00:25, 1077.42 examples/s]26898 examples [00:25, 1083.61 examples/s]27009 examples [00:25, 1090.42 examples/s]27119 examples [00:25, 1091.22 examples/s]27231 examples [00:25, 1099.18 examples/s]27341 examples [00:25, 1085.28 examples/s]27452 examples [00:25, 1090.13 examples/s]27562 examples [00:25, 1089.99 examples/s]27672 examples [00:25, 1087.40 examples/s]27781 examples [00:25, 1085.67 examples/s]27890 examples [00:26, 1072.25 examples/s]27998 examples [00:26, 1045.17 examples/s]28108 examples [00:26, 1058.99 examples/s]28215 examples [00:26, 1060.96 examples/s]28324 examples [00:26, 1069.32 examples/s]28432 examples [00:26, 1059.39 examples/s]28543 examples [00:26, 1072.97 examples/s]28652 examples [00:26, 1076.04 examples/s]28760 examples [00:26, 1073.59 examples/s]28871 examples [00:27, 1081.85 examples/s]28980 examples [00:27, 1075.56 examples/s]29090 examples [00:27, 1080.74 examples/s]29199 examples [00:27, 1080.75 examples/s]29310 examples [00:27, 1086.58 examples/s]29421 examples [00:27, 1092.03 examples/s]29531 examples [00:27, 1079.18 examples/s]29641 examples [00:27, 1082.50 examples/s]29750 examples [00:27, 1056.99 examples/s]29858 examples [00:27, 1063.37 examples/s]29965 examples [00:28, 1061.58 examples/s]30072 examples [00:28, 1006.17 examples/s]30181 examples [00:28, 1029.01 examples/s]30292 examples [00:28, 1050.18 examples/s]30402 examples [00:28, 1063.54 examples/s]30509 examples [00:28, 1008.30 examples/s]30615 examples [00:28, 1022.07 examples/s]30722 examples [00:28, 1034.20 examples/s]30829 examples [00:28, 1044.24 examples/s]30936 examples [00:28, 1051.00 examples/s]31044 examples [00:29, 1046.10 examples/s]31149 examples [00:29, 1020.76 examples/s]31252 examples [00:29, 966.13 examples/s] 31353 examples [00:29, 977.98 examples/s]31458 examples [00:29, 997.73 examples/s]31566 examples [00:29, 1020.55 examples/s]31671 examples [00:29, 1027.09 examples/s]31778 examples [00:29, 1039.37 examples/s]31883 examples [00:29, 1020.19 examples/s]31989 examples [00:30, 1030.10 examples/s]32097 examples [00:30, 1043.40 examples/s]32202 examples [00:30, 1042.52 examples/s]32307 examples [00:30, 1022.36 examples/s]32415 examples [00:30, 1038.91 examples/s]32520 examples [00:30, 1037.21 examples/s]32628 examples [00:30, 1049.37 examples/s]32735 examples [00:30, 1054.00 examples/s]32841 examples [00:30, 1051.70 examples/s]32949 examples [00:30, 1058.00 examples/s]33056 examples [00:31, 1061.42 examples/s]33163 examples [00:31, 1054.13 examples/s]33271 examples [00:31, 1060.95 examples/s]33380 examples [00:31, 1068.30 examples/s]33489 examples [00:31, 1072.46 examples/s]33597 examples [00:31, 1071.44 examples/s]33705 examples [00:31, 1062.83 examples/s]33812 examples [00:31, 1049.64 examples/s]33918 examples [00:31, 1045.70 examples/s]34026 examples [00:31, 1054.34 examples/s]34132 examples [00:32, 1053.60 examples/s]34239 examples [00:32, 1056.83 examples/s]34345 examples [00:32, 1049.72 examples/s]34451 examples [00:32, 1025.78 examples/s]34554 examples [00:32, 976.74 examples/s] 34662 examples [00:32, 1004.21 examples/s]34764 examples [00:32, 1004.74 examples/s]34868 examples [00:32, 1013.07 examples/s]34975 examples [00:32, 1029.25 examples/s]35082 examples [00:32, 1039.49 examples/s]35190 examples [00:33, 1049.47 examples/s]35298 examples [00:33, 1055.96 examples/s]35404 examples [00:33, 1023.06 examples/s]35512 examples [00:33, 1038.56 examples/s]35621 examples [00:33, 1052.22 examples/s]35731 examples [00:33, 1063.38 examples/s]35839 examples [00:33, 1066.63 examples/s]35946 examples [00:33, 1067.38 examples/s]36053 examples [00:33, 1064.91 examples/s]36163 examples [00:33, 1074.86 examples/s]36271 examples [00:34, 1053.22 examples/s]36377 examples [00:34, 1052.80 examples/s]36483 examples [00:34, 1047.09 examples/s]36594 examples [00:34, 1063.11 examples/s]36703 examples [00:34, 1068.47 examples/s]36810 examples [00:34, 1062.21 examples/s]36917 examples [00:34, 1061.07 examples/s]37024 examples [00:34, 1039.51 examples/s]37131 examples [00:34, 1048.12 examples/s]37238 examples [00:35, 1053.48 examples/s]37349 examples [00:35, 1067.75 examples/s]37457 examples [00:35, 1069.03 examples/s]37564 examples [00:35, 1062.15 examples/s]37676 examples [00:35, 1076.64 examples/s]37787 examples [00:35, 1085.67 examples/s]37896 examples [00:35, 1084.29 examples/s]38005 examples [00:35, 1067.48 examples/s]38115 examples [00:35, 1074.83 examples/s]38228 examples [00:35, 1089.22 examples/s]38340 examples [00:36, 1096.88 examples/s]38451 examples [00:36, 1099.93 examples/s]38562 examples [00:36, 1100.87 examples/s]38673 examples [00:36, 1095.16 examples/s]38787 examples [00:36, 1106.55 examples/s]38898 examples [00:36, 1106.23 examples/s]39009 examples [00:36, 1076.15 examples/s]39120 examples [00:36, 1085.09 examples/s]39229 examples [00:36, 1063.72 examples/s]39336 examples [00:36, 1060.19 examples/s]39448 examples [00:37, 1074.99 examples/s]39558 examples [00:37, 1081.25 examples/s]39670 examples [00:37, 1091.27 examples/s]39780 examples [00:37, 1078.11 examples/s]39890 examples [00:37, 1083.37 examples/s]40001 examples [00:37, 1037.31 examples/s]40111 examples [00:37, 1055.31 examples/s]40219 examples [00:37, 1062.11 examples/s]40328 examples [00:37, 1067.99 examples/s]40438 examples [00:37, 1076.53 examples/s]40546 examples [00:38, 1072.49 examples/s]40658 examples [00:38, 1085.72 examples/s]40768 examples [00:38, 1088.59 examples/s]40878 examples [00:38, 1089.52 examples/s]40988 examples [00:38, 1085.62 examples/s]41099 examples [00:38, 1092.28 examples/s]41209 examples [00:38, 1076.50 examples/s]41317 examples [00:38, 1067.43 examples/s]41424 examples [00:38, 1044.88 examples/s]41529 examples [00:38, 1036.68 examples/s]41640 examples [00:39, 1055.26 examples/s]41752 examples [00:39, 1070.90 examples/s]41860 examples [00:39, 1070.24 examples/s]41968 examples [00:39, 1065.70 examples/s]42076 examples [00:39, 1068.57 examples/s]42183 examples [00:39, 1066.72 examples/s]42295 examples [00:39, 1080.14 examples/s]42404 examples [00:39, 1082.23 examples/s]42513 examples [00:39, 1073.37 examples/s]42621 examples [00:40, 1040.27 examples/s]42726 examples [00:40, 1016.02 examples/s]42833 examples [00:40, 1029.80 examples/s]42943 examples [00:40, 1048.17 examples/s]43049 examples [00:40, 1051.41 examples/s]43156 examples [00:40, 1054.90 examples/s]43267 examples [00:40, 1068.45 examples/s]43375 examples [00:40, 1071.64 examples/s]43483 examples [00:40, 1070.39 examples/s]43591 examples [00:40, 1068.58 examples/s]43701 examples [00:41, 1077.54 examples/s]43811 examples [00:41, 1081.70 examples/s]43921 examples [00:41, 1086.82 examples/s]44030 examples [00:41, 1085.73 examples/s]44139 examples [00:41, 1082.45 examples/s]44249 examples [00:41, 1085.44 examples/s]44358 examples [00:41, 1079.40 examples/s]44466 examples [00:41, 1062.71 examples/s]44573 examples [00:41, 1041.61 examples/s]44680 examples [00:41, 1047.55 examples/s]44785 examples [00:42, 1048.20 examples/s]44890 examples [00:42, 1040.68 examples/s]45000 examples [00:42, 1057.60 examples/s]45110 examples [00:42, 1067.65 examples/s]45217 examples [00:42, 1067.47 examples/s]45327 examples [00:42, 1074.82 examples/s]45437 examples [00:42, 1081.69 examples/s]45549 examples [00:42, 1090.85 examples/s]45661 examples [00:42, 1098.71 examples/s]45771 examples [00:42, 1095.28 examples/s]45884 examples [00:43, 1103.59 examples/s]45995 examples [00:43, 1096.01 examples/s]46105 examples [00:43, 1090.76 examples/s]46215 examples [00:43, 1075.21 examples/s]46323 examples [00:43, 1063.26 examples/s]46430 examples [00:43, 1061.50 examples/s]46537 examples [00:43, 1047.81 examples/s]46645 examples [00:43, 1056.73 examples/s]46753 examples [00:43, 1063.12 examples/s]46860 examples [00:43, 1022.90 examples/s]46968 examples [00:44, 1036.29 examples/s]47076 examples [00:44, 1046.62 examples/s]47181 examples [00:44, 1017.84 examples/s]47285 examples [00:44, 1023.91 examples/s]47391 examples [00:44, 1034.33 examples/s]47496 examples [00:44, 1037.42 examples/s]47606 examples [00:44, 1053.25 examples/s]47712 examples [00:44, 1012.61 examples/s]47814 examples [00:44, 986.23 examples/s] 47925 examples [00:45, 1018.25 examples/s]48035 examples [00:45, 1040.80 examples/s]48144 examples [00:45, 1052.97 examples/s]48254 examples [00:45, 1064.32 examples/s]48362 examples [00:45, 1068.58 examples/s]48470 examples [00:45, 1068.93 examples/s]48581 examples [00:45, 1078.49 examples/s]48692 examples [00:45, 1086.43 examples/s]48801 examples [00:45, 1084.66 examples/s]48912 examples [00:45, 1089.32 examples/s]49023 examples [00:46, 1092.76 examples/s]49135 examples [00:46, 1100.46 examples/s]49246 examples [00:46, 1102.54 examples/s]49357 examples [00:46, 1095.45 examples/s]49467 examples [00:46, 1082.03 examples/s]49579 examples [00:46, 1091.89 examples/s]49691 examples [00:46, 1097.60 examples/s]49802 examples [00:46, 1099.59 examples/s]49912 examples [00:46, 1095.70 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 16%|█▌        | 7961/50000 [00:00<00:00, 79605.14 examples/s] 44%|████▎     | 21803/50000 [00:00<00:00, 91234.18 examples/s] 72%|███████▏  | 35828/50000 [00:00<00:00, 101920.09 examples/s] 99%|█████████▉| 49608/50000 [00:00<00:00, 110555.21 examples/s]                                                                0 examples [00:00, ? examples/s]92 examples [00:00, 914.14 examples/s]202 examples [00:00, 960.76 examples/s]312 examples [00:00, 997.39 examples/s]419 examples [00:00, 1015.61 examples/s]515 examples [00:00, 997.99 examples/s] 611 examples [00:00, 984.29 examples/s]723 examples [00:00, 1019.45 examples/s]836 examples [00:00, 1048.24 examples/s]948 examples [00:00, 1066.38 examples/s]1052 examples [00:01, 1039.07 examples/s]1158 examples [00:01, 1043.05 examples/s]1271 examples [00:01, 1066.47 examples/s]1383 examples [00:01, 1081.09 examples/s]1491 examples [00:01, 1070.25 examples/s]1603 examples [00:01, 1082.06 examples/s]1712 examples [00:01, 1082.99 examples/s]1823 examples [00:01, 1090.49 examples/s]1935 examples [00:01, 1098.86 examples/s]2045 examples [00:01, 1091.98 examples/s]2155 examples [00:02, 1092.77 examples/s]2267 examples [00:02, 1100.44 examples/s]2378 examples [00:02, 1099.80 examples/s]2490 examples [00:02, 1103.98 examples/s]2601 examples [00:02, 1104.86 examples/s]2712 examples [00:02, 1102.93 examples/s]2823 examples [00:02, 1094.22 examples/s]2934 examples [00:02, 1096.30 examples/s]3045 examples [00:02, 1099.94 examples/s]3156 examples [00:02, 1099.38 examples/s]3266 examples [00:03, 1096.72 examples/s]3376 examples [00:03, 1089.03 examples/s]3486 examples [00:03, 1089.78 examples/s]3597 examples [00:03, 1095.62 examples/s]3707 examples [00:03, 1071.24 examples/s]3816 examples [00:03, 1076.15 examples/s]3924 examples [00:03, 1056.28 examples/s]4035 examples [00:03, 1070.75 examples/s]4146 examples [00:03, 1082.08 examples/s]4258 examples [00:03, 1092.79 examples/s]4371 examples [00:04, 1101.46 examples/s]4482 examples [00:04, 1097.78 examples/s]4592 examples [00:04, 1091.35 examples/s]4702 examples [00:04, 1089.79 examples/s]4813 examples [00:04, 1093.08 examples/s]4925 examples [00:04, 1099.22 examples/s]5035 examples [00:04, 1096.65 examples/s]5149 examples [00:04, 1106.67 examples/s]5261 examples [00:04, 1108.38 examples/s]5377 examples [00:04, 1122.18 examples/s]5493 examples [00:05, 1132.45 examples/s]5607 examples [00:05, 1101.26 examples/s]5718 examples [00:05, 1082.30 examples/s]5830 examples [00:05, 1091.57 examples/s]5940 examples [00:05, 1092.50 examples/s]6051 examples [00:05, 1096.95 examples/s]6162 examples [00:05, 1098.88 examples/s]6274 examples [00:05, 1102.63 examples/s]6385 examples [00:05, 1100.17 examples/s]6496 examples [00:05, 1100.38 examples/s]6608 examples [00:06, 1103.74 examples/s]6719 examples [00:06, 1104.34 examples/s]6830 examples [00:06, 1099.10 examples/s]6942 examples [00:06, 1103.27 examples/s]7053 examples [00:06, 1101.85 examples/s]7164 examples [00:06, 1101.28 examples/s]7275 examples [00:06, 1057.85 examples/s]7382 examples [00:06, 1049.98 examples/s]7489 examples [00:06, 1054.44 examples/s]7598 examples [00:07, 1063.73 examples/s]7708 examples [00:07, 1073.53 examples/s]7818 examples [00:07, 1079.85 examples/s]7927 examples [00:07, 1079.20 examples/s]8037 examples [00:07, 1083.52 examples/s]8146 examples [00:07, 1081.92 examples/s]8257 examples [00:07, 1089.13 examples/s]8366 examples [00:07, 1086.01 examples/s]8477 examples [00:07, 1092.48 examples/s]8587 examples [00:07, 1079.68 examples/s]8696 examples [00:08, 1082.09 examples/s]8808 examples [00:08, 1091.66 examples/s]8918 examples [00:08, 1087.83 examples/s]9028 examples [00:08, 1089.48 examples/s]9140 examples [00:08, 1096.04 examples/s]9250 examples [00:08, 1087.96 examples/s]9360 examples [00:08, 1089.16 examples/s]9470 examples [00:08, 1091.45 examples/s]9580 examples [00:08, 1079.16 examples/s]9692 examples [00:08, 1088.55 examples/s]9801 examples [00:09, 1087.67 examples/s]9911 examples [00:09, 1090.43 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete9888B6/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete9888B6/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb7d6f6c950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb7d6f6c950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb7d6f6c950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb822959390>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb75d4e80f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb7d6f6c950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb7d6f6c950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb7d6f6c950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb7d45af320>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb7d45af0f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fb759368488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fb759368488> 

  function with postional parmater data_info <function split_train_valid at 0x7fb759368488> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fb759368598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fb759368598> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fb759368598> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.6.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.18.5)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.4.5.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=0bf21c691c9a2cfac41983d4e2785f9e65855518b91a6b8608638df87c386da9
  Stored in directory: /tmp/pip-ephem-wheel-cache-or0r_4ll/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:04<123:43:53, 1.94kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:04<86:51:40, 2.76kB/s] .vector_cache/glove.6B.zip:   0%|          | 147k/862M [00:04<60:52:12, 3.93kB/s] .vector_cache/glove.6B.zip:   0%|          | 369k/862M [00:04<42:37:57, 5.62kB/s].vector_cache/glove.6B.zip:   0%|          | 934k/862M [00:04<29:50:09, 8.02kB/s].vector_cache/glove.6B.zip:   0%|          | 2.11M/862M [00:04<20:51:47, 11.5kB/s].vector_cache/glove.6B.zip:   0%|          | 3.25M/862M [00:04<14:35:28, 16.4kB/s].vector_cache/glove.6B.zip:   1%|          | 5.48M/862M [00:04<10:11:26, 23.4kB/s].vector_cache/glove.6B.zip:   1%|          | 8.27M/862M [00:05<7:06:45, 33.3kB/s] .vector_cache/glove.6B.zip:   1%|          | 10.6M/862M [00:05<4:58:05, 47.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.4M/862M [00:05<3:28:09, 68.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.5M/862M [00:05<2:25:33, 97.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.8M/862M [00:05<1:42:04, 138kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 18.0M/862M [00:05<1:11:41, 196kB/s].vector_cache/glove.6B.zip:   2%|▏         | 19.3M/862M [00:05<50:26, 279kB/s]  .vector_cache/glove.6B.zip:   2%|▏         | 20.6M/862M [00:05<35:36, 394kB/s].vector_cache/glove.6B.zip:   3%|▎         | 21.7M/862M [00:05<25:15, 555kB/s].vector_cache/glove.6B.zip:   3%|▎         | 22.9M/862M [00:06<18:00, 777kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.3M/862M [00:06<12:53, 1.08MB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.8M/862M [00:06<09:18, 1.50MB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.2M/862M [00:06<06:49, 2.04MB/s].vector_cache/glove.6B.zip:   3%|▎         | 28.6M/862M [00:06<05:04, 2.74MB/s].vector_cache/glove.6B.zip:   3%|▎         | 30.0M/862M [00:06<03:50, 3.61MB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.4M/862M [00:06<02:59, 4.63MB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.1M/862M [00:06<02:20, 5.90MB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.8M/862M [00:06<01:52, 7.37MB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.8M/862M [00:06<01:30, 9.09MB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.4M/862M [00:07<01:13, 11.3MB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.0M/862M [00:07<01:00, 13.6MB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.7M/862M [00:07<00:51, 15.9MB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.0M/862M [00:07<00:46, 17.5MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.1M/862M [00:07<00:54, 15.0MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.9M/862M [00:07<00:51, 15.7MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.8M/862M [00:07<00:48, 16.6MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:08<03:57, 3.40MB/s].vector_cache/glove.6B.zip:   6%|▋         | 54.2M/862M [00:08<02:59, 4.50MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:08<02:20, 5.76MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:13<30:54, 434kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:13<35:24, 379kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:13<27:57, 480kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.4M/862M [00:13<20:19, 660kB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.9M/862M [00:13<14:28, 924kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:15<13:26, 994kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:15<12:23, 1.08MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:15<13:41, 976kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:15<15:18, 872kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:15<16:14, 822kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:15<17:29, 763kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.2M/862M [00:15<17:42, 754kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.3M/862M [00:16<17:22, 769kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.3M/862M [00:16<16:56, 788kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.4M/862M [00:16<15:49, 843kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.6M/862M [00:16<14:08, 943kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:16<13:27, 991kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.8M/862M [00:16<13:57, 955kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.9M/862M [00:16<12:49, 1.04MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.1M/862M [00:16<12:19, 1.08MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.2M/862M [00:16<11:07, 1.20MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.4M/862M [00:17<09:42, 1.37MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.7M/862M [00:17<08:28, 1.57MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.9M/862M [00:17<07:27, 1.79MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.2M/862M [00:17<06:57, 1.91MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.5M/862M [00:17<06:09, 2.16MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:17<05:34, 2.38MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:17<05:04, 2.62MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:17<04:36, 2.89MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:31<3:10:17, 69.8kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:31<2:27:04, 90.3kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:31<1:46:06, 125kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.5M/862M [00:31<1:15:29, 176kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:31<53:57, 246kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 66.2M/862M [00:31<38:47, 342kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.6M/862M [00:31<28:14, 470kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.0M/862M [00:32<20:40, 641kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.5M/862M [00:32<15:22, 862kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.9M/862M [00:32<11:37, 1.14MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:32<08:50, 1.50MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:35<38:18, 345kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:35<40:32, 326kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:36<31:35, 418kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:36<23:09, 570kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.7M/862M [00:36<18:28, 715kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.9M/862M [00:36<15:14, 866kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.0M/862M [00:36<13:49, 955kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:36<13:09, 1.00MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.3M/862M [00:36<11:54, 1.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.5M/862M [00:36<10:59, 1.20MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.7M/862M [00:36<09:40, 1.36MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.9M/862M [00:36<08:26, 1.56MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.2M/862M [00:37<07:22, 1.79MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.4M/862M [00:37<07:06, 1.85MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.6M/862M [00:37<06:39, 1.98MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.9M/862M [00:37<06:09, 2.14MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:37<05:39, 2.33MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:37<04:47, 2.75MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:39<24:26, 538kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:39<33:34, 392kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:40<32:08, 409kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:40<28:53, 455kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:40<26:22, 499kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:40<25:13, 521kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:40<22:44, 578kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:40<20:36, 638kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:40<19:31, 673kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:40<18:36, 706kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.8M/862M [00:40<18:00, 729kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:41<17:14, 762kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.0M/862M [00:41<16:38, 789kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.1M/862M [00:41<15:50, 829kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.2M/862M [00:41<15:02, 873kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:41<14:34, 901kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.4M/862M [00:41<14:21, 914kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.5M/862M [00:41<13:47, 952kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.6M/862M [00:41<12:13, 1.07MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.8M/862M [00:41<11:05, 1.18MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.9M/862M [00:41<11:19, 1.16MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.0M/862M [00:42<11:07, 1.18MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.2M/862M [00:42<11:48, 1.11MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.3M/862M [00:42<12:00, 1.09MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.4M/862M [00:42<12:27, 1.05MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.5M/862M [00:42<11:43, 1.12MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.8M/862M [00:42<09:22, 1.40MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:42<07:23, 1.77MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:42<05:59, 2.18MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:42<04:55, 2.65MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.9M/862M [00:43<04:06, 3.18MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.6M/862M [00:43<03:30, 3.72MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.6M/862M [00:48<6:06:48, 35.6kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.7M/862M [00:48<4:30:02, 48.4kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.8M/862M [00:48<3:12:16, 67.9kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.9M/862M [00:48<2:17:13, 95.1kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.1M/862M [00:48<1:39:47, 131kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:48<1:13:56, 177kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.3M/862M [00:49<55:18, 236kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 79.4M/862M [00:49<42:15, 309kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.5M/862M [00:49<33:19, 391kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.6M/862M [00:49<26:01, 501kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.0M/862M [00:49<19:27, 670kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.3M/862M [00:49<14:56, 872kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:49<12:46, 1.02MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:49<12:40, 1.03MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:49<12:21, 1.05MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:50<12:20, 1.05MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:50<12:34, 1.04MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:50<12:35, 1.03MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:50<13:01, 999kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:50<13:00, 1.00MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:50<12:54, 1.01MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:50<12:28, 1.04MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:50<12:02, 1.08MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:50<10:36, 1.23MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.9M/862M [00:50<09:27, 1.38MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.0M/862M [00:51<09:20, 1.39MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.2M/862M [00:51<09:11, 1.41MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:51<08:31, 1.53MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.5M/862M [00:51<08:20, 1.56MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.8M/862M [00:51<07:13, 1.80MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.1M/862M [00:51<06:52, 1.89MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.2M/862M [00:51<07:14, 1.79MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.3M/862M [00:51<08:45, 1.48MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.4M/862M [00:51<10:08, 1.28MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.5M/862M [00:51<11:06, 1.17MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.6M/862M [00:52<12:15, 1.06MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.7M/862M [00:52<13:17, 977kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 83.8M/862M [00:52<14:17, 907kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.9M/862M [00:52<14:07, 918kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.0M/862M [00:52<14:17, 908kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.1M/862M [00:52<13:15, 978kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.3M/862M [00:52<11:32, 1.12MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:52<10:11, 1.27MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:52<08:47, 1.47MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:52<07:01, 1.85MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.2M/862M [00:53<05:17, 2.45MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.5M/862M [00:53<03:59, 3.24MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:53<03:01, 4.27MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:53<25:50, 499kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:53<18:55, 680kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:53<14:28, 889kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.4M/862M [00:54<11:16, 1.14MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.8M/862M [00:54<09:00, 1.43MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.2M/862M [00:54<07:18, 1.76MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.6M/862M [00:54<06:08, 2.09MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.9M/862M [00:54<05:19, 2.41MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.4M/862M [00:54<04:31, 2.84MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:54<04:02, 3.18MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:55<10:20, 1.24MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:55<09:51, 1.30MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:55<09:15, 1.38MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:56<08:54, 1.44MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:56<08:13, 1.56MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.4M/862M [00:56<06:59, 1.83MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.1M/862M [00:56<05:28, 2.34MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.6M/862M [00:56<04:04, 3.13MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [01:01<21:54, 582kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [01:01<28:20, 450kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [01:01<22:31, 566kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.5M/862M [01:01<16:32, 770kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.8M/862M [01:01<11:50, 1.07MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [01:01<08:24, 1.51MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [01:06<1:13:05, 173kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [01:06<1:02:26, 203kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [01:06<47:06, 269kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [01:06<36:04, 351kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [01:06<27:15, 464kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [01:06<20:39, 612kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [01:07<14:59, 843kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [01:07<10:41, 1.18MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [01:07<08:31, 1.47MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [01:07<06:18, 1.99MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [01:09<06:58, 1.79MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [01:09<10:32, 1.19MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [01:10<08:51, 1.41MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [01:10<06:34, 1.90MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [01:11<07:04, 1.76MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 116M/862M [01:11<08:00, 1.55MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [01:11<08:03, 1.54MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [01:12<06:28, 1.92MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [01:12<04:54, 2.53MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [01:12<03:36, 3.42MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [01:13<16:16, 760kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [01:13<17:25, 710kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [01:13<18:04, 684kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [01:14<18:17, 676kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [01:14<17:53, 691kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [01:14<16:02, 770kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [01:14<13:38, 906kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [01:14<10:46, 1.15MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [01:14<08:01, 1.54MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [01:14<05:53, 2.09MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [01:15<07:15, 1.69MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [01:15<06:04, 2.02MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [01:15<04:39, 2.63MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [01:16<03:25, 3.57MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [01:17<11:02, 1.11MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [01:17<14:19, 853kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [01:17<12:40, 964kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [01:17<09:55, 1.23MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [01:18<07:18, 1.67MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [01:18<05:14, 2.32MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [01:19<33:28, 363kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [01:19<27:31, 441kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [01:19<21:44, 559kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [01:19<16:59, 715kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [01:20<12:52, 943kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [01:20<09:20, 1.30MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [01:20<06:44, 1.80MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [01:21<13:55, 867kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [01:21<11:54, 1.01MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [01:21<09:18, 1.30MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [01:21<06:49, 1.77MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [01:43<36:11, 332kB/s] .vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [01:43<37:39, 319kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [01:43<29:15, 411kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [01:43<21:17, 564kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [01:43<15:12, 788kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [01:45<13:50, 864kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [01:45<20:21, 587kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [01:45<18:24, 649kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [01:45<16:02, 744kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [01:46<13:40, 873kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [01:46<11:03, 1.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [01:46<08:32, 1.40MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [01:46<06:24, 1.86MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [01:46<05:11, 2.30MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [01:46<04:21, 2.73MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [01:46<03:46, 3.15MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [01:46<03:26, 3.45MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [01:47<15:20, 774kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [01:47<11:34, 1.03MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [01:47<08:42, 1.36MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [01:47<06:38, 1.79MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [01:48<05:20, 2.22MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [01:48<05:32, 2.14MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [01:48<04:58, 2.38MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [01:48<04:14, 2.78MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [01:48<03:48, 3.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [01:48<03:24, 3.46MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [01:54<15:03:35, 13.1kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [01:54<10:44:39, 18.3kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [01:55<7:34:02, 26.0kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [01:55<5:18:27, 37.0kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [01:55<3:42:55, 52.8kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [01:55<2:35:56, 75.4kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [01:56<1:55:05, 102kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [01:56<1:22:32, 142kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [01:56<1:00:30, 194kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [01:57<45:00, 261kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [01:57<34:04, 344kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [01:57<26:01, 451kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [01:57<20:36, 569kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [01:57<16:59, 690kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [01:57<14:13, 824kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [01:57<11:54, 984kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [01:57<08:56, 1.31MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [01:57<06:28, 1.80MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [01:58<09:21, 1.25MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [01:58<06:58, 1.67MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [01:58<05:23, 2.16MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:59<04:37, 2.52MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [01:59<03:37, 3.20MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [01:59<02:56, 3.94MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [02:03<44:33, 260kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [02:03<43:01, 270kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [02:03<32:58, 352kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [02:03<23:44, 488kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [02:03<16:48, 688kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [02:09<23:08, 499kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [02:09<26:25, 437kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [02:09<21:23, 539kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [02:09<15:38, 736kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [02:09<11:04, 1.04MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [02:10<08:41, 1.32MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [02:10<06:31, 1.75MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [02:10<05:01, 2.27MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [02:10<03:56, 2.89MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [02:10<03:20, 3.40MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [02:11<02:51, 3.99MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [02:12<13:52, 820kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [02:12<14:26, 788kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [02:12<13:00, 874kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [02:13<10:33, 1.08MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [02:13<08:05, 1.40MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [02:13<06:03, 1.87MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [02:13<04:34, 2.48MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [02:13<03:29, 3.23MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [02:17<1:24:38, 134kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [02:17<1:11:19, 158kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [02:17<52:05, 217kB/s]  .vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [02:17<38:06, 296kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [02:17<27:09, 416kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [02:17<19:10, 587kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [02:26<34:48, 323kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [02:26<35:02, 321kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [02:26<30:33, 368kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [02:27<23:18, 482kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [02:27<16:53, 664kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [02:27<12:04, 927kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [02:27<08:33, 1.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [02:33<28:19, 393kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [02:33<30:15, 368kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [02:34<27:20, 408kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [02:34<23:45, 469kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [02:34<20:22, 547kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [02:34<17:59, 619kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [02:34<16:48, 663kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [02:34<13:21, 833kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [02:34<09:53, 1.12MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [02:34<07:08, 1.55MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [02:38<13:53, 797kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [02:38<20:44, 534kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [02:38<17:08, 646kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [02:38<12:34, 880kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [02:38<09:06, 1.21MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [02:52<26:36, 413kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [02:52<30:16, 363kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [02:52<23:50, 461kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [02:53<17:19, 635kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [02:53<12:22, 887kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [02:53<09:34, 1.14MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [02:53<06:54, 1.58MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [02:53<05:07, 2.12MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [02:54<03:53, 2.80MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [03:02<1:12:46, 149kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [03:03<1:02:08, 175kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [03:03<46:06, 236kB/s]  .vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [03:03<32:53, 330kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [03:03<23:04, 469kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [03:03<16:40, 648kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [03:03<11:51, 908kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [03:03<08:36, 1.25MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [03:03<06:18, 1.70MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [03:05<14:38, 733kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [03:05<12:46, 840kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [03:05<11:50, 905kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [03:05<10:40, 1.00MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [03:05<10:02, 1.07MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [03:06<09:17, 1.15MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [03:06<07:22, 1.45MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [03:06<05:29, 1.95MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [03:06<04:09, 2.57MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [03:07<06:17, 1.70MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [03:07<05:19, 2.00MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [03:07<04:45, 2.24MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [03:07<04:15, 2.50MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [03:07<03:55, 2.71MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [03:07<03:35, 2.96MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [03:08<03:28, 3.06MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [03:08<03:15, 3.25MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [03:08<03:04, 3.46MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [03:08<02:44, 3.87MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [03:09<07:10, 1.48MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [03:09<06:34, 1.61MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [03:09<05:44, 1.84MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [03:09<04:36, 2.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [03:09<03:35, 2.95MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [03:09<02:54, 3.62MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [03:10<02:21, 4.46MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [03:14<34:44, 303kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [03:14<35:16, 298kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [03:15<26:38, 395kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [03:15<21:04, 499kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [03:15<15:51, 663kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [03:15<11:45, 893kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [03:15<08:26, 1.24MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [03:15<06:06, 1.71MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [03:19<2:21:48, 73.7kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [03:19<1:49:30, 95.4kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [03:19<1:22:46, 126kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [03:19<1:01:34, 170kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [03:19<46:20, 225kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [03:20<34:34, 302kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [03:20<25:09, 415kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [03:20<17:55, 582kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [03:20<12:43, 818kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [03:20<09:43, 1.07MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [03:21<06:53, 1.50MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [03:22<08:37, 1.19MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [03:23<09:30, 1.08MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [03:23<07:31, 1.37MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [03:23<05:27, 1.88MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [03:26<09:15, 1.10MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [03:26<17:22, 588kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [03:26<14:35, 700kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [03:26<10:49, 944kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [03:26<07:52, 1.29MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [03:34<15:21, 661kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [03:34<20:22, 498kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [03:34<20:10, 503kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [03:34<18:26, 550kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [03:34<16:50, 603kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [03:34<14:53, 681kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [03:34<12:51, 789kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [03:35<11:46, 861kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [03:35<09:20, 1.08MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [03:35<07:06, 1.43MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [03:35<05:16, 1.91MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [03:35<03:55, 2.57MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [03:37<19:51, 508kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [03:38<23:22, 431kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [03:38<19:51, 508kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [03:38<15:57, 631kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [03:38<13:04, 771kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [03:38<10:03, 1.00MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [03:38<07:28, 1.35MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [03:38<05:31, 1.82MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [03:38<04:06, 2.44MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [03:45<55:40, 180kB/s] .vector_cache/glove.6B.zip:  30%|███       | 261M/862M [03:45<49:02, 204kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [03:45<36:43, 273kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [03:45<26:17, 381kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [03:46<18:27, 539kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [03:49<31:14, 318kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [03:49<32:18, 308kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [03:49<25:11, 395kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [03:49<18:16, 544kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [03:49<13:03, 760kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 269M/862M [03:51<11:37, 849kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 269M/862M [03:51<14:48, 667kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [03:51<14:56, 661kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [03:51<14:10, 697kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [03:52<12:27, 793kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [03:52<09:56, 993kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [03:52<07:33, 1.30MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [03:52<05:41, 1.73MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [03:52<04:24, 2.23MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [03:52<03:30, 2.81MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [03:52<02:50, 3.45MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [03:53<34:04, 288kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [03:53<25:08, 390kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [03:53<18:27, 531kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [03:53<13:16, 738kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [03:54<09:34, 1.02MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [03:54<06:52, 1.42MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [03:55<24:18, 401kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [03:55<19:02, 512kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [03:55<14:13, 685kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [03:55<10:18, 943kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [03:56<07:17, 1.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [03:57<31:30, 307kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [03:57<24:22, 397kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [03:57<18:08, 533kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [03:57<13:11, 732kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [03:57<09:21, 1.03MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [03:59<10:07, 949kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [03:59<09:56, 965kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [03:59<08:34, 1.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [03:59<06:32, 1.46MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [03:59<04:46, 2.00MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [04:11<20:39, 462kB/s] .vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [04:11<24:26, 390kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [04:11<19:21, 492kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [04:11<14:05, 676kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [04:11<10:01, 947kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [04:17<15:36, 606kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [04:17<21:21, 443kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [04:17<19:00, 498kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [04:17<17:15, 548kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [04:17<15:42, 602kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [04:17<14:50, 637kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [04:17<13:50, 684kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [04:17<12:34, 752kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [04:17<12:05, 782kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [04:18<12:13, 773kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [04:18<12:11, 775kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [04:18<12:24, 761kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [04:18<12:11, 775kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [04:18<10:19, 914kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [04:18<07:53, 1.19MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [04:18<05:50, 1.61MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [04:18<04:21, 2.16MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [04:18<03:18, 2.83MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [04:19<03:32, 2.64MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [04:19<02:56, 3.18MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [04:19<02:32, 3.67MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [04:19<02:10, 4.28MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [04:19<01:56, 4.80MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [04:19<01:48, 5.17MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [04:19<01:43, 5.38MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [04:21<16:29, 564kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [04:21<14:43, 632kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [04:21<14:19, 649kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [04:21<11:58, 776kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [04:21<08:56, 1.04MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [04:21<06:27, 1.43MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [04:21<04:43, 1.96MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [04:23<08:54, 1.04MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [04:23<07:27, 1.24MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [04:23<06:41, 1.38MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [04:23<06:18, 1.46MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [04:23<05:55, 1.55MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [04:23<05:07, 1.80MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [04:23<04:23, 2.10MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [04:23<03:27, 2.67MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [04:24<02:39, 3.46MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [04:25<05:52, 1.56MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [04:25<05:15, 1.74MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [04:25<04:17, 2.14MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [04:25<03:33, 2.57MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [04:25<02:50, 3.22MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [04:25<02:19, 3.93MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [04:25<02:01, 4.49MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [04:42<1:14:40, 122kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [04:42<1:00:49, 150kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [04:42<47:45, 190kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [04:42<35:40, 255kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [04:42<25:44, 353kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [04:42<18:16, 496kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [04:42<12:51, 702kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [04:54<1:48:05, 83.5kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [04:54<1:23:47, 108kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [04:54<1:00:55, 148kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [04:54<43:57, 205kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [04:54<31:20, 288kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [04:54<22:10, 406kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [04:55<15:41, 572kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [04:55<11:10, 802kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [04:55<09:12, 971kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [04:55<06:45, 1.32MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [04:55<04:54, 1.81MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [04:57<06:00, 1.48MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [04:57<08:01, 1.11MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [04:57<08:02, 1.10MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [04:58<06:46, 1.31MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [04:58<05:07, 1.72MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [04:58<03:45, 2.35MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [04:59<05:15, 1.67MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [04:59<05:05, 1.73MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [04:59<04:37, 1.90MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [04:59<03:58, 2.21MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [05:00<03:08, 2.79MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [05:00<02:38, 3.31MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [05:01<04:06, 2.13MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [05:01<05:30, 1.59MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [05:01<06:00, 1.45MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [05:01<05:46, 1.51MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [05:02<04:53, 1.78MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [05:02<03:49, 2.28MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 341M/862M [05:02<02:59, 2.91MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [05:02<02:21, 3.69MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [05:03<05:12, 1.66MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [05:03<06:17, 1.38MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [05:03<06:01, 1.44MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [05:03<04:51, 1.78MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [05:04<03:46, 2.29MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [05:04<02:50, 3.04MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [05:05<04:30, 1.90MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [05:05<06:18, 1.36MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [05:05<05:45, 1.49MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [05:05<04:28, 1.91MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [05:06<03:18, 2.59MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [05:06<02:27, 3.47MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [05:07<27:04, 315kB/s] .vector_cache/glove.6B.zip:  41%|████      | 351M/862M [05:07<21:32, 395kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [05:07<15:41, 542kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [05:08<11:05, 765kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [05:08<07:56, 1.07MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [05:09<16:51, 501kB/s] .vector_cache/glove.6B.zip:  41%|████      | 355M/862M [05:09<14:18, 591kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [05:09<10:29, 805kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [05:09<07:42, 1.09MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [05:10<05:32, 1.52MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [05:20<24:05, 348kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [05:20<24:16, 345kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [05:20<19:40, 426kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [05:20<15:10, 552kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [05:20<11:05, 755kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [05:20<07:54, 1.06MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [05:21<06:46, 1.23MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [05:21<05:16, 1.57MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [05:21<04:24, 1.88MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [05:21<03:45, 2.20MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [05:21<03:22, 2.45MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [05:22<02:55, 2.83MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [05:22<02:42, 3.06MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [05:22<02:30, 3.29MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [05:22<02:25, 3.40MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [05:22<02:19, 3.56MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [05:23<07:00, 1.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [05:23<06:39, 1.24MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [05:23<05:46, 1.43MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [05:23<04:43, 1.74MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [05:23<03:59, 2.06MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [05:23<03:19, 2.47MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [05:24<03:02, 2.70MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [05:24<02:38, 3.11MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [05:24<02:11, 3.74MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [05:24<01:48, 4.53MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [05:25<32:44, 250kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [05:25<24:02, 340kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [05:25<17:30, 466kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [05:25<12:51, 634kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [05:25<09:40, 842kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [05:25<07:17, 1.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [05:26<05:26, 1.49MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [05:26<04:11, 1.93MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [05:27<06:10, 1.31MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [05:27<06:48, 1.19MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [05:27<07:52, 1.03MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [05:27<08:41, 932kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [05:27<08:43, 929kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [05:27<07:06, 1.14MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [05:28<05:39, 1.43MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [05:28<04:22, 1.85MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [05:28<03:30, 2.30MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [05:28<02:37, 3.07MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [05:29<05:23, 1.49MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [05:29<04:27, 1.80MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [05:29<03:38, 2.20MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [05:29<02:58, 2.69MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [05:29<02:16, 3.52MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [05:33<07:13, 1.10MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [05:33<11:47, 675kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [05:33<10:12, 780kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [05:33<07:36, 1.05MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [05:33<05:33, 1.43MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [05:33<04:08, 1.92MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [05:34<03:12, 2.46MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [05:35<07:11, 1.10MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [05:35<07:30, 1.05MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [05:35<07:07, 1.11MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [05:35<06:30, 1.21MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [05:35<05:24, 1.46MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [05:35<04:07, 1.91MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [05:35<03:03, 2.57MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [05:54<32:32, 241kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [05:54<30:52, 254kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [05:54<23:26, 334kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [05:54<16:55, 462kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [05:55<12:02, 648kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [05:55<08:30, 913kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [05:57<23:22, 332kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [05:57<22:56, 338kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [05:57<18:26, 421kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [05:57<14:13, 545kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [05:57<10:42, 724kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [05:58<07:45, 997kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [05:58<05:37, 1.37MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [05:58<04:05, 1.88MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [05:58<03:40, 2.09MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [05:58<02:43, 2.81MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [05:59<01:59, 3.82MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [06:00<13:40, 556kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [06:00<13:24, 566kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [06:01<10:21, 733kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [06:01<07:25, 1.02MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [06:01<05:19, 1.42MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [06:02<07:51, 959kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [06:02<07:22, 1.02MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [06:03<05:37, 1.34MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [06:03<04:01, 1.86MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [06:04<05:37, 1.33MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [06:04<05:48, 1.29MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [06:05<04:28, 1.66MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [06:05<03:14, 2.29MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [06:06<04:48, 1.54MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [06:06<05:36, 1.32MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [06:07<04:28, 1.65MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [06:07<03:15, 2.25MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [06:11<08:12, 893kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [06:11<13:23, 547kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [06:11<11:07, 658kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [06:11<08:09, 897kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [06:11<05:55, 1.23MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [06:16<08:47, 825kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [06:16<13:40, 530kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [06:16<11:19, 640kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [06:16<08:20, 869kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [06:17<05:57, 1.21MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [06:20<09:00, 798kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [06:20<12:21, 582kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [06:21<10:26, 688kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [06:21<07:40, 933kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [06:21<05:28, 1.30MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [06:21<04:37, 1.54MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [06:21<03:26, 2.06MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [06:22<02:51, 2.48MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [06:22<02:29, 2.85MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [06:22<02:10, 3.25MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [06:22<01:49, 3.86MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [06:26<10:39, 661kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [06:26<14:28, 487kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [06:26<11:47, 598kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [06:27<08:39, 812kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [06:27<06:08, 1.14MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [06:28<07:16, 959kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [06:28<08:02, 868kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [06:28<07:38, 913kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [06:28<06:35, 1.06MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [06:29<05:26, 1.28MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [06:29<04:13, 1.65MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [06:29<03:13, 2.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [06:29<02:22, 2.92MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [06:31<09:33, 722kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [06:31<14:04, 491kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [06:31<12:09, 568kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [06:31<09:56, 695kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [06:31<07:38, 903kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [06:32<05:43, 1.20MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [06:32<04:31, 1.52MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [06:32<03:40, 1.87MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [06:32<03:10, 2.17MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [06:32<02:49, 2.43MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [06:32<02:33, 2.69MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [06:32<02:18, 2.97MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [06:32<02:03, 3.32MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [06:33<08:42, 785kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [06:33<06:52, 995kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [06:33<05:38, 1.21MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [06:33<04:44, 1.44MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [06:33<04:04, 1.67MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [06:33<03:22, 2.02MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [06:34<02:55, 2.33MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [06:34<02:25, 2.81MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [06:34<01:55, 3.52MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [06:35<03:58, 1.71MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [06:35<03:34, 1.89MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [06:35<02:48, 2.41MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [06:35<02:05, 3.21MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [06:37<03:21, 2.00MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [06:37<04:23, 1.52MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [06:37<03:39, 1.83MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [06:37<02:53, 2.31MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [06:37<02:07, 3.14MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [06:42<08:40, 765kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [06:42<12:52, 515kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [06:42<10:37, 624kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [06:42<07:47, 850kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [06:42<05:33, 1.19MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [06:44<05:35, 1.17MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [06:44<05:55, 1.11MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [06:44<05:27, 1.20MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [06:44<04:31, 1.45MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [06:44<03:33, 1.84MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [06:44<02:51, 2.29MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [06:44<02:14, 2.92MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [06:46<03:07, 2.08MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [06:46<03:06, 2.09MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [06:46<02:33, 2.53MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [06:46<01:58, 3.27MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [06:46<01:30, 4.26MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [06:48<05:52, 1.10MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [06:48<10:40, 602kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [06:48<09:01, 712kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [06:48<06:38, 966kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [06:49<04:50, 1.32MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [06:49<03:28, 1.83MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [06:56<43:01, 148kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [06:56<55:16, 115kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [06:56<41:38, 153kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [06:56<29:50, 213kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [06:56<21:03, 301kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [06:56<14:50, 426kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [06:57<10:29, 600kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [07:04<1:38:47, 63.7kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [07:04<1:15:44, 83.1kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [07:04<54:30, 115kB/s]   .vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [07:04<38:29, 163kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [07:04<26:54, 232kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [07:04<18:51, 330kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [07:04<13:20, 465kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [07:05<11:01, 561kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [07:06<08:07, 760kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [07:06<05:55, 1.04MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [07:06<04:18, 1.42MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [07:06<03:07, 1.96MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [07:07<10:02, 609kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [07:08<09:28, 644kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [07:08<07:18, 836kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [07:08<05:22, 1.13MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [07:08<03:52, 1.57MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [07:12<07:08, 847kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [07:12<11:14, 537kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [07:12<09:18, 649kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [07:12<06:49, 883kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [07:13<04:52, 1.23MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [07:14<05:29, 1.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [07:14<05:28, 1.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [07:14<04:59, 1.20MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [07:14<04:15, 1.40MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [07:14<03:21, 1.78MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [07:14<02:31, 2.36MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [07:15<01:52, 3.16MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [07:23<23:53, 247kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [07:23<21:46, 271kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [07:23<16:43, 353kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [07:23<12:00, 490kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [07:23<08:28, 692kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [07:25<07:36, 766kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [07:25<08:12, 710kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [07:25<07:53, 739kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [07:25<07:24, 787kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [07:25<06:13, 937kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [07:25<04:42, 1.24MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [07:25<03:36, 1.61MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [07:25<02:47, 2.07MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [07:25<02:10, 2.66MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [07:26<01:45, 3.28MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [07:27<07:11, 801kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [07:27<09:23, 614kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [07:28<09:38, 598kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [07:28<07:53, 730kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [07:28<05:57, 966kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [07:28<04:19, 1.33MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [07:28<03:08, 1.82MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [07:30<05:02, 1.13MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [07:30<05:12, 1.09MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [07:30<04:45, 1.20MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [07:30<03:34, 1.59MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [07:30<02:38, 2.15MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [07:30<01:56, 2.90MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [07:32<07:21, 765kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [07:32<07:03, 798kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [07:32<05:19, 1.06MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [07:32<03:59, 1.41MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [07:32<02:55, 1.91MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [07:32<02:08, 2.59MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [07:34<16:48, 331kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [07:34<13:39, 407kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [07:34<10:07, 549kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [07:34<07:18, 759kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [07:34<05:16, 1.05MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [07:34<03:51, 1.43MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [07:34<02:52, 1.91MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [07:36<13:05, 419kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [07:36<11:10, 491kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [07:36<08:58, 611kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [07:36<06:49, 802kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [07:36<05:02, 1.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [07:36<03:37, 1.50MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [07:36<02:41, 2.02MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [07:38<06:02, 898kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [07:38<05:07, 1.06MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [07:38<03:49, 1.41MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [07:38<02:46, 1.94MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [07:38<02:05, 2.57MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [07:40<05:05, 1.05MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [07:40<05:23, 992kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [07:40<04:14, 1.26MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [07:40<03:11, 1.67MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [07:40<02:24, 2.21MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [07:40<01:52, 2.83MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [07:40<01:28, 3.57MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [07:42<08:00, 660kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [07:42<06:44, 783kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [07:42<05:12, 1.01MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [07:42<03:46, 1.39MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [07:42<02:47, 1.88MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [07:42<02:04, 2.52MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [07:44<10:02, 519kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [07:44<11:30, 453kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [07:44<09:04, 574kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [07:44<06:57, 748kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [07:44<05:09, 1.01MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [07:44<03:50, 1.35MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [07:45<02:57, 1.75MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [07:45<02:19, 2.22MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [07:45<01:53, 2.73MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [07:45<01:34, 3.28MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [07:46<24:54, 207kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [07:46<17:58, 286kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [07:46<12:39, 404kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [07:46<08:57, 569kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [07:46<06:26, 789kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [07:48<08:57, 566kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [07:48<08:15, 614kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [07:48<06:16, 807kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [07:48<04:30, 1.12MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [07:48<03:16, 1.53MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [07:48<02:25, 2.07MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [07:50<11:13, 446kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [07:50<08:39, 577kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [07:50<06:36, 757kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [07:50<04:47, 1.04MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [07:50<03:27, 1.43MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [07:50<02:33, 1.93MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [07:59<23:38, 209kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [07:59<21:27, 230kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [07:59<16:11, 305kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [07:59<11:33, 426kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [07:59<08:10, 599kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [08:04<08:52, 549kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [08:04<11:14, 433kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [08:04<09:01, 539kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [08:04<06:33, 740kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [08:05<04:41, 1.03MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [08:07<05:11, 925kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [08:07<08:59, 534kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [08:07<07:59, 600kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [08:07<06:16, 764kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [08:08<04:40, 1.03MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [08:08<03:25, 1.39MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [08:08<02:33, 1.86MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [08:08<01:53, 2.51MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [08:09<11:45, 402kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [08:09<09:20, 506kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [08:09<07:20, 644kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [08:09<05:43, 824kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [08:10<04:11, 1.12MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [08:10<03:03, 1.54MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [08:10<02:12, 2.11MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [08:14<1:09:47, 66.8kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [08:14<53:33, 87.0kB/s]  .vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [08:14<38:36, 121kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [08:14<27:14, 171kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [08:14<19:04, 242kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [08:14<13:23, 344kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [08:16<12:54, 356kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [08:16<10:15, 448kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [08:16<07:53, 582kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [08:16<05:42, 802kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [08:16<04:03, 1.12MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [08:19<05:39, 799kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [08:19<08:35, 527kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [08:20<07:04, 639kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [08:20<05:12, 866kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [08:20<03:42, 1.21MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [08:21<03:51, 1.15MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [08:21<04:22, 1.02MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [08:21<04:09, 1.07MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [08:22<03:40, 1.21MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [08:22<03:02, 1.46MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [08:22<02:30, 1.77MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [08:22<02:08, 2.06MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [08:22<01:51, 2.38MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [08:22<01:35, 2.76MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [08:22<01:17, 3.43MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [08:28<07:23, 594kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [08:28<08:50, 496kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [08:28<07:19, 599kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [08:28<05:39, 773kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [08:28<04:19, 1.01MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [08:28<03:13, 1.35MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [08:28<02:23, 1.81MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [08:29<01:47, 2.42MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 604M/862M [08:29<01:22, 3.12MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [08:29<01:08, 3.75MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [08:31<11:03, 388kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [08:32<12:14, 350kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [08:32<09:21, 458kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [08:32<07:06, 603kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [08:32<05:03, 844kB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [08:32<03:37, 1.17MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [08:33<04:09, 1.02MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [08:34<03:37, 1.16MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [08:34<02:48, 1.50MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [08:34<02:04, 2.02MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [08:34<01:36, 2.61MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [08:34<01:17, 3.24MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [08:34<01:03, 3.90MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [08:37<37:27, 111kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [08:37<30:31, 136kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [08:37<22:20, 186kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [08:37<15:49, 262kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [08:37<11:06, 371kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [08:39<08:37, 473kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [08:39<07:56, 514kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [08:39<07:08, 572kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [08:39<06:22, 640kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [08:39<06:08, 665kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [08:39<05:23, 757kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [08:39<04:14, 960kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [08:39<03:11, 1.27MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [08:40<02:28, 1.64MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [08:40<01:55, 2.10MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [08:40<01:26, 2.79MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [08:46<13:54, 289kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [08:46<13:08, 306kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [08:46<10:11, 394kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [08:46<07:20, 545kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [08:46<05:11, 767kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [08:48<04:36, 857kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [08:48<05:10, 762kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [08:48<04:55, 802kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [08:48<04:09, 949kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [08:48<03:15, 1.21MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [08:48<02:31, 1.55MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [08:49<02:02, 1.92MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [08:49<01:43, 2.28MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [08:49<01:30, 2.60MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [08:49<01:23, 2.81MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [08:49<01:18, 2.99MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [08:49<01:13, 3.17MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [08:49<01:09, 3.33MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [08:50<03:46, 1.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [08:50<03:01, 1.28MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [08:50<02:21, 1.64MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [08:50<01:52, 2.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [08:50<01:28, 2.61MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [08:50<01:13, 3.14MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [08:51<01:03, 3.62MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [08:51<00:56, 4.06MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [08:52<05:03, 754kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [08:52<04:27, 853kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [08:52<03:58, 958kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [08:52<03:04, 1.24MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [08:52<02:18, 1.65MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [08:52<01:46, 2.13MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [08:52<01:27, 2.58MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [08:53<01:11, 3.17MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [08:53<01:01, 3.66MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [09:01<36:44, 102kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [09:02<29:26, 127kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [09:02<21:21, 175kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [09:02<15:07, 246kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [09:02<10:36, 349kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [09:18<17:16, 212kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [09:18<15:48, 232kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [09:19<11:59, 306kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [09:19<08:43, 420kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [09:19<06:16, 582kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [09:19<04:30, 807kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [09:19<03:16, 1.11MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [09:23<05:32, 651kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [09:23<07:03, 511kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [09:23<05:49, 619kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [09:23<04:31, 796kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [09:23<03:21, 1.07MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [09:24<02:32, 1.41MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [09:24<01:59, 1.80MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [09:24<01:35, 2.23MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [09:24<01:19, 2.68MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [09:24<01:06, 3.18MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [09:24<00:58, 3.62MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [09:24<00:51, 4.07MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [09:24<00:46, 4.51MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [09:24<00:43, 4.86MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [09:24<00:41, 5.04MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [09:25<00:39, 5.29MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [09:25<00:37, 5.51MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [09:25<00:32, 6.44MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [09:30<27:05, 127kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [09:31<22:31, 153kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [09:31<16:40, 207kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [09:31<11:50, 290kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [09:31<08:19, 410kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [09:33<06:47, 498kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [09:33<08:12, 412kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [09:33<06:33, 516kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [09:33<04:45, 707kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [09:34<03:26, 973kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [09:34<02:28, 1.35MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [09:34<01:50, 1.80MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [09:35<04:05, 809kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [09:35<03:47, 873kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [09:35<03:06, 1.07MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [09:35<02:25, 1.36MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [09:35<01:49, 1.80MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [09:36<01:21, 2.41MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [09:36<01:03, 3.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [09:37<02:45, 1.18MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [09:37<02:43, 1.19MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [09:37<02:13, 1.46MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [09:37<01:38, 1.96MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [09:37<01:13, 2.63MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [09:38<00:56, 3.38MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [09:39<04:15, 746kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [09:39<04:18, 736kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [09:39<03:37, 876kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [09:39<02:40, 1.18MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [09:39<01:59, 1.59MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [09:40<01:27, 2.14MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [09:40<01:06, 2.81MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [09:41<06:32, 475kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [09:41<05:32, 561kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [09:41<04:36, 673kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [09:41<03:34, 866kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [09:41<02:42, 1.14MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [09:41<02:05, 1.48MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [09:42<01:38, 1.86MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [09:42<01:22, 2.24MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [09:42<01:09, 2.64MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [09:42<00:57, 3.17MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [09:43<02:01, 1.50MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [09:43<01:52, 1.62MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [09:43<01:32, 1.96MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [09:43<01:11, 2.54MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [09:43<00:55, 3.24MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [09:43<00:44, 4.03MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [09:44<00:37, 4.73MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [09:45<19:06, 155kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [09:45<14:14, 208kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [09:45<10:18, 287kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [09:45<07:20, 401kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [09:45<05:13, 561kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [09:45<03:45, 777kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [09:46<02:44, 1.06MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [09:46<02:03, 1.41MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [09:47<04:49, 601kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [09:47<04:03, 714kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [09:47<03:07, 924kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [09:47<02:19, 1.24MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [09:47<01:42, 1.67MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [09:47<01:17, 2.20MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [09:48<00:58, 2.92MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [09:49<11:37, 243kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [09:49<09:12, 307kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [09:49<07:01, 402kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [09:49<05:06, 551kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [09:49<03:38, 768kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [09:49<02:36, 1.06MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [09:50<01:53, 1.46MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [09:52<05:54, 466kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [09:52<06:54, 399kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [09:52<05:29, 502kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [09:52<03:58, 691kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [09:52<02:49, 964kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [10:03<06:15, 429kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [10:03<07:06, 378kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [10:03<05:36, 479kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [10:03<04:02, 661kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [10:03<02:54, 915kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [10:04<02:04, 1.27MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [10:04<02:11, 1.20MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [10:04<01:39, 1.57MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [10:05<01:15, 2.07MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [10:05<01:00, 2.57MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [10:05<00:49, 3.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [10:05<00:40, 3.81MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [10:09<05:04, 503kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [10:09<06:18, 404kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [10:09<05:23, 473kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [10:09<04:19, 589kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [10:09<03:13, 788kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [10:09<02:19, 1.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [10:09<01:38, 1.51MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [10:18<10:13, 243kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [10:18<09:17, 267kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [10:18<07:05, 349kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [10:18<05:05, 485kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [10:18<03:35, 680kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [10:26<05:08, 469kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [10:26<06:00, 401kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [10:26<04:46, 505kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [10:26<03:27, 695kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [10:26<02:27, 970kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [10:31<03:14, 724kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [10:31<04:41, 499kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [10:31<03:51, 608kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [10:31<02:51, 819kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [10:31<02:02, 1.13MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [10:31<01:27, 1.57MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [10:32<02:14, 1.01MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [10:33<01:45, 1.29MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [10:33<01:21, 1.67MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [10:33<01:02, 2.15MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [10:33<00:52, 2.58MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [10:33<00:43, 3.10MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [10:33<00:35, 3.74MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [10:38<05:21, 412kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [10:38<06:03, 364kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [10:38<04:48, 459kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [10:38<03:32, 621kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [10:39<02:31, 864kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [10:39<01:46, 1.21MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [10:40<02:28, 863kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [10:40<02:03, 1.04MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [10:40<01:39, 1.29MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [10:40<01:16, 1.67MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [10:40<00:57, 2.20MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [10:41<00:43, 2.86MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [10:41<00:33, 3.69MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [10:44<32:52, 63.0kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [10:45<25:11, 82.1kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [10:45<18:05, 114kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [10:45<12:44, 161kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [10:45<08:56, 229kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [10:45<06:17, 323kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [10:45<04:24, 456kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [10:46<04:14, 473kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [10:47<03:20, 598kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [10:47<02:37, 758kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [10:47<02:02, 973kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [10:47<01:33, 1.28MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [10:47<01:11, 1.66MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [10:47<00:54, 2.16MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [10:47<00:44, 2.65MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [10:47<00:34, 3.34MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [10:49<04:16, 452kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [10:49<04:10, 463kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [10:49<03:56, 489kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [10:49<03:21, 574kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [10:49<02:35, 742kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [10:50<01:55, 999kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [10:50<01:24, 1.35MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [10:50<01:04, 1.76MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [10:50<00:51, 2.21MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [10:50<00:41, 2.74MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [10:51<01:32, 1.20MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [10:51<01:21, 1.37MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [10:51<01:19, 1.40MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [10:51<01:18, 1.41MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [10:51<01:14, 1.50MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [10:51<01:07, 1.63MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [10:52<00:56, 1.95MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [10:52<00:43, 2.51MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [10:52<00:33, 3.25MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [10:54<01:43, 1.04MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [10:55<03:06, 578kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [10:55<02:38, 678kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [10:55<02:11, 819kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [10:55<01:43, 1.03MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [10:55<01:19, 1.34MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [10:55<00:59, 1.78MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [10:55<00:45, 2.30MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [10:55<00:35, 2.93MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [11:03<05:40, 304kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [11:03<05:15, 328kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [11:03<04:46, 361kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [11:03<03:59, 431kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [11:03<03:17, 524kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [11:03<02:31, 680kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [11:03<01:50, 925kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [11:03<01:19, 1.27MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [11:04<00:56, 1.76MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [11:04<01:51, 888kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [11:04<01:21, 1.21MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [11:04<01:01, 1.59MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [11:04<00:48, 2.01MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [11:05<00:38, 2.53MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [11:05<00:30, 3.11MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [11:05<00:26, 3.65MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [11:16<24:54, 63.8kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [11:16<19:04, 83.2kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [11:16<13:42, 116kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [11:17<09:36, 164kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [11:17<06:40, 233kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [11:24<06:23, 237kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [11:25<05:59, 253kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [11:25<04:32, 333kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [11:25<03:14, 465kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [11:25<02:15, 656kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [11:30<02:47, 521kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [11:30<03:18, 438kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [11:30<03:10, 457kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [11:30<02:40, 541kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [11:30<02:08, 674kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [11:30<01:35, 902kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [11:30<01:10, 1.22MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [11:30<00:50, 1.67MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [11:34<01:37, 854kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [11:34<02:36, 528kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [11:34<02:09, 640kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [11:34<01:33, 876kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [11:34<01:08, 1.19MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [11:34<00:47, 1.66MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [11:36<01:23, 944kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [11:36<01:09, 1.12MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [11:36<00:56, 1.38MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [11:36<00:44, 1.76MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [11:36<00:33, 2.29MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [11:36<00:27, 2.80MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [11:36<00:22, 3.30MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [11:36<00:19, 3.79MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [11:36<00:17, 4.16MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [11:41<1:10:52, 17.5kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [11:41<50:44, 24.5kB/s]  .vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [11:41<35:44, 34.7kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [11:42<24:56, 49.4kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [11:42<17:14, 70.4kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [11:43<11:53, 98.7kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [11:43<08:40, 135kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [11:43<06:30, 180kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [11:43<05:06, 229kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [11:44<04:04, 287kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [11:44<03:19, 351kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [11:44<02:45, 422kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [11:44<02:21, 493kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [11:44<02:06, 553kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [11:44<01:57, 592kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [11:44<01:51, 624kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [11:44<01:43, 670kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [11:44<01:36, 719kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [11:44<01:34, 734kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [11:45<01:32, 749kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [11:45<01:30, 768kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [11:45<01:27, 790kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [11:45<01:25, 812kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [11:45<01:26, 797kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [11:45<01:36, 716kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [11:45<01:36, 714kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [11:45<01:41, 680kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [11:45<01:44, 657kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [11:46<01:41, 675kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [11:46<01:31, 750kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [11:46<01:15, 901kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [11:46<01:00, 1.13MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [11:46<00:45, 1.49MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [11:46<00:33, 1.99MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [11:47<01:55, 572kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [11:47<01:21, 798kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [11:47<00:57, 1.11MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [11:47<00:41, 1.53MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [11:47<00:34, 1.80MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [11:47<00:25, 2.37MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [11:47<00:20, 2.99MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [11:47<00:16, 3.67MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [11:47<00:13, 4.40MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [11:49<00:42, 1.35MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [11:49<00:49, 1.16MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [11:49<00:49, 1.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [11:49<00:38, 1.48MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [11:49<00:28, 2.00MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [11:51<00:29, 1.84MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [11:51<00:35, 1.53MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [11:51<00:32, 1.63MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [11:51<00:27, 1.97MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [11:51<00:20, 2.57MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [11:51<00:14, 3.43MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [11:53<00:33, 1.49MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [11:53<00:40, 1.23MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [11:53<00:46, 1.06MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [11:53<00:49, 998kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [11:53<00:46, 1.05MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [11:53<00:38, 1.28MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [11:54<00:28, 1.68MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [11:54<00:21, 2.21MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [11:54<00:16, 2.80MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [11:54<00:13, 3.39MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [11:57<01:12, 633kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [11:57<01:37, 467kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [11:57<01:20, 565kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [11:57<01:01, 740kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [11:57<00:43, 1.01MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [11:57<00:30, 1.41MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [11:57<00:21, 1.93MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [11:59<04:03, 171kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [11:59<02:56, 234kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [11:59<02:03, 329kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [11:59<01:23, 466kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [12:12<02:55, 214kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [12:13<02:41, 232kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [12:13<02:01, 306kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [12:13<01:25, 426kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [12:13<00:57, 603kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [12:15<00:50, 662kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [12:15<00:48, 682kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [12:15<00:41, 791kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [12:15<00:32, 1.02MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [12:15<00:23, 1.38MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [12:15<00:16, 1.88MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [12:15<00:11, 2.50MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [12:20<01:46, 273kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [12:21<01:39, 292kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [12:21<01:16, 379kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [12:21<00:53, 525kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [12:21<00:34, 743kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [12:26<01:05, 381kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [12:26<01:12, 345kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [12:26<00:57, 436kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [12:26<00:42, 580kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [12:26<00:29, 806kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [12:26<00:19, 1.13MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [12:31<00:41, 506kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [12:31<00:50, 417kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [12:31<00:39, 522kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [12:32<00:27, 716kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [12:32<00:18, 1.00MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [12:36<00:23, 726kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [12:36<00:33, 495kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [12:36<00:30, 554kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [12:36<00:23, 701kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [12:36<00:17, 917kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [12:36<00:12, 1.23MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [12:37<00:09, 1.64MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [12:37<00:07, 2.02MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [12:37<00:05, 2.54MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [12:37<00:04, 3.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [12:38<00:13, 931kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [12:38<00:10, 1.13MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [12:38<00:08, 1.43MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [12:38<00:06, 1.80MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [12:38<00:04, 2.25MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [12:38<00:03, 2.89MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [12:38<00:03, 2.82MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [12:40<00:04, 1.83MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [12:40<00:05, 1.64MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [12:40<00:04, 1.96MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [12:40<00:02, 2.59MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [12:40<00:01, 3.38MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [12:40<00:01, 4.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [12:42<00:04, 986kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [12:42<00:04, 898kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [12:42<00:03, 1.16MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [12:42<00:02, 1.52MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [12:42<00:01, 2.01MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [12:42<00:00, 2.62MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [12:42<00:00, 3.29MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [12:51<00:00, 202kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [12:51<00:00, 221kB/s].vector_cache/glove.6B.zip: 862MB [12:51, 297kB/s]                           .vector_cache/glove.6B.zip: 862MB [12:51, 1.12MB/s]
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<84:49:48,  1.31it/s]  0%|          | 846/400000 [00:00<59:15:34,  1.87it/s]  0%|          | 1728/400000 [00:00<41:23:37,  2.67it/s]  1%|          | 2584/400000 [00:01<28:55:02,  3.82it/s]  1%|          | 3457/400000 [00:01<20:12:05,  5.45it/s]  1%|          | 4262/400000 [00:01<14:06:58,  7.79it/s]  1%|▏         | 5158/400000 [00:01<9:51:45, 11.12it/s]   2%|▏         | 6042/400000 [00:01<6:53:31, 15.88it/s]  2%|▏         | 6910/400000 [00:01<4:49:03, 22.66it/s]  2%|▏         | 7803/400000 [00:01<3:22:06, 32.34it/s]  2%|▏         | 8698/400000 [00:01<2:21:22, 46.13it/s]  2%|▏         | 9602/400000 [00:01<1:38:56, 65.76it/s]  3%|▎         | 10494/400000 [00:01<1:09:19, 93.64it/s]  3%|▎         | 11370/400000 [00:02<48:38, 133.16it/s]   3%|▎         | 12271/400000 [00:02<34:11, 189.03it/s]  3%|▎         | 13170/400000 [00:02<24:05, 267.63it/s]  4%|▎         | 14072/400000 [00:02<17:02, 377.52it/s]  4%|▎         | 14963/400000 [00:02<12:07, 529.34it/s]  4%|▍         | 15844/400000 [00:02<08:41, 736.76it/s]  4%|▍         | 16737/400000 [00:02<06:17, 1016.57it/s]  4%|▍         | 17617/400000 [00:02<04:36, 1382.93it/s]  5%|▍         | 18506/400000 [00:02<03:25, 1852.04it/s]  5%|▍         | 19397/400000 [00:02<02:36, 2429.20it/s]  5%|▌         | 20281/400000 [00:03<02:02, 3099.30it/s]  5%|▌         | 21161/400000 [00:03<01:38, 3832.57it/s]  6%|▌         | 22042/400000 [00:03<01:21, 4614.21it/s]  6%|▌         | 22941/400000 [00:03<01:09, 5402.40it/s]  6%|▌         | 23840/400000 [00:03<01:01, 6105.10it/s]  6%|▌         | 24721/400000 [00:03<00:57, 6558.61it/s]  6%|▋         | 25619/400000 [00:03<00:52, 7135.25it/s]  7%|▋         | 26487/400000 [00:03<00:50, 7406.32it/s]  7%|▋         | 27363/400000 [00:03<00:47, 7765.30it/s]  7%|▋         | 28228/400000 [00:03<00:46, 8011.02it/s]  7%|▋         | 29099/400000 [00:04<00:45, 8207.58it/s]  7%|▋         | 29985/400000 [00:04<00:44, 8368.79it/s]  8%|▊         | 30854/400000 [00:04<00:43, 8460.81it/s]  8%|▊         | 31722/400000 [00:04<00:43, 8456.72it/s]  8%|▊         | 32609/400000 [00:04<00:42, 8576.58it/s]  8%|▊         | 33478/400000 [00:04<00:42, 8595.43it/s]  9%|▊         | 34346/400000 [00:04<00:42, 8611.97it/s]  9%|▉         | 35213/400000 [00:04<00:45, 8105.94it/s]  9%|▉         | 36115/400000 [00:04<00:43, 8357.50it/s]  9%|▉         | 36998/400000 [00:05<00:42, 8491.44it/s]  9%|▉         | 37867/400000 [00:05<00:42, 8548.66it/s] 10%|▉         | 38752/400000 [00:05<00:41, 8635.55it/s] 10%|▉         | 39657/400000 [00:05<00:41, 8755.38it/s] 10%|█         | 40566/400000 [00:05<00:40, 8851.45it/s] 10%|█         | 41457/400000 [00:05<00:40, 8868.64it/s] 11%|█         | 42353/400000 [00:05<00:40, 8893.12it/s] 11%|█         | 43244/400000 [00:05<00:40, 8776.93it/s] 11%|█         | 44149/400000 [00:05<00:40, 8854.65it/s] 11%|█▏        | 45045/400000 [00:05<00:39, 8884.43it/s] 11%|█▏        | 45951/400000 [00:06<00:39, 8933.57it/s] 12%|█▏        | 46845/400000 [00:06<00:39, 8883.57it/s] 12%|█▏        | 47734/400000 [00:06<00:39, 8877.29it/s] 12%|█▏        | 48647/400000 [00:06<00:39, 8950.48it/s] 12%|█▏        | 49543/400000 [00:06<00:39, 8799.06it/s] 13%|█▎        | 50427/400000 [00:06<00:39, 8811.19it/s] 13%|█▎        | 51318/400000 [00:06<00:39, 8840.10it/s] 13%|█▎        | 52227/400000 [00:06<00:39, 8912.24it/s] 13%|█▎        | 53129/400000 [00:06<00:38, 8941.88it/s] 14%|█▎        | 54031/400000 [00:06<00:38, 8964.31it/s] 14%|█▎        | 54928/400000 [00:07<00:38, 8854.92it/s] 14%|█▍        | 55814/400000 [00:07<00:38, 8851.75it/s] 14%|█▍        | 56707/400000 [00:07<00:38, 8873.68it/s] 14%|█▍        | 57629/400000 [00:07<00:38, 8973.79it/s] 15%|█▍        | 58527/400000 [00:07<00:38, 8862.91it/s] 15%|█▍        | 59418/400000 [00:07<00:38, 8875.17it/s] 15%|█▌        | 60316/400000 [00:07<00:38, 8905.61it/s] 15%|█▌        | 61208/400000 [00:07<00:38, 8908.49it/s] 16%|█▌        | 62111/400000 [00:07<00:37, 8943.31it/s] 16%|█▌        | 63016/400000 [00:07<00:37, 8974.32it/s] 16%|█▌        | 63914/400000 [00:08<00:37, 8970.99it/s] 16%|█▌        | 64812/400000 [00:08<00:37, 8840.44it/s] 16%|█▋        | 65708/400000 [00:08<00:37, 8873.36it/s] 17%|█▋        | 66598/400000 [00:08<00:37, 8878.60it/s] 17%|█▋        | 67487/400000 [00:08<00:37, 8845.91it/s] 17%|█▋        | 68399/400000 [00:08<00:37, 8924.14it/s] 17%|█▋        | 69298/400000 [00:08<00:36, 8943.13it/s] 18%|█▊        | 70193/400000 [00:08<00:37, 8904.87it/s] 18%|█▊        | 71084/400000 [00:08<00:37, 8867.05it/s] 18%|█▊        | 71971/400000 [00:08<00:37, 8735.96it/s] 18%|█▊        | 72882/400000 [00:09<00:36, 8844.30it/s] 18%|█▊        | 73794/400000 [00:09<00:36, 8923.72it/s] 19%|█▊        | 74694/400000 [00:09<00:36, 8944.99it/s] 19%|█▉        | 75607/400000 [00:09<00:36, 8998.81it/s] 19%|█▉        | 76510/400000 [00:09<00:35, 8994.54it/s] 19%|█▉        | 77416/400000 [00:09<00:35, 9011.56it/s] 20%|█▉        | 78325/400000 [00:09<00:35, 9033.30it/s] 20%|█▉        | 79229/400000 [00:09<00:35, 8993.84it/s] 20%|██        | 80129/400000 [00:09<00:35, 8974.57it/s] 20%|██        | 81027/400000 [00:09<00:36, 8791.14it/s] 20%|██        | 81928/400000 [00:10<00:35, 8854.17it/s] 21%|██        | 82815/400000 [00:10<00:35, 8823.49it/s] 21%|██        | 83699/400000 [00:10<00:35, 8827.52it/s] 21%|██        | 84604/400000 [00:10<00:35, 8890.96it/s] 21%|██▏       | 85494/400000 [00:10<00:36, 8654.66it/s] 22%|██▏       | 86394/400000 [00:10<00:35, 8753.07it/s] 22%|██▏       | 87271/400000 [00:10<00:35, 8750.10it/s] 22%|██▏       | 88148/400000 [00:10<00:37, 8316.02it/s] 22%|██▏       | 89047/400000 [00:10<00:36, 8506.90it/s] 22%|██▏       | 89907/400000 [00:11<00:36, 8529.12it/s] 23%|██▎       | 90801/400000 [00:11<00:35, 8647.98it/s] 23%|██▎       | 91669/400000 [00:11<00:36, 8385.56it/s] 23%|██▎       | 92517/400000 [00:11<00:36, 8411.75it/s] 23%|██▎       | 93416/400000 [00:11<00:35, 8575.96it/s] 24%|██▎       | 94318/400000 [00:11<00:35, 8703.82it/s] 24%|██▍       | 95217/400000 [00:11<00:34, 8786.15it/s] 24%|██▍       | 96098/400000 [00:11<00:35, 8669.40it/s] 24%|██▍       | 96989/400000 [00:11<00:34, 8738.18it/s] 24%|██▍       | 97892/400000 [00:11<00:34, 8823.25it/s] 25%|██▍       | 98793/400000 [00:12<00:33, 8877.57it/s] 25%|██▍       | 99682/400000 [00:12<00:34, 8815.50it/s] 25%|██▌       | 100565/400000 [00:12<00:34, 8721.77it/s] 25%|██▌       | 101456/400000 [00:12<00:34, 8776.88it/s] 26%|██▌       | 102335/400000 [00:12<00:33, 8769.28it/s] 26%|██▌       | 103213/400000 [00:12<00:34, 8689.92it/s] 26%|██▌       | 104117/400000 [00:12<00:33, 8790.37it/s] 26%|██▋       | 105010/400000 [00:12<00:33, 8830.58it/s] 26%|██▋       | 105894/400000 [00:12<00:33, 8807.23it/s] 27%|██▋       | 106778/400000 [00:12<00:33, 8815.81it/s] 27%|██▋       | 107682/400000 [00:13<00:32, 8880.66it/s] 27%|██▋       | 108584/400000 [00:13<00:32, 8919.43it/s] 27%|██▋       | 109477/400000 [00:13<00:32, 8877.54it/s] 28%|██▊       | 110365/400000 [00:13<00:32, 8787.76it/s] 28%|██▊       | 111245/400000 [00:13<00:32, 8775.29it/s] 28%|██▊       | 112166/400000 [00:13<00:32, 8900.58it/s] 28%|██▊       | 113065/400000 [00:13<00:32, 8926.53it/s] 28%|██▊       | 113959/400000 [00:13<00:32, 8904.34it/s] 29%|██▊       | 114850/400000 [00:13<00:32, 8852.89it/s] 29%|██▉       | 115761/400000 [00:13<00:31, 8927.24it/s] 29%|██▉       | 116670/400000 [00:14<00:31, 8972.53it/s] 29%|██▉       | 117568/400000 [00:14<00:31, 8902.52it/s] 30%|██▉       | 118459/400000 [00:14<00:31, 8869.13it/s] 30%|██▉       | 119347/400000 [00:14<00:31, 8849.26it/s] 30%|███       | 120251/400000 [00:14<00:31, 8904.75it/s] 30%|███       | 121144/400000 [00:14<00:31, 8910.01it/s] 31%|███       | 122046/400000 [00:14<00:31, 8941.36it/s] 31%|███       | 122941/400000 [00:14<00:31, 8882.01it/s] 31%|███       | 123838/400000 [00:14<00:31, 8906.88it/s] 31%|███       | 124729/400000 [00:14<00:30, 8883.79it/s] 31%|███▏      | 125618/400000 [00:15<00:31, 8808.46it/s] 32%|███▏      | 126506/400000 [00:15<00:30, 8828.73it/s] 32%|███▏      | 127394/400000 [00:15<00:30, 8841.26it/s] 32%|███▏      | 128279/400000 [00:15<00:30, 8787.50it/s] 32%|███▏      | 129188/400000 [00:15<00:30, 8876.00it/s] 33%|███▎      | 130076/400000 [00:15<00:30, 8822.83it/s] 33%|███▎      | 130962/400000 [00:15<00:30, 8833.26it/s] 33%|███▎      | 131846/400000 [00:15<00:30, 8707.92it/s] 33%|███▎      | 132718/400000 [00:15<00:31, 8607.98it/s] 33%|███▎      | 133593/400000 [00:15<00:30, 8647.36it/s] 34%|███▎      | 134459/400000 [00:16<00:30, 8617.02it/s] 34%|███▍      | 135322/400000 [00:16<00:31, 8481.29it/s] 34%|███▍      | 136182/400000 [00:16<00:30, 8513.25it/s] 34%|███▍      | 137044/400000 [00:16<00:30, 8544.54it/s] 34%|███▍      | 137919/400000 [00:16<00:30, 8604.51it/s] 35%|███▍      | 138780/400000 [00:16<00:30, 8544.81it/s] 35%|███▍      | 139662/400000 [00:16<00:30, 8625.13it/s] 35%|███▌      | 140525/400000 [00:16<00:30, 8376.90it/s] 35%|███▌      | 141365/400000 [00:16<00:31, 8324.04it/s] 36%|███▌      | 142236/400000 [00:16<00:30, 8434.80it/s] 36%|███▌      | 143081/400000 [00:17<00:31, 8212.80it/s] 36%|███▌      | 143952/400000 [00:17<00:30, 8354.04it/s] 36%|███▌      | 144817/400000 [00:17<00:30, 8439.27it/s] 36%|███▋      | 145697/400000 [00:17<00:29, 8542.46it/s] 37%|███▋      | 146581/400000 [00:17<00:29, 8629.14it/s] 37%|███▋      | 147447/400000 [00:17<00:29, 8636.30it/s] 37%|███▋      | 148329/400000 [00:17<00:28, 8690.29it/s] 37%|███▋      | 149207/400000 [00:17<00:28, 8715.03it/s] 38%|███▊      | 150080/400000 [00:17<00:28, 8665.09it/s] 38%|███▊      | 150969/400000 [00:17<00:28, 8730.72it/s] 38%|███▊      | 151861/400000 [00:18<00:28, 8786.08it/s] 38%|███▊      | 152740/400000 [00:18<00:28, 8681.59it/s] 38%|███▊      | 153609/400000 [00:18<00:28, 8675.26it/s] 39%|███▊      | 154477/400000 [00:18<00:29, 8407.93it/s] 39%|███▉      | 155320/400000 [00:18<00:29, 8300.32it/s] 39%|███▉      | 156183/400000 [00:18<00:29, 8396.30it/s] 39%|███▉      | 157061/400000 [00:18<00:28, 8506.61it/s] 39%|███▉      | 157930/400000 [00:18<00:28, 8559.99it/s] 40%|███▉      | 158812/400000 [00:18<00:27, 8636.33it/s] 40%|███▉      | 159718/400000 [00:19<00:27, 8757.86it/s] 40%|████      | 160621/400000 [00:19<00:27, 8836.42it/s] 40%|████      | 161506/400000 [00:19<00:27, 8793.20it/s] 41%|████      | 162396/400000 [00:19<00:26, 8822.58it/s] 41%|████      | 163292/400000 [00:19<00:26, 8861.30it/s] 41%|████      | 164190/400000 [00:19<00:26, 8894.10it/s] 41%|████▏     | 165083/400000 [00:19<00:26, 8904.38it/s] 41%|████▏     | 165974/400000 [00:19<00:26, 8875.94it/s] 42%|████▏     | 166862/400000 [00:19<00:26, 8862.34it/s] 42%|████▏     | 167749/400000 [00:19<00:26, 8858.78it/s] 42%|████▏     | 168641/400000 [00:20<00:26, 8874.14it/s] 42%|████▏     | 169529/400000 [00:20<00:25, 8873.25it/s] 43%|████▎     | 170423/400000 [00:20<00:25, 8891.48it/s] 43%|████▎     | 171313/400000 [00:20<00:25, 8857.71it/s] 43%|████▎     | 172202/400000 [00:20<00:25, 8866.90it/s] 43%|████▎     | 173089/400000 [00:20<00:25, 8861.99it/s] 43%|████▎     | 173985/400000 [00:20<00:25, 8888.77it/s] 44%|████▎     | 174882/400000 [00:20<00:25, 8911.29it/s] 44%|████▍     | 175774/400000 [00:20<00:26, 8424.18it/s] 44%|████▍     | 176683/400000 [00:20<00:25, 8611.78it/s] 44%|████▍     | 177572/400000 [00:21<00:25, 8692.17it/s] 45%|████▍     | 178445/400000 [00:21<00:25, 8608.08it/s] 45%|████▍     | 179347/400000 [00:21<00:25, 8725.20it/s] 45%|████▌     | 180222/400000 [00:21<00:25, 8716.47it/s] 45%|████▌     | 181112/400000 [00:21<00:24, 8770.16it/s] 46%|████▌     | 182020/400000 [00:21<00:24, 8858.72it/s] 46%|████▌     | 182922/400000 [00:21<00:24, 8903.80it/s] 46%|████▌     | 183822/400000 [00:21<00:24, 8930.01it/s] 46%|████▌     | 184722/400000 [00:21<00:24, 8947.93it/s] 46%|████▋     | 185618/400000 [00:21<00:23, 8951.02it/s] 47%|████▋     | 186514/400000 [00:22<00:24, 8895.07it/s] 47%|████▋     | 187404/400000 [00:22<00:23, 8896.20it/s] 47%|████▋     | 188294/400000 [00:22<00:23, 8887.79it/s] 47%|████▋     | 189184/400000 [00:22<00:23, 8890.66it/s] 48%|████▊     | 190074/400000 [00:22<00:24, 8607.71it/s] 48%|████▊     | 190959/400000 [00:22<00:24, 8678.99it/s] 48%|████▊     | 191865/400000 [00:22<00:23, 8789.19it/s] 48%|████▊     | 192754/400000 [00:22<00:23, 8818.65it/s] 48%|████▊     | 193646/400000 [00:22<00:23, 8847.42it/s] 49%|████▊     | 194532/400000 [00:22<00:23, 8839.39it/s] 49%|████▉     | 195417/400000 [00:23<00:23, 8833.41it/s] 49%|████▉     | 196301/400000 [00:23<00:23, 8806.46it/s] 49%|████▉     | 197182/400000 [00:23<00:23, 8752.79it/s] 50%|████▉     | 198058/400000 [00:23<00:23, 8677.84it/s] 50%|████▉     | 198927/400000 [00:23<00:23, 8548.63it/s] 50%|████▉     | 199813/400000 [00:23<00:23, 8638.56it/s] 50%|█████     | 200709/400000 [00:23<00:22, 8732.01it/s] 50%|█████     | 201607/400000 [00:23<00:22, 8802.23it/s] 51%|█████     | 202500/400000 [00:23<00:22, 8837.57it/s] 51%|█████     | 203395/400000 [00:23<00:22, 8868.90it/s] 51%|█████     | 204287/400000 [00:24<00:22, 8881.36it/s] 51%|█████▏    | 205176/400000 [00:24<00:22, 8762.29it/s] 52%|█████▏    | 206087/400000 [00:24<00:21, 8861.08it/s] 52%|█████▏    | 206974/400000 [00:24<00:21, 8859.33it/s] 52%|█████▏    | 207861/400000 [00:24<00:22, 8423.87it/s] 52%|█████▏    | 208799/400000 [00:24<00:22, 8687.56it/s] 52%|█████▏    | 209699/400000 [00:24<00:21, 8778.26it/s] 53%|█████▎    | 210627/400000 [00:24<00:21, 8920.42it/s] 53%|█████▎    | 211542/400000 [00:24<00:20, 8986.76it/s] 53%|█████▎    | 212444/400000 [00:25<00:22, 8489.70it/s] 53%|█████▎    | 213354/400000 [00:25<00:21, 8662.65it/s] 54%|█████▎    | 214269/400000 [00:25<00:21, 8802.88it/s] 54%|█████▍    | 215186/400000 [00:25<00:20, 8907.49it/s] 54%|█████▍    | 216081/400000 [00:25<00:20, 8914.87it/s] 54%|█████▍    | 216976/400000 [00:25<00:20, 8883.63it/s] 54%|█████▍    | 217884/400000 [00:25<00:20, 8940.32it/s] 55%|█████▍    | 218787/400000 [00:25<00:20, 8965.31it/s] 55%|█████▍    | 219688/400000 [00:25<00:20, 8977.74it/s] 55%|█████▌    | 220597/400000 [00:25<00:19, 9005.30it/s] 55%|█████▌    | 221499/400000 [00:26<00:20, 8783.47it/s] 56%|█████▌    | 222379/400000 [00:26<00:20, 8776.33it/s] 56%|█████▌    | 223272/400000 [00:26<00:20, 8820.56it/s] 56%|█████▌    | 224155/400000 [00:26<00:20, 8741.13it/s] 56%|█████▋    | 225030/400000 [00:26<00:20, 8386.02it/s] 56%|█████▋    | 225911/400000 [00:26<00:20, 8506.60it/s] 57%|█████▋    | 226824/400000 [00:26<00:19, 8682.46it/s] 57%|█████▋    | 227739/400000 [00:26<00:19, 8817.04it/s] 57%|█████▋    | 228624/400000 [00:26<00:19, 8812.10it/s] 57%|█████▋    | 229508/400000 [00:26<00:19, 8810.28it/s] 58%|█████▊    | 230397/400000 [00:27<00:19, 8833.29it/s] 58%|█████▊    | 231291/400000 [00:27<00:19, 8864.84it/s] 58%|█████▊    | 232179/400000 [00:27<00:19, 8661.90it/s] 58%|█████▊    | 233047/400000 [00:27<00:19, 8574.96it/s] 58%|█████▊    | 233941/400000 [00:27<00:19, 8679.81it/s] 59%|█████▊    | 234842/400000 [00:27<00:18, 8773.77it/s] 59%|█████▉    | 235740/400000 [00:27<00:18, 8834.05it/s] 59%|█████▉    | 236629/400000 [00:27<00:18, 8848.34it/s] 59%|█████▉    | 237533/400000 [00:27<00:18, 8903.85it/s] 60%|█████▉    | 238424/400000 [00:27<00:18, 8889.98it/s] 60%|█████▉    | 239324/400000 [00:28<00:18, 8920.71it/s] 60%|██████    | 240217/400000 [00:28<00:18, 8848.98it/s] 60%|██████    | 241129/400000 [00:28<00:17, 8928.30it/s] 61%|██████    | 242024/400000 [00:28<00:17, 8933.27it/s] 61%|██████    | 242927/400000 [00:28<00:17, 8962.07it/s] 61%|██████    | 243829/400000 [00:28<00:17, 8978.94it/s] 61%|██████    | 244731/400000 [00:28<00:17, 8989.75it/s] 61%|██████▏   | 245631/400000 [00:28<00:17, 8920.57it/s] 62%|██████▏   | 246529/400000 [00:28<00:17, 8936.53it/s] 62%|██████▏   | 247423/400000 [00:28<00:17, 8911.13it/s] 62%|██████▏   | 248315/400000 [00:29<00:17, 8848.44it/s] 62%|██████▏   | 249201/400000 [00:29<00:17, 8744.92it/s] 63%|██████▎   | 250083/400000 [00:29<00:17, 8764.65it/s] 63%|██████▎   | 250977/400000 [00:29<00:16, 8815.41it/s] 63%|██████▎   | 251859/400000 [00:29<00:16, 8803.37it/s] 63%|██████▎   | 252740/400000 [00:29<00:17, 8601.02it/s] 63%|██████▎   | 253647/400000 [00:29<00:16, 8736.29it/s] 64%|██████▎   | 254548/400000 [00:29<00:16, 8816.03it/s] 64%|██████▍   | 255440/400000 [00:29<00:16, 8844.93it/s] 64%|██████▍   | 256326/400000 [00:29<00:16, 8725.27it/s] 64%|██████▍   | 257209/400000 [00:30<00:16, 8755.97it/s] 65%|██████▍   | 258086/400000 [00:30<00:16, 8756.46it/s] 65%|██████▍   | 258963/400000 [00:30<00:16, 8705.77it/s] 65%|██████▍   | 259853/400000 [00:30<00:15, 8761.94it/s] 65%|██████▌   | 260730/400000 [00:30<00:16, 8530.52it/s] 65%|██████▌   | 261585/400000 [00:30<00:16, 8514.27it/s] 66%|██████▌   | 262479/400000 [00:30<00:15, 8637.47it/s] 66%|██████▌   | 263372/400000 [00:30<00:15, 8720.42it/s] 66%|██████▌   | 264246/400000 [00:30<00:16, 8343.34it/s] 66%|██████▋   | 265116/400000 [00:31<00:15, 8445.71it/s] 67%|██████▋   | 266001/400000 [00:31<00:15, 8561.23it/s] 67%|██████▋   | 266877/400000 [00:31<00:15, 8619.48it/s] 67%|██████▋   | 267778/400000 [00:31<00:15, 8731.85it/s] 67%|██████▋   | 268668/400000 [00:31<00:14, 8780.34it/s] 67%|██████▋   | 269551/400000 [00:31<00:14, 8792.80it/s] 68%|██████▊   | 270447/400000 [00:31<00:14, 8840.00it/s] 68%|██████▊   | 271340/400000 [00:31<00:14, 8863.91it/s] 68%|██████▊   | 272227/400000 [00:31<00:14, 8836.83it/s] 68%|██████▊   | 273125/400000 [00:31<00:14, 8879.22it/s] 69%|██████▊   | 274014/400000 [00:32<00:14, 8820.37it/s] 69%|██████▊   | 274897/400000 [00:32<00:14, 8809.30it/s] 69%|██████▉   | 275779/400000 [00:32<00:14, 8750.84it/s] 69%|██████▉   | 276667/400000 [00:32<00:14, 8788.34it/s] 69%|██████▉   | 277547/400000 [00:32<00:14, 8687.16it/s] 70%|██████▉   | 278417/400000 [00:32<00:14, 8578.67it/s] 70%|██████▉   | 279276/400000 [00:32<00:14, 8541.93it/s] 70%|███████   | 280175/400000 [00:32<00:13, 8671.30it/s] 70%|███████   | 281083/400000 [00:32<00:13, 8788.89it/s] 70%|███████   | 281963/400000 [00:32<00:13, 8775.88it/s] 71%|███████   | 282842/400000 [00:33<00:13, 8770.03it/s] 71%|███████   | 283735/400000 [00:33<00:13, 8817.28it/s] 71%|███████   | 284618/400000 [00:33<00:13, 8787.65it/s] 71%|███████▏  | 285498/400000 [00:33<00:13, 8781.77it/s] 72%|███████▏  | 286377/400000 [00:33<00:13, 8727.33it/s] 72%|███████▏  | 287265/400000 [00:33<00:12, 8771.91it/s] 72%|███████▏  | 288146/400000 [00:33<00:12, 8781.40it/s] 72%|███████▏  | 289025/400000 [00:33<00:12, 8562.38it/s] 72%|███████▏  | 289883/400000 [00:33<00:12, 8481.37it/s] 73%|███████▎  | 290782/400000 [00:33<00:12, 8625.15it/s] 73%|███████▎  | 291646/400000 [00:34<00:12, 8519.76it/s] 73%|███████▎  | 292500/400000 [00:34<00:12, 8495.08it/s] 73%|███████▎  | 293408/400000 [00:34<00:12, 8662.48it/s] 74%|███████▎  | 294319/400000 [00:34<00:12, 8790.54it/s] 74%|███████▍  | 295218/400000 [00:34<00:11, 8847.39it/s] 74%|███████▍  | 296126/400000 [00:34<00:11, 8915.42it/s] 74%|███████▍  | 297019/400000 [00:34<00:11, 8803.64it/s] 74%|███████▍  | 297901/400000 [00:34<00:11, 8708.53it/s] 75%|███████▍  | 298776/400000 [00:34<00:11, 8718.62it/s] 75%|███████▍  | 299677/400000 [00:34<00:11, 8803.42it/s] 75%|███████▌  | 300558/400000 [00:35<00:11, 8789.75it/s] 75%|███████▌  | 301451/400000 [00:35<00:11, 8830.29it/s] 76%|███████▌  | 302363/400000 [00:35<00:10, 8914.98it/s] 76%|███████▌  | 303255/400000 [00:35<00:10, 8878.89it/s] 76%|███████▌  | 304144/400000 [00:35<00:10, 8861.63it/s] 76%|███████▋  | 305035/400000 [00:35<00:10, 8873.90it/s] 76%|███████▋  | 305930/400000 [00:35<00:10, 8894.37it/s] 77%|███████▋  | 306820/400000 [00:35<00:10, 8834.17it/s] 77%|███████▋  | 307704/400000 [00:35<00:10, 8810.24it/s] 77%|███████▋  | 308586/400000 [00:35<00:10, 8805.94it/s] 77%|███████▋  | 309467/400000 [00:36<00:10, 8764.84it/s] 78%|███████▊  | 310344/400000 [00:36<00:10, 8667.87it/s] 78%|███████▊  | 311251/400000 [00:36<00:10, 8784.24it/s] 78%|███████▊  | 312142/400000 [00:36<00:09, 8820.39it/s] 78%|███████▊  | 313031/400000 [00:36<00:09, 8838.80it/s] 78%|███████▊  | 313916/400000 [00:36<00:09, 8790.78it/s] 79%|███████▊  | 314796/400000 [00:36<00:09, 8764.71it/s] 79%|███████▉  | 315712/400000 [00:36<00:09, 8877.11it/s] 79%|███████▉  | 316601/400000 [00:36<00:09, 8687.91it/s] 79%|███████▉  | 317494/400000 [00:36<00:09, 8758.50it/s] 80%|███████▉  | 318371/400000 [00:37<00:09, 8742.26it/s] 80%|███████▉  | 319246/400000 [00:37<00:09, 8679.47it/s] 80%|████████  | 320145/400000 [00:37<00:09, 8768.81it/s] 80%|████████  | 321023/400000 [00:37<00:09, 8764.72it/s] 80%|████████  | 321923/400000 [00:37<00:08, 8831.69it/s] 81%|████████  | 322807/400000 [00:37<00:08, 8684.45it/s] 81%|████████  | 323692/400000 [00:37<00:08, 8733.19it/s] 81%|████████  | 324572/400000 [00:37<00:08, 8752.82it/s] 81%|████████▏ | 325448/400000 [00:37<00:08, 8748.55it/s] 82%|████████▏ | 326328/400000 [00:37<00:08, 8761.67it/s] 82%|████████▏ | 327239/400000 [00:38<00:08, 8862.63it/s] 82%|████████▏ | 328131/400000 [00:38<00:08, 8878.79it/s] 82%|████████▏ | 329043/400000 [00:38<00:07, 8949.26it/s] 82%|████████▏ | 329954/400000 [00:38<00:07, 8994.41it/s] 83%|████████▎ | 330861/400000 [00:38<00:07, 9015.75it/s] 83%|████████▎ | 331781/400000 [00:38<00:07, 9069.88it/s] 83%|████████▎ | 332689/400000 [00:38<00:07, 9024.17it/s] 83%|████████▎ | 333592/400000 [00:38<00:07, 8819.98it/s] 84%|████████▎ | 334476/400000 [00:38<00:07, 8547.08it/s] 84%|████████▍ | 335360/400000 [00:39<00:07, 8631.39it/s] 84%|████████▍ | 336247/400000 [00:39<00:07, 8699.60it/s] 84%|████████▍ | 337119/400000 [00:39<00:07, 8583.10it/s] 85%|████████▍ | 338014/400000 [00:39<00:07, 8687.26it/s] 85%|████████▍ | 338927/400000 [00:39<00:06, 8814.73it/s] 85%|████████▍ | 339844/400000 [00:39<00:06, 8916.39it/s] 85%|████████▌ | 340755/400000 [00:39<00:06, 8971.90it/s] 85%|████████▌ | 341654/400000 [00:39<00:06, 8961.84it/s] 86%|████████▌ | 342555/400000 [00:39<00:06, 8973.91it/s] 86%|████████▌ | 343477/400000 [00:39<00:06, 9044.73it/s] 86%|████████▌ | 344382/400000 [00:40<00:06, 9033.93it/s] 86%|████████▋ | 345286/400000 [00:40<00:06, 8944.31it/s] 87%|████████▋ | 346181/400000 [00:40<00:06, 8774.15it/s] 87%|████████▋ | 347092/400000 [00:40<00:05, 8870.11it/s] 87%|████████▋ | 348004/400000 [00:40<00:05, 8941.73it/s] 87%|████████▋ | 348925/400000 [00:40<00:05, 9018.53it/s] 87%|████████▋ | 349828/400000 [00:40<00:05, 9012.45it/s] 88%|████████▊ | 350730/400000 [00:40<00:05, 8904.41it/s] 88%|████████▊ | 351635/400000 [00:40<00:05, 8944.93it/s] 88%|████████▊ | 352530/400000 [00:40<00:05, 8946.23it/s] 88%|████████▊ | 353429/400000 [00:41<00:05, 8958.76it/s] 89%|████████▊ | 354326/400000 [00:41<00:05, 8866.80it/s] 89%|████████▉ | 355214/400000 [00:41<00:05, 8824.31it/s] 89%|████████▉ | 356097/400000 [00:41<00:04, 8810.39it/s] 89%|████████▉ | 356979/400000 [00:41<00:04, 8662.17it/s] 89%|████████▉ | 357857/400000 [00:41<00:04, 8695.77it/s] 90%|████████▉ | 358751/400000 [00:41<00:04, 8765.21it/s] 90%|████████▉ | 359629/400000 [00:41<00:04, 8727.51it/s] 90%|█████████ | 360503/400000 [00:41<00:04, 8683.28it/s] 90%|█████████ | 361403/400000 [00:41<00:04, 8773.11it/s] 91%|█████████ | 362293/400000 [00:42<00:04, 8810.72it/s] 91%|█████████ | 363175/400000 [00:42<00:04, 8754.29it/s] 91%|█████████ | 364051/400000 [00:42<00:04, 8643.85it/s] 91%|█████████ | 364930/400000 [00:42<00:04, 8685.53it/s] 91%|█████████▏| 365820/400000 [00:42<00:03, 8748.59it/s] 92%|█████████▏| 366696/400000 [00:42<00:03, 8705.98it/s] 92%|█████████▏| 367567/400000 [00:42<00:03, 8602.54it/s] 92%|█████████▏| 368435/400000 [00:42<00:03, 8623.42it/s] 92%|█████████▏| 369298/400000 [00:42<00:03, 8606.77it/s] 93%|█████████▎| 370159/400000 [00:42<00:03, 8461.43it/s] 93%|█████████▎| 371060/400000 [00:43<00:03, 8617.84it/s] 93%|█████████▎| 371929/400000 [00:43<00:03, 8637.33it/s] 93%|█████████▎| 372794/400000 [00:43<00:03, 8568.18it/s] 93%|█████████▎| 373652/400000 [00:43<00:03, 8467.10it/s] 94%|█████████▎| 374500/400000 [00:43<00:03, 7994.66it/s] 94%|█████████▍| 375306/400000 [00:43<00:03, 7883.58it/s] 94%|█████████▍| 376181/400000 [00:43<00:02, 8122.65it/s] 94%|█████████▍| 377061/400000 [00:43<00:02, 8312.74it/s] 94%|█████████▍| 377959/400000 [00:43<00:02, 8500.02it/s] 95%|█████████▍| 378840/400000 [00:44<00:02, 8588.01it/s] 95%|█████████▍| 379731/400000 [00:44<00:02, 8680.83it/s] 95%|█████████▌| 380602/400000 [00:44<00:02, 8457.93it/s] 95%|█████████▌| 381451/400000 [00:44<00:02, 8433.62it/s] 96%|█████████▌| 382328/400000 [00:44<00:02, 8529.42it/s] 96%|█████████▌| 383208/400000 [00:44<00:01, 8607.56it/s] 96%|█████████▌| 384093/400000 [00:44<00:01, 8678.52it/s] 96%|█████████▌| 384991/400000 [00:44<00:01, 8765.44it/s] 96%|█████████▋| 385885/400000 [00:44<00:01, 8814.70it/s] 97%|█████████▋| 386793/400000 [00:44<00:01, 8892.59it/s] 97%|█████████▋| 387683/400000 [00:45<00:01, 8893.86it/s] 97%|█████████▋| 388573/400000 [00:45<00:01, 8772.15it/s] 97%|█████████▋| 389461/400000 [00:45<00:01, 8802.18it/s] 98%|█████████▊| 390342/400000 [00:45<00:01, 8726.83it/s] 98%|█████████▊| 391216/400000 [00:45<00:01, 8689.66it/s] 98%|█████████▊| 392093/400000 [00:45<00:00, 8711.56it/s] 98%|█████████▊| 392983/400000 [00:45<00:00, 8765.97it/s] 98%|█████████▊| 393881/400000 [00:45<00:00, 8827.98it/s] 99%|█████████▊| 394765/400000 [00:45<00:00, 8744.36it/s] 99%|█████████▉| 395640/400000 [00:45<00:00, 8700.78it/s] 99%|█████████▉| 396526/400000 [00:46<00:00, 8747.71it/s] 99%|█████████▉| 397402/400000 [00:46<00:00, 8638.89it/s]100%|█████████▉| 398283/400000 [00:46<00:00, 8688.49it/s]100%|█████████▉| 399153/400000 [00:46<00:00, 8665.16it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8615.30it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fb75818a128>, <torchtext.data.dataset.TabularDataset object at 0x7fb75818a278>, <torchtext.vocab.Vocab object at 0x7fb75818a198>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 14.11 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 17.05 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 17.05 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 13.30 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 13.30 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.64 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.64 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.18 file/s]2020-06-17 18:24:41.578856: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-17 18:24:41.583240: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-06-17 18:24:41.583403: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56426c6c9ca0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-17 18:24:41.583417: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:01, 161108.73it/s] 76%|███████▌  | 7544832/9912422 [00:00<00:10, 229943.23it/s]9920512it [00:00, 44584403.75it/s]                           
0it [00:00, ?it/s]32768it [00:00, 691662.91it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1022833.72it/s]1654784it [00:00, 11064086.07it/s]                           
0it [00:00, ?it/s]8192it [00:00, 226071.73it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
