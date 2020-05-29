
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/28498baaa949d998f88ab5972ee2a9e4cca1c784', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': '28498baaa949d998f88ab5972ee2a9e4cca1c784', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/28498baaa949d998f88ab5972ee2a9e4cca1c784

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/28498baaa949d998f88ab5972ee2a9e4cca1c784

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/28498baaa949d998f88ab5972ee2a9e4cca1c784

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f9559ee8598> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f9559ee8598> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f95c54af0d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f95c54af0d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f95df201ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f95df201ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f9572d288c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f9572d288c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f9572d288c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 445, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 257, in compute
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 445, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 257, in compute
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 445, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 257, in compute
    obj_preprocessor = preprocessor_func(**args, data_info=self.data_info)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/preprocess/generic.py", line 607, in __init__
    data            = np.load( file_path,**args.get("numpy_loader_args", {}))
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 428, in load
    fid = open(os_fspath(file), "rb")
FileNotFoundError: [Errno 2] No such file or directory: 'train/mlmodels/dataset/text/imdb.npz'
Using TensorFlow backend.
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 445, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 297, in compute
    out_tmp = preprocessor_func(input_tmp, **args)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 92, in pickle_dump
    with open(kwargs["path"], "wb") as fi:
FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:15, 130766.46it/s] 44%|████▍     | 4382720/9912422 [00:00<00:29, 186569.76it/s]9920512it [00:00, 30812120.84it/s]                           
0it [00:00, ?it/s]32768it [00:00, 333344.54it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s] 70%|███████   | 1155072/1648877 [00:00<00:00, 11547289.65it/s]1654784it [00:00, 5081629.30it/s]                              
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 73223.14it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9559dc2c88>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9559dbaf28>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f9572d28510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f9572d28510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f9572d28510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:26,  1.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:26,  1.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:25,  1.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:25,  1.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:24,  1.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:24,  1.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<00:59,  2.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:59,  2.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:59,  2.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:58,  2.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:58,  2.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:58,  2.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:57,  2.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   7%|▋         | 12/162 [00:00<00:40,  3.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:40,  3.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:40,  3.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:40,  3.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:40,  3.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:39,  3.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:39,  3.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:28,  5.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:28,  5.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:28,  5.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:27,  5.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:27,  5.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:27,  5.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:27,  5.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:00<00:19,  6.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:19,  6.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:19,  6.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:19,  6.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:19,  6.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:19,  6.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:19,  6.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:01<00:13,  9.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:13,  9.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:13,  9.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:13,  9.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:13,  9.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:13,  9.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:13,  9.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 36/162 [00:01<00:10, 12.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:10, 12.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:09, 12.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:09, 12.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:09, 12.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:09, 12.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:09, 12.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:01<00:07, 16.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:07, 16.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:07, 16.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:07, 16.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:07, 16.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:07, 16.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:07, 16.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:05, 20.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:05, 20.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 20.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 20.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 20.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 20.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 20.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 20.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 26.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 26.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 26.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 26.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 26.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 26.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 26.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 26.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 26.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 32.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 32.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:02, 32.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:02, 32.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 32.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 32.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 32.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 32.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 32.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 39.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 39.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 39.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 39.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 39.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 39.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 39.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 39.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 39.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 46.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 46.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 46.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 46.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 46.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 46.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 46.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 46.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 46.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 52.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 52.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 52.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 52.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 52.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 52.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 52.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 52.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 52.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 58.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 58.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 58.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 58.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 58.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 58.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 58.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 58.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 58.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:00, 63.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:00, 63.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:00, 63.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:00, 63.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:00, 63.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:00, 63.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 63.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 63.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 63.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 66.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 66.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 66.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 66.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 66.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 66.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 66.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 66.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 66.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 69.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 69.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 69.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 69.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 69.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 69.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 69.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 69.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 69.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 71.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 71.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 71.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 71.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 71.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 71.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 71.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 71.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 71.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 71.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 71.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 71.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 71.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 71.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 71.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 71.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 71.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 71.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 69.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 69.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 69.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 69.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 69.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 69.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 69.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 69.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 69.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 69.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 69.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 69.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 69.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 69.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 69.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 69.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 69.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 69.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 70.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 70.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 70.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 70.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.95s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.95s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.95s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.99 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.22s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.95s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 70.99 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.22s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.22s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 31.03 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.22s/ url]
0 examples [00:00, ? examples/s]2020-05-29 21:14:36.077463: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-29 21:14:36.082765: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-29 21:14:36.082896: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5631b691c750 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-29 21:14:36.082911: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
67 examples [00:00, 665.19 examples/s]164 examples [00:00, 732.79 examples/s]259 examples [00:00, 786.75 examples/s]354 examples [00:00, 829.19 examples/s]448 examples [00:00, 857.65 examples/s]543 examples [00:00, 882.93 examples/s]641 examples [00:00, 907.63 examples/s]734 examples [00:00, 912.95 examples/s]830 examples [00:00, 925.80 examples/s]921 examples [00:01, 920.73 examples/s]1012 examples [00:01, 904.57 examples/s]1105 examples [00:01, 909.13 examples/s]1196 examples [00:01, 908.95 examples/s]1290 examples [00:01, 916.96 examples/s]1382 examples [00:01, 911.08 examples/s]1475 examples [00:01, 915.82 examples/s]1570 examples [00:01, 925.71 examples/s]1665 examples [00:01, 930.67 examples/s]1761 examples [00:01, 937.85 examples/s]1855 examples [00:02, 934.31 examples/s]1949 examples [00:02, 925.81 examples/s]2045 examples [00:02, 935.65 examples/s]2142 examples [00:02, 943.57 examples/s]2237 examples [00:02, 942.35 examples/s]2332 examples [00:02, 933.72 examples/s]2427 examples [00:02, 937.15 examples/s]2522 examples [00:02, 938.59 examples/s]2619 examples [00:02, 946.58 examples/s]2714 examples [00:02, 944.30 examples/s]2809 examples [00:03, 937.92 examples/s]2903 examples [00:03, 898.06 examples/s]2994 examples [00:03, 881.62 examples/s]3083 examples [00:03, 866.35 examples/s]3170 examples [00:03, 867.06 examples/s]3261 examples [00:03, 875.97 examples/s]3351 examples [00:03, 882.71 examples/s]3443 examples [00:03, 891.20 examples/s]3535 examples [00:03, 897.95 examples/s]3625 examples [00:03, 873.97 examples/s]3716 examples [00:04, 883.12 examples/s]3807 examples [00:04, 890.36 examples/s]3899 examples [00:04, 897.70 examples/s]3989 examples [00:04, 887.91 examples/s]4083 examples [00:04, 901.17 examples/s]4179 examples [00:04, 916.15 examples/s]4272 examples [00:04, 917.66 examples/s]4364 examples [00:04, 917.31 examples/s]4456 examples [00:04, 914.64 examples/s]4548 examples [00:04, 906.12 examples/s]4639 examples [00:05, 906.91 examples/s]4734 examples [00:05, 917.97 examples/s]4829 examples [00:05, 925.96 examples/s]4922 examples [00:05, 919.46 examples/s]5017 examples [00:05, 926.84 examples/s]5113 examples [00:05, 934.32 examples/s]5209 examples [00:05, 941.02 examples/s]5304 examples [00:05, 942.82 examples/s]5399 examples [00:05, 929.42 examples/s]5493 examples [00:06, 922.72 examples/s]5591 examples [00:06, 936.45 examples/s]5685 examples [00:06, 936.61 examples/s]5779 examples [00:06, 931.66 examples/s]5873 examples [00:06, 921.90 examples/s]5966 examples [00:06, 921.02 examples/s]6060 examples [00:06, 925.75 examples/s]6156 examples [00:06, 935.37 examples/s]6251 examples [00:06, 938.69 examples/s]6347 examples [00:06, 944.47 examples/s]6442 examples [00:07, 921.51 examples/s]6535 examples [00:07, 922.63 examples/s]6628 examples [00:07, 891.93 examples/s]6720 examples [00:07, 900.13 examples/s]6813 examples [00:07, 906.13 examples/s]6905 examples [00:07, 908.78 examples/s]6997 examples [00:07, 910.78 examples/s]7089 examples [00:07, 912.35 examples/s]7181 examples [00:07, 912.81 examples/s]7273 examples [00:07, 899.90 examples/s]7368 examples [00:08, 911.18 examples/s]7463 examples [00:08, 920.22 examples/s]7556 examples [00:08, 898.81 examples/s]7647 examples [00:08, 890.56 examples/s]7739 examples [00:08, 897.94 examples/s]7834 examples [00:08, 912.90 examples/s]7926 examples [00:08, 909.71 examples/s]8023 examples [00:08, 924.44 examples/s]8116 examples [00:08, 917.14 examples/s]8208 examples [00:08, 911.85 examples/s]8300 examples [00:09, 885.16 examples/s]8394 examples [00:09, 899.78 examples/s]8487 examples [00:09, 908.61 examples/s]8579 examples [00:09, 909.09 examples/s]8671 examples [00:09, 909.47 examples/s]8766 examples [00:09, 919.24 examples/s]8861 examples [00:09, 927.53 examples/s]8954 examples [00:09, 925.80 examples/s]9051 examples [00:09, 937.18 examples/s]9145 examples [00:09, 933.52 examples/s]9240 examples [00:10, 937.19 examples/s]9334 examples [00:10, 936.14 examples/s]9430 examples [00:10, 940.65 examples/s]9525 examples [00:10, 924.54 examples/s]9618 examples [00:10, 921.20 examples/s]9713 examples [00:10, 928.03 examples/s]9806 examples [00:10, 927.88 examples/s]9903 examples [00:10, 937.30 examples/s]9997 examples [00:10, 935.08 examples/s]10091 examples [00:11, 865.30 examples/s]10184 examples [00:11, 883.00 examples/s]10277 examples [00:11, 896.07 examples/s]10369 examples [00:11, 900.72 examples/s]10463 examples [00:11, 910.54 examples/s]10555 examples [00:11, 879.72 examples/s]10645 examples [00:11, 884.08 examples/s]10740 examples [00:11, 901.98 examples/s]10836 examples [00:11, 917.06 examples/s]10928 examples [00:11, 915.46 examples/s]11020 examples [00:12, 893.24 examples/s]11119 examples [00:12, 919.48 examples/s]11212 examples [00:12, 914.56 examples/s]11306 examples [00:12, 909.83 examples/s]11398 examples [00:12, 910.00 examples/s]11492 examples [00:12, 918.06 examples/s]11585 examples [00:12, 919.85 examples/s]11680 examples [00:12, 927.83 examples/s]11773 examples [00:12, 927.31 examples/s]11867 examples [00:12, 928.84 examples/s]11960 examples [00:13, 923.56 examples/s]12053 examples [00:13, 909.91 examples/s]12145 examples [00:13, 887.33 examples/s]12234 examples [00:13, 835.86 examples/s]12319 examples [00:13, 820.79 examples/s]12402 examples [00:13, 819.54 examples/s]12490 examples [00:13, 835.54 examples/s]12577 examples [00:13, 844.76 examples/s]12665 examples [00:13, 852.79 examples/s]12753 examples [00:14, 858.32 examples/s]12846 examples [00:14, 877.79 examples/s]12939 examples [00:14, 892.76 examples/s]13029 examples [00:14, 849.30 examples/s]13116 examples [00:14, 854.26 examples/s]13205 examples [00:14, 863.42 examples/s]13299 examples [00:14, 882.93 examples/s]13393 examples [00:14, 898.11 examples/s]13487 examples [00:14, 909.25 examples/s]13580 examples [00:14, 914.76 examples/s]13672 examples [00:15, 914.89 examples/s]13764 examples [00:15, 914.34 examples/s]13858 examples [00:15, 920.92 examples/s]13955 examples [00:15, 934.80 examples/s]14049 examples [00:15, 915.73 examples/s]14141 examples [00:15, 911.77 examples/s]14235 examples [00:15, 918.74 examples/s]14328 examples [00:15, 920.57 examples/s]14424 examples [00:15, 929.18 examples/s]14517 examples [00:15, 927.29 examples/s]14611 examples [00:16, 930.85 examples/s]14705 examples [00:16, 917.18 examples/s]14801 examples [00:16, 928.00 examples/s]14896 examples [00:16, 934.09 examples/s]14994 examples [00:16, 945.51 examples/s]15089 examples [00:16, 938.59 examples/s]15183 examples [00:16, 905.86 examples/s]15276 examples [00:16, 911.26 examples/s]15368 examples [00:16, 906.05 examples/s]15459 examples [00:16, 897.73 examples/s]15551 examples [00:17, 902.05 examples/s]15645 examples [00:17, 911.10 examples/s]15742 examples [00:17, 927.37 examples/s]15837 examples [00:17, 932.59 examples/s]15931 examples [00:17, 932.61 examples/s]16025 examples [00:17, 926.54 examples/s]16118 examples [00:17, 903.76 examples/s]16213 examples [00:17, 916.22 examples/s]16305 examples [00:17, 913.90 examples/s]16400 examples [00:17, 923.68 examples/s]16493 examples [00:18, 916.96 examples/s]16586 examples [00:18, 918.89 examples/s]16680 examples [00:18, 923.87 examples/s]16773 examples [00:18, 892.83 examples/s]16864 examples [00:18, 896.81 examples/s]16954 examples [00:18, 882.70 examples/s]17043 examples [00:18, 879.27 examples/s]17134 examples [00:18, 888.17 examples/s]17228 examples [00:18, 901.67 examples/s]17322 examples [00:19, 911.05 examples/s]17414 examples [00:19, 912.44 examples/s]17506 examples [00:19, 904.82 examples/s]17599 examples [00:19, 910.27 examples/s]17691 examples [00:19, 909.34 examples/s]17785 examples [00:19, 915.49 examples/s]17877 examples [00:19, 906.81 examples/s]17970 examples [00:19, 911.58 examples/s]18062 examples [00:19, 913.93 examples/s]18154 examples [00:19, 911.92 examples/s]18246 examples [00:20, 912.16 examples/s]18338 examples [00:20, 896.36 examples/s]18432 examples [00:20, 908.99 examples/s]18528 examples [00:20, 921.75 examples/s]18623 examples [00:20, 929.70 examples/s]18718 examples [00:20, 935.64 examples/s]18812 examples [00:20, 930.85 examples/s]18906 examples [00:20, 913.96 examples/s]19000 examples [00:20, 920.64 examples/s]19093 examples [00:20, 910.25 examples/s]19185 examples [00:21, 910.68 examples/s]19277 examples [00:21, 900.82 examples/s]19369 examples [00:21, 902.72 examples/s]19462 examples [00:21, 909.31 examples/s]19553 examples [00:21, 901.92 examples/s]19644 examples [00:21, 904.09 examples/s]19735 examples [00:21, 897.46 examples/s]19828 examples [00:21, 903.76 examples/s]19923 examples [00:21, 917.12 examples/s]20015 examples [00:21, 868.44 examples/s]20107 examples [00:22, 878.17 examples/s]20201 examples [00:22, 893.28 examples/s]20293 examples [00:22, 900.20 examples/s]20390 examples [00:22, 918.81 examples/s]20487 examples [00:22, 931.44 examples/s]20581 examples [00:22, 929.54 examples/s]20675 examples [00:22, 928.37 examples/s]20770 examples [00:22, 933.45 examples/s]20864 examples [00:22, 913.52 examples/s]20958 examples [00:23, 920.68 examples/s]21051 examples [00:23, 922.57 examples/s]21149 examples [00:23, 936.87 examples/s]21244 examples [00:23, 937.76 examples/s]21338 examples [00:23, 919.42 examples/s]21432 examples [00:23, 923.27 examples/s]21526 examples [00:23, 927.32 examples/s]21619 examples [00:23, 918.54 examples/s]21713 examples [00:23, 924.70 examples/s]21806 examples [00:23, 919.92 examples/s]21899 examples [00:24, 919.48 examples/s]21994 examples [00:24, 925.74 examples/s]22087 examples [00:24, 926.33 examples/s]22180 examples [00:24, 926.80 examples/s]22273 examples [00:24, 919.45 examples/s]22367 examples [00:24, 924.07 examples/s]22462 examples [00:24, 929.53 examples/s]22555 examples [00:24, 919.83 examples/s]22648 examples [00:24, 920.47 examples/s]22741 examples [00:24, 917.32 examples/s]22833 examples [00:25, 908.06 examples/s]22924 examples [00:25, 887.75 examples/s]23013 examples [00:25, 868.47 examples/s]23109 examples [00:25, 891.86 examples/s]23206 examples [00:25, 913.53 examples/s]23298 examples [00:25, 912.65 examples/s]23390 examples [00:25, 907.24 examples/s]23485 examples [00:25, 918.41 examples/s]23577 examples [00:25, 903.35 examples/s]23668 examples [00:25, 883.87 examples/s]23757 examples [00:26, 860.79 examples/s]23844 examples [00:26, 862.53 examples/s]23935 examples [00:26, 873.62 examples/s]24023 examples [00:26, 862.49 examples/s]24115 examples [00:26, 876.66 examples/s]24203 examples [00:26, 874.25 examples/s]24291 examples [00:26, 860.30 examples/s]24378 examples [00:26, 862.09 examples/s]24473 examples [00:26, 885.37 examples/s]24568 examples [00:27, 902.33 examples/s]24659 examples [00:27, 903.45 examples/s]24752 examples [00:27, 908.81 examples/s]24844 examples [00:27, 905.68 examples/s]24935 examples [00:27, 883.35 examples/s]25024 examples [00:27, 880.88 examples/s]25113 examples [00:27, 880.32 examples/s]25203 examples [00:27, 884.90 examples/s]25294 examples [00:27, 890.78 examples/s]25384 examples [00:27, 884.62 examples/s]25473 examples [00:28, 864.04 examples/s]25564 examples [00:28, 875.28 examples/s]25653 examples [00:28, 879.55 examples/s]25746 examples [00:28, 891.00 examples/s]25836 examples [00:28, 880.35 examples/s]25925 examples [00:28, 880.09 examples/s]26015 examples [00:28, 885.62 examples/s]26105 examples [00:28, 888.24 examples/s]26194 examples [00:28, 875.81 examples/s]26288 examples [00:28, 892.11 examples/s]26380 examples [00:29, 897.48 examples/s]26470 examples [00:29, 868.98 examples/s]26558 examples [00:29, 869.54 examples/s]26649 examples [00:29, 878.90 examples/s]26742 examples [00:29, 891.64 examples/s]26836 examples [00:29, 903.53 examples/s]26927 examples [00:29, 896.03 examples/s]27020 examples [00:29, 903.46 examples/s]27115 examples [00:29, 916.76 examples/s]27207 examples [00:29, 917.69 examples/s]27299 examples [00:30, 909.44 examples/s]27391 examples [00:30, 905.81 examples/s]27483 examples [00:30, 908.29 examples/s]27575 examples [00:30, 909.14 examples/s]27666 examples [00:30, 893.09 examples/s]27760 examples [00:30, 903.86 examples/s]27851 examples [00:30, 904.00 examples/s]27942 examples [00:30, 902.64 examples/s]28034 examples [00:30, 905.85 examples/s]28133 examples [00:30, 927.33 examples/s]28229 examples [00:31, 935.98 examples/s]28325 examples [00:31, 940.26 examples/s]28420 examples [00:31, 933.64 examples/s]28514 examples [00:31, 932.59 examples/s]28611 examples [00:31, 942.46 examples/s]28706 examples [00:31, 941.50 examples/s]28801 examples [00:31, 939.60 examples/s]28895 examples [00:31, 906.18 examples/s]28988 examples [00:31, 910.70 examples/s]29081 examples [00:32, 915.11 examples/s]29173 examples [00:32, 896.45 examples/s]29263 examples [00:32, 896.13 examples/s]29353 examples [00:32, 895.77 examples/s]29448 examples [00:32, 910.10 examples/s]29544 examples [00:32, 923.24 examples/s]29640 examples [00:32, 933.32 examples/s]29734 examples [00:32, 931.23 examples/s]29828 examples [00:32, 927.17 examples/s]29922 examples [00:32, 928.45 examples/s]30015 examples [00:33, 869.39 examples/s]30107 examples [00:33, 883.29 examples/s]30199 examples [00:33, 893.00 examples/s]30289 examples [00:33, 886.95 examples/s]30379 examples [00:33, 885.06 examples/s]30471 examples [00:33, 893.54 examples/s]30566 examples [00:33, 907.19 examples/s]30657 examples [00:33, 905.82 examples/s]30748 examples [00:33, 896.94 examples/s]30838 examples [00:33, 878.62 examples/s]30927 examples [00:34, 870.47 examples/s]31018 examples [00:34, 880.32 examples/s]31107 examples [00:34, 856.26 examples/s]31193 examples [00:34, 849.08 examples/s]31281 examples [00:34, 855.69 examples/s]31376 examples [00:34, 879.81 examples/s]31468 examples [00:34, 890.75 examples/s]31558 examples [00:34, 891.17 examples/s]31648 examples [00:34, 875.11 examples/s]31741 examples [00:34, 890.57 examples/s]31831 examples [00:35, 891.86 examples/s]31921 examples [00:35, 870.08 examples/s]32009 examples [00:35, 871.32 examples/s]32097 examples [00:35, 843.63 examples/s]32191 examples [00:35, 868.57 examples/s]32283 examples [00:35, 881.90 examples/s]32376 examples [00:35, 895.06 examples/s]32468 examples [00:35, 899.76 examples/s]32559 examples [00:35, 892.69 examples/s]32649 examples [00:36, 886.29 examples/s]32742 examples [00:36, 896.86 examples/s]32834 examples [00:36, 902.66 examples/s]32926 examples [00:36, 907.11 examples/s]33017 examples [00:36, 907.18 examples/s]33109 examples [00:36, 910.51 examples/s]33205 examples [00:36, 921.89 examples/s]33300 examples [00:36, 928.99 examples/s]33393 examples [00:36, 925.07 examples/s]33486 examples [00:36, 920.36 examples/s]33582 examples [00:37, 931.77 examples/s]33676 examples [00:37, 923.78 examples/s]33771 examples [00:37, 927.84 examples/s]33864 examples [00:37, 925.80 examples/s]33957 examples [00:37, 920.48 examples/s]34052 examples [00:37, 928.71 examples/s]34150 examples [00:37, 942.71 examples/s]34245 examples [00:37, 941.63 examples/s]34340 examples [00:37, 942.95 examples/s]34435 examples [00:37, 926.84 examples/s]34528 examples [00:38, 926.99 examples/s]34625 examples [00:38, 937.40 examples/s]34720 examples [00:38, 939.24 examples/s]34815 examples [00:38, 941.95 examples/s]34910 examples [00:38, 900.51 examples/s]35003 examples [00:38, 907.07 examples/s]35096 examples [00:38, 913.74 examples/s]35188 examples [00:38, 914.01 examples/s]35281 examples [00:38, 917.28 examples/s]35373 examples [00:38, 908.30 examples/s]35464 examples [00:39, 902.44 examples/s]35560 examples [00:39, 917.76 examples/s]35654 examples [00:39, 923.36 examples/s]35747 examples [00:39, 907.16 examples/s]35838 examples [00:39, 891.08 examples/s]35928 examples [00:39, 890.79 examples/s]36018 examples [00:39, 872.60 examples/s]36111 examples [00:39, 887.10 examples/s]36205 examples [00:39, 901.11 examples/s]36297 examples [00:40, 905.49 examples/s]36391 examples [00:40, 914.99 examples/s]36485 examples [00:40, 921.18 examples/s]36578 examples [00:40, 921.37 examples/s]36674 examples [00:40, 929.13 examples/s]36767 examples [00:40, 927.02 examples/s]36863 examples [00:40, 934.85 examples/s]36959 examples [00:40, 940.81 examples/s]37055 examples [00:40, 944.71 examples/s]37150 examples [00:40, 932.49 examples/s]37244 examples [00:41, 929.61 examples/s]37338 examples [00:41, 927.45 examples/s]37432 examples [00:41, 930.11 examples/s]37526 examples [00:41, 929.11 examples/s]37621 examples [00:41, 935.12 examples/s]37715 examples [00:41, 891.66 examples/s]37811 examples [00:41, 910.19 examples/s]37903 examples [00:41, 912.41 examples/s]37996 examples [00:41, 916.86 examples/s]38088 examples [00:41, 912.15 examples/s]38180 examples [00:42, 912.62 examples/s]38277 examples [00:42, 926.63 examples/s]38373 examples [00:42, 935.56 examples/s]38467 examples [00:42, 931.91 examples/s]38561 examples [00:42, 927.10 examples/s]38654 examples [00:42, 921.00 examples/s]38750 examples [00:42, 930.01 examples/s]38844 examples [00:42, 929.96 examples/s]38938 examples [00:42, 924.87 examples/s]39031 examples [00:42, 925.53 examples/s]39124 examples [00:43, 913.74 examples/s]39216 examples [00:43, 912.24 examples/s]39308 examples [00:43, 912.75 examples/s]39400 examples [00:43, 914.08 examples/s]39492 examples [00:43, 914.35 examples/s]39584 examples [00:43, 910.64 examples/s]39676 examples [00:43, 907.04 examples/s]39768 examples [00:43, 910.30 examples/s]39860 examples [00:43, 912.40 examples/s]39953 examples [00:43, 916.50 examples/s]40045 examples [00:44, 836.86 examples/s]40143 examples [00:44, 875.21 examples/s]40236 examples [00:44, 890.06 examples/s]40327 examples [00:44, 890.41 examples/s]40417 examples [00:44, 881.37 examples/s]40506 examples [00:44, 866.92 examples/s]40594 examples [00:44, 867.41 examples/s]40686 examples [00:44, 882.20 examples/s]40777 examples [00:44, 889.48 examples/s]40867 examples [00:45, 883.58 examples/s]40956 examples [00:45, 881.37 examples/s]41049 examples [00:45, 895.02 examples/s]41142 examples [00:45, 903.92 examples/s]41236 examples [00:45, 913.19 examples/s]41328 examples [00:45, 914.56 examples/s]41420 examples [00:45, 910.22 examples/s]41512 examples [00:45, 911.75 examples/s]41605 examples [00:45, 915.24 examples/s]41697 examples [00:45, 895.97 examples/s]41788 examples [00:46, 900.02 examples/s]41879 examples [00:46, 900.94 examples/s]41972 examples [00:46, 909.40 examples/s]42064 examples [00:46, 911.14 examples/s]42159 examples [00:46, 919.62 examples/s]42252 examples [00:46, 917.76 examples/s]42344 examples [00:46, 894.99 examples/s]42439 examples [00:46, 909.77 examples/s]42531 examples [00:46, 910.93 examples/s]42623 examples [00:46, 861.84 examples/s]42711 examples [00:47, 866.83 examples/s]42804 examples [00:47, 883.77 examples/s]42901 examples [00:47, 905.92 examples/s]42994 examples [00:47, 912.97 examples/s]43088 examples [00:47, 919.99 examples/s]43181 examples [00:47, 914.75 examples/s]43274 examples [00:47, 916.55 examples/s]43371 examples [00:47, 929.14 examples/s]43465 examples [00:47, 925.12 examples/s]43560 examples [00:47, 931.45 examples/s]43654 examples [00:48, 926.68 examples/s]43747 examples [00:48, 925.95 examples/s]43841 examples [00:48, 929.86 examples/s]43935 examples [00:48, 923.32 examples/s]44028 examples [00:48, 911.29 examples/s]44120 examples [00:48, 898.55 examples/s]44212 examples [00:48, 903.73 examples/s]44309 examples [00:48, 920.93 examples/s]44402 examples [00:48, 919.64 examples/s]44495 examples [00:49, 895.94 examples/s]44585 examples [00:49, 893.71 examples/s]44675 examples [00:49, 891.87 examples/s]44768 examples [00:49, 900.97 examples/s]44861 examples [00:49, 906.26 examples/s]44952 examples [00:49, 896.38 examples/s]45044 examples [00:49, 901.45 examples/s]45135 examples [00:49, 899.32 examples/s]45225 examples [00:49, 897.92 examples/s]45317 examples [00:49, 901.74 examples/s]45408 examples [00:50, 899.92 examples/s]45499 examples [00:50, 872.11 examples/s]45591 examples [00:50, 883.37 examples/s]45680 examples [00:50, 882.96 examples/s]45772 examples [00:50, 892.74 examples/s]45866 examples [00:50, 904.68 examples/s]45957 examples [00:50, 885.82 examples/s]46047 examples [00:50, 889.83 examples/s]46140 examples [00:50, 898.66 examples/s]46231 examples [00:50, 901.71 examples/s]46322 examples [00:51, 889.67 examples/s]46413 examples [00:51, 893.82 examples/s]46504 examples [00:51, 897.88 examples/s]46595 examples [00:51, 900.96 examples/s]46689 examples [00:51, 909.89 examples/s]46781 examples [00:51, 884.55 examples/s]46875 examples [00:51, 899.29 examples/s]46966 examples [00:51, 891.67 examples/s]47058 examples [00:51, 898.53 examples/s]47150 examples [00:51, 902.33 examples/s]47241 examples [00:52, 877.80 examples/s]47330 examples [00:52, 881.06 examples/s]47423 examples [00:52, 893.94 examples/s]47516 examples [00:52, 901.78 examples/s]47610 examples [00:52, 911.62 examples/s]47703 examples [00:52, 914.61 examples/s]47795 examples [00:52, 892.89 examples/s]47887 examples [00:52, 900.04 examples/s]47979 examples [00:52, 904.37 examples/s]48073 examples [00:52, 914.05 examples/s]48165 examples [00:53, 888.13 examples/s]48255 examples [00:53, 884.25 examples/s]48344 examples [00:53, 881.41 examples/s]48434 examples [00:53, 883.92 examples/s]48523 examples [00:53, 883.70 examples/s]48613 examples [00:53, 886.29 examples/s]48702 examples [00:53, 848.12 examples/s]48790 examples [00:53, 857.17 examples/s]48884 examples [00:53, 879.04 examples/s]48973 examples [00:54, 874.43 examples/s]49065 examples [00:54, 887.30 examples/s]49157 examples [00:54, 895.81 examples/s]49249 examples [00:54, 900.15 examples/s]49340 examples [00:54, 878.05 examples/s]49431 examples [00:54, 886.23 examples/s]49520 examples [00:54, 880.77 examples/s]49609 examples [00:54, 866.56 examples/s]49703 examples [00:54, 886.46 examples/s]49797 examples [00:54, 899.29 examples/s]49888 examples [00:55, 891.28 examples/s]49981 examples [00:55, 901.51 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6712/50000 [00:00<00:00, 67113.82 examples/s] 36%|███▌      | 17946/50000 [00:00<00:00, 76332.75 examples/s] 60%|██████    | 30164/50000 [00:00<00:00, 86015.49 examples/s] 85%|████████▍ | 42267/50000 [00:00<00:00, 94189.42 examples/s]                                                               0 examples [00:00, ? examples/s]73 examples [00:00, 728.59 examples/s]163 examples [00:00, 769.39 examples/s]256 examples [00:00, 809.99 examples/s]343 examples [00:00, 825.69 examples/s]433 examples [00:00, 846.59 examples/s]524 examples [00:00, 864.07 examples/s]612 examples [00:00, 866.32 examples/s]703 examples [00:00, 876.17 examples/s]798 examples [00:00, 895.93 examples/s]890 examples [00:01, 902.72 examples/s]981 examples [00:01, 904.30 examples/s]1071 examples [00:01, 888.95 examples/s]1163 examples [00:01, 895.25 examples/s]1255 examples [00:01, 900.76 examples/s]1346 examples [00:01, 901.64 examples/s]1436 examples [00:01, 708.07 examples/s]1533 examples [00:01, 768.02 examples/s]1627 examples [00:01, 810.53 examples/s]1713 examples [00:02, 817.93 examples/s]1806 examples [00:02, 846.54 examples/s]1900 examples [00:02, 869.97 examples/s]1993 examples [00:02, 886.14 examples/s]2084 examples [00:02, 891.87 examples/s]2176 examples [00:02, 896.46 examples/s]2270 examples [00:02, 908.47 examples/s]2365 examples [00:02, 919.27 examples/s]2458 examples [00:02, 916.30 examples/s]2550 examples [00:02, 913.62 examples/s]2642 examples [00:03, 903.65 examples/s]2733 examples [00:03, 873.58 examples/s]2826 examples [00:03, 887.54 examples/s]2918 examples [00:03, 894.64 examples/s]3012 examples [00:03, 907.66 examples/s]3103 examples [00:03, 869.74 examples/s]3193 examples [00:03, 878.36 examples/s]3283 examples [00:03, 882.28 examples/s]3375 examples [00:03, 891.35 examples/s]3465 examples [00:03, 890.91 examples/s]3555 examples [00:04, 891.17 examples/s]3645 examples [00:04, 890.23 examples/s]3735 examples [00:04, 880.43 examples/s]3824 examples [00:04, 847.86 examples/s]3917 examples [00:04, 868.91 examples/s]4006 examples [00:04, 873.75 examples/s]4097 examples [00:04, 881.63 examples/s]4191 examples [00:04, 896.65 examples/s]4281 examples [00:04, 894.48 examples/s]4373 examples [00:04, 899.57 examples/s]4464 examples [00:05, 895.14 examples/s]4561 examples [00:05, 914.23 examples/s]4656 examples [00:05, 922.67 examples/s]4749 examples [00:05, 920.03 examples/s]4844 examples [00:05, 927.20 examples/s]4937 examples [00:05, 922.26 examples/s]5031 examples [00:05, 927.42 examples/s]5124 examples [00:05, 925.04 examples/s]5217 examples [00:05, 919.64 examples/s]5309 examples [00:05, 909.33 examples/s]5400 examples [00:06, 903.88 examples/s]5493 examples [00:06, 910.43 examples/s]5586 examples [00:06, 914.90 examples/s]5679 examples [00:06, 918.04 examples/s]5773 examples [00:06, 923.06 examples/s]5866 examples [00:06, 921.73 examples/s]5960 examples [00:06, 923.96 examples/s]6055 examples [00:06, 931.46 examples/s]6149 examples [00:06, 925.24 examples/s]6243 examples [00:07, 926.18 examples/s]6336 examples [00:07, 917.28 examples/s]6429 examples [00:07, 918.74 examples/s]6521 examples [00:07, 893.35 examples/s]6611 examples [00:07, 881.55 examples/s]6700 examples [00:07, 875.27 examples/s]6789 examples [00:07, 879.25 examples/s]6883 examples [00:07, 896.60 examples/s]6973 examples [00:07, 875.58 examples/s]7065 examples [00:07, 887.67 examples/s]7154 examples [00:08, 876.69 examples/s]7242 examples [00:08, 862.74 examples/s]7329 examples [00:08, 854.27 examples/s]7420 examples [00:08, 867.76 examples/s]7513 examples [00:08, 883.97 examples/s]7605 examples [00:08, 893.57 examples/s]7695 examples [00:08, 880.43 examples/s]7787 examples [00:08, 890.99 examples/s]7879 examples [00:08, 898.36 examples/s]7975 examples [00:08, 913.57 examples/s]8068 examples [00:09, 916.70 examples/s]8160 examples [00:09, 916.72 examples/s]8257 examples [00:09, 930.62 examples/s]8351 examples [00:09, 921.22 examples/s]8444 examples [00:09, 923.48 examples/s]8541 examples [00:09, 934.19 examples/s]8635 examples [00:09, 926.64 examples/s]8728 examples [00:09, 906.70 examples/s]8820 examples [00:09, 907.52 examples/s]8912 examples [00:09, 909.32 examples/s]9006 examples [00:10, 915.58 examples/s]9098 examples [00:10, 885.11 examples/s]9189 examples [00:10, 890.99 examples/s]9279 examples [00:10, 887.46 examples/s]9369 examples [00:10, 890.08 examples/s]9461 examples [00:10, 897.34 examples/s]9554 examples [00:10, 906.49 examples/s]9648 examples [00:10, 914.17 examples/s]9740 examples [00:10, 880.42 examples/s]9831 examples [00:11, 885.88 examples/s]9920 examples [00:11, 880.24 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteG4NHFJ/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteG4NHFJ/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f9572d288c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f9572d288c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f9572d288c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f94fc1cb198>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f94fc184940>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f9572d288c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f9572d288c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f9572d288c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f95590ea240>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f95590ea2e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f94f40aa268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f94f40aa268> 

  function with postional parmater data_info <function split_train_valid at 0x7f94f40aa268> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f94f40aa378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f94f40aa378> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f94f40aa378> , (data_info, **args) 
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<23:12:19, 10.3kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:28:30, 14.5kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:35:08, 20.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:07:03, 29.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.39M/862M [00:01<5:40:07, 42.1kB/s].vector_cache/glove.6B.zip:   1%|          | 6.43M/862M [00:01<3:57:23, 60.1kB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.3M/862M [00:01<2:45:19, 85.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.1M/862M [00:01<1:55:19, 122kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.2M/862M [00:01<1:20:19, 175kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.9M/862M [00:01<56:44, 247kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 27.2M/862M [00:01<39:31, 352kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.9M/862M [00:02<27:32, 502kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.5M/862M [00:02<19:18, 712kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.9M/862M [00:02<13:32, 1.01MB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.2M/862M [00:02<09:32, 1.43MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.8M/862M [00:02<06:43, 2.01MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.2M/862M [00:02<06:06, 2.21MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:04<06:10, 2.18MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:05<07:57, 1.69MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:05<06:21, 2.11MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.1M/862M [00:05<04:39, 2.88MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:06<07:37, 1.75MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:07<07:09, 1.86MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.0M/862M [00:07<05:27, 2.44MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:08<06:21, 2.09MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:09<06:03, 2.19MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.3M/862M [00:09<04:39, 2.85MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:10<05:56, 2.23MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:11<07:24, 1.79MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.7M/862M [00:11<05:50, 2.26MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.5M/862M [00:11<04:17, 3.07MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:12<07:08, 1.84MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:13<06:31, 2.02MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.8M/862M [00:13<04:55, 2.66MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:15<06:35, 1.98MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:15<07:17, 1.79MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:15<05:46, 2.26MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:17<06:08, 2.12MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:17<05:37, 2.32MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.3M/862M [00:17<04:15, 3.04MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:19<06:01, 2.15MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:19<06:52, 1.88MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.4M/862M [00:19<05:21, 2.41MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.5M/862M [00:19<03:55, 3.28MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:20<08:44, 1.47MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:21<07:26, 1.73MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.5M/862M [00:21<05:29, 2.34MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<06:50, 1.87MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:23<07:24, 1.73MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.6M/862M [00:23<05:50, 2.19MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<06:08, 2.08MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:25<05:35, 2.28MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.7M/862M [00:25<04:13, 3.01MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:55, 2.14MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<06:43, 1.88MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<05:13, 2.42MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<03:51, 3.28MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:38, 1.65MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:37, 1.90MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<04:57, 2.54MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:23, 1.96MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<07:01, 1.79MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<05:29, 2.28MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<03:57, 3.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<16:53, 738kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<13:06, 951kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<09:24, 1.32MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<07:23, 1.68MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<9:23:57, 22.0kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<6:34:59, 31.4kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:34<4:35:29, 44.8kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<3:41:07, 55.8kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<2:37:14, 78.4kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<1:50:33, 111kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<1:19:04, 155kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<56:35, 217kB/s]  .vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<39:51, 307kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<30:38, 398kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<23:55, 510kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<17:20, 702kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<12:13, 992kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<11:50:07, 17.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<8:18:03, 24.3kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<5:48:10, 34.7kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<4:05:50, 49.0kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<2:54:28, 69.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<2:02:31, 98.3kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:44<1:25:35, 140kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<1:10:47, 169kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<50:46, 236kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<35:43, 335kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<27:44, 430kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<20:36, 578kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<14:42, 808kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<13:04, 907kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<10:20, 1.14MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<07:28, 1.58MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<08:02, 1.47MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:48, 1.73MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<05:03, 2.32MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<06:22, 1.84MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:50, 1.71MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:23, 2.17MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:38, 2.06MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:10, 2.25MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<03:54, 2.97MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<05:24, 2.14MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<04:57, 2.33MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<03:42, 3.11MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<05:18, 2.17MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<06:02, 1.90MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<04:42, 2.44MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<03:30, 3.27MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<06:08, 1.86MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<05:28, 2.09MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<04:07, 2.77MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<05:31, 2.06MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<06:10, 1.84MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<04:52, 2.33MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<03:32, 3.19MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<10:51:08, 17.4kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<7:36:40, 24.7kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<5:19:13, 35.3kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<3:45:23, 49.8kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<2:40:00, 70.2kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<1:52:26, 99.7kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<1:18:30, 142kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<1:08:30, 163kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<49:03, 227kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<34:32, 322kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<26:42, 415kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<20:56, 530kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<15:08, 732kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<10:41, 1.03MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<16:38, 663kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<12:46, 862kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 203M/862M [01:14<09:12, 1.19MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<09:00, 1.22MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<07:24, 1.48MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<05:26, 2.01MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<06:24, 1.70MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<06:41, 1.63MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<05:08, 2.12MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<03:45, 2.89MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<07:07, 1.52MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<06:06, 1.77MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<04:32, 2.37MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:41, 1.89MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<06:15, 1.72MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<04:56, 2.18MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<03:34, 2.99MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<1:19:07, 135kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<56:27, 189kB/s]  .vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<39:41, 268kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<30:11, 352kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<23:16, 456kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<16:48, 630kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<13:25, 786kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<10:16, 1.03MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<07:23, 1.42MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<05:20, 1.96MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<1:00:33, 173kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<44:30, 235kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<31:34, 331kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:30<22:07, 471kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<27:22, 380kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<20:13, 514kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<14:34, 713kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<10:18, 1.00MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<1:31:34, 113kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<1:08:17, 151kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<48:49, 212kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<34:21, 300kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<26:18, 391kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<22:34, 455kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<16:39, 616kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<12:00, 853kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<08:35, 1.19MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<10:35, 963kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<11:35, 880kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<09:05, 1.12MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<06:35, 1.54MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<06:58, 1.45MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<08:26, 1.20MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<06:50, 1.48MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<04:58, 2.03MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<05:56, 1.69MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<07:41, 1.31MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<06:15, 1.61MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<04:34, 2.19MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<05:42, 1.75MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<07:14, 1.38MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<05:48, 1.72MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<04:15, 2.34MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<03:13, 3.08MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<08:47, 1.13MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<09:22, 1.06MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<07:23, 1.34MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<05:23, 1.83MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<06:17, 1.57MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<07:38, 1.29MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<06:08, 1.60MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<04:30, 2.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<05:43, 1.71MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<07:24, 1.32MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<06:00, 1.63MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<04:25, 2.20MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<05:31, 1.76MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<06:49, 1.43MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<05:31, 1.76MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<04:02, 2.40MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<05:35, 1.73MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<06:52, 1.40MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<05:24, 1.78MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<03:58, 2.42MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<02:58, 3.22MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<18:41, 512kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<15:59, 599kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<11:47, 811kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<08:26, 1.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<08:32, 1.11MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<09:03, 1.05MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<07:05, 1.34MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<05:09, 1.84MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<06:16, 1.50MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<07:05, 1.33MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<05:33, 1.69MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<04:02, 2.33MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [01:59<03:01, 3.10MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<1:28:57, 105kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<1:05:06, 144kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<46:09, 203kB/s]  .vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<32:24, 288kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<25:24, 366kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<20:25, 455kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<14:58, 621kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<10:37, 871kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<10:08, 910kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<09:44, 947kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<07:23, 1.25MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<05:21, 1.72MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<06:24, 1.43MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<07:16, 1.26MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<05:44, 1.59MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<04:10, 2.18MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<05:43, 1.59MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<06:36, 1.38MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<05:17, 1.72MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<03:51, 2.34MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<05:24, 1.67MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<06:22, 1.42MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<05:02, 1.79MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<03:44, 2.40MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<05:12, 1.72MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<06:12, 1.44MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<04:53, 1.83MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<03:36, 2.48MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<05:06, 1.74MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<06:17, 1.41MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<05:03, 1.75MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<03:42, 2.39MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<05:09, 1.71MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<06:08, 1.43MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<04:56, 1.78MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<03:37, 2.42MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<05:15, 1.66MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<06:10, 1.42MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<04:52, 1.79MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<03:35, 2.43MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<05:06, 1.70MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<06:12, 1.40MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<04:53, 1.77MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<03:35, 2.40MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<05:01, 1.71MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<05:58, 1.44MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<04:42, 1.83MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<03:26, 2.49MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<02:36, 3.28MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<18:10, 470kB/s] .vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<15:09, 563kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<11:13, 760kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<07:59, 1.06MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<08:14, 1.03MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<08:11, 1.03MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<06:15, 1.35MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<04:32, 1.86MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<05:38, 1.49MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<06:21, 1.32MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<05:03, 1.66MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:42, 2.25MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<05:01, 1.66MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<06:02, 1.38MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<04:50, 1.72MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:32, 2.35MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<04:54, 1.68MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<06:06, 1.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<04:57, 1.66MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:37, 2.27MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<04:45, 1.72MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<05:42, 1.43MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<04:29, 1.82MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:18, 2.47MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:37<04:16, 1.90MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:37<05:11, 1.57MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:37<04:11, 1.94MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<03:05, 2.62MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<04:44, 1.70MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<05:29, 1.47MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:18, 1.87MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<03:10, 2.53MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<04:46, 1.67MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<05:37, 1.42MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:24, 1.81MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<03:14, 2.45MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<04:43, 1.68MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<05:26, 1.45MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:43<04:20, 1.82MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:08, 2.50MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<04:52, 1.61MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<05:39, 1.38MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<04:27, 1.76MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:13, 2.42MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<04:42, 1.65MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<05:11, 1.50MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<04:07, 1.88MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:01, 2.56MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<05:01, 1.53MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<05:29, 1.40MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<04:15, 1.80MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<03:06, 2.46MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<05:12, 1.47MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<05:35, 1.36MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:25, 1.72MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:13, 2.35MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<04:57, 1.53MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<05:18, 1.42MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<04:10, 1.81MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:02, 2.48MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<05:25, 1.38MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:55<05:32, 1.35MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<04:18, 1.74MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<03:07, 2.39MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<06:24, 1.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<06:17, 1.18MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<04:51, 1.52MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:31, 2.10MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<05:30, 1.34MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<05:44, 1.28MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<04:29, 1.64MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:15, 2.24MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<05:11, 1.40MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<05:10, 1.41MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<03:58, 1.83MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<02:52, 2.51MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<07:18, 987kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<06:53, 1.05MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<05:14, 1.37MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<03:45, 1.91MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<06:09, 1.16MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<05:45, 1.24MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<04:23, 1.63MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<03:09, 2.24MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<08:57, 791kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<07:59, 886kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<05:57, 1.19MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<04:14, 1.66MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<03:51, 1.81MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<5:06:30, 22.9kB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<3:35:00, 32.6kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<2:30:04, 46.5kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<1:46:16, 65.4kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<1:17:27, 89.7kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<54:50, 127kB/s]   .vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<38:24, 180kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<26:55, 256kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<24:14, 284kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<18:09, 379kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<12:59, 528kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<09:07, 747kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<6:36:37, 17.2kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<4:38:43, 24.4kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<3:14:55, 34.8kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<2:15:43, 49.7kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<1:40:21, 67.2kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<1:11:46, 93.8kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<50:30, 133kB/s]   .vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:16<35:13, 190kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<28:14, 236kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<20:50, 320kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<14:49, 449kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<11:28, 575kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<09:26, 699kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<06:57, 947kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<04:55, 1.33MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<13:42, 477kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<10:58, 594kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<08:00, 813kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:22<05:39, 1.14MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<34:40, 186kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<25:12, 256kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<17:47, 362kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:24<12:29, 513kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<13:03, 489kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<10:20, 618kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<07:31, 846kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<06:16, 1.01MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<05:40, 1.11MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<04:16, 1.47MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<03:59, 1.57MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<04:01, 1.56MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:04, 2.03MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<02:15, 2.74MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:25, 1.81MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:38, 1.69MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<02:51, 2.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<02:58, 2.05MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<03:20, 1.83MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<02:37, 2.32MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<01:54, 3.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<05:38, 1.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<05:07, 1.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:52, 1.55MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:36<02:45, 2.17MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<1:58:30, 50.5kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<1:24:04, 71.1kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<59:01, 101kB/s]   .vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<41:55, 141kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<30:32, 194kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<21:37, 273kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<15:57, 366kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<12:18, 475kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<08:53, 656kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<07:06, 812kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<06:09, 938kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<04:35, 1.26MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<04:06, 1.39MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<03:59, 1.43MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:04, 1.85MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<03:03, 1.85MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<03:14, 1.74MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:32, 2.21MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:50<02:40, 2.08MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<03:00, 1.85MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<02:22, 2.34MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<01:42, 3.23MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<15:12, 362kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<11:43, 469kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<08:27, 648kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<06:45, 804kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<05:47, 938kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<04:18, 1.26MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<03:03, 1.76MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<08:24, 638kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<06:58, 769kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<05:07, 1.04MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<04:25, 1.20MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<04:07, 1.28MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<03:06, 1.70MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:16, 2.32MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<03:07, 1.68MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<03:14, 1.61MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [04:00<02:31, 2.06MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<01:49, 2.84MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<04:51, 1.06MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<04:24, 1.17MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<03:20, 1.54MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<02:22, 2.15MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<09:16, 549kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<07:29, 679kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<05:29, 925kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<03:51, 1.30MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<11:04, 454kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<08:45, 572kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<06:22, 786kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<05:13, 948kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<04:37, 1.07MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<03:28, 1.42MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<03:12, 1.52MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<03:14, 1.51MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<02:30, 1.95MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<01:47, 2.70MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<12:29, 385kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:11<09:43, 495kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<07:01, 683kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<05:38, 842kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<04:53, 970kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<03:38, 1.30MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<03:17, 1.42MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<03:15, 1.44MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<02:29, 1.87MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:29, 1.85MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:38, 1.75MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:04, 2.22MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:10, 2.09MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:24, 1.88MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<01:52, 2.42MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:19<01:21, 3.30MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:34, 1.25MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:24, 1.31MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:36, 1.71MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:31, 1.75MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:37, 1.68MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:03, 2.14MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:07, 2.04MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:22, 1.83MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:52, 2.31MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<01:59, 2.14MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<01:58, 2.16MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<01:29, 2.83MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<01:51, 2.27MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:03, 2.04MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<01:37, 2.58MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<01:10, 3.51MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:31<03:32, 1.17MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<03:11, 1.29MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:23, 1.72MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<01:42, 2.39MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<04:21, 932kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:46, 1.07MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:49, 1.43MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:37, 1.52MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:33, 1.56MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:56, 2.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:35<01:24, 2.79MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<02:26, 1.61MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<02:24, 1.62MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:50, 2.13MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<01:20, 2.90MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<02:39, 1.45MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<02:31, 1.52MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:56, 1.98MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<01:58, 1.91MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<02:04, 1.82MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:36, 2.33MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<01:44, 2.13MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<01:53, 1.95MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:29, 2.48MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:45<01:38, 2.21MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:45<02:25, 1.50MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:01, 1.79MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:28, 2.44MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:47<01:56, 1.84MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<02:00, 1.78MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:33, 2.28MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<01:40, 2.10MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<01:48, 1.94MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:25, 2.46MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<01:33, 2.20MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<01:43, 2.00MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:21, 2.53MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<01:30, 2.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<01:38, 2.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:17, 2.58MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:27, 2.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<01:27, 2.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:07, 2.94MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:24, 2.30MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:57<02:02, 1.58MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:40, 1.93MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:14, 2.58MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:30, 2.10MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<01:36, 1.96MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:15, 2.50MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:23, 2.21MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:01<01:31, 2.03MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:11, 2.58MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:20, 2.25MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<01:27, 2.08MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:07, 2.67MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<00:49, 3.61MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:03, 1.44MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:57, 1.51MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:28, 2.00MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:03, 2.76MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<04:05, 706kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<04:02, 715kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<03:10, 910kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<02:39, 1.08MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:07<02:15, 1.27MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<02:00, 1.43MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:52, 1.53MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:41, 1.69MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:35, 1.79MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:31, 1.87MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<01:27, 1.95MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<01:25, 2.00MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<01:23, 2.05MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<01:22, 2.07MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:20, 2.11MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:19, 2.14MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:17, 2.17MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<01:17, 2.18MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<01:17, 2.19MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<01:16, 2.21MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:15, 2.22MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:15, 2.23MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:14, 2.24MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:13, 2.27MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<01:12, 2.29MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<01:15, 2.22MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<01:10, 2.36MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<01:08, 2.43MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:14, 2.24MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:12, 1.25MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:52, 1.47MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:38, 1.67MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:27, 1.89MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:22, 2.00MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:17, 2.13MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:13, 2.24MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:10, 2.33MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:07, 2.41MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:03, 2.56MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:09, 2.34MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<01:05, 2.49MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:03, 2.53MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:39, 1.01MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:18, 1.16MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:54, 1.40MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:37, 1.64MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<01:25, 1.87MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:13<01:16, 2.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:09, 2.29MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:06, 2.39MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:02, 2.52MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:00, 2.62MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<00:58, 2.71MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<00:55, 2.83MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<04:25, 592kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<03:28, 751kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:41, 967kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:08, 1.21MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<01:45, 1.48MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:28, 1.76MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:16, 2.02MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:09, 2.23MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:03, 2.43MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<00:59, 2.60MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<00:55, 2.77MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<03:20, 765kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<02:59, 853kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<02:20, 1.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:52, 1.35MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<01:33, 1.63MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:18, 1.92MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:17<01:08, 2.20MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:00, 2.47MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<00:53, 2.80MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<00:51, 2.92MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:57, 1.27MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:53, 1.31MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:32, 1.60MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:15, 1.95MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:01, 2.40MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<00:51, 2.87MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<00:49, 2.98MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<00:43, 3.32MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<00:42, 3.39MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<01:42, 1.42MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<01:58, 1.22MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:36, 1.50MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:14, 1.92MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:00, 2.37MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<00:49, 2.85MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<00:42, 3.30MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<01:39, 1.42MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<02:25, 965kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<02:00, 1.16MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:31, 1.52MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:10, 1.98MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<00:54, 2.54MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<00:43, 3.14MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<03:14, 702kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<03:16, 694kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<02:32, 892kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:52, 1.20MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<01:23, 1.61MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:02, 2.14MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:58, 1.12MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<02:13, 994kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:43, 1.28MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:15, 1.73MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:00, 2.16MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:26<00:45, 2.86MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:50, 1.16MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:59, 1.07MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:33, 1.36MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:09, 1.83MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<00:51, 2.44MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:39, 1.25MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<02:17, 900kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:53, 1.09MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:23, 1.46MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<01:00, 1.99MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<01:27, 1.37MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<01:59, 1.00MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:38, 1.21MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:12, 1.63MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:32<00:52, 2.21MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:34<01:45, 1.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:34<02:01, 957kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:36, 1.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:10, 1.62MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<00:50, 2.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:36<03:58, 469kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<03:28, 534kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<02:36, 712kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:50, 992kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<01:17, 1.39MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<03:11, 562kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<02:51, 626kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<02:07, 839kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:30, 1.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:26, 1.19MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:32, 1.12MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:10, 1.45MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:50, 1.99MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:40<00:36, 2.70MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:42<02:30, 662kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:42<02:13, 744kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:39, 988kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<01:10, 1.37MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<01:21, 1.17MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<01:21, 1.17MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:02, 1.51MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:44, 2.06MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:46<01:03, 1.43MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:46<01:10, 1.28MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:54, 1.64MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:39, 2.26MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:48<00:53, 1.61MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<00:57, 1.52MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:44, 1.93MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:31, 2.66MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<01:20, 1.03MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<01:16, 1.08MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<00:57, 1.43MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:41, 1.96MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:47, 1.64MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:49, 1.58MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:38, 2.03MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<00:26, 2.80MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<02:24, 514kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<01:58, 627kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<01:26, 849kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:59, 1.19MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<01:22, 851kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<01:08, 1.02MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:50, 1.37MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:34, 1.91MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<01:19, 831kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<01:11, 921kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:53, 1.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:36, 1.71MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:00<01:07, 917kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:00<01:01, 1.00MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:45, 1.34MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:31, 1.86MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:02<00:42, 1.35MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:42, 1.35MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:32, 1.74MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:22, 2.41MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<01:02, 856kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:56, 949kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:42, 1.25MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:30, 1.70MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:21, 2.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<01:42, 484kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<01:23, 592kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<01:00, 804kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:40, 1.13MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:56, 794kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:50, 898kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:37, 1.19MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:25, 1.66MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:48, 843kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<00:43, 941kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:31, 1.26MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:21, 1.75MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:32, 1.14MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:12<00:31, 1.18MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:23, 1.56MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:15, 2.16MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:27, 1.20MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:14<00:26, 1.23MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:19, 1.61MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:13, 2.23MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:19, 1.45MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:17, 1.60MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:12, 2.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:12, 1.93MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:12, 1.92MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:09, 2.42MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:06, 3.25MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:11, 1.81MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:11, 1.72MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:08, 2.19MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:07, 2.07MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:08, 1.87MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:06, 2.36MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:05, 2.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:08, 1.39MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:06, 1.66MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:04, 2.26MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:04, 1.84MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:26<00:04, 1.74MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:03, 2.23MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:01, 2.09MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 1.88MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 2.38MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 578/400000 [00:00<01:09, 5779.07it/s]  0%|          | 1327/400000 [00:00<01:04, 6202.96it/s]  1%|          | 2076/400000 [00:00<01:00, 6539.12it/s]  1%|          | 2841/400000 [00:00<00:58, 6835.48it/s]  1%|          | 3595/400000 [00:00<00:56, 7031.83it/s]  1%|          | 4408/400000 [00:00<00:53, 7327.24it/s]  1%|▏         | 5190/400000 [00:00<00:52, 7468.14it/s]  1%|▏         | 5942/400000 [00:00<00:52, 7482.72it/s]  2%|▏         | 6723/400000 [00:00<00:51, 7577.22it/s]  2%|▏         | 7508/400000 [00:01<00:51, 7656.44it/s]  2%|▏         | 8295/400000 [00:01<00:50, 7716.85it/s]  2%|▏         | 9107/400000 [00:01<00:49, 7832.84it/s]  2%|▏         | 9922/400000 [00:01<00:49, 7925.27it/s]  3%|▎         | 10752/400000 [00:01<00:48, 8033.85it/s]  3%|▎         | 11572/400000 [00:01<00:48, 8081.64it/s]  3%|▎         | 12379/400000 [00:01<00:48, 7977.63it/s]  3%|▎         | 13181/400000 [00:01<00:48, 7989.20it/s]  3%|▎         | 13991/400000 [00:01<00:48, 8021.32it/s]  4%|▎         | 14793/400000 [00:01<00:48, 7999.36it/s]  4%|▍         | 15627/400000 [00:02<00:47, 8098.32it/s]  4%|▍         | 16437/400000 [00:02<00:48, 7921.39it/s]  4%|▍         | 17242/400000 [00:02<00:48, 7956.73it/s]  5%|▍         | 18039/400000 [00:02<00:47, 7959.02it/s]  5%|▍         | 18856/400000 [00:02<00:47, 8015.24it/s]  5%|▍         | 19658/400000 [00:02<00:47, 7930.01it/s]  5%|▌         | 20452/400000 [00:02<00:50, 7590.58it/s]  5%|▌         | 21215/400000 [00:02<00:50, 7507.80it/s]  5%|▌         | 21969/400000 [00:02<00:50, 7431.65it/s]  6%|▌         | 22721/400000 [00:02<00:50, 7454.85it/s]  6%|▌         | 23468/400000 [00:03<00:50, 7447.43it/s]  6%|▌         | 24216/400000 [00:03<00:50, 7457.15it/s]  6%|▌         | 24963/400000 [00:03<00:50, 7416.84it/s]  6%|▋         | 25729/400000 [00:03<00:49, 7487.98it/s]  7%|▋         | 26479/400000 [00:03<00:50, 7414.10it/s]  7%|▋         | 27239/400000 [00:03<00:49, 7467.40it/s]  7%|▋         | 28008/400000 [00:03<00:49, 7530.94it/s]  7%|▋         | 28788/400000 [00:03<00:48, 7608.75it/s]  7%|▋         | 29557/400000 [00:03<00:48, 7632.33it/s]  8%|▊         | 30353/400000 [00:03<00:47, 7726.63it/s]  8%|▊         | 31156/400000 [00:04<00:47, 7812.68it/s]  8%|▊         | 31960/400000 [00:04<00:46, 7878.48it/s]  8%|▊         | 32749/400000 [00:04<00:46, 7878.90it/s]  8%|▊         | 33538/400000 [00:04<00:46, 7806.42it/s]  9%|▊         | 34320/400000 [00:04<00:47, 7712.82it/s]  9%|▉         | 35092/400000 [00:04<00:47, 7679.78it/s]  9%|▉         | 35884/400000 [00:04<00:46, 7749.87it/s]  9%|▉         | 36680/400000 [00:04<00:46, 7809.23it/s]  9%|▉         | 37519/400000 [00:04<00:45, 7972.39it/s] 10%|▉         | 38335/400000 [00:04<00:45, 8025.85it/s] 10%|▉         | 39174/400000 [00:05<00:44, 8130.18it/s] 10%|▉         | 39988/400000 [00:05<00:45, 7924.60it/s] 10%|█         | 40802/400000 [00:05<00:44, 7987.56it/s] 10%|█         | 41603/400000 [00:05<00:44, 7984.56it/s] 11%|█         | 42403/400000 [00:05<00:45, 7906.40it/s] 11%|█         | 43198/400000 [00:05<00:45, 7917.67it/s] 11%|█         | 43995/400000 [00:05<00:44, 7931.86it/s] 11%|█         | 44793/400000 [00:05<00:44, 7943.25it/s] 11%|█▏        | 45588/400000 [00:05<00:44, 7931.74it/s] 12%|█▏        | 46382/400000 [00:05<00:44, 7928.38it/s] 12%|█▏        | 47175/400000 [00:06<00:44, 7901.46it/s] 12%|█▏        | 47975/400000 [00:06<00:44, 7930.53it/s] 12%|█▏        | 48826/400000 [00:06<00:43, 8094.03it/s] 12%|█▏        | 49657/400000 [00:06<00:42, 8154.97it/s] 13%|█▎        | 50479/400000 [00:06<00:42, 8173.77it/s] 13%|█▎        | 51297/400000 [00:06<00:43, 7970.32it/s] 13%|█▎        | 52096/400000 [00:06<00:44, 7746.59it/s] 13%|█▎        | 52891/400000 [00:06<00:44, 7805.02it/s] 13%|█▎        | 53678/400000 [00:06<00:44, 7821.77it/s] 14%|█▎        | 54467/400000 [00:06<00:44, 7841.11it/s] 14%|█▍        | 55264/400000 [00:07<00:43, 7877.08it/s] 14%|█▍        | 56053/400000 [00:07<00:44, 7781.63it/s] 14%|█▍        | 56863/400000 [00:07<00:43, 7874.30it/s] 14%|█▍        | 57679/400000 [00:07<00:43, 7957.46it/s] 15%|█▍        | 58514/400000 [00:07<00:42, 8069.33it/s] 15%|█▍        | 59322/400000 [00:07<00:43, 7813.97it/s] 15%|█▌        | 60106/400000 [00:07<00:43, 7782.31it/s] 15%|█▌        | 60891/400000 [00:07<00:43, 7800.87it/s] 15%|█▌        | 61686/400000 [00:07<00:43, 7842.39it/s] 16%|█▌        | 62508/400000 [00:08<00:42, 7949.85it/s] 16%|█▌        | 63333/400000 [00:08<00:41, 8037.19it/s] 16%|█▌        | 64138/400000 [00:08<00:42, 7942.28it/s] 16%|█▌        | 64950/400000 [00:08<00:41, 7993.24it/s] 16%|█▋        | 65773/400000 [00:08<00:41, 8062.73it/s] 17%|█▋        | 66580/400000 [00:08<00:41, 8050.90it/s] 17%|█▋        | 67386/400000 [00:08<00:41, 8002.44it/s] 17%|█▋        | 68187/400000 [00:08<00:41, 7907.13it/s] 17%|█▋        | 68979/400000 [00:08<00:42, 7845.86it/s] 17%|█▋        | 69808/400000 [00:08<00:41, 7973.45it/s] 18%|█▊        | 70607/400000 [00:09<00:41, 7886.18it/s] 18%|█▊        | 71397/400000 [00:09<00:41, 7859.88it/s] 18%|█▊        | 72184/400000 [00:09<00:42, 7709.58it/s] 18%|█▊        | 72957/400000 [00:09<00:42, 7683.05it/s] 18%|█▊        | 73751/400000 [00:09<00:42, 7757.50it/s] 19%|█▊        | 74537/400000 [00:09<00:41, 7784.44it/s] 19%|█▉        | 75316/400000 [00:09<00:41, 7755.98it/s] 19%|█▉        | 76092/400000 [00:09<00:42, 7667.85it/s] 19%|█▉        | 76860/400000 [00:09<00:42, 7657.95it/s] 19%|█▉        | 77631/400000 [00:09<00:42, 7672.50it/s] 20%|█▉        | 78432/400000 [00:10<00:41, 7768.68it/s] 20%|█▉        | 79254/400000 [00:10<00:40, 7896.50it/s] 20%|██        | 80073/400000 [00:10<00:40, 7981.26it/s] 20%|██        | 80874/400000 [00:10<00:39, 7989.20it/s] 20%|██        | 81674/400000 [00:10<00:40, 7953.70it/s] 21%|██        | 82470/400000 [00:10<00:41, 7708.05it/s] 21%|██        | 83247/400000 [00:10<00:40, 7725.87it/s] 21%|██        | 84043/400000 [00:10<00:40, 7793.93it/s] 21%|██        | 84824/400000 [00:10<00:40, 7786.28it/s] 21%|██▏       | 85604/400000 [00:10<00:41, 7661.63it/s] 22%|██▏       | 86385/400000 [00:11<00:40, 7704.84it/s] 22%|██▏       | 87201/400000 [00:11<00:39, 7833.49it/s] 22%|██▏       | 87986/400000 [00:11<00:40, 7796.77it/s] 22%|██▏       | 88800/400000 [00:11<00:39, 7894.79it/s] 22%|██▏       | 89639/400000 [00:11<00:38, 8034.57it/s] 23%|██▎       | 90444/400000 [00:11<00:38, 8009.62it/s] 23%|██▎       | 91276/400000 [00:11<00:38, 8099.45it/s] 23%|██▎       | 92087/400000 [00:11<00:38, 8007.76it/s] 23%|██▎       | 92889/400000 [00:11<00:38, 7974.27it/s] 23%|██▎       | 93711/400000 [00:11<00:38, 8046.41it/s] 24%|██▎       | 94538/400000 [00:12<00:37, 8111.39it/s] 24%|██▍       | 95350/400000 [00:12<00:38, 7995.64it/s] 24%|██▍       | 96151/400000 [00:12<00:38, 7941.84it/s] 24%|██▍       | 96965/400000 [00:12<00:37, 7999.06it/s] 24%|██▍       | 97766/400000 [00:12<00:37, 7964.49it/s] 25%|██▍       | 98563/400000 [00:12<00:37, 7956.10it/s] 25%|██▍       | 99359/400000 [00:12<00:38, 7776.60it/s] 25%|██▌       | 100148/400000 [00:12<00:38, 7808.53it/s] 25%|██▌       | 100930/400000 [00:12<00:38, 7781.32it/s] 25%|██▌       | 101715/400000 [00:12<00:38, 7800.40it/s] 26%|██▌       | 102496/400000 [00:13<00:38, 7742.17it/s] 26%|██▌       | 103271/400000 [00:13<00:38, 7706.34it/s] 26%|██▌       | 104066/400000 [00:13<00:38, 7776.91it/s] 26%|██▌       | 104880/400000 [00:13<00:37, 7881.19it/s] 26%|██▋       | 105685/400000 [00:13<00:37, 7931.07it/s] 27%|██▋       | 106527/400000 [00:13<00:36, 8068.92it/s] 27%|██▋       | 107371/400000 [00:13<00:35, 8176.54it/s] 27%|██▋       | 108190/400000 [00:13<00:36, 8065.08it/s] 27%|██▋       | 109040/400000 [00:13<00:35, 8187.20it/s] 27%|██▋       | 109860/400000 [00:13<00:35, 8173.38it/s] 28%|██▊       | 110710/400000 [00:14<00:34, 8267.66it/s] 28%|██▊       | 111542/400000 [00:14<00:34, 8282.61it/s] 28%|██▊       | 112371/400000 [00:14<00:35, 8075.83it/s] 28%|██▊       | 113181/400000 [00:14<00:36, 7966.54it/s] 28%|██▊       | 113980/400000 [00:14<00:36, 7901.07it/s] 29%|██▊       | 114787/400000 [00:14<00:35, 7950.42it/s] 29%|██▉       | 115583/400000 [00:14<00:35, 7942.25it/s] 29%|██▉       | 116378/400000 [00:14<00:36, 7857.78it/s] 29%|██▉       | 117175/400000 [00:14<00:35, 7888.91it/s] 30%|██▉       | 118008/400000 [00:15<00:35, 8014.57it/s] 30%|██▉       | 118851/400000 [00:15<00:34, 8132.62it/s] 30%|██▉       | 119666/400000 [00:15<00:34, 8132.25it/s] 30%|███       | 120480/400000 [00:15<00:34, 7994.17it/s] 30%|███       | 121297/400000 [00:15<00:34, 8044.59it/s] 31%|███       | 122103/400000 [00:15<00:34, 7997.17it/s] 31%|███       | 122945/400000 [00:15<00:34, 8118.45it/s] 31%|███       | 123758/400000 [00:15<00:35, 7823.27it/s] 31%|███       | 124544/400000 [00:15<00:35, 7674.36it/s] 31%|███▏      | 125315/400000 [00:15<00:35, 7673.44it/s] 32%|███▏      | 126099/400000 [00:16<00:35, 7720.06it/s] 32%|███▏      | 126900/400000 [00:16<00:35, 7802.01it/s] 32%|███▏      | 127715/400000 [00:16<00:34, 7901.23it/s] 32%|███▏      | 128507/400000 [00:16<00:34, 7877.34it/s] 32%|███▏      | 129311/400000 [00:16<00:34, 7924.57it/s] 33%|███▎      | 130105/400000 [00:16<00:34, 7875.42it/s] 33%|███▎      | 130916/400000 [00:16<00:33, 7943.37it/s] 33%|███▎      | 131711/400000 [00:16<00:34, 7831.41it/s] 33%|███▎      | 132495/400000 [00:16<00:34, 7723.96it/s] 33%|███▎      | 133269/400000 [00:16<00:35, 7547.21it/s] 34%|███▎      | 134063/400000 [00:17<00:34, 7657.43it/s] 34%|███▎      | 134867/400000 [00:17<00:34, 7767.26it/s] 34%|███▍      | 135691/400000 [00:17<00:33, 7902.46it/s] 34%|███▍      | 136483/400000 [00:17<00:33, 7850.63it/s] 34%|███▍      | 137334/400000 [00:17<00:32, 8034.72it/s] 35%|███▍      | 138143/400000 [00:17<00:32, 8049.93it/s] 35%|███▍      | 138950/400000 [00:17<00:32, 7998.97it/s] 35%|███▍      | 139751/400000 [00:17<00:32, 7929.22it/s] 35%|███▌      | 140545/400000 [00:17<00:32, 7917.23it/s] 35%|███▌      | 141349/400000 [00:17<00:32, 7951.60it/s] 36%|███▌      | 142145/400000 [00:18<00:32, 7939.72it/s] 36%|███▌      | 142940/400000 [00:18<00:32, 7872.00it/s] 36%|███▌      | 143728/400000 [00:18<00:32, 7807.60it/s] 36%|███▌      | 144517/400000 [00:18<00:32, 7829.78it/s] 36%|███▋      | 145338/400000 [00:18<00:32, 7939.32it/s] 37%|███▋      | 146133/400000 [00:18<00:32, 7787.41it/s] 37%|███▋      | 146939/400000 [00:18<00:32, 7864.73it/s] 37%|███▋      | 147750/400000 [00:18<00:31, 7935.64it/s] 37%|███▋      | 148545/400000 [00:18<00:31, 7896.95it/s] 37%|███▋      | 149336/400000 [00:18<00:31, 7876.33it/s] 38%|███▊      | 150125/400000 [00:19<00:31, 7808.88it/s] 38%|███▊      | 150908/400000 [00:19<00:31, 7814.21it/s] 38%|███▊      | 151690/400000 [00:19<00:32, 7645.52it/s] 38%|███▊      | 152456/400000 [00:19<00:32, 7558.66it/s] 38%|███▊      | 153241/400000 [00:19<00:32, 7643.16it/s] 39%|███▊      | 154022/400000 [00:19<00:31, 7691.36it/s] 39%|███▊      | 154811/400000 [00:19<00:31, 7747.49it/s] 39%|███▉      | 155587/400000 [00:19<00:31, 7745.99it/s] 39%|███▉      | 156363/400000 [00:19<00:31, 7701.69it/s] 39%|███▉      | 157165/400000 [00:20<00:31, 7790.83it/s] 39%|███▉      | 157982/400000 [00:20<00:30, 7899.64it/s] 40%|███▉      | 158773/400000 [00:20<00:30, 7820.72it/s] 40%|███▉      | 159573/400000 [00:20<00:30, 7872.64it/s] 40%|████      | 160361/400000 [00:20<00:30, 7838.04it/s] 40%|████      | 161148/400000 [00:20<00:30, 7845.49it/s] 40%|████      | 161956/400000 [00:20<00:30, 7911.05it/s] 41%|████      | 162810/400000 [00:20<00:29, 8088.76it/s] 41%|████      | 163621/400000 [00:20<00:29, 8024.80it/s] 41%|████      | 164425/400000 [00:20<00:31, 7509.39it/s] 41%|████▏     | 165184/400000 [00:21<00:32, 7298.23it/s] 41%|████▏     | 165947/400000 [00:21<00:31, 7393.35it/s] 42%|████▏     | 166764/400000 [00:21<00:30, 7609.96it/s] 42%|████▏     | 167592/400000 [00:21<00:29, 7797.70it/s] 42%|████▏     | 168395/400000 [00:21<00:29, 7864.33it/s] 42%|████▏     | 169230/400000 [00:21<00:28, 8003.87it/s] 43%|████▎     | 170064/400000 [00:21<00:28, 8101.05it/s] 43%|████▎     | 170917/400000 [00:21<00:27, 8223.70it/s] 43%|████▎     | 171742/400000 [00:21<00:28, 8111.09it/s] 43%|████▎     | 172555/400000 [00:21<00:28, 8060.90it/s] 43%|████▎     | 173383/400000 [00:22<00:27, 8123.81it/s] 44%|████▎     | 174198/400000 [00:22<00:27, 8130.72it/s] 44%|████▍     | 175012/400000 [00:22<00:28, 8015.01it/s] 44%|████▍     | 175815/400000 [00:22<00:28, 7975.24it/s] 44%|████▍     | 176614/400000 [00:22<00:28, 7898.99it/s] 44%|████▍     | 177438/400000 [00:22<00:27, 7996.79it/s] 45%|████▍     | 178239/400000 [00:22<00:27, 7961.78it/s] 45%|████▍     | 179036/400000 [00:22<00:28, 7798.29it/s] 45%|████▍     | 179826/400000 [00:22<00:28, 7825.36it/s] 45%|████▌     | 180610/400000 [00:22<00:28, 7727.37it/s] 45%|████▌     | 181448/400000 [00:23<00:27, 7912.17it/s] 46%|████▌     | 182276/400000 [00:23<00:27, 8016.36it/s] 46%|████▌     | 183107/400000 [00:23<00:26, 8100.10it/s] 46%|████▌     | 183922/400000 [00:23<00:26, 8114.30it/s] 46%|████▌     | 184735/400000 [00:23<00:27, 7926.67it/s] 46%|████▋     | 185542/400000 [00:23<00:26, 7966.77it/s] 47%|████▋     | 186340/400000 [00:23<00:26, 7936.28it/s] 47%|████▋     | 187151/400000 [00:23<00:26, 7987.00it/s] 47%|████▋     | 187963/400000 [00:23<00:26, 8025.91it/s] 47%|████▋     | 188767/400000 [00:24<00:26, 7897.85it/s] 47%|████▋     | 189558/400000 [00:24<00:26, 7892.70it/s] 48%|████▊     | 190352/400000 [00:24<00:26, 7905.08it/s] 48%|████▊     | 191164/400000 [00:24<00:26, 7967.40it/s] 48%|████▊     | 191962/400000 [00:24<00:26, 7939.75it/s] 48%|████▊     | 192757/400000 [00:24<00:26, 7920.29it/s] 48%|████▊     | 193555/400000 [00:24<00:26, 7937.97it/s] 49%|████▊     | 194349/400000 [00:24<00:25, 7922.35it/s] 49%|████▉     | 195162/400000 [00:24<00:25, 7983.43it/s] 49%|████▉     | 195983/400000 [00:24<00:25, 8049.23it/s] 49%|████▉     | 196789/400000 [00:25<00:25, 8050.86it/s] 49%|████▉     | 197595/400000 [00:25<00:25, 8047.69it/s] 50%|████▉     | 198400/400000 [00:25<00:25, 8034.71it/s] 50%|████▉     | 199204/400000 [00:25<00:25, 7983.97it/s] 50%|█████     | 200003/400000 [00:25<00:25, 7860.20it/s] 50%|█████     | 200794/400000 [00:25<00:25, 7874.16it/s] 50%|█████     | 201603/400000 [00:25<00:24, 7936.48it/s] 51%|█████     | 202400/400000 [00:25<00:24, 7944.40it/s] 51%|█████     | 203200/400000 [00:25<00:24, 7959.54it/s] 51%|█████     | 204008/400000 [00:25<00:24, 7995.10it/s] 51%|█████     | 204808/400000 [00:26<00:24, 7853.89it/s] 51%|█████▏    | 205595/400000 [00:26<00:25, 7672.62it/s] 52%|█████▏    | 206364/400000 [00:26<00:25, 7656.48it/s] 52%|█████▏    | 207192/400000 [00:26<00:24, 7831.82it/s] 52%|█████▏    | 208009/400000 [00:26<00:24, 7928.55it/s] 52%|█████▏    | 208811/400000 [00:26<00:24, 7955.67it/s] 52%|█████▏    | 209644/400000 [00:26<00:23, 8063.80it/s] 53%|█████▎    | 210452/400000 [00:26<00:23, 7963.65it/s] 53%|█████▎    | 211287/400000 [00:26<00:23, 8074.37it/s] 53%|█████▎    | 212096/400000 [00:26<00:23, 8065.83it/s] 53%|█████▎    | 212942/400000 [00:27<00:22, 8179.74it/s] 53%|█████▎    | 213761/400000 [00:27<00:23, 7983.39it/s] 54%|█████▎    | 214562/400000 [00:27<00:23, 7874.13it/s] 54%|█████▍    | 215351/400000 [00:27<00:23, 7715.27it/s] 54%|█████▍    | 216125/400000 [00:27<00:23, 7719.49it/s] 54%|█████▍    | 216957/400000 [00:27<00:23, 7889.50it/s] 54%|█████▍    | 217752/400000 [00:27<00:23, 7907.17it/s] 55%|█████▍    | 218544/400000 [00:27<00:22, 7890.91it/s] 55%|█████▍    | 219334/400000 [00:27<00:23, 7793.07it/s] 55%|█████▌    | 220115/400000 [00:27<00:23, 7621.83it/s] 55%|█████▌    | 220895/400000 [00:28<00:23, 7673.04it/s] 55%|█████▌    | 221712/400000 [00:28<00:22, 7813.39it/s] 56%|█████▌    | 222515/400000 [00:28<00:22, 7874.43it/s] 56%|█████▌    | 223304/400000 [00:28<00:22, 7799.54it/s] 56%|█████▌    | 224085/400000 [00:28<00:22, 7665.66it/s] 56%|█████▌    | 224863/400000 [00:28<00:22, 7696.90it/s] 56%|█████▋    | 225665/400000 [00:28<00:22, 7790.66it/s] 57%|█████▋    | 226464/400000 [00:28<00:22, 7847.04it/s] 57%|█████▋    | 227293/400000 [00:28<00:21, 7974.43it/s] 57%|█████▋    | 228092/400000 [00:28<00:21, 7883.09it/s] 57%|█████▋    | 228884/400000 [00:29<00:21, 7893.22it/s] 57%|█████▋    | 229675/400000 [00:29<00:21, 7896.06it/s] 58%|█████▊    | 230466/400000 [00:29<00:21, 7878.06it/s] 58%|█████▊    | 231255/400000 [00:29<00:21, 7767.85it/s] 58%|█████▊    | 232033/400000 [00:29<00:21, 7637.58it/s] 58%|█████▊    | 232825/400000 [00:29<00:21, 7717.93it/s] 58%|█████▊    | 233598/400000 [00:29<00:21, 7679.81it/s] 59%|█████▊    | 234367/400000 [00:29<00:21, 7589.74it/s] 59%|█████▉    | 235181/400000 [00:29<00:21, 7746.15it/s] 59%|█████▉    | 235962/400000 [00:29<00:21, 7762.52it/s] 59%|█████▉    | 236748/400000 [00:30<00:20, 7789.69it/s] 59%|█████▉    | 237589/400000 [00:30<00:20, 7963.56it/s] 60%|█████▉    | 238387/400000 [00:30<00:20, 7830.88it/s] 60%|█████▉    | 239207/400000 [00:30<00:20, 7935.76it/s] 60%|██████    | 240002/400000 [00:30<00:20, 7934.91it/s] 60%|██████    | 240797/400000 [00:30<00:20, 7798.14it/s] 60%|██████    | 241579/400000 [00:30<00:20, 7762.99it/s] 61%|██████    | 242362/400000 [00:30<00:20, 7780.89it/s] 61%|██████    | 243142/400000 [00:30<00:20, 7786.57it/s] 61%|██████    | 243922/400000 [00:31<00:20, 7618.16it/s] 61%|██████    | 244685/400000 [00:31<00:20, 7601.78it/s] 61%|██████▏   | 245446/400000 [00:31<00:20, 7595.84it/s] 62%|██████▏   | 246207/400000 [00:31<00:20, 7521.60it/s] 62%|██████▏   | 247023/400000 [00:31<00:19, 7701.89it/s] 62%|██████▏   | 247795/400000 [00:31<00:19, 7704.33it/s] 62%|██████▏   | 248581/400000 [00:31<00:19, 7748.62it/s] 62%|██████▏   | 249378/400000 [00:31<00:19, 7812.02it/s] 63%|██████▎   | 250190/400000 [00:31<00:18, 7900.23it/s] 63%|██████▎   | 251014/400000 [00:31<00:18, 7998.14it/s] 63%|██████▎   | 251815/400000 [00:32<00:18, 7889.62it/s] 63%|██████▎   | 252647/400000 [00:32<00:18, 8012.72it/s] 63%|██████▎   | 253466/400000 [00:32<00:18, 8064.04it/s] 64%|██████▎   | 254293/400000 [00:32<00:17, 8123.95it/s] 64%|██████▍   | 255107/400000 [00:32<00:17, 8085.97it/s] 64%|██████▍   | 255917/400000 [00:32<00:17, 8050.46it/s] 64%|██████▍   | 256759/400000 [00:32<00:17, 8156.22it/s] 64%|██████▍   | 257576/400000 [00:32<00:17, 8079.29it/s] 65%|██████▍   | 258385/400000 [00:32<00:17, 8067.22it/s] 65%|██████▍   | 259193/400000 [00:32<00:17, 8042.72it/s] 65%|██████▍   | 259998/400000 [00:33<00:17, 8023.74it/s] 65%|██████▌   | 260830/400000 [00:33<00:17, 8109.35it/s] 65%|██████▌   | 261642/400000 [00:33<00:17, 8029.10it/s] 66%|██████▌   | 262460/400000 [00:33<00:17, 8072.89it/s] 66%|██████▌   | 263268/400000 [00:33<00:16, 8058.73it/s] 66%|██████▌   | 264075/400000 [00:33<00:17, 7976.96it/s] 66%|██████▌   | 264896/400000 [00:33<00:16, 8039.37it/s] 66%|██████▋   | 265709/400000 [00:33<00:16, 8064.66it/s] 67%|██████▋   | 266516/400000 [00:33<00:16, 8051.02it/s] 67%|██████▋   | 267322/400000 [00:33<00:16, 8040.86it/s] 67%|██████▋   | 268127/400000 [00:34<00:16, 8013.86it/s] 67%|██████▋   | 268942/400000 [00:34<00:16, 8052.85it/s] 67%|██████▋   | 269772/400000 [00:34<00:16, 8124.41it/s] 68%|██████▊   | 270585/400000 [00:34<00:16, 8000.91it/s] 68%|██████▊   | 271386/400000 [00:34<00:16, 7938.86it/s] 68%|██████▊   | 272181/400000 [00:34<00:16, 7903.02it/s] 68%|██████▊   | 272972/400000 [00:34<00:16, 7743.91it/s] 68%|██████▊   | 273748/400000 [00:34<00:16, 7736.05it/s] 69%|██████▊   | 274523/400000 [00:34<00:16, 7682.74it/s] 69%|██████▉   | 275339/400000 [00:34<00:15, 7819.69it/s] 69%|██████▉   | 276127/400000 [00:35<00:15, 7836.68it/s] 69%|██████▉   | 276925/400000 [00:35<00:15, 7876.93it/s] 69%|██████▉   | 277714/400000 [00:35<00:15, 7805.77it/s] 70%|██████▉   | 278501/400000 [00:35<00:15, 7822.64it/s] 70%|██████▉   | 279297/400000 [00:35<00:15, 7862.23it/s] 70%|███████   | 280084/400000 [00:35<00:15, 7736.09it/s] 70%|███████   | 280902/400000 [00:35<00:15, 7859.39it/s] 70%|███████   | 281724/400000 [00:35<00:14, 7964.08it/s] 71%|███████   | 282540/400000 [00:35<00:14, 8021.47it/s] 71%|███████   | 283343/400000 [00:35<00:14, 7907.89it/s] 71%|███████   | 284135/400000 [00:36<00:14, 7860.73it/s] 71%|███████   | 284922/400000 [00:36<00:14, 7836.40it/s] 71%|███████▏  | 285731/400000 [00:36<00:14, 7910.22it/s] 72%|███████▏  | 286538/400000 [00:36<00:14, 7954.57it/s] 72%|███████▏  | 287334/400000 [00:36<00:14, 7675.51it/s] 72%|███████▏  | 288104/400000 [00:36<00:14, 7524.59it/s] 72%|███████▏  | 288859/400000 [00:36<00:14, 7473.40it/s] 72%|███████▏  | 289644/400000 [00:36<00:14, 7582.24it/s] 73%|███████▎  | 290472/400000 [00:36<00:14, 7776.99it/s] 73%|███████▎  | 291303/400000 [00:37<00:13, 7928.55it/s] 73%|███████▎  | 292099/400000 [00:37<00:13, 7875.77it/s] 73%|███████▎  | 292931/400000 [00:37<00:13, 8001.68it/s] 73%|███████▎  | 293733/400000 [00:37<00:13, 7913.29it/s] 74%|███████▎  | 294546/400000 [00:37<00:13, 7976.18it/s] 74%|███████▍  | 295345/400000 [00:37<00:13, 7941.38it/s] 74%|███████▍  | 296140/400000 [00:37<00:13, 7930.67it/s] 74%|███████▍  | 296954/400000 [00:37<00:12, 7991.02it/s] 74%|███████▍  | 297754/400000 [00:37<00:13, 7762.98it/s] 75%|███████▍  | 298533/400000 [00:37<00:13, 7644.17it/s] 75%|███████▍  | 299349/400000 [00:38<00:12, 7790.85it/s] 75%|███████▌  | 300130/400000 [00:38<00:12, 7790.23it/s] 75%|███████▌  | 300953/400000 [00:38<00:12, 7914.20it/s] 75%|███████▌  | 301746/400000 [00:38<00:12, 7878.72it/s] 76%|███████▌  | 302567/400000 [00:38<00:12, 7972.66it/s] 76%|███████▌  | 303366/400000 [00:38<00:12, 7955.87it/s] 76%|███████▌  | 304163/400000 [00:38<00:12, 7853.85it/s] 76%|███████▌  | 304975/400000 [00:38<00:11, 7931.43it/s] 76%|███████▋  | 305816/400000 [00:38<00:11, 8067.97it/s] 77%|███████▋  | 306657/400000 [00:38<00:11, 8164.44it/s] 77%|███████▋  | 307480/400000 [00:39<00:11, 8181.83it/s] 77%|███████▋  | 308299/400000 [00:39<00:11, 8120.32it/s] 77%|███████▋  | 309112/400000 [00:39<00:11, 8075.93it/s] 77%|███████▋  | 309921/400000 [00:39<00:11, 8050.93it/s] 78%|███████▊  | 310727/400000 [00:39<00:11, 8034.85it/s] 78%|███████▊  | 311531/400000 [00:39<00:11, 7898.18it/s] 78%|███████▊  | 312322/400000 [00:39<00:11, 7625.80it/s] 78%|███████▊  | 313121/400000 [00:39<00:11, 7728.38it/s] 78%|███████▊  | 313942/400000 [00:39<00:10, 7865.33it/s] 79%|███████▊  | 314740/400000 [00:39<00:10, 7897.07it/s] 79%|███████▉  | 315535/400000 [00:40<00:10, 7911.70it/s] 79%|███████▉  | 316348/400000 [00:40<00:10, 7975.14it/s] 79%|███████▉  | 317164/400000 [00:40<00:10, 8028.05it/s] 79%|███████▉  | 317993/400000 [00:40<00:10, 8103.62it/s] 80%|███████▉  | 318819/400000 [00:40<00:09, 8149.68it/s] 80%|███████▉  | 319635/400000 [00:40<00:09, 8051.19it/s] 80%|████████  | 320441/400000 [00:40<00:10, 7864.37it/s] 80%|████████  | 321229/400000 [00:40<00:10, 7800.06it/s] 81%|████████  | 322011/400000 [00:40<00:10, 7794.18it/s] 81%|████████  | 322792/400000 [00:40<00:09, 7738.73it/s] 81%|████████  | 323622/400000 [00:41<00:09, 7894.11it/s] 81%|████████  | 324413/400000 [00:41<00:09, 7774.92it/s] 81%|████████▏ | 325192/400000 [00:41<00:09, 7603.29it/s] 81%|████████▏ | 325955/400000 [00:41<00:09, 7573.76it/s] 82%|████████▏ | 326758/400000 [00:41<00:09, 7703.80it/s] 82%|████████▏ | 327543/400000 [00:41<00:09, 7746.00it/s] 82%|████████▏ | 328319/400000 [00:41<00:09, 7566.91it/s] 82%|████████▏ | 329112/400000 [00:41<00:09, 7671.38it/s] 82%|████████▏ | 329902/400000 [00:41<00:09, 7737.52it/s] 83%|████████▎ | 330692/400000 [00:42<00:08, 7783.54it/s] 83%|████████▎ | 331512/400000 [00:42<00:08, 7902.76it/s] 83%|████████▎ | 332326/400000 [00:42<00:08, 7970.95it/s] 83%|████████▎ | 333124/400000 [00:42<00:08, 7944.51it/s] 83%|████████▎ | 333920/400000 [00:42<00:08, 7906.17it/s] 84%|████████▎ | 334712/400000 [00:42<00:08, 7804.18it/s] 84%|████████▍ | 335530/400000 [00:42<00:08, 7910.87it/s] 84%|████████▍ | 336322/400000 [00:42<00:08, 7782.69it/s] 84%|████████▍ | 337108/400000 [00:42<00:08, 7804.72it/s] 84%|████████▍ | 337890/400000 [00:42<00:07, 7804.70it/s] 85%|████████▍ | 338671/400000 [00:43<00:07, 7677.62it/s] 85%|████████▍ | 339454/400000 [00:43<00:07, 7720.76it/s] 85%|████████▌ | 340257/400000 [00:43<00:07, 7808.10it/s] 85%|████████▌ | 341075/400000 [00:43<00:07, 7915.69it/s] 85%|████████▌ | 341887/400000 [00:43<00:07, 7974.49it/s] 86%|████████▌ | 342686/400000 [00:43<00:07, 7863.69it/s] 86%|████████▌ | 343487/400000 [00:43<00:07, 7905.14it/s] 86%|████████▌ | 344280/400000 [00:43<00:07, 7911.82it/s] 86%|████████▋ | 345103/400000 [00:43<00:06, 8003.67it/s] 86%|████████▋ | 345911/400000 [00:43<00:06, 8024.72it/s] 87%|████████▋ | 346714/400000 [00:44<00:06, 7941.34it/s] 87%|████████▋ | 347509/400000 [00:44<00:06, 7857.08it/s] 87%|████████▋ | 348296/400000 [00:44<00:06, 7844.90it/s] 87%|████████▋ | 349097/400000 [00:44<00:06, 7892.01it/s] 87%|████████▋ | 349887/400000 [00:44<00:06, 7892.46it/s] 88%|████████▊ | 350677/400000 [00:44<00:06, 7771.74it/s] 88%|████████▊ | 351455/400000 [00:44<00:06, 7595.58it/s] 88%|████████▊ | 352254/400000 [00:44<00:06, 7707.07it/s] 88%|████████▊ | 353074/400000 [00:44<00:05, 7848.52it/s] 88%|████████▊ | 353871/400000 [00:44<00:05, 7884.02it/s] 89%|████████▊ | 354707/400000 [00:45<00:05, 8018.83it/s] 89%|████████▉ | 355511/400000 [00:45<00:05, 7986.72it/s] 89%|████████▉ | 356333/400000 [00:45<00:05, 8053.01it/s] 89%|████████▉ | 357151/400000 [00:45<00:05, 8090.58it/s] 89%|████████▉ | 357964/400000 [00:45<00:05, 8099.24it/s] 90%|████████▉ | 358775/400000 [00:45<00:05, 8065.16it/s] 90%|████████▉ | 359582/400000 [00:45<00:05, 7853.02it/s] 90%|█████████ | 360369/400000 [00:45<00:05, 7687.31it/s] 90%|█████████ | 361178/400000 [00:45<00:04, 7802.91it/s] 90%|█████████ | 361984/400000 [00:45<00:04, 7876.29it/s] 91%|█████████ | 362777/400000 [00:46<00:04, 7890.87it/s] 91%|█████████ | 363591/400000 [00:46<00:04, 7962.70it/s] 91%|█████████ | 364389/400000 [00:46<00:04, 7864.81it/s] 91%|█████████▏| 365218/400000 [00:46<00:04, 7986.71it/s] 92%|█████████▏| 366036/400000 [00:46<00:04, 8043.66it/s] 92%|█████████▏| 366842/400000 [00:46<00:04, 8046.23it/s] 92%|█████████▏| 367648/400000 [00:46<00:04, 7954.06it/s] 92%|█████████▏| 368445/400000 [00:46<00:04, 7862.68it/s] 92%|█████████▏| 369232/400000 [00:46<00:04, 7653.48it/s] 92%|█████████▎| 370000/400000 [00:47<00:03, 7565.16it/s] 93%|█████████▎| 370798/400000 [00:47<00:03, 7680.43it/s] 93%|█████████▎| 371601/400000 [00:47<00:03, 7779.53it/s] 93%|█████████▎| 372381/400000 [00:47<00:03, 7721.73it/s] 93%|█████████▎| 373159/400000 [00:47<00:03, 7737.00it/s] 93%|█████████▎| 373965/400000 [00:47<00:03, 7830.05it/s] 94%|█████████▎| 374789/400000 [00:47<00:03, 7946.52it/s] 94%|█████████▍| 375585/400000 [00:47<00:03, 7868.85it/s] 94%|█████████▍| 376407/400000 [00:47<00:02, 7969.80it/s] 94%|█████████▍| 377205/400000 [00:47<00:02, 7825.22it/s] 94%|█████████▍| 377989/400000 [00:48<00:02, 7817.36it/s] 95%|█████████▍| 378818/400000 [00:48<00:02, 7952.09it/s] 95%|█████████▍| 379615/400000 [00:48<00:02, 7909.09it/s] 95%|█████████▌| 380407/400000 [00:48<00:02, 7841.52it/s] 95%|█████████▌| 381192/400000 [00:48<00:02, 7774.16it/s] 95%|█████████▌| 381993/400000 [00:48<00:02, 7843.16it/s] 96%|█████████▌| 382778/400000 [00:48<00:02, 7752.65it/s] 96%|█████████▌| 383569/400000 [00:48<00:02, 7798.01it/s] 96%|█████████▌| 384366/400000 [00:48<00:01, 7847.25it/s] 96%|█████████▋| 385174/400000 [00:48<00:01, 7915.07it/s] 96%|█████████▋| 386000/400000 [00:49<00:01, 8014.08it/s] 97%|█████████▋| 386808/400000 [00:49<00:01, 8033.25it/s] 97%|█████████▋| 387612/400000 [00:49<00:01, 7947.44it/s] 97%|█████████▋| 388408/400000 [00:49<00:01, 7875.11it/s] 97%|█████████▋| 389203/400000 [00:49<00:01, 7896.80it/s] 97%|█████████▋| 389994/400000 [00:49<00:01, 7840.96it/s] 98%|█████████▊| 390779/400000 [00:49<00:01, 7565.47it/s] 98%|█████████▊| 391538/400000 [00:49<00:01, 7421.23it/s] 98%|█████████▊| 392319/400000 [00:49<00:01, 7532.40it/s] 98%|█████████▊| 393122/400000 [00:49<00:00, 7674.86it/s] 98%|█████████▊| 393892/400000 [00:50<00:00, 7631.84it/s] 99%|█████████▊| 394705/400000 [00:50<00:00, 7773.37it/s] 99%|█████████▉| 395485/400000 [00:50<00:00, 7752.51it/s] 99%|█████████▉| 396282/400000 [00:50<00:00, 7814.26it/s] 99%|█████████▉| 397073/400000 [00:50<00:00, 7841.67it/s] 99%|█████████▉| 397858/400000 [00:50<00:00, 7672.32it/s]100%|█████████▉| 398654/400000 [00:50<00:00, 7753.51it/s]100%|█████████▉| 399449/400000 [00:50<00:00, 7809.01it/s]100%|█████████▉| 399999/400000 [00:50<00:00, 7868.46it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f94ea4b07b8>, <torchtext.data.dataset.TabularDataset object at 0x7f94ea4b0908>, <torchtext.vocab.Vocab object at 0x7f94ea4b0828>), {}) 

