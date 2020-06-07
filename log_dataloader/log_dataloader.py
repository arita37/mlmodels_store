
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'a4f6eb9a7161522868de1f62953382979dca62d3', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/a4f6eb9a7161522868de1f62953382979dca62d3

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f352dd51620> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f352dd51620> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f35993180d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f35993180d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f35b3045ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f35b3045ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f3546b93950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f3546b93950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f3546b93950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 39%|███▉      | 3874816/9912422 [00:00<00:00, 36875538.87it/s]9920512it [00:00, 33722792.51it/s]                             
0it [00:00, ?it/s]32768it [00:00, 460148.36it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 7900579.68it/s]          
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 73096.03it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f35440d5b00>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f35440d9da0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f3546b93598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f3546b93598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f3546b93598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:54,  2.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:54,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:53,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:53,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:53,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:52,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:52,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:37,  4.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:37,  4.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:37,  4.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:36,  4.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:36,  4.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:36,  4.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:36,  4.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:35,  4.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:35,  4.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:25,  5.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:25,  5.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:25,  5.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:25,  5.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:24,  5.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:24,  5.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:24,  5.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:24,  5.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:24,  5.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:00<00:17,  8.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:17,  8.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:17,  8.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:17,  8.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:16,  8.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:16,  8.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:16,  8.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:16,  8.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:16,  8.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:16,  8.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:00<00:11, 11.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:11, 11.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:11, 11.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:11, 11.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:11, 11.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:11, 11.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:11, 11.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:11, 11.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:11, 11.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:00<00:08, 14.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:08, 14.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:08, 14.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:08, 14.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:08, 14.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:07, 14.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:07, 14.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:07, 14.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:07, 14.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:00<00:05, 19.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:05, 19.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:05, 19.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:05, 19.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 19.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 19.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 19.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 19.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 19.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 25.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 25.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 25.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 25.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 25.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 25.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 25.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 25.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 25.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 31.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 31.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 31.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 31.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 31.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 31.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 31.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 31.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 31.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 36.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 36.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 36.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 36.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 36.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 36.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 36.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 36.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 42.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 42.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 42.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 42.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 42.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 42.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 42.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 42.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 47.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 47.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 47.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 47.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 47.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 47.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 47.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 47.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 51.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 51.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 51.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 51.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 51.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 51.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 51.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 51.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 54.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 54.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 54.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 54.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 54.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 54.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 54.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 54.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 55.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 55.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 55.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 55.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 55.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 55.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 55.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 55.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 58.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 58.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 58.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 58.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 58.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 58.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 58.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 58.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 60.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 60.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 60.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 60.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 60.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 60.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 60.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 60.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 60.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 60.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 60.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 60.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 60.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 60.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 60.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 60.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 61.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 61.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 61.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 61.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 61.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 61.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 61.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 61.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 62.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 62.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 62.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 62.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 62.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 62.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 62.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 62.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 62.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 62.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 62.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 62.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 62.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 62.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 62.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 62.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 62.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 62.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 62.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 62.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 62.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 62.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 62.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 62.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.72s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.72s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 62.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.72s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 62.31 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.14s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.72s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 62.31 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.14s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.14s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 31.50 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.14s/ url]
0 examples [00:00, ? examples/s]2020-06-07 00:09:59.490547: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-07 00:09:59.505286: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-06-07 00:09:59.505478: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d74fe9e6a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-07 00:09:59.505501: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
22 examples [00:00, 218.61 examples/s]118 examples [00:00, 284.45 examples/s]210 examples [00:00, 358.80 examples/s]297 examples [00:00, 435.53 examples/s]389 examples [00:00, 516.91 examples/s]482 examples [00:00, 596.20 examples/s]577 examples [00:00, 670.80 examples/s]673 examples [00:00, 737.25 examples/s]765 examples [00:00, 783.85 examples/s]854 examples [00:01, 812.58 examples/s]949 examples [00:01, 848.79 examples/s]1040 examples [00:01, 859.75 examples/s]1133 examples [00:01, 878.94 examples/s]1228 examples [00:01, 896.56 examples/s]1320 examples [00:01, 901.28 examples/s]1414 examples [00:01, 910.69 examples/s]1508 examples [00:01, 918.94 examples/s]1604 examples [00:01, 929.80 examples/s]1698 examples [00:01, 906.16 examples/s]1791 examples [00:02, 911.67 examples/s]1883 examples [00:02, 909.30 examples/s]1977 examples [00:02, 916.47 examples/s]2073 examples [00:02, 926.87 examples/s]2168 examples [00:02, 931.48 examples/s]2262 examples [00:02, 920.38 examples/s]2355 examples [00:02, 891.48 examples/s]2445 examples [00:02, 880.58 examples/s]2540 examples [00:02, 898.55 examples/s]2632 examples [00:02, 903.34 examples/s]2724 examples [00:03, 907.46 examples/s]2817 examples [00:03, 912.16 examples/s]2910 examples [00:03, 916.45 examples/s]3005 examples [00:03, 925.55 examples/s]3098 examples [00:03, 887.27 examples/s]3192 examples [00:03, 900.71 examples/s]3286 examples [00:03, 912.14 examples/s]3378 examples [00:03, 896.73 examples/s]3472 examples [00:03, 906.66 examples/s]3563 examples [00:03, 896.92 examples/s]3656 examples [00:04, 905.86 examples/s]3747 examples [00:04, 890.12 examples/s]3837 examples [00:04, 891.93 examples/s]3927 examples [00:04, 894.29 examples/s]4019 examples [00:04, 901.18 examples/s]4114 examples [00:04, 913.23 examples/s]4208 examples [00:04, 919.33 examples/s]4301 examples [00:04, 920.82 examples/s]4394 examples [00:04, 913.56 examples/s]4487 examples [00:04, 918.39 examples/s]4580 examples [00:05, 919.80 examples/s]4674 examples [00:05, 924.81 examples/s]4767 examples [00:05, 917.18 examples/s]4860 examples [00:05, 918.50 examples/s]4952 examples [00:05, 915.10 examples/s]5044 examples [00:05, 897.82 examples/s]5138 examples [00:05, 909.27 examples/s]5232 examples [00:05, 917.70 examples/s]5327 examples [00:05, 925.37 examples/s]5420 examples [00:06, 899.10 examples/s]5514 examples [00:06, 910.73 examples/s]5609 examples [00:06, 920.22 examples/s]5702 examples [00:06, 919.52 examples/s]5797 examples [00:06, 927.40 examples/s]5890 examples [00:06, 924.10 examples/s]5984 examples [00:06, 926.62 examples/s]6079 examples [00:06, 930.97 examples/s]6174 examples [00:06, 935.12 examples/s]6269 examples [00:06, 937.81 examples/s]6363 examples [00:07, 932.53 examples/s]6457 examples [00:07, 926.11 examples/s]6551 examples [00:07, 929.77 examples/s]6646 examples [00:07, 933.32 examples/s]6741 examples [00:07, 937.66 examples/s]6835 examples [00:07, 931.11 examples/s]6929 examples [00:07, 922.16 examples/s]7024 examples [00:07, 929.20 examples/s]7118 examples [00:07, 930.16 examples/s]7214 examples [00:07, 936.28 examples/s]7308 examples [00:08, 931.28 examples/s]7403 examples [00:08, 934.58 examples/s]7498 examples [00:08, 936.85 examples/s]7592 examples [00:08, 922.47 examples/s]7685 examples [00:08, 904.02 examples/s]7777 examples [00:08, 908.30 examples/s]7868 examples [00:08, 894.17 examples/s]7962 examples [00:08, 906.40 examples/s]8056 examples [00:08, 915.04 examples/s]8150 examples [00:08, 920.94 examples/s]8243 examples [00:09, 908.72 examples/s]8337 examples [00:09, 916.52 examples/s]8431 examples [00:09, 922.76 examples/s]8524 examples [00:09, 924.68 examples/s]8619 examples [00:09, 931.32 examples/s]8713 examples [00:09, 922.45 examples/s]8807 examples [00:09, 925.57 examples/s]8901 examples [00:09, 929.13 examples/s]8996 examples [00:09, 932.85 examples/s]9090 examples [00:09, 916.06 examples/s]9182 examples [00:10, 917.06 examples/s]9275 examples [00:10, 919.30 examples/s]9368 examples [00:10, 921.19 examples/s]9461 examples [00:10, 918.65 examples/s]9553 examples [00:10, 897.88 examples/s]9644 examples [00:10, 898.90 examples/s]9734 examples [00:10, 887.22 examples/s]9824 examples [00:10, 889.61 examples/s]9914 examples [00:10, 886.08 examples/s]10003 examples [00:11, 823.79 examples/s]10092 examples [00:11, 842.45 examples/s]10186 examples [00:11, 867.82 examples/s]10281 examples [00:11, 888.91 examples/s]10371 examples [00:11, 888.41 examples/s]10461 examples [00:11, 891.73 examples/s]10551 examples [00:11, 893.06 examples/s]10641 examples [00:11, 880.38 examples/s]10734 examples [00:11, 894.70 examples/s]10827 examples [00:11, 902.06 examples/s]10922 examples [00:12, 913.81 examples/s]11014 examples [00:12, 904.33 examples/s]11107 examples [00:12, 911.80 examples/s]11202 examples [00:12, 920.96 examples/s]11297 examples [00:12, 927.03 examples/s]11392 examples [00:12, 930.99 examples/s]11486 examples [00:12, 927.58 examples/s]11580 examples [00:12, 930.73 examples/s]11674 examples [00:12, 929.26 examples/s]11768 examples [00:12, 931.80 examples/s]11863 examples [00:13, 934.55 examples/s]11957 examples [00:13, 925.73 examples/s]12051 examples [00:13, 928.79 examples/s]12144 examples [00:13, 928.68 examples/s]12237 examples [00:13, 924.49 examples/s]12330 examples [00:13, 915.33 examples/s]12422 examples [00:13, 915.21 examples/s]12516 examples [00:13, 919.88 examples/s]12613 examples [00:13, 933.36 examples/s]12709 examples [00:13, 940.32 examples/s]12805 examples [00:14, 943.74 examples/s]12900 examples [00:14, 926.99 examples/s]12993 examples [00:14, 927.69 examples/s]13086 examples [00:14, 928.12 examples/s]13179 examples [00:14, 921.70 examples/s]13274 examples [00:14, 926.82 examples/s]13367 examples [00:14, 899.52 examples/s]13458 examples [00:14, 868.08 examples/s]13553 examples [00:14, 889.05 examples/s]13647 examples [00:14, 902.61 examples/s]13739 examples [00:15, 906.82 examples/s]13833 examples [00:15, 914.61 examples/s]13928 examples [00:15, 923.82 examples/s]14024 examples [00:15, 931.53 examples/s]14119 examples [00:15, 934.46 examples/s]14214 examples [00:15, 935.46 examples/s]14308 examples [00:15, 926.65 examples/s]14403 examples [00:15, 930.99 examples/s]14498 examples [00:15, 935.19 examples/s]14592 examples [00:16, 915.64 examples/s]14684 examples [00:16, 914.69 examples/s]14776 examples [00:16, 909.65 examples/s]14871 examples [00:16, 919.93 examples/s]14966 examples [00:16, 926.29 examples/s]15060 examples [00:16, 929.78 examples/s]15154 examples [00:16, 930.76 examples/s]15248 examples [00:16, 915.90 examples/s]15340 examples [00:16, 914.80 examples/s]15433 examples [00:16, 918.75 examples/s]15527 examples [00:17, 923.67 examples/s]15620 examples [00:17, 871.09 examples/s]15711 examples [00:17, 880.13 examples/s]15802 examples [00:17, 886.15 examples/s]15896 examples [00:17, 899.83 examples/s]15989 examples [00:17, 908.55 examples/s]16081 examples [00:17, 907.12 examples/s]16172 examples [00:17, 900.14 examples/s]16263 examples [00:17, 884.34 examples/s]16357 examples [00:17, 900.15 examples/s]16451 examples [00:18, 909.52 examples/s]16543 examples [00:18, 885.20 examples/s]16637 examples [00:18, 899.37 examples/s]16731 examples [00:18, 910.85 examples/s]16823 examples [00:18, 905.17 examples/s]16914 examples [00:18, 897.94 examples/s]17007 examples [00:18, 907.05 examples/s]17100 examples [00:18, 912.29 examples/s]17192 examples [00:18, 914.40 examples/s]17287 examples [00:18, 923.57 examples/s]17381 examples [00:19, 927.20 examples/s]17474 examples [00:19, 912.44 examples/s]17568 examples [00:19, 919.16 examples/s]17660 examples [00:19, 915.47 examples/s]17754 examples [00:19, 921.15 examples/s]17848 examples [00:19, 923.98 examples/s]17941 examples [00:19, 901.89 examples/s]18036 examples [00:19, 913.92 examples/s]18128 examples [00:19, 883.63 examples/s]18222 examples [00:20, 898.95 examples/s]18316 examples [00:20, 908.81 examples/s]18408 examples [00:20, 910.31 examples/s]18501 examples [00:20, 913.62 examples/s]18593 examples [00:20, 911.92 examples/s]18686 examples [00:20, 914.89 examples/s]18778 examples [00:20, 907.16 examples/s]18869 examples [00:20, 905.51 examples/s]18960 examples [00:20, 889.39 examples/s]19052 examples [00:20, 896.62 examples/s]19145 examples [00:21, 905.59 examples/s]19236 examples [00:21, 891.60 examples/s]19326 examples [00:21, 892.46 examples/s]19420 examples [00:21, 905.72 examples/s]19514 examples [00:21, 913.58 examples/s]19609 examples [00:21, 922.98 examples/s]19703 examples [00:21, 925.36 examples/s]19796 examples [00:21, 917.36 examples/s]19891 examples [00:21, 925.07 examples/s]19985 examples [00:21, 929.06 examples/s]20078 examples [00:22, 892.22 examples/s]20168 examples [00:22, 894.52 examples/s]20258 examples [00:22, 893.58 examples/s]20353 examples [00:22, 907.17 examples/s]20448 examples [00:22, 917.35 examples/s]20542 examples [00:22, 922.72 examples/s]20635 examples [00:22, 920.13 examples/s]20728 examples [00:22, 912.66 examples/s]20820 examples [00:22, 883.29 examples/s]20914 examples [00:22, 898.78 examples/s]21009 examples [00:23, 912.19 examples/s]21103 examples [00:23, 919.91 examples/s]21196 examples [00:23, 913.31 examples/s]21288 examples [00:23, 911.97 examples/s]21382 examples [00:23, 919.55 examples/s]21475 examples [00:23, 921.01 examples/s]21568 examples [00:23, 905.30 examples/s]21659 examples [00:23, 899.52 examples/s]21750 examples [00:23, 874.07 examples/s]21843 examples [00:24, 888.28 examples/s]21938 examples [00:24, 904.46 examples/s]22030 examples [00:24, 906.96 examples/s]22121 examples [00:24, 907.73 examples/s]22212 examples [00:24, 908.04 examples/s]22303 examples [00:24, 902.67 examples/s]22394 examples [00:24, 903.72 examples/s]22485 examples [00:24, 892.60 examples/s]22575 examples [00:24, 893.99 examples/s]22667 examples [00:24, 901.54 examples/s]22762 examples [00:25, 913.12 examples/s]22854 examples [00:25, 913.42 examples/s]22946 examples [00:25, 870.87 examples/s]23034 examples [00:25, 851.28 examples/s]23123 examples [00:25, 860.84 examples/s]23210 examples [00:25, 852.70 examples/s]23300 examples [00:25, 866.12 examples/s]23390 examples [00:25, 874.16 examples/s]23478 examples [00:25, 869.74 examples/s]23571 examples [00:25, 885.59 examples/s]23665 examples [00:26, 900.46 examples/s]23760 examples [00:26, 912.85 examples/s]23852 examples [00:26, 907.11 examples/s]23943 examples [00:26, 894.05 examples/s]24037 examples [00:26, 905.97 examples/s]24130 examples [00:26, 911.29 examples/s]24224 examples [00:26, 917.53 examples/s]24316 examples [00:26, 915.33 examples/s]24408 examples [00:26, 911.19 examples/s]24500 examples [00:26, 894.41 examples/s]24594 examples [00:27, 905.76 examples/s]24685 examples [00:27, 906.09 examples/s]24778 examples [00:27, 910.60 examples/s]24871 examples [00:27, 915.50 examples/s]24966 examples [00:27, 924.09 examples/s]25060 examples [00:27, 926.20 examples/s]25153 examples [00:27, 925.62 examples/s]25246 examples [00:27, 872.39 examples/s]25336 examples [00:27, 878.53 examples/s]25427 examples [00:28, 882.83 examples/s]25520 examples [00:28, 896.04 examples/s]25611 examples [00:28, 900.18 examples/s]25702 examples [00:28, 892.58 examples/s]25792 examples [00:28, 880.89 examples/s]25884 examples [00:28, 891.53 examples/s]25977 examples [00:28, 900.99 examples/s]26068 examples [00:28, 899.84 examples/s]26159 examples [00:28, 862.41 examples/s]26253 examples [00:28, 882.10 examples/s]26346 examples [00:29, 893.35 examples/s]26440 examples [00:29, 905.61 examples/s]26533 examples [00:29, 910.11 examples/s]26625 examples [00:29, 908.83 examples/s]26717 examples [00:29, 911.97 examples/s]26809 examples [00:29, 913.79 examples/s]26901 examples [00:29, 915.38 examples/s]26993 examples [00:29, 901.47 examples/s]27084 examples [00:29, 856.89 examples/s]27175 examples [00:29, 870.56 examples/s]27263 examples [00:30, 861.90 examples/s]27356 examples [00:30, 879.50 examples/s]27447 examples [00:30, 887.52 examples/s]27536 examples [00:30, 862.13 examples/s]27628 examples [00:30, 875.67 examples/s]27719 examples [00:30, 883.53 examples/s]27811 examples [00:30, 890.34 examples/s]27901 examples [00:30, 875.31 examples/s]27992 examples [00:30, 883.38 examples/s]28085 examples [00:30, 894.39 examples/s]28178 examples [00:31, 904.36 examples/s]28270 examples [00:31, 907.86 examples/s]28361 examples [00:31, 886.83 examples/s]28450 examples [00:31, 878.28 examples/s]28543 examples [00:31, 891.35 examples/s]28635 examples [00:31, 899.34 examples/s]28726 examples [00:31, 888.15 examples/s]28818 examples [00:31, 897.23 examples/s]28908 examples [00:31, 894.09 examples/s]28998 examples [00:32, 893.02 examples/s]29088 examples [00:32, 893.32 examples/s]29180 examples [00:32, 899.34 examples/s]29270 examples [00:32, 896.59 examples/s]29360 examples [00:32, 896.44 examples/s]29451 examples [00:32, 897.92 examples/s]29542 examples [00:32, 900.13 examples/s]29634 examples [00:32, 903.46 examples/s]29725 examples [00:32, 888.76 examples/s]29814 examples [00:32, 882.92 examples/s]29903 examples [00:33, 872.73 examples/s]29991 examples [00:33, 866.39 examples/s]30078 examples [00:33, 819.74 examples/s]30167 examples [00:33, 837.31 examples/s]30253 examples [00:33, 842.08 examples/s]30345 examples [00:33, 861.32 examples/s]30437 examples [00:33, 875.84 examples/s]30527 examples [00:33, 882.51 examples/s]30617 examples [00:33, 887.38 examples/s]30706 examples [00:33, 867.71 examples/s]30793 examples [00:34, 867.80 examples/s]30886 examples [00:34, 883.23 examples/s]30979 examples [00:34, 896.61 examples/s]31069 examples [00:34, 882.20 examples/s]31160 examples [00:34, 887.95 examples/s]31254 examples [00:34, 900.44 examples/s]31345 examples [00:34, 851.31 examples/s]31434 examples [00:34, 861.13 examples/s]31522 examples [00:34, 864.69 examples/s]31609 examples [00:34, 855.06 examples/s]31704 examples [00:35, 879.47 examples/s]31798 examples [00:35, 896.28 examples/s]31891 examples [00:35, 904.25 examples/s]31982 examples [00:35, 886.91 examples/s]32072 examples [00:35, 889.40 examples/s]32166 examples [00:35, 901.87 examples/s]32260 examples [00:35, 912.10 examples/s]32354 examples [00:35, 917.61 examples/s]32447 examples [00:35, 919.73 examples/s]32540 examples [00:36, 908.17 examples/s]32633 examples [00:36, 914.41 examples/s]32728 examples [00:36, 924.42 examples/s]32823 examples [00:36, 930.14 examples/s]32917 examples [00:36, 922.16 examples/s]33010 examples [00:36, 916.69 examples/s]33102 examples [00:36, 913.20 examples/s]33194 examples [00:36, 910.75 examples/s]33286 examples [00:36, 911.04 examples/s]33378 examples [00:36, 877.57 examples/s]33468 examples [00:37, 882.10 examples/s]33560 examples [00:37, 891.95 examples/s]33654 examples [00:37, 904.05 examples/s]33748 examples [00:37, 913.96 examples/s]33840 examples [00:37, 886.88 examples/s]33933 examples [00:37, 897.88 examples/s]34027 examples [00:37, 909.55 examples/s]34121 examples [00:37, 915.92 examples/s]34214 examples [00:37, 917.74 examples/s]34309 examples [00:37, 925.35 examples/s]34402 examples [00:38, 918.41 examples/s]34497 examples [00:38, 927.23 examples/s]34591 examples [00:38, 928.87 examples/s]34684 examples [00:38, 910.33 examples/s]34776 examples [00:38, 903.66 examples/s]34867 examples [00:38, 904.71 examples/s]34962 examples [00:38, 917.72 examples/s]35056 examples [00:38, 923.97 examples/s]35149 examples [00:38, 920.98 examples/s]35242 examples [00:38, 922.58 examples/s]35335 examples [00:39, 897.01 examples/s]35429 examples [00:39, 906.84 examples/s]35520 examples [00:39, 906.14 examples/s]35613 examples [00:39, 911.76 examples/s]35706 examples [00:39, 916.37 examples/s]35798 examples [00:39, 902.45 examples/s]35889 examples [00:39, 903.73 examples/s]35982 examples [00:39, 908.84 examples/s]36073 examples [00:39, 881.32 examples/s]36165 examples [00:40, 892.31 examples/s]36255 examples [00:40, 893.37 examples/s]36345 examples [00:40, 892.82 examples/s]36439 examples [00:40, 906.36 examples/s]36531 examples [00:40, 908.08 examples/s]36622 examples [00:40, 856.47 examples/s]36714 examples [00:40, 871.97 examples/s]36806 examples [00:40, 885.54 examples/s]36900 examples [00:40, 900.38 examples/s]36991 examples [00:40, 898.67 examples/s]37082 examples [00:41, 898.76 examples/s]37175 examples [00:41, 906.17 examples/s]37269 examples [00:41, 914.37 examples/s]37363 examples [00:41, 921.09 examples/s]37457 examples [00:41, 924.89 examples/s]37550 examples [00:41, 919.52 examples/s]37643 examples [00:41, 903.66 examples/s]37736 examples [00:41, 908.96 examples/s]37828 examples [00:41, 911.66 examples/s]37922 examples [00:41, 917.79 examples/s]38014 examples [00:42, 916.57 examples/s]38106 examples [00:42, 903.04 examples/s]38200 examples [00:42, 912.94 examples/s]38294 examples [00:42, 919.67 examples/s]38387 examples [00:42, 913.36 examples/s]38479 examples [00:42, 914.94 examples/s]38573 examples [00:42, 920.72 examples/s]38666 examples [00:42, 923.10 examples/s]38760 examples [00:42, 926.71 examples/s]38853 examples [00:42, 919.46 examples/s]38945 examples [00:43, 919.33 examples/s]39041 examples [00:43, 930.12 examples/s]39135 examples [00:43, 932.07 examples/s]39232 examples [00:43, 940.87 examples/s]39327 examples [00:43, 930.76 examples/s]39421 examples [00:43, 900.70 examples/s]39515 examples [00:43, 910.74 examples/s]39607 examples [00:43, 897.17 examples/s]39700 examples [00:43, 905.94 examples/s]39793 examples [00:43, 912.30 examples/s]39885 examples [00:44, 908.47 examples/s]39976 examples [00:44, 905.33 examples/s]40067 examples [00:44, 873.15 examples/s]40155 examples [00:44, 866.50 examples/s]40250 examples [00:44, 888.75 examples/s]40343 examples [00:44, 899.54 examples/s]40434 examples [00:44, 897.08 examples/s]40528 examples [00:44, 907.76 examples/s]40622 examples [00:44, 915.06 examples/s]40717 examples [00:45, 922.36 examples/s]40810 examples [00:45, 909.36 examples/s]40902 examples [00:45, 904.75 examples/s]40993 examples [00:45, 903.34 examples/s]41088 examples [00:45, 914.72 examples/s]41182 examples [00:45, 921.50 examples/s]41275 examples [00:45, 917.76 examples/s]41367 examples [00:45, 876.80 examples/s]41461 examples [00:45, 892.65 examples/s]41555 examples [00:45, 905.61 examples/s]41649 examples [00:46, 913.26 examples/s]41741 examples [00:46, 913.69 examples/s]41834 examples [00:46, 918.52 examples/s]41927 examples [00:46, 919.02 examples/s]42019 examples [00:46, 909.19 examples/s]42111 examples [00:46, 892.15 examples/s]42202 examples [00:46, 895.31 examples/s]42292 examples [00:46, 871.62 examples/s]42387 examples [00:46, 891.79 examples/s]42480 examples [00:46, 902.13 examples/s]42574 examples [00:47, 912.65 examples/s]42666 examples [00:47, 909.25 examples/s]42760 examples [00:47, 916.88 examples/s]42855 examples [00:47, 924.09 examples/s]42949 examples [00:47, 928.13 examples/s]43043 examples [00:47, 928.88 examples/s]43136 examples [00:47, 901.60 examples/s]43227 examples [00:47, 888.95 examples/s]43318 examples [00:47, 894.70 examples/s]43408 examples [00:47, 894.46 examples/s]43500 examples [00:48, 901.01 examples/s]43592 examples [00:48, 904.55 examples/s]43686 examples [00:48, 913.27 examples/s]43778 examples [00:48, 914.43 examples/s]43870 examples [00:48, 909.30 examples/s]43961 examples [00:48, 875.99 examples/s]44053 examples [00:48, 886.20 examples/s]44145 examples [00:48, 893.60 examples/s]44239 examples [00:48, 905.59 examples/s]44330 examples [00:49, 903.34 examples/s]44421 examples [00:49, 900.55 examples/s]44512 examples [00:49, 895.14 examples/s]44602 examples [00:49, 895.40 examples/s]44693 examples [00:49, 899.51 examples/s]44787 examples [00:49, 908.88 examples/s]44878 examples [00:49, 883.53 examples/s]44967 examples [00:49, 879.10 examples/s]45059 examples [00:49, 889.60 examples/s]45150 examples [00:49, 894.07 examples/s]45244 examples [00:50, 906.38 examples/s]45339 examples [00:50, 917.62 examples/s]45431 examples [00:50, 913.97 examples/s]45525 examples [00:50, 921.09 examples/s]45619 examples [00:50, 926.68 examples/s]45712 examples [00:50, 913.06 examples/s]45804 examples [00:50, 893.36 examples/s]45894 examples [00:50, 886.55 examples/s]45986 examples [00:50, 895.14 examples/s]46080 examples [00:50, 905.87 examples/s]46174 examples [00:51, 913.73 examples/s]46267 examples [00:51, 915.84 examples/s]46359 examples [00:51, 910.08 examples/s]46453 examples [00:51, 917.27 examples/s]46547 examples [00:51, 922.50 examples/s]46640 examples [00:51, 923.37 examples/s]46733 examples [00:51, 922.31 examples/s]46826 examples [00:51, 908.88 examples/s]46917 examples [00:51, 898.82 examples/s]47012 examples [00:51, 910.97 examples/s]47104 examples [00:52, 895.44 examples/s]47194 examples [00:52, 888.93 examples/s]47286 examples [00:52, 895.63 examples/s]47378 examples [00:52, 901.13 examples/s]47469 examples [00:52, 902.77 examples/s]47562 examples [00:52, 909.22 examples/s]47653 examples [00:52, 865.15 examples/s]47740 examples [00:52, 865.60 examples/s]47832 examples [00:52, 880.31 examples/s]47921 examples [00:52, 881.45 examples/s]48010 examples [00:53, 882.97 examples/s]48099 examples [00:53, 879.54 examples/s]48188 examples [00:53, 879.92 examples/s]48283 examples [00:53, 897.79 examples/s]48374 examples [00:53, 898.71 examples/s]48466 examples [00:53, 903.33 examples/s]48557 examples [00:53, 903.05 examples/s]48649 examples [00:53, 907.14 examples/s]48742 examples [00:53, 912.29 examples/s]48836 examples [00:54, 919.84 examples/s]48929 examples [00:54, 921.68 examples/s]49022 examples [00:54, 917.87 examples/s]49114 examples [00:54, 917.44 examples/s]49206 examples [00:54, 901.27 examples/s]49299 examples [00:54, 908.64 examples/s]49392 examples [00:54, 914.44 examples/s]49484 examples [00:54, 886.11 examples/s]49577 examples [00:54, 898.41 examples/s]49671 examples [00:54, 908.59 examples/s]49763 examples [00:55, 902.40 examples/s]49857 examples [00:55, 911.34 examples/s]49949 examples [00:55, 895.79 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6376/50000 [00:00<00:00, 63758.24 examples/s] 35%|███▌      | 17556/50000 [00:00<00:00, 73193.28 examples/s] 57%|█████▋    | 28381/50000 [00:00<00:00, 81068.43 examples/s] 79%|███████▊  | 39293/50000 [00:00<00:00, 87842.66 examples/s]                                                               0 examples [00:00, ? examples/s]80 examples [00:00, 795.08 examples/s]174 examples [00:00, 833.56 examples/s]270 examples [00:00, 867.02 examples/s]364 examples [00:00, 885.85 examples/s]458 examples [00:00, 901.06 examples/s]553 examples [00:00, 913.05 examples/s]650 examples [00:00, 927.62 examples/s]737 examples [00:00, 727.37 examples/s]829 examples [00:00, 774.12 examples/s]925 examples [00:01, 820.39 examples/s]1021 examples [00:01, 855.67 examples/s]1117 examples [00:01, 882.65 examples/s]1208 examples [00:01, 889.59 examples/s]1299 examples [00:01, 894.96 examples/s]1394 examples [00:01, 908.99 examples/s]1486 examples [00:01, 907.82 examples/s]1581 examples [00:01, 919.01 examples/s]1675 examples [00:01, 922.78 examples/s]1768 examples [00:01, 918.88 examples/s]1863 examples [00:02, 925.24 examples/s]1956 examples [00:02, 918.13 examples/s]2051 examples [00:02, 926.88 examples/s]2146 examples [00:02, 931.78 examples/s]2240 examples [00:02, 925.90 examples/s]2335 examples [00:02, 927.44 examples/s]2428 examples [00:02, 925.62 examples/s]2523 examples [00:02, 931.82 examples/s]2617 examples [00:02, 929.01 examples/s]2710 examples [00:03, 888.13 examples/s]2805 examples [00:03, 904.08 examples/s]2900 examples [00:03, 916.76 examples/s]2994 examples [00:03, 921.26 examples/s]3087 examples [00:03, 902.52 examples/s]3178 examples [00:03, 890.69 examples/s]3272 examples [00:03, 903.18 examples/s]3367 examples [00:03, 915.37 examples/s]3462 examples [00:03, 923.78 examples/s]3555 examples [00:03, 918.72 examples/s]3647 examples [00:04, 917.28 examples/s]3739 examples [00:04, 907.75 examples/s]3830 examples [00:04, 905.51 examples/s]3921 examples [00:04, 901.58 examples/s]4012 examples [00:04, 897.63 examples/s]4102 examples [00:04, 832.13 examples/s]4196 examples [00:04, 860.42 examples/s]4291 examples [00:04, 883.25 examples/s]4381 examples [00:04, 874.78 examples/s]4470 examples [00:04, 879.24 examples/s]4564 examples [00:05, 894.93 examples/s]4654 examples [00:05, 892.49 examples/s]4747 examples [00:05, 902.87 examples/s]4842 examples [00:05, 913.93 examples/s]4934 examples [00:05, 904.32 examples/s]5025 examples [00:05, 895.59 examples/s]5117 examples [00:05, 901.60 examples/s]5208 examples [00:05, 893.22 examples/s]5300 examples [00:05, 899.02 examples/s]5390 examples [00:06, 852.81 examples/s]5482 examples [00:06, 869.60 examples/s]5578 examples [00:06, 893.25 examples/s]5670 examples [00:06, 900.12 examples/s]5762 examples [00:06, 904.51 examples/s]5853 examples [00:06, 903.11 examples/s]5948 examples [00:06, 915.19 examples/s]6043 examples [00:06, 924.29 examples/s]6136 examples [00:06, 924.91 examples/s]6229 examples [00:06, 926.09 examples/s]6322 examples [00:07, 894.68 examples/s]6415 examples [00:07, 903.94 examples/s]6506 examples [00:07, 900.57 examples/s]6597 examples [00:07, 883.16 examples/s]6691 examples [00:07, 897.46 examples/s]6781 examples [00:07, 877.95 examples/s]6870 examples [00:07, 878.77 examples/s]6963 examples [00:07, 892.58 examples/s]7053 examples [00:07, 894.62 examples/s]7146 examples [00:07, 903.69 examples/s]7237 examples [00:08, 881.93 examples/s]7332 examples [00:08, 899.42 examples/s]7427 examples [00:08, 912.05 examples/s]7519 examples [00:08, 914.07 examples/s]7611 examples [00:08, 907.50 examples/s]7702 examples [00:08, 896.97 examples/s]7797 examples [00:08, 910.22 examples/s]7889 examples [00:08, 899.97 examples/s]7984 examples [00:08, 911.70 examples/s]8076 examples [00:08, 912.61 examples/s]8168 examples [00:09, 882.87 examples/s]8260 examples [00:09, 893.26 examples/s]8355 examples [00:09, 908.24 examples/s]8449 examples [00:09, 916.94 examples/s]8542 examples [00:09, 918.72 examples/s]8634 examples [00:09, 913.11 examples/s]8727 examples [00:09, 915.85 examples/s]8820 examples [00:09, 918.58 examples/s]8913 examples [00:09, 920.13 examples/s]9006 examples [00:10, 919.09 examples/s]9098 examples [00:10, 902.72 examples/s]9191 examples [00:10, 910.30 examples/s]9285 examples [00:10, 918.04 examples/s]9380 examples [00:10, 925.24 examples/s]9474 examples [00:10, 929.53 examples/s]9567 examples [00:10, 919.80 examples/s]9662 examples [00:10, 925.75 examples/s]9757 examples [00:10, 930.11 examples/s]9851 examples [00:10, 928.94 examples/s]9945 examples [00:11, 929.47 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteRWDKT7/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteRWDKT7/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f3546b93950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f3546b93950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f3546b93950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f34cbff5940>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f34d001aeb8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f3546b93950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f3546b93950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f3546b93950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3546b80c50>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f352cf52128>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f34be5801e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f34be5801e0> 

  function with postional parmater data_info <function split_train_valid at 0x7f34be5801e0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f34be5802f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f34be5802f0> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f34be5802f0> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=d117db5075eeb7d20532cc7195d1113f029aca8dbf294e1f18e440f790af713c
  Stored in directory: /tmp/pip-ephem-wheel-cache-50njxy8g/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:04<134:49:13, 1.78kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:04<94:36:12, 2.53kB/s] .vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:04<66:15:53, 3.61kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:05<46:21:46, 5.16kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:05<32:21:15, 7.37kB/s].vector_cache/glove.6B.zip:   1%|          | 9.76M/862M [00:05<22:29:15, 10.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.1M/862M [00:05<15:40:57, 15.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.2M/862M [00:05<10:54:47, 21.5kB/s].vector_cache/glove.6B.zip:   3%|▎         | 21.9M/862M [00:05<7:36:27, 30.7kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 27.7M/862M [00:05<5:17:25, 43.8kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.6M/862M [00:05<3:40:42, 62.6kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.4M/862M [00:05<2:33:41, 89.3kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.7M/862M [00:06<1:47:07, 127kB/s] .vector_cache/glove.6B.zip:   6%|▌         | 47.6M/862M [00:06<1:14:37, 182kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.4M/862M [00:06<52:05, 259kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 51.5M/862M [00:06<43:10, 313kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.7M/862M [00:08<32:00, 420kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:08<26:03, 516kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:08<19:08, 701kB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.7M/862M [00:08<13:35, 985kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.8M/862M [00:10<15:17, 874kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:10<12:15, 1.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.6M/862M [00:10<08:53, 1.50MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:12<09:01, 1.47MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:12<09:08, 1.45MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:12<06:59, 1.90MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.4M/862M [00:12<05:02, 2.63MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:14<12:56, 1.02MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:14<10:26, 1.27MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.0M/862M [00:14<07:35, 1.74MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:16<08:23, 1.57MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:16<08:32, 1.54MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:16<06:33, 2.01MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.4M/862M [00:16<04:45, 2.76MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:18<10:10, 1.29MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:18<08:27, 1.55MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.3M/862M [00:18<06:14, 2.09MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:20<07:21, 1.77MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:20<07:31, 1.73MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:20<06:35, 1.98MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:20<04:53, 2.66MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.6M/862M [00:20<03:45, 3.46MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:22<08:01, 1.61MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:22<10:35, 1.22MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:22<08:29, 1.53MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.7M/862M [00:22<06:16, 2.06MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:22<04:37, 2.78MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:24<1:19:56, 161kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:24<1:00:46, 212kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:24<43:34, 296kB/s]  .vector_cache/glove.6B.zip:  11%|█         | 90.6M/862M [00:24<30:45, 418kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.9M/862M [00:24<21:52, 587kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:26<20:28, 626kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:26<19:39, 652kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:26<15:00, 854kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.0M/862M [00:26<10:48, 1.18MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:26<07:47, 1.64MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:28<2:53:59, 73.3kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:28<2:06:58, 100kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:28<1:30:04, 141kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.2M/862M [00:28<1:03:12, 201kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:28<44:20, 286kB/s]   .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:30<56:48, 223kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:30<44:31, 285kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:30<32:24, 391kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:30<22:55, 552kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:30<16:20, 773kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:32<17:20, 727kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:32<16:52, 747kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:32<12:52, 979kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:32<09:19, 1.35MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:32<06:49, 1.84MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:34<10:42, 1.17MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:34<12:36, 995kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:34<09:51, 1.27MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:34<07:13, 1.73MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:34<05:16, 2.37MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:36<14:54, 837kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:36<15:29, 805kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:36<11:53, 1.05MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:36<08:38, 1.44MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:36<06:21, 1.95MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:38<10:15, 1.21MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:38<12:14, 1.01MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:38<09:35, 1.29MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:38<07:02, 1.76MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:38<05:11, 2.38MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:40<08:35, 1.43MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:40<10:38, 1.16MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:40<08:39, 1.42MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:40<06:20, 1.94MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:40<04:41, 2.62MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:41<11:25, 1.07MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:42<12:38, 970kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:42<09:53, 1.24MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:42<07:10, 1.71MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:42<05:20, 2.29MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:43<08:49, 1.38MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:44<10:54, 1.12MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:44<08:46, 1.39MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:44<06:26, 1.89MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:45<07:14, 1.67MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:46<09:39, 1.26MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:46<07:42, 1.57MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:46<05:42, 2.12MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:46<04:10, 2.89MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:47<44:14, 273kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:48<35:09, 343kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:48<25:30, 472kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:48<18:07, 663kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:48<12:57, 926kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:49<15:18, 783kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:50<14:53, 805kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:50<11:20, 1.06MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:50<08:13, 1.45MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:51<08:35, 1.39MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:52<10:09, 1.17MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:52<07:56, 1.50MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:52<05:50, 2.03MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:53<07:00, 1.69MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:53<09:02, 1.31MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:54<07:19, 1.62MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:54<05:21, 2.21MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:55<06:40, 1.76MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:55<08:16, 1.42MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:56<06:36, 1.78MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:56<04:56, 2.38MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:57<06:25, 1.82MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:57<08:18, 1.41MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:58<06:35, 1.77MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:58<04:52, 2.40MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:58<03:36, 3.22MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:59<2:10:19, 89.3kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:59<1:34:33, 123kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:00<1:06:50, 174kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [01:00<46:54, 247kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:00<32:58, 351kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:01<46:23, 250kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:01<35:36, 325kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:02<25:35, 452kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:02<18:06, 637kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:03<16:20, 704kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:03<14:34, 789kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:04<11:00, 1.04MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:04<07:53, 1.45MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:05<09:04, 1.26MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:05<09:17, 1.23MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:06<07:14, 1.58MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:06<05:14, 2.17MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:07<07:51, 1.45MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:07<08:35, 1.32MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:07<06:41, 1.70MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:08<04:51, 2.33MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:09<06:54, 1.64MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:09<07:35, 1.49MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:09<05:54, 1.91MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:10<04:20, 2.60MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:11<06:12, 1.81MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:11<07:24, 1.52MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:11<05:49, 1.92MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:12<04:14, 2.63MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:13<06:30, 1.71MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:13<06:56, 1.61MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:13<05:26, 2.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:14<03:56, 2.82MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:15<12:01, 922kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:15<11:07, 996kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:15<08:28, 1.31MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:16<06:05, 1.81MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:17<09:49, 1.12MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:17<08:56, 1.23MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:17<06:46, 1.62MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:18<04:51, 2.26MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:19<23:13, 471kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:19<18:18, 598kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:19<13:14, 826kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:19<09:22, 1.16MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:21<14:25, 755kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:21<12:03, 902kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:21<08:55, 1.22MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:23<08:02, 1.34MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:23<08:02, 1.34MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:23<06:14, 1.73MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:23<04:30, 2.38MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:25<13:03, 822kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:25<11:35, 927kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:25<08:42, 1.23MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:25<06:12, 1.72MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:27<14:52, 718kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:27<12:49, 832kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:27<09:29, 1.12MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:27<06:44, 1.57MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:29<10:50, 979kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:29<09:54, 1.07MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:29<07:24, 1.43MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:29<05:24, 1.96MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:31<06:29, 1.62MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:31<06:52, 1.53MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:31<05:23, 1.95MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:31<03:53, 2.70MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:33<13:20, 785kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:33<11:39, 897kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:33<08:42, 1.20MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:33<06:13, 1.67MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:35<12:49, 811kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:35<10:30, 988kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:35<07:39, 1.35MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:35<05:27, 1.89MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:37<25:49, 400kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:37<20:05, 514kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:37<14:28, 712kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:37<10:12, 1.01MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:39<16:38, 617kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:39<13:35, 755kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:39<09:58, 1.03MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<08:36, 1.18MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<10:10, 1.00MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<07:59, 1.27MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:41<05:50, 1.74MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:43<06:27, 1.57MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:43<06:30, 1.56MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:43<04:58, 2.03MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:43<03:34, 2.81MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:45<13:43, 733kB/s] .vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:45<11:29, 875kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:45<08:31, 1.18MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:47<07:34, 1.32MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:47<07:15, 1.38MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:47<05:29, 1.81MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:47<03:55, 2.52MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:49<20:00, 496kB/s] .vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:49<15:55, 622kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:49<11:37, 851kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:51<09:43, 1.01MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:51<08:45, 1.12MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:51<06:36, 1.49MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:53<06:12, 1.58MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:53<06:19, 1.54MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:53<04:48, 2.03MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:53<03:34, 2.73MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:55<05:18, 1.83MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:55<07:27, 1.30MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:55<05:59, 1.62MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:55<04:37, 2.10MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:57<04:48, 2.00MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:57<05:10, 1.86MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:57<04:05, 2.35MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:57<02:57, 3.23MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:59<12:36, 759kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:59<10:43, 892kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:59<07:57, 1.20MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:01<07:06, 1.34MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:01<06:49, 1.39MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:01<05:14, 1.81MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:03<05:12, 1.81MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:03<05:29, 1.72MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:03<04:17, 2.19MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:05<04:31, 2.07MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:05<05:02, 1.86MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:05<03:58, 2.35MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:06<04:17, 2.16MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:07<04:49, 1.93MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:07<03:49, 2.43MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:08<04:10, 2.21MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:09<04:43, 1.95MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:09<03:45, 2.45MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:10<04:07, 2.22MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:11<04:40, 1.96MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:11<03:42, 2.46MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:12<04:04, 2.23MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:13<04:37, 1.97MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:13<03:40, 2.47MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:14<04:02, 2.23MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:14<06:12, 1.45MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:15<05:04, 1.77MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:15<03:47, 2.37MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:16<04:22, 2.05MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:16<04:50, 1.85MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:17<03:45, 2.38MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:17<02:43, 3.27MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:18<10:39, 834kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:18<09:11, 966kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:19<06:51, 1.29MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:20<06:13, 1.42MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:20<06:05, 1.45MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:21<04:41, 1.88MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:22<04:42, 1.86MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:22<05:00, 1.75MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:22<03:51, 2.26MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:23<02:47, 3.11MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:24<08:42, 996kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:24<07:47, 1.11MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:24<05:48, 1.49MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:25<04:09, 2.08MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:26<08:13, 1.05MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:26<07:24, 1.16MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:26<05:32, 1.55MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:26<03:56, 2.16MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:28<20:07, 424kB/s] .vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:28<15:46, 541kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:28<11:22, 749kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:28<08:02, 1.06MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:30<10:10, 832kB/s] .vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:30<08:46, 965kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:30<06:32, 1.29MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:32<05:56, 1.41MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:32<05:47, 1.45MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:32<04:24, 1.90MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:32<03:10, 2.63MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:34<08:27, 986kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:34<07:32, 1.10MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:34<05:37, 1.48MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:34<04:01, 2.06MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:36<08:55, 926kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:36<07:51, 1.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:36<05:53, 1.40MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:38<05:26, 1.50MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:38<05:26, 1.51MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:38<04:12, 1.95MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:40<04:15, 1.91MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:40<06:02, 1.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:40<05:01, 1.62MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:40<03:41, 2.19MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:42<04:29, 1.79MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:42<04:43, 1.70MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:42<03:37, 2.22MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:42<02:39, 3.02MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:44<05:17, 1.51MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:44<05:17, 1.51MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:44<04:01, 1.98MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:44<02:55, 2.71MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:46<05:09, 1.53MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:46<05:05, 1.55MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:46<03:56, 2.00MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:48<04:02, 1.95MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:48<04:21, 1.80MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:48<03:23, 2.31MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:48<02:26, 3.19MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:50<19:21, 402kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:50<15:04, 516kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:50<10:54, 712kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:52<08:51, 871kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:52<07:42, 1.00MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:52<05:46, 1.33MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:54<05:16, 1.45MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:54<05:11, 1.47MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:54<04:00, 1.91MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:56<04:01, 1.88MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:56<04:19, 1.75MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:56<03:19, 2.28MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:56<02:25, 3.11MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:58<05:27, 1.38MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:58<05:17, 1.42MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:58<04:03, 1.85MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:00<04:02, 1.84MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:00<04:17, 1.74MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:00<03:21, 2.22MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:02<03:32, 2.08MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:02<03:54, 1.88MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:02<03:02, 2.42MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:02<02:12, 3.31MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:03<06:24, 1.14MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:04<05:51, 1.25MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:04<04:26, 1.64MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:05<04:16, 1.69MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:06<04:20, 1.67MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:06<03:43, 1.94MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:06<02:46, 2.60MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:07<03:26, 2.08MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:08<03:47, 1.89MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:08<03:20, 2.14MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:08<02:29, 2.87MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:09<03:18, 2.14MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:09<03:44, 1.90MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:10<03:14, 2.19MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:10<02:26, 2.90MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:11<03:08, 2.24MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:11<03:27, 2.04MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:12<03:04, 2.28MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:12<02:18, 3.03MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:13<03:06, 2.24MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:13<03:31, 1.97MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:14<02:49, 2.47MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:15<03:05, 2.23MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:15<03:30, 1.96MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:16<02:46, 2.47MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:17<03:03, 2.23MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:17<03:28, 1.96MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:17<02:43, 2.50MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:18<01:59, 3.40MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:19<04:41, 1.44MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:19<04:36, 1.47MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:19<03:32, 1.90MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:20<02:32, 2.63MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:21<07:16, 918kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:21<06:23, 1.04MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:21<04:48, 1.39MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:22<03:24, 1.94MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:23<10:26, 634kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:23<08:35, 770kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:23<06:16, 1.05MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:23<04:28, 1.47MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:25<05:51, 1.12MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:25<05:23, 1.22MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:25<04:02, 1.62MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:25<02:56, 2.21MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:27<04:10, 1.55MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:27<04:11, 1.55MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:27<03:14, 2.00MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:29<03:18, 1.94MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:29<03:33, 1.80MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:29<02:44, 2.33MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:29<02:00, 3.18MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:31<04:38, 1.37MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:31<04:26, 1.43MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:31<03:21, 1.88MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:31<02:26, 2.59MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:33<04:28, 1.40MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:33<04:21, 1.44MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:33<03:21, 1.87MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:35<03:20, 1.85MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:35<03:33, 1.74MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:35<02:45, 2.25MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:35<01:58, 3.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:37<27:15, 225kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:37<20:16, 303kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:37<14:25, 424kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:37<10:05, 602kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:39<16:00, 379kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:39<12:23, 490kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:39<08:54, 680kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:39<06:16, 958kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:41<07:17, 822kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:41<06:16, 955kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:41<04:38, 1.29MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:41<03:17, 1.80MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:43<07:23, 802kB/s] .vector_cache/glove.6B.zip:  59%|█████▊    | 507M/862M [03:43<06:14, 949kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:43<04:37, 1.28MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:45<04:12, 1.39MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:45<04:04, 1.44MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:45<03:05, 1.89MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:45<02:12, 2.63MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:47<50:16, 115kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:47<36:19, 159kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:47<25:36, 226kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:47<17:53, 321kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:49<15:27, 370kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:49<11:55, 480kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:49<08:36, 663kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:51<06:54, 819kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:51<05:56, 953kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:51<04:24, 1.28MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:53<03:59, 1.40MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:53<03:43, 1.50MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:53<02:52, 1.94MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:53<02:04, 2.67MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:55<04:22, 1.26MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:55<04:08, 1.33MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:55<03:06, 1.77MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:55<02:15, 2.43MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:56<03:51, 1.41MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:57<03:45, 1.45MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:57<02:53, 1.87MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:58<02:53, 1.86MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:59<03:04, 1.75MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:59<02:24, 2.23MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:00<02:32, 2.09MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:01<02:49, 1.88MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:01<02:12, 2.40MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:01<01:34, 3.32MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:02<1:44:11, 50.3kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:03<1:13:52, 71.0kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:03<51:47, 101kB/s]   .vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:03<36:02, 144kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:04<29:00, 178kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:04<21:18, 243kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:05<15:07, 341kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:06<11:19, 451kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:06<08:55, 572kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:07<06:28, 786kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:08<05:19, 948kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:08<04:42, 1.07MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:09<03:29, 1.44MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:09<02:29, 2.00MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:10<04:19, 1.15MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:10<03:59, 1.24MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:11<03:01, 1.64MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:12<02:54, 1.69MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:12<02:59, 1.64MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:12<02:19, 2.10MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:14<02:24, 2.01MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:14<02:37, 1.84MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:14<02:01, 2.37MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:15<01:28, 3.23MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:16<03:21, 1.42MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:16<03:16, 1.45MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:16<02:31, 1.88MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:18<02:31, 1.86MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:18<02:42, 1.74MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:18<02:07, 2.21MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:20<02:13, 2.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:20<02:10, 2.13MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:20<01:39, 2.79MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:22<02:01, 2.25MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:22<02:14, 2.03MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:22<01:45, 2.57MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:24<01:59, 2.26MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:24<02:11, 2.04MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:24<01:42, 2.61MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:24<01:15, 3.53MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:26<03:19, 1.33MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:26<03:28, 1.27MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:26<02:39, 1.65MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:26<01:55, 2.28MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:28<02:39, 1.63MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:28<02:37, 1.65MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:28<01:59, 2.18MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:28<01:29, 2.90MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:30<02:18, 1.86MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:30<02:19, 1.83MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:30<01:48, 2.35MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:32<01:58, 2.13MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:32<02:07, 1.98MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:32<01:39, 2.52MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:34<01:51, 2.23MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:34<02:00, 2.07MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:34<01:33, 2.65MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:34<01:07, 3.64MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:36<30:53, 132kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:36<22:18, 183kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:36<15:42, 258kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:36<10:56, 367kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:38<10:17, 390kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:38<07:53, 507kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:38<05:39, 706kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:38<03:57, 997kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:40<09:38, 409kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:40<07:23, 532kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:40<05:18, 739kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:40<03:43, 1.04MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:42<05:55, 652kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:42<04:49, 802kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:42<03:31, 1.09MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:44<03:05, 1.23MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:44<03:24, 1.12MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:44<02:39, 1.43MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:44<01:55, 1.95MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:46<02:20, 1.59MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:46<02:17, 1.62MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:46<01:45, 2.11MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:48<01:50, 1.99MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:48<01:55, 1.90MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:48<01:28, 2.47MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:48<01:04, 3.37MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:50<03:47, 947kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:50<03:16, 1.10MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:50<02:26, 1.47MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:50<01:42, 2.06MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:52<1:09:57, 50.4kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:52<49:32, 71.1kB/s]  .vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:52<34:41, 101kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:54<24:31, 141kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:54<17:45, 194kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:54<12:29, 275kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:54<08:42, 391kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:56<08:01, 422kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:56<06:12, 546kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:56<04:27, 757kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:57<03:38, 911kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:58<03:42, 896kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:58<02:52, 1.15MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:58<02:03, 1.59MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:59<02:19, 1.40MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [05:00<02:11, 1.48MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [05:00<01:39, 1.96MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:00<01:10, 2.71MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:01<04:19, 737kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:02<03:34, 890kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:02<02:36, 1.21MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:03<02:20, 1.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:04<02:11, 1.42MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:04<01:39, 1.87MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:05<01:39, 1.83MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:05<02:13, 1.37MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:06<01:46, 1.71MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:06<01:17, 2.31MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:07<01:42, 1.74MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:07<01:43, 1.73MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:08<01:18, 2.26MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:08<00:57, 3.06MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:09<01:50, 1.57MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:09<01:47, 1.62MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:10<01:22, 2.10MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:11<01:26, 1.98MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:11<01:29, 1.91MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:12<01:08, 2.47MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:12<00:50, 3.33MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:13<01:40, 1.65MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:13<01:39, 1.67MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:13<01:16, 2.16MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:15<01:20, 2.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:15<01:24, 1.91MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:15<01:04, 2.49MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:16<00:47, 3.35MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:17<01:35, 1.65MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:17<01:34, 1.67MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:17<01:11, 2.19MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:18<00:51, 2.99MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:19<01:51, 1.39MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:19<01:44, 1.47MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:19<01:19, 1.92MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:21<01:20, 1.87MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:21<01:21, 1.84MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:21<01:02, 2.36MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:23<01:08, 2.13MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:23<01:13, 1.99MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:23<00:57, 2.52MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:23<00:41, 3.39MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:25<02:39, 889kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:25<02:25, 970kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:25<01:48, 1.29MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:25<01:17, 1.79MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:27<01:34, 1.45MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:27<01:39, 1.37MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:27<01:16, 1.78MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:27<00:54, 2.45MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:29<01:25, 1.56MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:29<01:32, 1.44MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:29<01:11, 1.86MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:29<00:50, 2.57MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:31<01:27, 1.47MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:31<01:31, 1.41MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:31<01:09, 1.83MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:31<00:49, 2.53MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:33<01:42, 1.22MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:33<01:39, 1.25MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:33<01:16, 1.63MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:33<00:53, 2.26MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:35<02:15, 888kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:35<02:00, 998kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:35<01:30, 1.32MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:35<01:03, 1.84MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:37<02:20, 827kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:37<02:04, 935kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:37<01:32, 1.24MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:37<01:05, 1.73MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:39<02:37, 712kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:39<02:13, 836kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:39<01:39, 1.12MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:39<01:09, 1.57MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:41<02:56, 611kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:41<02:25, 740kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:41<01:46, 1.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:41<01:14, 1.42MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:43<01:49, 951kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:43<01:40, 1.04MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:43<01:22, 1.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:43<01:00, 1.70MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:45<01:01, 1.63MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:45<01:04, 1.54MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:45<00:50, 1.95MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:45<00:36, 2.67MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:47<01:39, 964kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:47<01:31, 1.04MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:47<01:08, 1.39MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:47<00:48, 1.91MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:49<01:00, 1.52MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:49<01:03, 1.44MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:49<00:49, 1.83MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:49<00:35, 2.51MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:51<01:23, 1.04MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:51<01:19, 1.10MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:51<00:59, 1.44MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:51<00:41, 2.00MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:53<01:29, 929kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:53<01:19, 1.05MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:53<00:58, 1.41MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:53<00:40, 1.96MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:55<01:20, 987kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:55<01:15, 1.04MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:55<01:01, 1.28MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:55<00:44, 1.73MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:55<00:31, 2.40MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:57<01:34, 795kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:57<01:23, 899kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:57<01:01, 1.21MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:57<00:43, 1.67MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:59<00:50, 1.41MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:59<00:52, 1.34MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:59<00:41, 1.69MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:59<00:30, 2.26MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:01<00:33, 1.96MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:01<00:36, 1.81MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:01<00:28, 2.33MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:01<00:20, 3.11MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:03<00:32, 1.94MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:03<00:37, 1.65MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:03<00:29, 2.10MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:03<00:21, 2.82MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:05<00:29, 1.99MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:05<00:33, 1.74MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:05<00:25, 2.23MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:05<00:18, 3.05MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:07<00:38, 1.42MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:07<00:39, 1.37MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:07<00:29, 1.79MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:07<00:20, 2.47MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:09<00:37, 1.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:09<00:37, 1.33MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:09<00:28, 1.71MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:09<00:19, 2.35MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:11<00:48, 941kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:11<00:44, 1.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:11<00:32, 1.37MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:11<00:22, 1.88MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:13<00:27, 1.51MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:13<00:26, 1.53MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:13<00:20, 1.98MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:13<00:13, 2.73MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:15<01:51, 335kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:15<01:24, 438kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:15<00:59, 608kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:15<00:39, 859kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:17<01:35, 348kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:17<01:13, 453kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:17<00:51, 628kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:17<00:33, 886kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:19<01:29, 325kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:19<01:09, 419kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:19<00:49, 577kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:19<00:31, 816kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:21<00:37, 661kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:21<00:31, 780kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:21<00:23, 1.05MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:21<00:14, 1.46MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:23<00:26, 798kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:23<00:22, 901kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:23<00:16, 1.19MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:23<00:10, 1.67MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:25<00:17, 930kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:25<00:16, 1.02MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:25<00:11, 1.34MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:25<00:07, 1.86MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:26<00:13, 906kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:27<00:11, 1.04MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:27<00:08, 1.40MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:27<00:04, 1.96MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:28<00:18, 453kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:29<00:14, 559kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:29<00:09, 768kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:29<00:05, 1.08MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:30<00:04, 1.03MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:31<00:03, 1.09MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:31<00:02, 1.42MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:31<00:00, 1.98MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:32<00:00, 911kB/s] .vector_cache/glove.6B.zip: 862MB [06:32, 2.19MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<35:21:55,  3.14it/s]  0%|          | 661/400000 [00:00<24:43:12,  4.49it/s]  0%|          | 1309/400000 [00:00<17:16:51,  6.41it/s]  1%|          | 2003/400000 [00:00<12:04:49,  9.15it/s]  1%|          | 2721/400000 [00:00<8:26:44, 13.07it/s]   1%|          | 3386/400000 [00:00<5:54:25, 18.65it/s]  1%|          | 4089/400000 [00:00<4:07:56, 26.61it/s]  1%|          | 4795/400000 [00:01<2:53:31, 37.96it/s]  1%|▏         | 5502/400000 [00:01<2:01:31, 54.10it/s]  2%|▏         | 6213/400000 [00:01<1:25:11, 77.04it/s]  2%|▏         | 6937/400000 [00:01<59:47, 109.55it/s]   2%|▏         | 7655/400000 [00:01<42:03, 155.48it/s]  2%|▏         | 8353/400000 [00:01<29:41, 219.89it/s]  2%|▏         | 9085/400000 [00:01<21:00, 310.13it/s]  2%|▏         | 9804/400000 [00:01<14:56, 435.00it/s]  3%|▎         | 10523/400000 [00:01<10:43, 605.72it/s]  3%|▎         | 11256/400000 [00:01<07:45, 835.70it/s]  3%|▎         | 11973/400000 [00:02<05:41, 1135.47it/s]  3%|▎         | 12684/400000 [00:02<04:16, 1512.91it/s]  3%|▎         | 13409/400000 [00:02<03:14, 1983.80it/s]  4%|▎         | 14140/400000 [00:02<02:32, 2538.43it/s]  4%|▎         | 14860/400000 [00:02<02:02, 3149.98it/s]  4%|▍         | 15576/400000 [00:02<01:41, 3774.03it/s]  4%|▍         | 16295/400000 [00:02<01:27, 4401.13it/s]  4%|▍         | 17009/400000 [00:02<01:17, 4911.83it/s]  4%|▍         | 17742/400000 [00:02<01:10, 5449.75it/s]  5%|▍         | 18460/400000 [00:02<01:04, 5873.98it/s]  5%|▍         | 19173/400000 [00:03<01:02, 6095.74it/s]  5%|▍         | 19896/400000 [00:03<00:59, 6395.14it/s]  5%|▌         | 20624/400000 [00:03<00:57, 6635.62it/s]  5%|▌         | 21337/400000 [00:03<00:58, 6524.35it/s]  6%|▌         | 22025/400000 [00:03<00:58, 6477.25it/s]  6%|▌         | 22744/400000 [00:03<00:56, 6674.86it/s]  6%|▌         | 23455/400000 [00:03<00:55, 6797.64it/s]  6%|▌         | 24170/400000 [00:03<00:54, 6898.00it/s]  6%|▌         | 24874/400000 [00:03<00:54, 6938.14it/s]  6%|▋         | 25575/400000 [00:03<00:55, 6799.33it/s]  7%|▋         | 26286/400000 [00:04<00:54, 6889.35it/s]  7%|▋         | 26980/400000 [00:04<00:54, 6806.37it/s]  7%|▋         | 27665/400000 [00:04<00:54, 6816.77it/s]  7%|▋         | 28397/400000 [00:04<00:53, 6959.32it/s]  7%|▋         | 29125/400000 [00:04<00:52, 7049.70it/s]  7%|▋         | 29832/400000 [00:04<00:53, 6928.75it/s]  8%|▊         | 30566/400000 [00:04<00:52, 7045.32it/s]  8%|▊         | 31297/400000 [00:04<00:51, 7121.82it/s]  8%|▊         | 32038/400000 [00:04<00:51, 7203.39it/s]  8%|▊         | 32760/400000 [00:04<00:50, 7206.93it/s]  8%|▊         | 33482/400000 [00:05<00:51, 7146.18it/s]  9%|▊         | 34198/400000 [00:05<00:52, 6978.34it/s]  9%|▊         | 34927/400000 [00:05<00:51, 7068.54it/s]  9%|▉         | 35663/400000 [00:05<00:50, 7152.47it/s]  9%|▉         | 36391/400000 [00:05<00:50, 7188.14it/s]  9%|▉         | 37111/400000 [00:05<00:50, 7144.07it/s]  9%|▉         | 37827/400000 [00:05<00:50, 7141.64it/s] 10%|▉         | 38542/400000 [00:05<00:51, 7078.14it/s] 10%|▉         | 39251/400000 [00:05<00:51, 7009.08it/s] 10%|▉         | 39953/400000 [00:06<00:52, 6865.06it/s] 10%|█         | 40656/400000 [00:06<00:51, 6912.56it/s] 10%|█         | 41386/400000 [00:06<00:51, 7022.24it/s] 11%|█         | 42100/400000 [00:06<00:50, 7056.89it/s] 11%|█         | 42807/400000 [00:06<00:51, 6932.00it/s] 11%|█         | 43518/400000 [00:06<00:51, 6982.99it/s] 11%|█         | 44244/400000 [00:06<00:50, 7063.30it/s] 11%|█         | 44979/400000 [00:06<00:49, 7145.82it/s] 11%|█▏        | 45716/400000 [00:06<00:49, 7209.87it/s] 12%|█▏        | 46456/400000 [00:06<00:48, 7265.68it/s] 12%|█▏        | 47184/400000 [00:07<00:49, 7059.24it/s] 12%|█▏        | 47892/400000 [00:07<00:50, 6975.49it/s] 12%|█▏        | 48626/400000 [00:07<00:49, 7080.27it/s] 12%|█▏        | 49357/400000 [00:07<00:49, 7144.80it/s] 13%|█▎        | 50097/400000 [00:07<00:48, 7218.91it/s] 13%|█▎        | 50834/400000 [00:07<00:48, 7260.62it/s] 13%|█▎        | 51561/400000 [00:07<00:49, 7091.13it/s] 13%|█▎        | 52303/400000 [00:07<00:48, 7186.05it/s] 13%|█▎        | 53044/400000 [00:07<00:47, 7250.33it/s] 13%|█▎        | 53779/400000 [00:07<00:47, 7279.48it/s] 14%|█▎        | 54508/400000 [00:08<00:47, 7258.00it/s] 14%|█▍        | 55235/400000 [00:08<00:47, 7221.49it/s] 14%|█▍        | 55958/400000 [00:08<00:48, 7091.90it/s] 14%|█▍        | 56683/400000 [00:08<00:48, 7138.57it/s] 14%|█▍        | 57402/400000 [00:08<00:47, 7153.82it/s] 15%|█▍        | 58118/400000 [00:08<00:48, 7111.80it/s] 15%|█▍        | 58830/400000 [00:08<00:48, 7101.22it/s] 15%|█▍        | 59541/400000 [00:08<00:48, 7091.07it/s] 15%|█▌        | 60251/400000 [00:08<00:49, 6882.23it/s] 15%|█▌        | 60969/400000 [00:08<00:48, 6967.13it/s] 15%|█▌        | 61667/400000 [00:09<00:48, 6957.17it/s] 16%|█▌        | 62393/400000 [00:09<00:47, 7043.42it/s] 16%|█▌        | 63119/400000 [00:09<00:47, 7106.44it/s] 16%|█▌        | 63847/400000 [00:09<00:46, 7157.27it/s] 16%|█▌        | 64564/400000 [00:09<00:48, 6955.28it/s] 16%|█▋        | 65282/400000 [00:09<00:47, 7018.44it/s] 16%|█▋        | 65989/400000 [00:09<00:47, 7030.44it/s] 17%|█▋        | 66693/400000 [00:09<00:47, 7004.95it/s] 17%|█▋        | 67395/400000 [00:09<00:48, 6908.34it/s] 17%|█▋        | 68099/400000 [00:09<00:47, 6944.80it/s] 17%|█▋        | 68795/400000 [00:10<00:49, 6712.01it/s] 17%|█▋        | 69504/400000 [00:10<00:48, 6820.72it/s] 18%|█▊        | 70216/400000 [00:10<00:47, 6905.35it/s] 18%|█▊        | 70929/400000 [00:10<00:47, 6968.75it/s] 18%|█▊        | 71657/400000 [00:10<00:46, 7057.15it/s] 18%|█▊        | 72364/400000 [00:10<00:47, 6949.03it/s] 18%|█▊        | 73061/400000 [00:10<00:48, 6706.24it/s] 18%|█▊        | 73793/400000 [00:10<00:47, 6879.04it/s] 19%|█▊        | 74484/400000 [00:10<00:47, 6851.59it/s] 19%|█▉        | 75204/400000 [00:11<00:46, 6949.85it/s] 19%|█▉        | 75901/400000 [00:11<00:46, 6917.37it/s] 19%|█▉        | 76631/400000 [00:11<00:46, 7025.19it/s] 19%|█▉        | 77335/400000 [00:11<00:46, 6874.34it/s] 20%|█▉        | 78063/400000 [00:11<00:46, 6989.71it/s] 20%|█▉        | 78772/400000 [00:11<00:45, 7018.40it/s] 20%|█▉        | 79476/400000 [00:11<00:46, 6900.53it/s] 20%|██        | 80168/400000 [00:11<00:46, 6830.28it/s] 20%|██        | 80882/400000 [00:11<00:46, 6918.21it/s] 20%|██        | 81575/400000 [00:11<00:46, 6872.73it/s] 21%|██        | 82270/400000 [00:12<00:46, 6894.88it/s] 21%|██        | 82971/400000 [00:12<00:45, 6927.83it/s] 21%|██        | 83693/400000 [00:12<00:45, 7010.37it/s] 21%|██        | 84420/400000 [00:12<00:44, 7084.58it/s] 21%|██▏       | 85130/400000 [00:12<00:45, 6924.03it/s] 21%|██▏       | 85824/400000 [00:12<00:45, 6840.12it/s] 22%|██▏       | 86510/400000 [00:12<00:47, 6665.80it/s] 22%|██▏       | 87238/400000 [00:12<00:45, 6838.15it/s] 22%|██▏       | 87950/400000 [00:12<00:45, 6918.31it/s] 22%|██▏       | 88668/400000 [00:12<00:44, 6992.22it/s] 22%|██▏       | 89398/400000 [00:13<00:43, 7079.11it/s] 23%|██▎       | 90108/400000 [00:13<00:44, 7037.04it/s] 23%|██▎       | 90813/400000 [00:13<00:44, 6981.22it/s] 23%|██▎       | 91531/400000 [00:13<00:43, 7037.69it/s] 23%|██▎       | 92237/400000 [00:13<00:43, 7042.69it/s] 23%|██▎       | 92942/400000 [00:13<00:43, 6982.19it/s] 23%|██▎       | 93641/400000 [00:13<00:45, 6680.14it/s] 24%|██▎       | 94320/400000 [00:13<00:45, 6712.54it/s] 24%|██▎       | 94994/400000 [00:13<00:46, 6561.55it/s] 24%|██▍       | 95703/400000 [00:13<00:45, 6709.05it/s] 24%|██▍       | 96423/400000 [00:14<00:44, 6847.27it/s] 24%|██▍       | 97111/400000 [00:14<00:44, 6817.86it/s] 24%|██▍       | 97835/400000 [00:14<00:43, 6937.89it/s] 25%|██▍       | 98567/400000 [00:14<00:42, 7047.13it/s] 25%|██▍       | 99274/400000 [00:14<00:43, 6943.00it/s] 25%|██▌       | 100011/400000 [00:14<00:42, 7065.59it/s] 25%|██▌       | 100720/400000 [00:14<00:42, 7028.17it/s] 25%|██▌       | 101431/400000 [00:14<00:42, 7050.96it/s] 26%|██▌       | 102164/400000 [00:14<00:41, 7131.01it/s] 26%|██▌       | 102878/400000 [00:15<00:41, 7117.61it/s] 26%|██▌       | 103591/400000 [00:15<00:42, 6901.64it/s] 26%|██▌       | 104283/400000 [00:15<00:43, 6827.27it/s] 26%|██▌       | 104968/400000 [00:15<00:43, 6747.47it/s] 26%|██▋       | 105671/400000 [00:15<00:43, 6828.27it/s] 27%|██▋       | 106386/400000 [00:15<00:42, 6918.96it/s] 27%|██▋       | 107100/400000 [00:15<00:41, 6982.61it/s] 27%|██▋       | 107800/400000 [00:15<00:43, 6742.59it/s] 27%|██▋       | 108488/400000 [00:15<00:42, 6780.97it/s] 27%|██▋       | 109188/400000 [00:15<00:42, 6843.68it/s] 27%|██▋       | 109924/400000 [00:16<00:41, 6989.04it/s] 28%|██▊       | 110662/400000 [00:16<00:40, 7101.08it/s] 28%|██▊       | 111374/400000 [00:16<00:40, 7060.55it/s] 28%|██▊       | 112082/400000 [00:16<00:41, 6936.71it/s] 28%|██▊       | 112812/400000 [00:16<00:40, 7040.75it/s] 28%|██▊       | 113549/400000 [00:16<00:40, 7134.33it/s] 29%|██▊       | 114264/400000 [00:16<00:40, 7091.56it/s] 29%|██▊       | 114975/400000 [00:16<00:41, 6864.46it/s] 29%|██▉       | 115692/400000 [00:16<00:40, 6950.75it/s] 29%|██▉       | 116389/400000 [00:16<00:41, 6754.62it/s] 29%|██▉       | 117126/400000 [00:17<00:40, 6926.49it/s] 29%|██▉       | 117822/400000 [00:17<00:40, 6891.65it/s] 30%|██▉       | 118514/400000 [00:17<00:40, 6870.79it/s] 30%|██▉       | 119240/400000 [00:17<00:40, 6982.22it/s] 30%|██▉       | 119943/400000 [00:17<00:40, 6995.32it/s] 30%|███       | 120644/400000 [00:17<00:40, 6911.45it/s] 30%|███       | 121337/400000 [00:17<00:41, 6770.47it/s] 31%|███       | 122048/400000 [00:17<00:40, 6866.18it/s] 31%|███       | 122777/400000 [00:17<00:39, 6985.50it/s] 31%|███       | 123508/400000 [00:17<00:39, 7078.36it/s] 31%|███       | 124248/400000 [00:18<00:38, 7170.82it/s] 31%|███       | 124967/400000 [00:18<00:38, 7073.23it/s] 31%|███▏      | 125676/400000 [00:18<00:38, 7039.63it/s] 32%|███▏      | 126409/400000 [00:18<00:38, 7124.01it/s] 32%|███▏      | 127123/400000 [00:18<00:39, 6938.81it/s] 32%|███▏      | 127858/400000 [00:18<00:38, 7055.21it/s] 32%|███▏      | 128572/400000 [00:18<00:38, 7079.92it/s] 32%|███▏      | 129282/400000 [00:18<00:39, 6898.66it/s] 33%|███▎      | 130014/400000 [00:18<00:38, 7017.56it/s] 33%|███▎      | 130753/400000 [00:19<00:37, 7122.64it/s] 33%|███▎      | 131488/400000 [00:19<00:37, 7188.57it/s] 33%|███▎      | 132222/400000 [00:19<00:37, 7231.78it/s] 33%|███▎      | 132947/400000 [00:19<00:37, 7155.94it/s] 33%|███▎      | 133664/400000 [00:19<00:38, 6929.47it/s] 34%|███▎      | 134388/400000 [00:19<00:37, 6998.58it/s] 34%|███▍      | 135097/400000 [00:19<00:37, 7023.11it/s] 34%|███▍      | 135801/400000 [00:19<00:37, 7004.80it/s] 34%|███▍      | 136503/400000 [00:19<00:38, 6932.63it/s] 34%|███▍      | 137246/400000 [00:19<00:37, 7072.97it/s] 34%|███▍      | 137955/400000 [00:20<00:37, 6992.46it/s] 35%|███▍      | 138687/400000 [00:20<00:36, 7086.24it/s] 35%|███▍      | 139427/400000 [00:20<00:36, 7175.13it/s] 35%|███▌      | 140146/400000 [00:20<00:36, 7146.65it/s] 35%|███▌      | 140887/400000 [00:20<00:35, 7222.75it/s] 35%|███▌      | 141610/400000 [00:20<00:35, 7187.61it/s] 36%|███▌      | 142330/400000 [00:20<00:36, 7043.73it/s] 36%|███▌      | 143066/400000 [00:20<00:36, 7135.08it/s] 36%|███▌      | 143781/400000 [00:20<00:35, 7138.89it/s] 36%|███▌      | 144523/400000 [00:20<00:35, 7218.56it/s] 36%|███▋      | 145263/400000 [00:21<00:35, 7269.21it/s] 36%|███▋      | 146000/400000 [00:21<00:34, 7297.00it/s] 37%|███▋      | 146731/400000 [00:21<00:35, 7102.55it/s] 37%|███▋      | 147456/400000 [00:21<00:35, 7144.05it/s] 37%|███▋      | 148172/400000 [00:21<00:35, 7099.65it/s] 37%|███▋      | 148909/400000 [00:21<00:34, 7177.35it/s] 37%|███▋      | 149640/400000 [00:21<00:34, 7214.78it/s] 38%|███▊      | 150372/400000 [00:21<00:34, 7244.23it/s] 38%|███▊      | 151097/400000 [00:21<00:36, 6912.03it/s] 38%|███▊      | 151838/400000 [00:21<00:35, 7052.62it/s] 38%|███▊      | 152571/400000 [00:22<00:34, 7127.01it/s] 38%|███▊      | 153287/400000 [00:22<00:35, 7012.22it/s] 39%|███▊      | 154009/400000 [00:22<00:34, 7070.94it/s] 39%|███▊      | 154735/400000 [00:22<00:34, 7125.67it/s] 39%|███▉      | 155449/400000 [00:22<00:34, 7011.69it/s] 39%|███▉      | 156173/400000 [00:22<00:34, 7078.54it/s] 39%|███▉      | 156913/400000 [00:22<00:33, 7171.46it/s] 39%|███▉      | 157632/400000 [00:22<00:34, 7127.14it/s] 40%|███▉      | 158346/400000 [00:22<00:33, 7127.83it/s] 40%|███▉      | 159069/400000 [00:22<00:33, 7154.71it/s] 40%|███▉      | 159785/400000 [00:23<00:34, 7039.20it/s] 40%|████      | 160518/400000 [00:23<00:33, 7123.28it/s] 40%|████      | 161232/400000 [00:23<00:33, 7101.02it/s] 40%|████      | 161943/400000 [00:23<00:33, 7084.38it/s] 41%|████      | 162652/400000 [00:23<00:33, 7024.28it/s] 41%|████      | 163355/400000 [00:23<00:34, 6932.21it/s] 41%|████      | 164049/400000 [00:23<00:35, 6718.23it/s] 41%|████      | 164761/400000 [00:23<00:34, 6832.79it/s] 41%|████▏     | 165500/400000 [00:23<00:33, 6990.64it/s] 42%|████▏     | 166240/400000 [00:24<00:32, 7106.37it/s] 42%|████▏     | 166972/400000 [00:24<00:32, 7168.15it/s] 42%|████▏     | 167691/400000 [00:24<00:32, 7088.06it/s] 42%|████▏     | 168402/400000 [00:24<00:33, 6855.75it/s] 42%|████▏     | 169126/400000 [00:24<00:33, 6965.33it/s] 42%|████▏     | 169858/400000 [00:24<00:32, 7065.92it/s] 43%|████▎     | 170590/400000 [00:24<00:32, 7138.91it/s] 43%|████▎     | 171329/400000 [00:24<00:31, 7211.38it/s] 43%|████▎     | 172052/400000 [00:24<00:33, 6774.00it/s] 43%|████▎     | 172736/400000 [00:24<00:33, 6778.60it/s] 43%|████▎     | 173438/400000 [00:25<00:33, 6847.70it/s] 44%|████▎     | 174171/400000 [00:25<00:32, 6984.77it/s] 44%|████▎     | 174903/400000 [00:25<00:31, 7080.59it/s] 44%|████▍     | 175614/400000 [00:25<00:31, 7081.79it/s] 44%|████▍     | 176324/400000 [00:25<00:32, 6949.30it/s] 44%|████▍     | 177026/400000 [00:25<00:31, 6969.97it/s] 44%|████▍     | 177762/400000 [00:25<00:31, 7082.03it/s] 45%|████▍     | 178502/400000 [00:25<00:30, 7171.93it/s] 45%|████▍     | 179221/400000 [00:25<00:31, 7058.31it/s] 45%|████▍     | 179962/400000 [00:25<00:30, 7158.78it/s] 45%|████▌     | 180680/400000 [00:26<00:31, 7037.51it/s] 45%|████▌     | 181398/400000 [00:26<00:30, 7078.83it/s] 46%|████▌     | 182118/400000 [00:26<00:30, 7114.33it/s] 46%|████▌     | 182831/400000 [00:26<00:30, 7042.70it/s] 46%|████▌     | 183569/400000 [00:26<00:30, 7137.82it/s] 46%|████▌     | 184285/400000 [00:26<00:30, 7141.52it/s] 46%|████▋     | 185000/400000 [00:26<00:31, 6925.38it/s] 46%|████▋     | 185717/400000 [00:26<00:30, 6995.38it/s] 47%|████▋     | 186418/400000 [00:26<00:30, 6958.21it/s] 47%|████▋     | 187152/400000 [00:26<00:30, 7066.79it/s] 47%|████▋     | 187892/400000 [00:27<00:29, 7163.08it/s] 47%|████▋     | 188631/400000 [00:27<00:29, 7228.61it/s] 47%|████▋     | 189355/400000 [00:27<00:29, 7055.53it/s] 48%|████▊     | 190083/400000 [00:27<00:29, 7118.33it/s] 48%|████▊     | 190806/400000 [00:27<00:29, 7150.72it/s] 48%|████▊     | 191527/400000 [00:27<00:29, 7166.91it/s] 48%|████▊     | 192245/400000 [00:27<00:29, 7161.16it/s] 48%|████▊     | 192983/400000 [00:27<00:28, 7224.67it/s] 48%|████▊     | 193706/400000 [00:27<00:29, 7080.70it/s] 49%|████▊     | 194439/400000 [00:28<00:28, 7152.68it/s] 49%|████▉     | 195175/400000 [00:28<00:28, 7212.67it/s] 49%|████▉     | 195897/400000 [00:28<00:28, 7176.70it/s] 49%|████▉     | 196629/400000 [00:28<00:28, 7216.34it/s] 49%|████▉     | 197352/400000 [00:28<00:28, 7178.86it/s] 50%|████▉     | 198071/400000 [00:28<00:28, 6992.44it/s] 50%|████▉     | 198773/400000 [00:28<00:28, 6997.63it/s] 50%|████▉     | 199511/400000 [00:28<00:28, 7107.05it/s] 50%|█████     | 200236/400000 [00:28<00:27, 7148.82it/s] 50%|█████     | 200952/400000 [00:28<00:28, 7084.69it/s] 50%|█████     | 201673/400000 [00:29<00:27, 7120.74it/s] 51%|█████     | 202386/400000 [00:29<00:28, 6939.46it/s] 51%|█████     | 203130/400000 [00:29<00:27, 7080.66it/s] 51%|█████     | 203866/400000 [00:29<00:27, 7159.87it/s] 51%|█████     | 204584/400000 [00:29<00:27, 7137.78it/s] 51%|█████▏    | 205325/400000 [00:29<00:26, 7215.67it/s] 52%|█████▏    | 206060/400000 [00:29<00:26, 7253.88it/s] 52%|█████▏    | 206787/400000 [00:29<00:27, 7099.41it/s] 52%|█████▏    | 207517/400000 [00:29<00:26, 7157.24it/s] 52%|█████▏    | 208248/400000 [00:29<00:26, 7201.80it/s] 52%|█████▏    | 208984/400000 [00:30<00:26, 7245.69it/s] 52%|█████▏    | 209727/400000 [00:30<00:26, 7299.51it/s] 53%|█████▎    | 210458/400000 [00:30<00:26, 7193.16it/s] 53%|█████▎    | 211179/400000 [00:30<00:26, 7061.97it/s] 53%|█████▎    | 211887/400000 [00:30<00:27, 6931.66it/s] 53%|█████▎    | 212623/400000 [00:30<00:26, 7052.78it/s] 53%|█████▎    | 213365/400000 [00:30<00:26, 7157.03it/s] 54%|█████▎    | 214105/400000 [00:30<00:25, 7225.15it/s] 54%|█████▎    | 214829/400000 [00:30<00:26, 7066.17it/s] 54%|█████▍    | 215538/400000 [00:30<00:26, 7056.20it/s] 54%|█████▍    | 216280/400000 [00:31<00:25, 7159.58it/s] 54%|█████▍    | 217023/400000 [00:31<00:25, 7236.75it/s] 54%|█████▍    | 217754/400000 [00:31<00:25, 7257.55it/s] 55%|█████▍    | 218494/400000 [00:31<00:24, 7296.91it/s] 55%|█████▍    | 219225/400000 [00:31<00:25, 7008.02it/s] 55%|█████▍    | 219957/400000 [00:31<00:25, 7097.05it/s] 55%|█████▌    | 220693/400000 [00:31<00:25, 7172.00it/s] 55%|█████▌    | 221422/400000 [00:31<00:24, 7204.44it/s] 56%|█████▌    | 222145/400000 [00:31<00:24, 7210.65it/s] 56%|█████▌    | 222867/400000 [00:31<00:24, 7113.95it/s] 56%|█████▌    | 223580/400000 [00:32<00:25, 6968.43it/s] 56%|█████▌    | 224317/400000 [00:32<00:24, 7083.55it/s] 56%|█████▋    | 225045/400000 [00:32<00:24, 7139.21it/s] 56%|█████▋    | 225774/400000 [00:32<00:24, 7182.78it/s] 57%|█████▋    | 226494/400000 [00:32<00:24, 7159.94it/s] 57%|█████▋    | 227230/400000 [00:32<00:23, 7218.20it/s] 57%|█████▋    | 227953/400000 [00:32<00:24, 7094.56it/s] 57%|█████▋    | 228693/400000 [00:32<00:23, 7181.95it/s] 57%|█████▋    | 229419/400000 [00:32<00:23, 7203.69it/s] 58%|█████▊    | 230147/400000 [00:33<00:23, 7223.87it/s] 58%|█████▊    | 230883/400000 [00:33<00:23, 7261.88it/s] 58%|█████▊    | 231617/400000 [00:33<00:23, 7283.67it/s] 58%|█████▊    | 232346/400000 [00:33<00:23, 7086.83it/s] 58%|█████▊    | 233077/400000 [00:33<00:23, 7150.21it/s] 58%|█████▊    | 233794/400000 [00:33<00:23, 6972.78it/s] 59%|█████▊    | 234518/400000 [00:33<00:23, 7048.32it/s] 59%|█████▉    | 235225/400000 [00:33<00:23, 6916.63it/s] 59%|█████▉    | 235953/400000 [00:33<00:23, 7020.97it/s] 59%|█████▉    | 236657/400000 [00:33<00:23, 6836.47it/s] 59%|█████▉    | 237392/400000 [00:34<00:23, 6980.70it/s] 60%|█████▉    | 238124/400000 [00:34<00:22, 7077.47it/s] 60%|█████▉    | 238866/400000 [00:34<00:22, 7174.77it/s] 60%|█████▉    | 239602/400000 [00:34<00:22, 7227.76it/s] 60%|██████    | 240327/400000 [00:34<00:22, 7122.72it/s] 60%|██████    | 241041/400000 [00:34<00:22, 7038.13it/s] 60%|██████    | 241778/400000 [00:34<00:22, 7133.49it/s] 61%|██████    | 242511/400000 [00:34<00:21, 7190.65it/s] 61%|██████    | 243249/400000 [00:34<00:21, 7245.50it/s] 61%|██████    | 243975/400000 [00:34<00:21, 7194.58it/s] 61%|██████    | 244696/400000 [00:35<00:21, 7067.70it/s] 61%|██████▏   | 245415/400000 [00:35<00:21, 7103.25it/s] 62%|██████▏   | 246140/400000 [00:35<00:21, 7144.71it/s] 62%|██████▏   | 246875/400000 [00:35<00:21, 7202.89it/s] 62%|██████▏   | 247596/400000 [00:35<00:21, 7182.76it/s] 62%|██████▏   | 248323/400000 [00:35<00:21, 7206.08it/s] 62%|██████▏   | 249044/400000 [00:35<00:21, 7068.82it/s] 62%|██████▏   | 249778/400000 [00:35<00:21, 7146.33it/s] 63%|██████▎   | 250512/400000 [00:35<00:20, 7201.64it/s] 63%|██████▎   | 251233/400000 [00:35<00:20, 7148.35it/s] 63%|██████▎   | 251968/400000 [00:36<00:20, 7203.52it/s] 63%|██████▎   | 252693/400000 [00:36<00:20, 7216.53it/s] 63%|██████▎   | 253415/400000 [00:36<00:20, 7074.77it/s] 64%|██████▎   | 254148/400000 [00:36<00:20, 7148.72it/s] 64%|██████▎   | 254864/400000 [00:36<00:20, 7139.62it/s] 64%|██████▍   | 255602/400000 [00:36<00:20, 7209.58it/s] 64%|██████▍   | 256330/400000 [00:36<00:19, 7228.30it/s] 64%|██████▍   | 257065/400000 [00:36<00:19, 7262.64it/s] 64%|██████▍   | 257792/400000 [00:36<00:20, 7080.51it/s] 65%|██████▍   | 258502/400000 [00:36<00:20, 7023.34it/s] 65%|██████▍   | 259206/400000 [00:37<00:20, 7023.66it/s] 65%|██████▍   | 259910/400000 [00:37<00:20, 6954.08it/s] 65%|██████▌   | 260613/400000 [00:37<00:19, 6974.63it/s] 65%|██████▌   | 261354/400000 [00:37<00:19, 7098.01it/s] 66%|██████▌   | 262065/400000 [00:37<00:19, 6928.76it/s] 66%|██████▌   | 262800/400000 [00:37<00:19, 7049.02it/s] 66%|██████▌   | 263538/400000 [00:37<00:19, 7143.06it/s] 66%|██████▌   | 264279/400000 [00:37<00:18, 7220.58it/s] 66%|██████▋   | 265006/400000 [00:37<00:18, 7232.40it/s] 66%|██████▋   | 265731/400000 [00:38<00:18, 7169.38it/s] 67%|██████▋   | 266449/400000 [00:38<00:19, 6948.02it/s] 67%|██████▋   | 267187/400000 [00:38<00:18, 7071.17it/s] 67%|██████▋   | 267923/400000 [00:38<00:18, 7152.33it/s] 67%|██████▋   | 268640/400000 [00:38<00:18, 7121.55it/s] 67%|██████▋   | 269354/400000 [00:38<00:18, 6953.13it/s] 68%|██████▊   | 270088/400000 [00:38<00:18, 7062.80it/s] 68%|██████▊   | 270796/400000 [00:38<00:19, 6694.63it/s] 68%|██████▊   | 271486/400000 [00:38<00:19, 6754.52it/s] 68%|██████▊   | 272204/400000 [00:38<00:18, 6874.08it/s] 68%|██████▊   | 272944/400000 [00:39<00:18, 7021.94it/s] 68%|██████▊   | 273676/400000 [00:39<00:17, 7106.67it/s] 69%|██████▊   | 274410/400000 [00:39<00:17, 7172.63it/s] 69%|██████▉   | 275130/400000 [00:39<00:17, 7059.23it/s] 69%|██████▉   | 275839/400000 [00:39<00:17, 7068.37it/s] 69%|██████▉   | 276571/400000 [00:39<00:17, 7141.54it/s] 69%|██████▉   | 277308/400000 [00:39<00:17, 7206.34it/s] 70%|██████▉   | 278046/400000 [00:39<00:16, 7255.93it/s] 70%|██████▉   | 278786/400000 [00:39<00:16, 7295.52it/s] 70%|██████▉   | 279517/400000 [00:39<00:17, 6858.66it/s] 70%|███████   | 280209/400000 [00:40<00:17, 6859.49it/s] 70%|███████   | 280943/400000 [00:40<00:17, 6994.41it/s] 70%|███████   | 281679/400000 [00:40<00:16, 7099.65it/s] 71%|███████   | 282414/400000 [00:40<00:16, 7171.23it/s] 71%|███████   | 283134/400000 [00:40<00:16, 7112.51it/s] 71%|███████   | 283847/400000 [00:40<00:16, 7103.62it/s] 71%|███████   | 284579/400000 [00:40<00:16, 7164.50it/s] 71%|███████▏  | 285312/400000 [00:40<00:15, 7213.06it/s] 72%|███████▏  | 286036/400000 [00:40<00:15, 7219.67it/s] 72%|███████▏  | 286759/400000 [00:40<00:16, 7051.42it/s] 72%|███████▏  | 287466/400000 [00:41<00:16, 6974.37it/s] 72%|███████▏  | 288167/400000 [00:41<00:16, 6983.69it/s] 72%|███████▏  | 288906/400000 [00:41<00:15, 7100.41it/s] 72%|███████▏  | 289642/400000 [00:41<00:15, 7173.95it/s] 73%|███████▎  | 290361/400000 [00:41<00:15, 7084.85it/s] 73%|███████▎  | 291071/400000 [00:41<00:15, 7041.43it/s] 73%|███████▎  | 291776/400000 [00:41<00:15, 6970.56it/s] 73%|███████▎  | 292484/400000 [00:41<00:15, 7001.33it/s] 73%|███████▎  | 293212/400000 [00:41<00:15, 7080.73it/s] 73%|███████▎  | 293934/400000 [00:41<00:14, 7121.39it/s] 74%|███████▎  | 294673/400000 [00:42<00:14, 7197.47it/s] 74%|███████▍  | 295394/400000 [00:42<00:14, 7201.06it/s] 74%|███████▍  | 296115/400000 [00:42<00:14, 7056.09it/s] 74%|███████▍  | 296853/400000 [00:42<00:14, 7148.86it/s] 74%|███████▍  | 297574/400000 [00:42<00:14, 7166.27it/s] 75%|███████▍  | 298292/400000 [00:42<00:14, 7152.38it/s] 75%|███████▍  | 299008/400000 [00:42<00:14, 7121.98it/s] 75%|███████▍  | 299745/400000 [00:42<00:13, 7194.49it/s] 75%|███████▌  | 300465/400000 [00:42<00:14, 7023.52it/s] 75%|███████▌  | 301169/400000 [00:43<00:14, 7019.26it/s] 75%|███████▌  | 301898/400000 [00:43<00:13, 7097.78it/s] 76%|███████▌  | 302609/400000 [00:43<00:13, 7093.63it/s] 76%|███████▌  | 303343/400000 [00:43<00:13, 7164.42it/s] 76%|███████▌  | 304074/400000 [00:43<00:13, 7205.20it/s] 76%|███████▌  | 304795/400000 [00:43<00:13, 6994.77it/s] 76%|███████▋  | 305534/400000 [00:43<00:13, 7107.09it/s] 77%|███████▋  | 306247/400000 [00:43<00:13, 7075.85it/s] 77%|███████▋  | 306976/400000 [00:43<00:13, 7137.37it/s] 77%|███████▋  | 307691/400000 [00:43<00:13, 7072.22it/s] 77%|███████▋  | 308400/400000 [00:44<00:13, 7003.04it/s] 77%|███████▋  | 309102/400000 [00:44<00:13, 6895.71it/s] 77%|███████▋  | 309832/400000 [00:44<00:12, 7011.98it/s] 78%|███████▊  | 310561/400000 [00:44<00:12, 7090.38it/s] 78%|███████▊  | 311295/400000 [00:44<00:12, 7162.94it/s] 78%|███████▊  | 312018/400000 [00:44<00:12, 7180.11it/s] 78%|███████▊  | 312753/400000 [00:44<00:12, 7228.70it/s] 78%|███████▊  | 313477/400000 [00:44<00:12, 7012.79it/s] 79%|███████▊  | 314220/400000 [00:44<00:12, 7132.17it/s] 79%|███████▊  | 314959/400000 [00:44<00:11, 7206.19it/s] 79%|███████▉  | 315682/400000 [00:45<00:11, 7159.74it/s] 79%|███████▉  | 316403/400000 [00:45<00:11, 7174.51it/s] 79%|███████▉  | 317122/400000 [00:45<00:11, 7130.90it/s] 79%|███████▉  | 317836/400000 [00:45<00:11, 6984.06it/s] 80%|███████▉  | 318568/400000 [00:45<00:11, 7078.20it/s] 80%|███████▉  | 319297/400000 [00:45<00:11, 7139.08it/s] 80%|████████  | 320040/400000 [00:45<00:11, 7220.79it/s] 80%|████████  | 320781/400000 [00:45<00:10, 7274.04it/s] 80%|████████  | 321510/400000 [00:45<00:10, 7264.72it/s] 81%|████████  | 322237/400000 [00:45<00:10, 7242.91it/s] 81%|████████  | 322962/400000 [00:46<00:10, 7190.10it/s] 81%|████████  | 323685/400000 [00:46<00:10, 7201.21it/s] 81%|████████  | 324413/400000 [00:46<00:10, 7222.38it/s] 81%|████████▏ | 325137/400000 [00:46<00:10, 7225.76it/s] 81%|████████▏ | 325860/400000 [00:46<00:10, 7179.56it/s] 82%|████████▏ | 326584/400000 [00:46<00:10, 7195.96it/s] 82%|████████▏ | 327313/400000 [00:46<00:10, 7223.45it/s] 82%|████████▏ | 328049/400000 [00:46<00:09, 7263.16it/s] 82%|████████▏ | 328784/400000 [00:46<00:09, 7286.49it/s] 82%|████████▏ | 329513/400000 [00:46<00:09, 7251.98it/s] 83%|████████▎ | 330239/400000 [00:47<00:09, 7036.85it/s] 83%|████████▎ | 330961/400000 [00:47<00:09, 7088.25it/s] 83%|████████▎ | 331695/400000 [00:47<00:09, 7160.99it/s] 83%|████████▎ | 332421/400000 [00:47<00:09, 7187.93it/s] 83%|████████▎ | 333157/400000 [00:47<00:09, 7238.32it/s] 83%|████████▎ | 333884/400000 [00:47<00:09, 7246.04it/s] 84%|████████▎ | 334610/400000 [00:47<00:09, 7145.88it/s] 84%|████████▍ | 335343/400000 [00:47<00:08, 7199.27it/s] 84%|████████▍ | 336081/400000 [00:47<00:08, 7252.04it/s] 84%|████████▍ | 336810/400000 [00:47<00:08, 7262.88it/s] 84%|████████▍ | 337542/400000 [00:48<00:08, 7279.09it/s] 85%|████████▍ | 338279/400000 [00:48<00:08, 7304.74it/s] 85%|████████▍ | 339010/400000 [00:48<00:08, 7181.98it/s] 85%|████████▍ | 339749/400000 [00:48<00:08, 7240.56it/s] 85%|████████▌ | 340474/400000 [00:48<00:08, 7196.70it/s] 85%|████████▌ | 341201/400000 [00:48<00:08, 7216.45it/s] 85%|████████▌ | 341923/400000 [00:48<00:08, 7172.22it/s] 86%|████████▌ | 342658/400000 [00:48<00:07, 7223.49it/s] 86%|████████▌ | 343381/400000 [00:48<00:07, 7128.77it/s] 86%|████████▌ | 344120/400000 [00:49<00:07, 7204.42it/s] 86%|████████▌ | 344846/400000 [00:49<00:07, 7219.31it/s] 86%|████████▋ | 345585/400000 [00:49<00:07, 7268.30it/s] 87%|████████▋ | 346328/400000 [00:49<00:07, 7313.42it/s] 87%|████████▋ | 347060/400000 [00:49<00:07, 7298.82it/s] 87%|████████▋ | 347791/400000 [00:49<00:07, 7238.02it/s] 87%|████████▋ | 348516/400000 [00:49<00:07, 7226.44it/s] 87%|████████▋ | 349260/400000 [00:49<00:06, 7286.44it/s] 87%|████████▋ | 349999/400000 [00:49<00:06, 7317.15it/s] 88%|████████▊ | 350741/400000 [00:49<00:06, 7344.93it/s] 88%|████████▊ | 351476/400000 [00:50<00:06, 7155.32it/s] 88%|████████▊ | 352196/400000 [00:50<00:06, 7167.20it/s] 88%|████████▊ | 352914/400000 [00:50<00:06, 7069.14it/s] 88%|████████▊ | 353650/400000 [00:50<00:06, 7151.92it/s] 89%|████████▊ | 354383/400000 [00:50<00:06, 7202.73it/s] 89%|████████▉ | 355104/400000 [00:50<00:06, 7203.16it/s] 89%|████████▉ | 355825/400000 [00:50<00:06, 7100.67it/s] 89%|████████▉ | 356545/400000 [00:50<00:06, 7129.80it/s] 89%|████████▉ | 357259/400000 [00:50<00:06, 6962.13it/s] 89%|████████▉ | 357979/400000 [00:50<00:05, 7030.10it/s] 90%|████████▉ | 358705/400000 [00:51<00:05, 7094.64it/s] 90%|████████▉ | 359424/400000 [00:51<00:05, 7120.45it/s] 90%|█████████ | 360137/400000 [00:51<00:05, 6956.38it/s] 90%|█████████ | 360872/400000 [00:51<00:05, 7067.42it/s] 90%|█████████ | 361608/400000 [00:51<00:05, 7151.97it/s] 91%|█████████ | 362344/400000 [00:51<00:05, 7211.04it/s] 91%|█████████ | 363069/400000 [00:51<00:05, 7221.01it/s] 91%|█████████ | 363802/400000 [00:51<00:04, 7250.48it/s] 91%|█████████ | 364528/400000 [00:51<00:04, 7100.66it/s] 91%|█████████▏| 365240/400000 [00:51<00:04, 7019.23it/s] 91%|█████████▏| 365976/400000 [00:52<00:04, 7115.67it/s] 92%|█████████▏| 366689/400000 [00:52<00:04, 7113.60it/s] 92%|█████████▏| 367418/400000 [00:52<00:04, 7164.78it/s] 92%|█████████▏| 368136/400000 [00:52<00:04, 7061.59it/s] 92%|█████████▏| 368843/400000 [00:52<00:04, 7015.03it/s] 92%|█████████▏| 369581/400000 [00:52<00:04, 7118.44it/s] 93%|█████████▎| 370301/400000 [00:52<00:04, 7142.47it/s] 93%|█████████▎| 371049/400000 [00:52<00:03, 7238.20it/s] 93%|█████████▎| 371786/400000 [00:52<00:03, 7277.07it/s] 93%|█████████▎| 372524/400000 [00:52<00:03, 7305.21it/s] 93%|█████████▎| 373255/400000 [00:53<00:03, 7242.79it/s] 93%|█████████▎| 373980/400000 [00:53<00:03, 7231.49it/s] 94%|█████████▎| 374710/400000 [00:53<00:03, 7250.57it/s] 94%|█████████▍| 375451/400000 [00:53<00:03, 7295.07it/s] 94%|█████████▍| 376188/400000 [00:53<00:03, 7315.88it/s] 94%|█████████▍| 376920/400000 [00:53<00:03, 6969.99it/s] 94%|█████████▍| 377621/400000 [00:53<00:03, 6684.48it/s] 95%|█████████▍| 378362/400000 [00:53<00:03, 6885.63it/s] 95%|█████████▍| 379101/400000 [00:53<00:02, 7028.06it/s] 95%|█████████▍| 379841/400000 [00:53<00:02, 7134.00it/s] 95%|█████████▌| 380558/400000 [00:54<00:02, 7133.59it/s] 95%|█████████▌| 381274/400000 [00:54<00:02, 7041.13it/s] 95%|█████████▌| 381997/400000 [00:54<00:02, 7094.88it/s] 96%|█████████▌| 382725/400000 [00:54<00:02, 7147.35it/s] 96%|█████████▌| 383447/400000 [00:54<00:02, 7168.08it/s] 96%|█████████▌| 384174/400000 [00:54<00:02, 7197.36it/s] 96%|█████████▌| 384895/400000 [00:54<00:02, 7184.43it/s] 96%|█████████▋| 385614/400000 [00:54<00:02, 7157.96it/s] 97%|█████████▋| 386352/400000 [00:54<00:01, 7221.76it/s] 97%|█████████▋| 387091/400000 [00:55<00:01, 7270.44it/s] 97%|█████████▋| 387835/400000 [00:55<00:01, 7320.44it/s] 97%|█████████▋| 388568/400000 [00:55<00:01, 7261.47it/s] 97%|█████████▋| 389304/400000 [00:55<00:01, 7288.32it/s] 98%|█████████▊| 390034/400000 [00:55<00:01, 7245.17it/s] 98%|█████████▊| 390773/400000 [00:55<00:01, 7286.97it/s] 98%|█████████▊| 391509/400000 [00:55<00:01, 7308.27it/s] 98%|█████████▊| 392240/400000 [00:55<00:01, 7254.31it/s] 98%|█████████▊| 392979/400000 [00:55<00:00, 7293.52it/s] 98%|█████████▊| 393709/400000 [00:55<00:00, 7223.28it/s] 99%|█████████▊| 394432/400000 [00:56<00:00, 7117.41it/s] 99%|█████████▉| 395166/400000 [00:56<00:00, 7182.39it/s] 99%|█████████▉| 395885/400000 [00:56<00:00, 7138.79it/s] 99%|█████████▉| 396600/400000 [00:56<00:00, 7119.87it/s] 99%|█████████▉| 397332/400000 [00:56<00:00, 7174.83it/s]100%|█████████▉| 398050/400000 [00:56<00:00, 7143.21it/s]100%|█████████▉| 398774/400000 [00:56<00:00, 7170.40it/s]100%|█████████▉| 399492/400000 [00:56<00:00, 6884.87it/s]100%|█████████▉| 399999/400000 [00:56<00:00, 7040.78it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f35b2dc8128>, <torchtext.data.dataset.TabularDataset object at 0x7f35b2dc8278>, <torchtext.vocab.Vocab object at 0x7f35b2dc8198>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.41 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.41 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  8.41 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.07 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.07 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  3.83 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  3.83 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.11 file/s]2020-06-07 00:19:31.730565: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-07 00:19:31.735027: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-06-07 00:19:31.735543: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55dbdbb41ce0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-07 00:19:31.735863: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 38%|███▊      | 3743744/9912422 [00:00<00:00, 37433191.81it/s]9920512it [00:00, 35162791.48it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 266730.36it/s]           
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 152338.58it/s]1654784it [00:00, 10968357.83it/s]                         
0it [00:00, ?it/s]8192it [00:00, 166035.60it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
