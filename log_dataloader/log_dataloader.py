
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f3a963c7598> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f3a963c7598> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f3b01077488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f3b01077488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f3b20295ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f3b20295ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f3aae3b0158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f3aae3b0158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f3aae3b0158> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  2%|▏         | 204800/9912422 [00:00<00:04, 2041421.59it/s] 29%|██▉       | 2916352/9912422 [00:00<00:02, 2822281.11it/s]9920512it [00:00, 23278864.37it/s]                            
0it [00:00, ?it/s] 57%|█████▋    | 16384/28881 [00:00<00:00, 150030.73it/s]32768it [00:00, 298565.50it/s]                           
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:14, 113275.48it/s] 40%|███▉      | 655360/1648877 [00:00<00:06, 160581.83it/s]1654784it [00:00, 6383226.37it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 74590.01it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3a95602668>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3aac7b3908>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f3aae3a7d90> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f3aae3a7d90> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f3aae3a7d90> , (data_info, **args) 

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

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:13,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:51,  2.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:51,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:51,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:50,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:50,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:50,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:49,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:49,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:49,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:48,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:48,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:34,  4.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:34,  4.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:34,  4.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:33,  4.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:33,  4.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:33,  4.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:33,  4.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:32,  4.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:32,  4.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:32,  4.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:00<00:22,  5.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:22,  5.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:22,  5.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:22,  5.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:22,  5.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:22,  5.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:22,  5.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:21,  5.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:21,  5.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:21,  5.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:21,  5.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:00<00:15,  8.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:15,  8.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:15,  8.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:15,  8.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:14,  8.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:14,  8.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:14,  8.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:14,  8.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:14,  8.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:14,  8.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:14,  8.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:00<00:10, 11.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:10, 11.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 11.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 11.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:09, 11.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:09, 11.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:09, 11.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:09, 11.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:09, 11.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:09, 11.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:09, 11.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 15.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 15.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 15.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 15.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 15.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 15.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 15.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 15.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 15.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 15.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:06, 15.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 20.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 20.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 20.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 20.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 20.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 20.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 20.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 20.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 20.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 20.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 20.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 26.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 26.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 26.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 26.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 26.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 26.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 26.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 26.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 26.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 26.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 26.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 33.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 33.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 33.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 33.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 33.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 33.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 33.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 33.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 33.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 33.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 33.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 41.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 41.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 41.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 41.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 41.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 41.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 41.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 41.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 41.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 41.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 41.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 49.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 49.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 49.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 49.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 49.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 49.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 49.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 49.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 49.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 49.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 57.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 57.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 57.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 57.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 57.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 57.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 57.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 57.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 57.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 57.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 63.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 63.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 63.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 63.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 63.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 63.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 63.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 63.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 63.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 63.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 69.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 69.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 69.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 69.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 69.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 69.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 69.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 69.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 69.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 69.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 74.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 74.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 74.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 74.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 74.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 74.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 74.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 74.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 74.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 74.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 78.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 81.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 81.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.28s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.28s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.28s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.47 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.59s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.28s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 81.47 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.59s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.59s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 35.31 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.59s/ url]
0 examples [00:00, ? examples/s]2020-08-10 12:07:17.130625: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-10 12:07:17.164027: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-10 12:07:17.164231: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5650c2910b00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-10 12:07:17.164250: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  9.36 examples/s]85 examples [00:00, 13.31 examples/s]181 examples [00:00, 18.90 examples/s]278 examples [00:00, 26.77 examples/s]375 examples [00:00, 37.80 examples/s]468 examples [00:00, 53.07 examples/s]563 examples [00:00, 74.02 examples/s]658 examples [00:00, 102.32 examples/s]752 examples [00:00, 139.66 examples/s]848 examples [00:01, 187.70 examples/s]944 examples [00:01, 247.32 examples/s]1043 examples [00:01, 318.89 examples/s]1137 examples [00:01, 393.33 examples/s]1232 examples [00:01, 476.74 examples/s]1325 examples [00:01, 557.54 examples/s]1418 examples [00:01, 633.30 examples/s]1513 examples [00:01, 702.81 examples/s]1607 examples [00:01, 758.22 examples/s]1705 examples [00:01, 813.44 examples/s]1803 examples [00:02, 854.87 examples/s]1900 examples [00:02, 885.71 examples/s]1996 examples [00:02, 902.46 examples/s]2096 examples [00:02, 927.18 examples/s]2193 examples [00:02, 934.22 examples/s]2290 examples [00:02, 942.23 examples/s]2387 examples [00:02, 935.57 examples/s]2482 examples [00:02, 930.95 examples/s]2579 examples [00:02, 940.75 examples/s]2677 examples [00:02, 949.40 examples/s]2776 examples [00:03, 958.73 examples/s]2873 examples [00:03, 923.46 examples/s]2966 examples [00:03, 875.35 examples/s]3059 examples [00:03, 889.87 examples/s]3150 examples [00:03, 894.74 examples/s]3240 examples [00:03, 881.08 examples/s]3336 examples [00:03, 901.18 examples/s]3431 examples [00:03, 913.36 examples/s]3525 examples [00:03, 920.48 examples/s]3620 examples [00:03, 928.50 examples/s]3719 examples [00:04, 943.89 examples/s]3814 examples [00:04, 928.52 examples/s]3908 examples [00:04, 915.97 examples/s]4008 examples [00:04, 939.00 examples/s]4107 examples [00:04, 952.45 examples/s]4203 examples [00:04, 953.95 examples/s]4299 examples [00:04, 954.34 examples/s]4395 examples [00:04, 947.78 examples/s]4490 examples [00:04, 931.13 examples/s]4584 examples [00:05, 933.75 examples/s]4682 examples [00:05, 945.69 examples/s]4777 examples [00:05, 936.35 examples/s]4871 examples [00:05, 934.86 examples/s]4971 examples [00:05, 951.48 examples/s]5072 examples [00:05, 965.77 examples/s]5172 examples [00:05, 972.79 examples/s]5270 examples [00:05, 960.65 examples/s]5367 examples [00:05, 952.25 examples/s]5464 examples [00:05, 956.84 examples/s]5561 examples [00:06, 959.35 examples/s]5658 examples [00:06, 962.19 examples/s]5755 examples [00:06, 954.99 examples/s]5851 examples [00:06, 944.16 examples/s]5947 examples [00:06, 946.74 examples/s]6042 examples [00:06, 941.87 examples/s]6137 examples [00:06, 941.52 examples/s]6234 examples [00:06, 947.72 examples/s]6329 examples [00:06, 940.26 examples/s]6424 examples [00:06, 938.18 examples/s]6518 examples [00:07, 935.01 examples/s]6613 examples [00:07, 937.84 examples/s]6709 examples [00:07, 943.87 examples/s]6806 examples [00:07, 948.37 examples/s]6902 examples [00:07, 950.23 examples/s]7002 examples [00:07, 963.43 examples/s]7101 examples [00:07, 970.87 examples/s]7200 examples [00:07, 975.31 examples/s]7298 examples [00:07, 973.19 examples/s]7396 examples [00:07, 969.25 examples/s]7496 examples [00:08, 977.99 examples/s]7597 examples [00:08, 986.22 examples/s]7696 examples [00:08, 984.70 examples/s]7795 examples [00:08, 971.18 examples/s]7893 examples [00:08, 966.06 examples/s]7991 examples [00:08, 967.65 examples/s]8089 examples [00:08, 970.21 examples/s]8187 examples [00:08, 966.35 examples/s]8284 examples [00:08, 960.54 examples/s]8382 examples [00:08, 963.66 examples/s]8479 examples [00:09, 959.42 examples/s]8575 examples [00:09, 939.91 examples/s]8670 examples [00:09, 937.85 examples/s]8766 examples [00:09, 942.36 examples/s]8861 examples [00:09, 929.73 examples/s]8958 examples [00:09, 941.37 examples/s]9057 examples [00:09, 953.14 examples/s]9153 examples [00:09, 953.84 examples/s]9249 examples [00:09, 951.99 examples/s]9345 examples [00:10, 914.11 examples/s]9440 examples [00:10, 923.84 examples/s]9535 examples [00:10, 930.04 examples/s]9631 examples [00:10, 938.68 examples/s]9726 examples [00:10, 935.59 examples/s]9822 examples [00:10, 940.38 examples/s]9919 examples [00:10, 948.55 examples/s]10014 examples [00:10, 889.82 examples/s]10111 examples [00:10, 911.95 examples/s]10206 examples [00:10, 921.99 examples/s]10306 examples [00:11, 942.45 examples/s]10405 examples [00:11, 954.32 examples/s]10501 examples [00:11, 950.52 examples/s]10597 examples [00:11, 944.17 examples/s]10692 examples [00:11, 937.02 examples/s]10788 examples [00:11, 942.53 examples/s]10883 examples [00:11, 929.30 examples/s]10977 examples [00:11, 906.89 examples/s]11070 examples [00:11, 913.29 examples/s]11163 examples [00:11, 917.92 examples/s]11261 examples [00:12, 934.35 examples/s]11358 examples [00:12, 942.49 examples/s]11455 examples [00:12, 949.73 examples/s]11553 examples [00:12, 956.97 examples/s]11649 examples [00:12, 949.87 examples/s]11745 examples [00:12, 947.46 examples/s]11841 examples [00:12, 949.84 examples/s]11939 examples [00:12, 957.04 examples/s]12035 examples [00:12, 953.03 examples/s]12131 examples [00:12, 940.15 examples/s]12230 examples [00:13, 953.02 examples/s]12328 examples [00:13, 958.44 examples/s]12424 examples [00:13, 956.73 examples/s]12520 examples [00:13, 943.86 examples/s]12615 examples [00:13, 942.51 examples/s]12713 examples [00:13, 952.80 examples/s]12810 examples [00:13, 955.33 examples/s]12913 examples [00:13, 975.28 examples/s]13011 examples [00:13, 964.03 examples/s]13108 examples [00:13, 946.25 examples/s]13204 examples [00:14, 948.92 examples/s]13300 examples [00:14, 950.87 examples/s]13396 examples [00:14, 953.29 examples/s]13492 examples [00:14, 938.40 examples/s]13587 examples [00:14, 941.72 examples/s]13686 examples [00:14, 954.63 examples/s]13787 examples [00:14, 970.40 examples/s]13885 examples [00:14, 964.36 examples/s]13982 examples [00:14, 964.41 examples/s]14079 examples [00:15, 954.25 examples/s]14175 examples [00:15, 954.94 examples/s]14271 examples [00:15, 935.55 examples/s]14366 examples [00:15, 938.47 examples/s]14460 examples [00:15, 935.03 examples/s]14554 examples [00:15, 934.86 examples/s]14652 examples [00:15, 945.46 examples/s]14748 examples [00:15, 949.55 examples/s]14844 examples [00:15, 945.05 examples/s]14939 examples [00:15, 944.61 examples/s]15034 examples [00:16, 938.39 examples/s]15129 examples [00:16, 938.98 examples/s]15226 examples [00:16, 946.39 examples/s]15322 examples [00:16, 949.47 examples/s]15418 examples [00:16, 950.35 examples/s]15514 examples [00:16, 948.21 examples/s]15609 examples [00:16, 943.31 examples/s]15705 examples [00:16, 947.56 examples/s]15801 examples [00:16, 950.01 examples/s]15897 examples [00:16, 943.03 examples/s]15992 examples [00:17, 933.13 examples/s]16087 examples [00:17, 936.56 examples/s]16183 examples [00:17, 940.85 examples/s]16278 examples [00:17, 941.30 examples/s]16373 examples [00:17, 936.10 examples/s]16467 examples [00:17, 926.89 examples/s]16562 examples [00:17, 932.81 examples/s]16656 examples [00:17, 931.30 examples/s]16752 examples [00:17, 938.91 examples/s]16848 examples [00:17, 942.26 examples/s]16943 examples [00:18, 922.14 examples/s]17039 examples [00:18, 931.75 examples/s]17136 examples [00:18, 942.15 examples/s]17231 examples [00:18, 943.72 examples/s]17326 examples [00:18, 945.24 examples/s]17421 examples [00:18, 931.17 examples/s]17515 examples [00:18, 932.54 examples/s]17614 examples [00:18, 947.52 examples/s]17715 examples [00:18, 963.35 examples/s]17813 examples [00:18, 967.61 examples/s]17910 examples [00:19, 961.97 examples/s]18007 examples [00:19, 942.21 examples/s]18102 examples [00:19, 932.25 examples/s]18196 examples [00:19, 930.92 examples/s]18290 examples [00:19, 931.56 examples/s]18385 examples [00:19, 935.36 examples/s]18480 examples [00:19, 939.41 examples/s]18575 examples [00:19, 942.42 examples/s]18670 examples [00:19, 938.69 examples/s]18765 examples [00:19, 939.90 examples/s]18860 examples [00:20, 928.25 examples/s]18955 examples [00:20, 934.47 examples/s]19053 examples [00:20, 945.00 examples/s]19148 examples [00:20, 938.49 examples/s]19245 examples [00:20, 945.62 examples/s]19344 examples [00:20, 957.00 examples/s]19443 examples [00:20, 966.09 examples/s]19540 examples [00:20, 939.69 examples/s]19635 examples [00:20, 938.59 examples/s]19730 examples [00:21, 938.98 examples/s]19824 examples [00:21, 935.04 examples/s]19918 examples [00:21, 925.45 examples/s]20011 examples [00:21, 845.38 examples/s]20108 examples [00:21, 878.42 examples/s]20202 examples [00:21, 894.58 examples/s]20293 examples [00:21, 889.86 examples/s]20388 examples [00:21, 905.67 examples/s]20482 examples [00:21, 914.44 examples/s]20576 examples [00:21, 920.64 examples/s]20671 examples [00:22, 928.72 examples/s]20765 examples [00:22, 921.33 examples/s]20862 examples [00:22, 933.40 examples/s]20959 examples [00:22, 943.21 examples/s]21054 examples [00:22, 933.06 examples/s]21151 examples [00:22, 941.79 examples/s]21246 examples [00:22, 911.12 examples/s]21339 examples [00:22, 916.53 examples/s]21434 examples [00:22, 924.22 examples/s]21530 examples [00:22, 933.74 examples/s]21626 examples [00:23, 940.24 examples/s]21721 examples [00:23, 930.67 examples/s]21815 examples [00:23, 932.15 examples/s]21909 examples [00:23, 924.65 examples/s]22003 examples [00:23, 928.64 examples/s]22099 examples [00:23, 937.47 examples/s]22195 examples [00:23, 942.83 examples/s]22290 examples [00:23, 944.86 examples/s]22385 examples [00:23, 935.27 examples/s]22479 examples [00:23, 932.07 examples/s]22574 examples [00:24, 934.62 examples/s]22668 examples [00:24, 918.91 examples/s]22760 examples [00:24, 893.86 examples/s]22852 examples [00:24, 901.10 examples/s]22943 examples [00:24, 903.24 examples/s]23037 examples [00:24, 911.35 examples/s]23129 examples [00:24, 906.16 examples/s]23228 examples [00:24, 928.71 examples/s]23327 examples [00:24, 944.46 examples/s]23422 examples [00:25, 943.42 examples/s]23517 examples [00:25, 938.82 examples/s]23611 examples [00:25, 931.98 examples/s]23707 examples [00:25, 939.97 examples/s]23802 examples [00:25, 939.14 examples/s]23897 examples [00:25, 940.22 examples/s]23992 examples [00:25, 940.68 examples/s]24089 examples [00:25, 948.24 examples/s]24185 examples [00:25, 950.50 examples/s]24281 examples [00:25, 949.62 examples/s]24379 examples [00:26, 957.05 examples/s]24480 examples [00:26, 970.39 examples/s]24578 examples [00:26, 956.73 examples/s]24674 examples [00:26, 957.31 examples/s]24770 examples [00:26, 950.37 examples/s]24866 examples [00:26, 946.12 examples/s]24961 examples [00:26, 942.05 examples/s]25056 examples [00:26, 926.42 examples/s]25150 examples [00:26, 928.22 examples/s]25244 examples [00:26, 930.89 examples/s]25338 examples [00:27, 932.53 examples/s]25434 examples [00:27, 938.73 examples/s]25528 examples [00:27, 935.71 examples/s]25624 examples [00:27, 940.94 examples/s]25719 examples [00:27, 940.44 examples/s]25815 examples [00:27, 945.69 examples/s]25911 examples [00:27, 948.22 examples/s]26006 examples [00:27, 936.56 examples/s]26102 examples [00:27, 942.08 examples/s]26197 examples [00:27, 943.75 examples/s]26293 examples [00:28, 948.15 examples/s]26390 examples [00:28, 953.05 examples/s]26486 examples [00:28, 945.89 examples/s]26581 examples [00:28, 943.47 examples/s]26676 examples [00:28, 941.68 examples/s]26771 examples [00:28, 939.57 examples/s]26866 examples [00:28, 941.10 examples/s]26961 examples [00:28, 940.84 examples/s]27056 examples [00:28, 940.80 examples/s]27158 examples [00:28, 960.94 examples/s]27255 examples [00:29, 952.84 examples/s]27351 examples [00:29, 937.74 examples/s]27445 examples [00:29, 931.40 examples/s]27539 examples [00:29, 928.21 examples/s]27632 examples [00:29, 924.93 examples/s]27732 examples [00:29, 944.25 examples/s]27830 examples [00:29, 951.68 examples/s]27926 examples [00:29, 940.65 examples/s]28021 examples [00:29, 943.05 examples/s]28120 examples [00:29, 954.20 examples/s]28219 examples [00:30, 962.94 examples/s]28316 examples [00:30, 957.64 examples/s]28415 examples [00:30, 966.55 examples/s]28515 examples [00:30, 973.67 examples/s]28613 examples [00:30, 974.17 examples/s]28712 examples [00:30, 976.37 examples/s]28810 examples [00:30, 970.43 examples/s]28908 examples [00:30, 972.53 examples/s]29006 examples [00:30, 967.03 examples/s]29103 examples [00:31, 959.75 examples/s]29205 examples [00:31, 976.63 examples/s]29303 examples [00:31, 973.85 examples/s]29401 examples [00:31, 975.41 examples/s]29499 examples [00:31, 971.28 examples/s]29597 examples [00:31, 958.46 examples/s]29693 examples [00:31, 949.31 examples/s]29788 examples [00:31, 929.50 examples/s]29882 examples [00:31, 930.59 examples/s]29977 examples [00:31, 934.13 examples/s]30071 examples [00:32, 878.94 examples/s]30170 examples [00:32, 908.19 examples/s]30263 examples [00:32, 912.85 examples/s]30362 examples [00:32, 934.68 examples/s]30461 examples [00:32, 949.12 examples/s]30558 examples [00:32, 954.20 examples/s]30659 examples [00:32, 968.56 examples/s]30757 examples [00:32, 960.47 examples/s]30854 examples [00:32, 958.45 examples/s]30950 examples [00:32, 955.93 examples/s]31047 examples [00:33, 960.04 examples/s]31144 examples [00:33, 956.20 examples/s]31240 examples [00:33, 946.06 examples/s]31335 examples [00:33, 943.76 examples/s]31430 examples [00:33, 935.63 examples/s]31526 examples [00:33, 940.42 examples/s]31622 examples [00:33, 945.63 examples/s]31717 examples [00:33, 937.84 examples/s]31814 examples [00:33, 945.91 examples/s]31909 examples [00:33, 938.39 examples/s]32003 examples [00:34, 936.29 examples/s]32099 examples [00:34, 940.92 examples/s]32194 examples [00:34, 931.56 examples/s]32288 examples [00:34, 926.68 examples/s]32384 examples [00:34, 934.41 examples/s]32480 examples [00:34, 941.11 examples/s]32575 examples [00:34, 942.86 examples/s]32670 examples [00:34, 933.51 examples/s]32764 examples [00:34, 929.98 examples/s]32858 examples [00:35, 909.50 examples/s]32951 examples [00:35, 913.65 examples/s]33046 examples [00:35, 921.91 examples/s]33139 examples [00:35, 907.17 examples/s]33233 examples [00:35, 915.16 examples/s]33331 examples [00:35, 933.56 examples/s]33427 examples [00:35, 941.28 examples/s]33522 examples [00:35, 943.80 examples/s]33617 examples [00:35, 921.20 examples/s]33710 examples [00:35, 903.48 examples/s]33804 examples [00:36, 913.90 examples/s]33901 examples [00:36, 928.64 examples/s]33997 examples [00:36, 937.64 examples/s]34091 examples [00:36, 937.78 examples/s]34185 examples [00:36, 935.85 examples/s]34279 examples [00:36, 933.57 examples/s]34378 examples [00:36, 947.34 examples/s]34473 examples [00:36, 909.47 examples/s]34565 examples [00:36, 907.99 examples/s]34659 examples [00:36, 915.54 examples/s]34754 examples [00:37, 923.64 examples/s]34850 examples [00:37, 933.79 examples/s]34947 examples [00:37, 943.81 examples/s]35043 examples [00:37, 947.51 examples/s]35138 examples [00:37, 947.75 examples/s]35235 examples [00:37, 953.56 examples/s]35331 examples [00:37, 953.45 examples/s]35427 examples [00:37, 951.09 examples/s]35523 examples [00:37, 943.50 examples/s]35619 examples [00:37, 948.07 examples/s]35714 examples [00:38, 945.93 examples/s]35813 examples [00:38, 957.62 examples/s]35912 examples [00:38, 966.40 examples/s]36009 examples [00:38, 952.73 examples/s]36108 examples [00:38, 961.91 examples/s]36205 examples [00:38, 961.00 examples/s]36302 examples [00:38, 959.61 examples/s]36398 examples [00:38, 955.76 examples/s]36494 examples [00:38, 949.91 examples/s]36590 examples [00:38, 935.33 examples/s]36687 examples [00:39, 944.02 examples/s]36783 examples [00:39, 946.64 examples/s]36880 examples [00:39, 951.57 examples/s]36976 examples [00:39, 935.79 examples/s]37075 examples [00:39, 951.07 examples/s]37175 examples [00:39, 963.18 examples/s]37274 examples [00:39, 970.47 examples/s]37372 examples [00:39, 961.68 examples/s]37469 examples [00:39, 952.32 examples/s]37565 examples [00:39, 945.75 examples/s]37661 examples [00:40, 949.66 examples/s]37757 examples [00:40, 943.68 examples/s]37855 examples [00:40, 953.04 examples/s]37951 examples [00:40, 948.62 examples/s]38046 examples [00:40, 948.40 examples/s]38142 examples [00:40, 950.75 examples/s]38239 examples [00:40, 953.11 examples/s]38335 examples [00:40, 952.45 examples/s]38431 examples [00:40, 945.50 examples/s]38527 examples [00:41, 947.49 examples/s]38625 examples [00:41, 956.35 examples/s]38721 examples [00:41, 956.04 examples/s]38818 examples [00:41, 960.17 examples/s]38915 examples [00:41, 951.01 examples/s]39014 examples [00:41, 959.85 examples/s]39111 examples [00:41, 954.59 examples/s]39209 examples [00:41, 960.37 examples/s]39306 examples [00:41, 941.52 examples/s]39401 examples [00:41, 919.63 examples/s]39500 examples [00:42, 935.68 examples/s]39600 examples [00:42, 952.31 examples/s]39698 examples [00:42, 959.33 examples/s]39796 examples [00:42, 963.39 examples/s]39893 examples [00:42, 956.83 examples/s]39991 examples [00:42, 962.87 examples/s]40088 examples [00:42, 913.03 examples/s]40186 examples [00:42, 931.21 examples/s]40280 examples [00:42, 893.04 examples/s]40370 examples [00:42, 894.69 examples/s]40464 examples [00:43, 907.33 examples/s]40559 examples [00:43, 917.21 examples/s]40652 examples [00:43, 918.35 examples/s]40747 examples [00:43, 924.96 examples/s]40840 examples [00:43, 923.82 examples/s]40933 examples [00:43, 919.04 examples/s]41025 examples [00:43, 915.18 examples/s]41120 examples [00:43, 924.87 examples/s]41215 examples [00:43, 930.61 examples/s]41309 examples [00:43, 928.13 examples/s]41405 examples [00:44, 937.06 examples/s]41499 examples [00:44, 935.56 examples/s]41595 examples [00:44, 942.18 examples/s]41690 examples [00:44, 908.71 examples/s]41782 examples [00:44, 907.00 examples/s]41873 examples [00:44, 893.77 examples/s]41965 examples [00:44, 900.05 examples/s]42059 examples [00:44, 911.06 examples/s]42155 examples [00:44, 922.52 examples/s]42248 examples [00:45, 917.84 examples/s]42345 examples [00:45, 930.58 examples/s]42439 examples [00:45, 916.08 examples/s]42533 examples [00:45, 920.48 examples/s]42627 examples [00:45, 925.00 examples/s]42720 examples [00:45, 923.12 examples/s]42813 examples [00:45, 924.57 examples/s]42909 examples [00:45, 932.73 examples/s]43003 examples [00:45, 927.89 examples/s]43096 examples [00:45, 922.93 examples/s]43189 examples [00:46, 908.53 examples/s]43281 examples [00:46, 909.47 examples/s]43376 examples [00:46, 918.52 examples/s]43473 examples [00:46, 932.49 examples/s]43570 examples [00:46, 942.07 examples/s]43666 examples [00:46, 947.09 examples/s]43764 examples [00:46, 955.00 examples/s]43860 examples [00:46, 904.27 examples/s]43954 examples [00:46, 914.26 examples/s]44049 examples [00:46, 923.27 examples/s]44142 examples [00:47, 918.33 examples/s]44235 examples [00:47, 916.57 examples/s]44329 examples [00:47, 921.97 examples/s]44422 examples [00:47, 905.86 examples/s]44515 examples [00:47, 911.67 examples/s]44607 examples [00:47, 903.69 examples/s]44701 examples [00:47, 912.81 examples/s]44796 examples [00:47, 922.31 examples/s]44889 examples [00:47, 913.04 examples/s]44984 examples [00:47, 922.94 examples/s]45077 examples [00:48, 919.08 examples/s]45173 examples [00:48, 928.92 examples/s]45268 examples [00:48, 932.95 examples/s]45365 examples [00:48, 942.28 examples/s]45462 examples [00:48, 948.78 examples/s]45558 examples [00:48, 950.02 examples/s]45655 examples [00:48, 955.14 examples/s]45751 examples [00:48, 953.16 examples/s]45847 examples [00:48, 925.28 examples/s]45941 examples [00:49, 928.79 examples/s]46035 examples [00:49, 927.36 examples/s]46137 examples [00:49, 951.08 examples/s]46233 examples [00:49, 949.74 examples/s]46329 examples [00:49, 951.40 examples/s]46425 examples [00:49, 906.70 examples/s]46517 examples [00:49, 883.82 examples/s]46612 examples [00:49, 901.19 examples/s]46706 examples [00:49, 909.78 examples/s]46802 examples [00:49, 923.70 examples/s]46898 examples [00:50, 932.89 examples/s]46992 examples [00:50, 926.35 examples/s]47088 examples [00:50, 934.41 examples/s]47185 examples [00:50, 943.97 examples/s]47284 examples [00:50, 957.01 examples/s]47380 examples [00:50, 948.82 examples/s]47475 examples [00:50, 934.39 examples/s]47571 examples [00:50, 939.78 examples/s]47666 examples [00:50, 934.75 examples/s]47760 examples [00:50, 928.77 examples/s]47856 examples [00:51, 935.95 examples/s]47950 examples [00:51, 934.50 examples/s]48044 examples [00:51, 933.66 examples/s]48138 examples [00:51, 926.99 examples/s]48231 examples [00:51, 927.59 examples/s]48326 examples [00:51, 932.78 examples/s]48420 examples [00:51, 925.59 examples/s]48515 examples [00:51, 930.16 examples/s]48609 examples [00:51, 927.40 examples/s]48702 examples [00:51, 926.78 examples/s]48797 examples [00:52, 931.67 examples/s]48891 examples [00:52, 916.81 examples/s]48987 examples [00:52, 927.44 examples/s]49083 examples [00:52, 936.64 examples/s]49177 examples [00:52, 934.38 examples/s]49271 examples [00:52, 933.56 examples/s]49365 examples [00:52, 921.81 examples/s]49458 examples [00:52, 911.48 examples/s]49555 examples [00:52, 926.41 examples/s]49648 examples [00:52, 907.31 examples/s]49739 examples [00:53, 907.52 examples/s]49830 examples [00:53, 905.60 examples/s]49924 examples [00:53, 913.58 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 6196/50000 [00:00<00:00, 61950.90 examples/s] 37%|███▋      | 18593/50000 [00:00<00:00, 72889.98 examples/s] 60%|██████    | 30011/50000 [00:00<00:00, 81759.09 examples/s] 85%|████████▍ | 42429/50000 [00:00<00:00, 91094.10 examples/s]                                                               0 examples [00:00, ? examples/s]69 examples [00:00, 689.68 examples/s]162 examples [00:00, 746.72 examples/s]255 examples [00:00, 792.46 examples/s]352 examples [00:00, 837.60 examples/s]444 examples [00:00, 858.47 examples/s]539 examples [00:00, 882.13 examples/s]621 examples [00:00, 861.42 examples/s]716 examples [00:00, 883.84 examples/s]812 examples [00:00, 902.85 examples/s]911 examples [00:01, 926.52 examples/s]1011 examples [00:01, 945.94 examples/s]1105 examples [00:01, 935.10 examples/s]1200 examples [00:01, 937.45 examples/s]1296 examples [00:01, 941.41 examples/s]1392 examples [00:01, 945.29 examples/s]1490 examples [00:01, 954.27 examples/s]1589 examples [00:01, 963.35 examples/s]1686 examples [00:01, 949.29 examples/s]1782 examples [00:01, 949.51 examples/s]1877 examples [00:02, 945.24 examples/s]1974 examples [00:02, 949.52 examples/s]2069 examples [00:02, 945.99 examples/s]2164 examples [00:02, 938.56 examples/s]2259 examples [00:02, 941.34 examples/s]2355 examples [00:02, 945.39 examples/s]2450 examples [00:02, 929.03 examples/s]2548 examples [00:02, 942.98 examples/s]2644 examples [00:02, 945.08 examples/s]2739 examples [00:02, 914.65 examples/s]2836 examples [00:03, 928.88 examples/s]2930 examples [00:03, 926.28 examples/s]3027 examples [00:03, 935.74 examples/s]3121 examples [00:03, 930.52 examples/s]3217 examples [00:03, 937.21 examples/s]3313 examples [00:03, 941.98 examples/s]3408 examples [00:03, 941.58 examples/s]3504 examples [00:03, 944.92 examples/s]3599 examples [00:03, 943.90 examples/s]3694 examples [00:03, 932.14 examples/s]3788 examples [00:04, 933.30 examples/s]3882 examples [00:04, 926.77 examples/s]3975 examples [00:04, 925.44 examples/s]4068 examples [00:04, 922.28 examples/s]4161 examples [00:04, 921.41 examples/s]4257 examples [00:04, 929.19 examples/s]4352 examples [00:04, 934.87 examples/s]4448 examples [00:04, 940.59 examples/s]4543 examples [00:04, 939.15 examples/s]4640 examples [00:04, 946.58 examples/s]4740 examples [00:05, 961.07 examples/s]4839 examples [00:05, 966.92 examples/s]4942 examples [00:05, 982.95 examples/s]5041 examples [00:05, 972.79 examples/s]5139 examples [00:05, 966.72 examples/s]5236 examples [00:05, 958.31 examples/s]5332 examples [00:05, 922.99 examples/s]5428 examples [00:05, 933.63 examples/s]5522 examples [00:05, 935.10 examples/s]5618 examples [00:05, 940.08 examples/s]5713 examples [00:06, 941.73 examples/s]5808 examples [00:06, 933.09 examples/s]5904 examples [00:06, 940.97 examples/s]6000 examples [00:06, 943.14 examples/s]6096 examples [00:06, 945.86 examples/s]6191 examples [00:06, 944.04 examples/s]6287 examples [00:06, 946.22 examples/s]6383 examples [00:06, 949.53 examples/s]6478 examples [00:06, 940.41 examples/s]6573 examples [00:07, 937.86 examples/s]6667 examples [00:07, 935.44 examples/s]6766 examples [00:07, 949.31 examples/s]6864 examples [00:07, 955.38 examples/s]6960 examples [00:07, 951.12 examples/s]7056 examples [00:07, 952.04 examples/s]7152 examples [00:07, 945.99 examples/s]7248 examples [00:07, 949.89 examples/s]7344 examples [00:07, 949.50 examples/s]7439 examples [00:07, 942.17 examples/s]7534 examples [00:08, 943.40 examples/s]7629 examples [00:08, 918.15 examples/s]7726 examples [00:08, 931.22 examples/s]7820 examples [00:08, 932.25 examples/s]7915 examples [00:08, 935.72 examples/s]8009 examples [00:08, 936.26 examples/s]8104 examples [00:08, 937.79 examples/s]8198 examples [00:08, 936.27 examples/s]8294 examples [00:08, 941.73 examples/s]8389 examples [00:08, 940.46 examples/s]8484 examples [00:09, 935.47 examples/s]8579 examples [00:09, 937.87 examples/s]8678 examples [00:09, 951.48 examples/s]8774 examples [00:09, 925.98 examples/s]8867 examples [00:09, 908.76 examples/s]8959 examples [00:09, 911.18 examples/s]9051 examples [00:09, 907.36 examples/s]9143 examples [00:09, 910.65 examples/s]9235 examples [00:09, 912.70 examples/s]9327 examples [00:09, 906.82 examples/s]9421 examples [00:10, 915.84 examples/s]9518 examples [00:10, 931.22 examples/s]9614 examples [00:10, 938.76 examples/s]9712 examples [00:10, 948.14 examples/s]9807 examples [00:10, 935.00 examples/s]9902 examples [00:10, 937.21 examples/s]9996 examples [00:10, 931.30 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete30BCAZ/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete30BCAZ/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f3aae3b0158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f3aae3b0158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f3aae3b0158> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3aac40f358>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3a6c67e2b0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f3aae3b0158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f3aae3b0158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f3aae3b0158> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3aac7b32e8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3aac7b31d0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 11.85 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 21.10 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.44 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.44 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.30 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.30 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.76 file/s]2020-08-10 12:08:29.252615: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-10 12:08:29.256353: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-10 12:08:29.256566: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ee3c48ff30 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-10 12:08:29.256584: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  1%|          | 73728/9912422 [00:00<00:15, 621033.95it/s]  3%|▎         | 278528/9912422 [00:00<00:12, 759143.29it/s] 25%|██▍       | 2433024/9912422 [00:00<00:07, 1065582.78it/s] 62%|██████▏   | 6127616/9912422 [00:00<00:02, 1501981.27it/s] 82%|████████▏ | 8118272/9912422 [00:00<00:00, 2076343.78it/s]9920512it [00:00, 13348326.35it/s]                            
0it [00:00, ?it/s] 57%|█████▋    | 16384/28881 [00:00<00:00, 156648.44it/s]32768it [00:00, 310940.12it/s]                           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  8%|▊         | 131072/1648877 [00:00<00:01, 1200413.59it/s]1654784it [00:00, 5351579.76it/s]                            
0it [00:00, ?it/s]8192it [00:00, 104017.35it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
