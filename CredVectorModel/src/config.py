class Config():
    pass

model_config = Config()

model_config.word_emb_size = 100
model_config.sen_emb_size = 150
model_config.sen_len = 50
model_config.batch_size = 20
model_config.sen_bidirectional = True

# CNN
model_config.sen_kernels = [2, 3, 4, 5, 6, 7]
model_config.no_kernels = 60

model_config.doc_emb_size = 300
model_config.doc_bidirectional = True

model_config.title_len = 20
model_config.title_emb_size = 150
model_config.title_bidirectional = True

model_config.batch_size = 20

project_config = Config()

project_config.xml_data = 'dataset/'
project_config.processed_data = 'dataset/'

project_config.training_folder = 'Training/'
project_config.training_outputs = 'Outputs/training_outputs'
project_config.validation_folder = 'Validation/'
project_config.validation_outputs = 'Outputs/validation_outputs'
project_config.meta_folder = 'Meta/'

project_config.glove_file = ''