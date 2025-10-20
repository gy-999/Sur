import os
import sys
sys.path.append(r'D:\python\cancer_survival\testPrediction\model')
import utils
from lifelines.utils import concordance_index
from testPrediction1.model.data_loader import MyDataset
from testPrediction1.model.data_loader import preprocess_clinical_data
from testPrediction1.model.model import Model
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index

m_length = 16
BATCH_SIZE = 4
EPOCH = 400
lr = 0.003
K = 3
data_path = utils.DATA_PATH
modalities_list = [['clinical','Edema','Tumor','Necrosis']]

utils.setup_seed(24)
device = utils.test_gpu()

for modalities in modalities_list:

        mydataset = MyDataset(modalities, data_path)
        test_c_index_arr = []
        val_c_index_arr = []
        combined_c_index_arr = []

        prepro_clin_data_X, _, prepro_clin_data_y, _ = preprocess_clinical_data(data_path['clinical'])
        prepro_clin_data_X.reset_index(drop=True, inplace=True)
        prepro_clin_data_y.reset_index(drop=True, inplace=True)


        train_size = 0.5
        val_size = 0.25
        test_size = 0.25


        x_train_val, x_test, y_train_val, y_test = train_test_split(prepro_clin_data_X, prepro_clin_data_y[10],
                                                                    test_size=test_size, random_state=24,
                                                                    stratify=prepro_clin_data_y[10])
        x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val,
                                                          test_size=val_size / (train_size + val_size), random_state=24,
                                                          stratify=y_train_val)


        train_indices = x_train.index.tolist()
        val_indices = x_val.index.tolist()
        test_indices = x_test.index.tolist()


        dataloaders = utils.get_dataloaders(mydataset, train_indices, val_indices, test_indices, BATCH_SIZE)


        survmodel = Model(
            modalities=modalities,
            m_length=m_length,
            dataloaders=dataloaders,
            fusion_method='attention',
            trade_off=0.3,
            mode='total',
            device=device
        )

        run_tag = utils.compose_run_tag(
            model=survmodel, lr=lr, dataloaders=dataloaders,
            log_dir='.training_logs/', suffix=''
        )

        fit_args = {
            'num_epochs': EPOCH,
            'lr': lr,
            'info_freq': 3,
            'log_dir': os.path.join('.training_logs/', run_tag),
            'lr_factor': 0.1,
            'scheduler_patience': 5,
        }

        survmodel.fit(**fit_args)

        fold_representation = {modality: [] for modality in modalities}


        survmodel.test()



        for data_test, data_label_test in dataloaders['test']:
            out_test, event_test, time_test = survmodel.predict(data_test, data_label_test)
            hazard_test, representation_test= out_test
            test_c_index = concordance_index(time_test.cpu().numpy(), -hazard_test['hazard'].detach().cpu().numpy(),
                                             event_test.cpu().numpy())
            test_c_index_arr.append(test_c_index.item())
        print(f'C-index on Test set: ', test_c_index.item())

