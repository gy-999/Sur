import os
import sys
sys.path.append(r'D:\python\cancer_survival\testPrediction\model')
import utils
from lifelines.utils import concordance_index
from testPrediction.model.data_loader import MyDataset
from testPrediction.model.data_loader import preprocess_clinical_data
from testPrediction.model.model import Model
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib.offsetbox import AnchoredText
# Const
m_length = 16
BATCH_SIZE = 160
EPOCH = 400
lr = 0.01
K = 5


data_path = utils.DATA_PATH

modalities_list = [['clinical' , 'Tumor']]
# , ,,,'Edema','Necrosis'
# setup random seedutils.setup_seed(24)
# detect cuda
device = utils.test_gpu()
# device = torch.device('cpu')

for modalities in modalities_list:
	if modalities[0] == 'clinical':
		lr = 0.01
	else:
		lr = 0.0005
	# create dataset
	mydataset = MyDataset(modalities, data_path)
	# create sampler
	prepro_clin_data_X, _, prepro_clin_data_y, _ = preprocess_clinical_data(data_path['clinical'])
	prepro_clin_data_X.reset_index(drop=True)
	prepro_clin_data_y.reset_index(drop=True)
	test_c_index_arr = []

	# 先分出20%作为测试集
	train_indices, test_indices = train_test_split(range(len(prepro_clin_data_X)), test_size=0.2, random_state=24)

	# 再从剩余的80%中分出25%作为验证集（即原始数据的20%）
	train_indices, val_indices = train_test_split(train_indices, test_size=0.25, random_state=24)

	# 根据索引创建数据加载器
	dataloaders = utils.get_dataloaders(mydataset, train_indices, val_indices, test_indices, BATCH_SIZE)

	# Create survival model
	survmodel = Model(
		modalities=modalities,
		m_length=m_length,
		dataloaders=dataloaders,
		fusion_method='attention',
		trade_off=0.2,
		mode='total',  # only_cox
		device=device)
	# Generate run tag
	run_tag = utils.compose_run_tag(
		model=survmodel, lr=lr, dataloaders=dataloaders,
		log_dir='.training_logs/', suffix=''
	)

	fit_args = {
		'num_epochs': EPOCH,
		'lr': lr,
		'info_freq': 2,
		'log_dir': os.path.join('.training_logs_risk/', run_tag),
		'lr_factor': 0.1,
		'scheduler_patience': 5,
	}
	# model fitting
	survmodel.fit(**fit_args)


	for data, data_label in dataloaders['test']:
		out, event, time = survmodel.predict(data, data_label)
		hazard, representation = out
		test_c_index = concordance_index(time.cpu().numpy(), -hazard['hazard'].detach().cpu().numpy(),
										 event.cpu().numpy())
		test_c_index_arr.append(test_c_index.item())
	print(f'C-index on Test set: ', test_c_index.item())
	print('Mean and std: ', utils.evaluate_model(test_c_index_arr))

	# 定义一个函数来绘制KM曲线
	def plot_km_curve(time, event, risk_scores, title):
		# 计算风险分数的中位数并分组
		median_risk = np.median(risk_scores)
		high_risk_idx = risk_scores >= median_risk
		low_risk_idx = risk_scores < median_risk

		# 使用布尔索引获取高低风险组的时间和事件数据
		time_high_risk = time[high_risk_idx]
		event_high_risk = event[high_risk_idx]
		time_low_risk = time[low_risk_idx]
		event_low_risk = event[low_risk_idx]

		# 使用KaplanMeierFitter进行拟合
		kmf_high_risk = KaplanMeierFitter()
		kmf_low_risk = KaplanMeierFitter()
		kmf_high_risk.fit(time_high_risk, event_observed=event_high_risk, label='High Risk')
		kmf_low_risk.fit(time_low_risk, event_observed=event_low_risk, label='Low Risk')

		# 计算p值
		results = logrank_test(time_high_risk, time_low_risk, event_observed_A=event_high_risk,
							   event_observed_B=event_low_risk)
		p_value = results.p_value

		# 创建图形和轴对象
		fig, ax = plt.subplots()

		# 绘制高风险和低风险组的生存曲线
		kmf_high_risk.plot(ax=ax, ci_show=False, color='red')
		kmf_low_risk.plot(ax=ax, ci_show=False, color='blue')

		# 添加标题和坐标轴标签
		ax.set_title(title)
		ax.set_xlabel('Time (Months)')
		ax.set_ylabel('Survival (percentage)')

		# 添加图例
		ax.legend()

		# 添加风险计数
		from lifelines.plotting import add_at_risk_counts
		add_at_risk_counts(kmf_high_risk, kmf_low_risk, ax=ax)

		# 显示p值
		if p_value < 0.001:
			anchored_text = AnchoredText(f'p-value<0.001', loc='center right')
			ax.add_artist(anchored_text)
		else:
			anchored_text = AnchoredText(f'p-value: {p_value:.4f}', loc='center right')
			ax.add_artist(anchored_text)

		# 显示图形
		plt.tight_layout()
		plt.show()


	# 处理训练数据
	time_list_train = []
	event_list_train = []
	risk_scores_list_train = []

	for data, data_label in dataloaders['train']:
		out, event, time = survmodel.predict(data, data_label)
		hazard, representation = out
		risk_scores = -hazard['hazard'].detach().cpu().numpy().flatten()  # Ensure risk_scores is 1D

		# Collect time, event, and risk score data
		time_list_train.extend(time.cpu().numpy())
		event_list_train.extend(event.cpu().numpy())
		risk_scores_list_train.extend(risk_scores)

	# Convert collected data to numpy arrays
	time_train = np.array(time_list_train)
	event_train = np.array(event_list_train)
	risk_scores_train = np.array(risk_scores_list_train)

	plot_km_curve(time_train, event_train, risk_scores_train, 'Training')

	# 处理验证数据
	time_list_val = []
	event_list_val = []
	risk_scores_list_val = []

	for data, data_label in dataloaders['val']:
		out, event, time = survmodel.predict(data, data_label)
		hazard, representation = out
		risk_scores = -hazard['hazard'].detach().cpu().numpy().flatten()  # Ensure risk_scores is 1D

		# Collect time, event, and risk score data
		time_list_val.extend(time.cpu().numpy())
		event_list_val.extend(event.cpu().numpy())
		risk_scores_list_val.extend(risk_scores)

	# Convert collected data to numpy arrays
	time_val = np.array(time_list_val)
	event_val = np.array(event_list_val)
	risk_scores_val = np.array(risk_scores_list_val)

	plot_km_curve(time_val, event_val, risk_scores_val, 'Validation')

	# 处理测试数据
	time_list_test = []
	event_list_test = []
	risk_scores_list_test = []

	for data, data_label in dataloaders['test']:
		out, event, time = survmodel.predict(data, data_label)
		hazard, representation = out
		risk_scores = -hazard['hazard'].detach().cpu().numpy().flatten()  # Ensure risk_scores is 1D

		# Collect time, event, and risk score data
		time_list_test.extend(time.cpu().numpy())
		event_list_test.extend(event.cpu().numpy())
		risk_scores_list_test.extend(risk_scores)

	# Convert collected data to numpy arrays
	time_test = np.array(time_list_test)
	event_test = np.array(event_list_test)
	risk_scores_test = np.array(risk_scores_list_test)

	plot_km_curve(time_test, event_test, risk_scores_test, 'Test')
	utils.save_5fold_results(test_c_index_arr, run_tag)


