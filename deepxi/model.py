## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from deepxi.gain import gfunc
from deepxi.network.selector import network_selector
from deepxi.inp_tgt import inp_tgt_selector
from deepxi.sig import InputTarget
from deepxi.utils import read_mat, read_wav, save_mat, save_wav
from hybrid.phoneme import Phoneme
from hybrid.analyze_label import ensures_dir
from tensorflow.keras.callbacks import Callback, CSVLogger, ModelCheckpoint
from tensorflow.keras.layers import Input, Masking
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.python.lib.io import file_io
from matplotlib import pyplot as plt
from tqdm import tqdm
import csv, math, os, pickle, random
import deepxi.se_batch as batch
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from hybrid.hybridMixMax import hybridMixMax,phoneme_lmfb_gaussians, phoneme_extract_gaussians
from asr_mfcc.base import logfbank
from mcra.mcra123 import mcra, mcra_2, imcra
from gmmn.gmmnoise import gmm_phoneme_noise, rt_vts_noise
from histogram2quantile.histogram import histogram
import soundfile as sf
import librosa
# [1] Nicolson, A. and Paliwal, K.K., 2019. Deep learning for
# 	  minimum mean-square error approaches to speech enhancement.
# 	  Speech Communication, 111, pp.44-55.

class DeepXi():
	"""
	Deep Xi model from [1].
	"""
	def __init__(
		self,
		N_d,
		N_s,
		K,
		f_s,
		inp_tgt_type,
		network_type,
		min_snr,
		max_snr,
		snr_inter,
		log_path,
		sample_dir=None,
		ver='VERSION_NAME',
		train_s_list=None,
		train_d_list=None,
		sample_size=None,
		reset_inp_tgt=False,
		**kwargs
		):
		"""
		Argument/s:
			N_d - window duration (samples).
			N_s - window shift (samples).
			K - number of frequency bins.
			f_s - sampling frequency.
			inp_tgt_type - input and target type.
			network_type - network type.
			min_snr - minimum SNR level for training.
			max_snr - maximum SNR level for training.
			stats_dir - path to save sample statistics.
			ver - version name.
			train_s_list - clean-speech training list to compute statistics.
			train_d_list - noise training list to compute statistics.
			sample_size - number of samples to compute the statistics from.
			kwargs - keyword arguments.
		"""
		self.inp_tgt_type = inp_tgt_type
		self.network_type = network_type
		self.min_snr = min_snr
		self.max_snr = max_snr
		self.snr_levels = list(range(self.min_snr, self.max_snr + 1, snr_inter))
		self.ver = ver
		self.train_s_list=train_s_list
		self.train_d_list=train_d_list

		inp_tgt_obj_path = sample_dir + '/' + self.ver + '_inp_tgt.p'
		if os.path.exists(inp_tgt_obj_path) and not reset_inp_tgt:
			with open(inp_tgt_obj_path, 'rb') as f:
				self.inp_tgt = pickle.load(f)
		else:
			self.inp_tgt = inp_tgt_selector(self.inp_tgt_type, N_d, N_s, K, f_s, **kwargs)
			if self.inp_tgt_type != "MagPhonme":
				s_sample, d_sample, x_sample, wav_len = self.sample(sample_size, sample_dir)
				self.inp_tgt.stats(s_sample, d_sample, x_sample, wav_len)
			with open(inp_tgt_obj_path, 'wb') as f:
				pickle.dump(self.inp_tgt, f, pickle.HIGHEST_PROTOCOL)

		self.inp = Input(name='inp', shape=[None, self.inp_tgt.n_feat], dtype='float32')
		self.network = network_selector(self.network_type, self.inp,
			self.inp_tgt.n_outp, **kwargs)

		self.model = Model(inputs=self.inp, outputs=self.network.outp)
		self.model.summary()
		if not os.path.exists(log_path + "/summary"):
			os.makedirs(log_path + "/summary")
		with open(log_path + "/summary/" + self.ver + ".txt", "w") as f:
			self.model.summary(print_fn=lambda x: f.write(x + '\n'))

	def train(
		self,
		train_s_list,
		train_d_list,
		mbatch_size,
		max_epochs,
		loss_fnc,
		log_path,
		model_path='model',
		val_s=None,
		val_d=None,
		val_s_len=None,
		val_d_len=None,
		val_snr=None,
		val_flag=True,
		val_save_path=None,
		resume_epoch=0,
		eval_example=False,
		save_model=True,
		):
		"""
		Deep Xi training.

		Argument/s:
			train_s_list - clean-speech training list.
			train_d_list - noise training list.
			model_path - model save path.
			val_s - clean-speech validation batch.
			val_d - noise validation batch.
			val_s_len - clean-speech validation sequence length batch.
			val_d_len - noise validation sequence length batch.
			val_snr - SNR validation batch.
			val_flag - perform validation.
			val_save_path - validation batch save path.
			mbatch_size - mini-batch size.
			max_epochs - maximum number of epochs.
			resume_epoch - epoch to resume training from.
			eval_example - evaluate a mini-batch of training examples.
			save_model - save architecture, weights, and training configuration.
			loss_fnc - loss function.
		"""
		self.train_s_list = train_s_list
		self.train_d_list = train_d_list
		self.mbatch_size = mbatch_size
		self.n_examples = len(self.train_s_list)
		self.n_iter = math.ceil(self.n_examples/mbatch_size)
		train_dataset0 = self.phoneme_mbatch_gen_aug(max_epochs - resume_epoch) #mbatch_gen
		train_dataset = self.dataset(max_epochs-resume_epoch)

		if val_flag:
			#val_set = self.val_batch(val_save_path, val_s, val_d, val_s_len, val_d_len, val_snr)
			val_set = self.phoneme_val_mbatch_gen(val_s)
			val_steps = len(val_set[0])
		else: val_set, val_steps = None, None

		if not os.path.exists(model_path): os.makedirs(model_path)
		if not os.path.exists(log_path + "/loss"): os.makedirs(log_path + "/loss")

		callbacks = []
		callbacks.append(CSVLogger(log_path + "/loss/" + self.ver + ".csv",
			separator=',', append=True))
		if save_model: callbacks.append(SaveWeights(model_path))

		if resume_epoch > 0: self.model.load_weights(model_path + "/epoch-" +
			str(resume_epoch-1) + "/variables/variables")

		if eval_example:
			print("Saving a mini-batch of training examples in .mat files...")
			inp_batch, tgt_batch, seq_mask_batch = list(train_dataset.take(1).as_numpy_iterator())[0]
			save_mat('./inp_batch.mat', inp_batch, 'inp_batch')
			save_mat('./tgt_batch.mat', tgt_batch, 'tgt_batch')
			save_mat('./seq_mask_batch.mat', seq_mask_batch, 'seq_mask_batch')
			print("Testing if add_noise() works correctly...")
			s, d, s_len, d_len, snr_tgt = self.wav_batch(self.train_s_list[0:mbatch_size],
				self.train_d_list[0:mbatch_size])
			(_, s, d) = self.inp_tgt.add_noise_batch(self.inp_tgt.normalise(s),
				self.inp_tgt.normalise(d), s_len, d_len, snr_tgt)
			for (i, _) in enumerate(s):
				snr_act = self.inp_tgt.snr_db(s[i][0:s_len[i]], d[i][0:d_len[i]])
				print('SNR target|actual: {:.2f}|{:.2f} (dB).'.format(snr_tgt[i], snr_act))

		if "MHA" in self.network_type:
			print("Using Transformer learning rate schedular.")
			lr_schedular = TransformerSchedular(self.network.d_model,
				self.network.warmup_steps)
			opt = Adam(learning_rate=lr_schedular, clipvalue=1.0, beta_1=0.9,
				beta_2=0.98, epsilon=1e-9)
		else: opt = Adam(learning_rate=0.001, clipvalue=1.0)

		if loss_fnc == "BinaryCrossentropy": loss = BinaryCrossentropy()
		elif loss_fnc == "MeanSquaredError": loss = MeanSquaredError()
		else: raise ValueError("Invalid loss function")

		self.model.compile(
			sample_weight_mode="temporal",
			loss=loss,
			optimizer=opt,
			metrics=['accuracy']
			)
		print("SNR levels used for training:")
		print(self.snr_levels)
		self.model.fit(
			x=train_dataset,
			initial_epoch=resume_epoch,
			epochs=max_epochs,
			steps_per_epoch=self.n_iter,
			callbacks=callbacks,
			validation_data=val_set,
			validation_steps=val_steps
			)

	def infer(
		self,
		test_x,
		test_x_len,
		test_x_base_names,
		test_epoch,
		model_path='model',
		out_type='y',
		gain='mmse-lsa',
		out_path='out',
		n_filters=40,
		saved_data_path=None,
		):
		"""
		Deep Xi inference. The specified 'out_type' is saved.

		Argument/s:
			test_x - noisy-speech test batch.
			test_x_len - noisy-speech test batch lengths.
			test_x_base_names - noisy-speech base names.
			test_epoch - epoch to test.
			model_path - path to model directory.
			out_type - output type (see deepxi/args.py).
			gain - gain function (see deepxi/args.py).
			out_path - path to save output files.
			saved_data_path - path to saved data necessary for enhancement.
		"""
		out_path_base = out_path
		if not isinstance(test_epoch, list): test_epoch = [test_epoch]
		if not isinstance(gain, list): gain = [gain]

		# The mel-scale filter bank is to compute an ideal binary mask (IBM)
		# estimate for log-spectral subband energies (LSSE).
		if out_type == 'subband_ibm_hat':
			mel_filter_bank = self.mel_filter_bank(n_filters)

		for e in test_epoch:
			if e < 1: raise ValueError("test_epoch must be greater than 0.")
			for g in gain:

				out_path = out_path_base + '/' + self.ver + '/' + 'e' + str(e) # output path.
				if out_type == 'xi_hat': out_path = out_path + '/xi_hat'
				elif out_type == 'gamma_hat': out_path = out_path + '/gamma_hat'
				elif out_type == 'mag_hat': out_path = out_path + '/mag_hat'
				elif out_type == 'y':
					if (self.inp_tgt_type == 'MagGain') or (self.inp_tgt_type == 'MagMag'):
						out_path = out_path + '/y'
					else: out_path = out_path + '/y/' + g
				elif out_type == 'deepmmse': out_path = out_path + '/deepmmse'
				elif out_type == 'ibm_hat': out_path = out_path + '/ibm_hat'
				elif out_type == 'subband_ibm_hat': out_path = out_path + '/subband_ibm_hat'
				elif out_type == 'cd_hat': out_path = out_path + '/cd_hat'
				else: raise ValueError('Invalid output type.')
				if not os.path.exists(out_path): os.makedirs(out_path)

				self.model.load_weights(model_path + '/epoch-' + str(e-1) +
					'/variables/variables' )

				print("Processing observations...")
				inp_batch, supplementary_batch, n_frames = self.observation_batch(test_x, test_x_len)

				print("Performing inference...")
				tgt_hat_batch = self.model.predict(inp_batch, batch_size=1, verbose=1)

				print("Saving outputs...")
				batch_size = len(test_x_len)
				for i in tqdm(range(batch_size)):
					base_name = test_x_base_names[i]
					inp = inp_batch[i,:n_frames[i],:]
					tgt_hat = tgt_hat_batch[i,:n_frames[i],:]

					# if tf.is_tensor(supplementary_batch):
					supplementary = supplementary_batch[i,:n_frames[i],:]

					if saved_data_path is not None:
						saved_data = read_mat(saved_data_path + '/' + base_name + '.mat')
						supplementary = (supplementary, saved_data)

					if out_type == 'xi_hat':
						xi_hat = self.inp_tgt.xi_hat(tgt_hat)
						save_mat(out_path + '/' + base_name + '.mat', xi_hat, 'xi_hat')
					elif out_type == 'gamma_hat':
						gamma_hat = self.inp_tgt.gamma_hat(tgt_hat)
						save_mat(out_path + '/' + base_name + '.mat', gamma_hat, 'gamma_hat')
					elif out_type == 'mag_hat':
						mag_hat = self.inp_tgt.mag_hat(tgt_hat)
						save_mat(out_path + '/' + base_name + '.mat', mag_hat, 'mag_hat')
					elif out_type == 'y':
						y = self.inp_tgt.enhanced_speech(inp, supplementary, tgt_hat, g).numpy()
						save_wav(out_path + '/' + base_name + '.wav', y, self.inp_tgt.f_s)
					elif out_type == 'deepmmse':
						xi_hat = self.inp_tgt.xi_hat(tgt_hat)
						d_PSD_hat = np.multiply(np.square(inp), gfunc(xi_hat, xi_hat+1.0,
							gtype='deepmmse'))
						save_mat(out_path + '/' + base_name + '.mat', d_PSD_hat, 'd_psd_hat')
					elif out_type == 'ibm_hat':
						xi_hat = self.inp_tgt.xi_hat(tgt_hat)
						ibm_hat = np.greater(xi_hat, 1.0).astype(bool)
						save_mat(out_path + '/' + base_name + '.mat', ibm_hat, 'ibm_hat')
					elif out_type == 'subband_ibm_hat':
						xi_hat = self.inp_tgt.xi_hat(tgt_hat)
						xi_hat_subband = np.matmul(xi_hat, mel_filter_bank.transpose())
						subband_ibm_hat = np.greater(xi_hat_subband, 1.0).astype(bool)
						save_mat(out_path + '/' + base_name + '.mat', subband_ibm_hat,
							'subband_ibm_hat')
					elif out_type == 'cd_hat':
						cd_hat = self.inp_tgt.cd_hat(tgt_hat)
						save_mat(out_path + '/' + base_name + '.mat', cd_hat, 'cd_hat')
					else: raise ValueError('Invalid output type.')

	def infer_pho(
		self,
		test_x,
		test_x_len,
		test_x_base_names,
		test_epoch,
		model_path='model',
		out_type='y',
		gain='mmse-lsa',
		out_path='out',
		n_filters=40,
		saved_data_path=None,
		):
		"""
		Deep Xi inference. The specified 'out_type' is saved.

		Argument/s:
			test_x - noisy-speech test batch.
			test_x_len - noisy-speech test batch lengths.
			test_x_base_names - noisy-speech base names.
			test_epoch - epoch to test.
			model_path - path to model directory.
			out_type - output type (see deepxi/args.py).
			gain - gain function (see deepxi/args.py).
			out_path - path to save output files.
			saved_data_path - path to saved data necessary for enhancement.
		"""
		out_path_base = out_path
		if not isinstance(test_epoch, list): test_epoch = [test_epoch]
		if not isinstance(gain, list): gain = [gain]
		#label_dir = '/home/devpath/datasets/shellcsv/test_labels_round1'
		label_dir = '/home/devpath/datasets/aidatatang/train_labels/ph_label_infer_small'
		ph_label_mapper = Phoneme(
			os.path.join('/home/devpath/golfbears/DeepXi/hybrid', 'initialfinal2phoneme-lexicon.txt'))
		for e in test_epoch:
			if e < 1: raise ValueError("test_epoch must be greater than 0.")
			for g in gain:

				out_path = out_path_base + '/' + self.ver + '/' + 'e' + str(e) # output path.
				if out_type == 'xi_hat': out_path = out_path + '/xi_hat'
				elif out_type == 'gamma_hat': out_path = out_path + '/gamma_hat'
				elif out_type == 'mag_hat': out_path = out_path + '/mag_hat'
				elif out_type == 'y':
					if (self.inp_tgt_type == 'MagGain') or (self.inp_tgt_type == 'MagMag'):
						out_path = out_path + '/y'
					else: out_path = out_path + '/y/' + g
				elif out_type == 'deepmmse': out_path = out_path + '/deepmmse'
				elif out_type == 'ibm_hat': out_path = out_path + '/ibm_hat'
				elif out_type == 'subband_ibm_hat': out_path = out_path + '/subband_ibm_hat'
				elif out_type == 'cd_hat': out_path = out_path + '/cd_hat'
				else: raise ValueError('Invalid output type.')
				if not os.path.exists(out_path): os.makedirs(out_path)

				self.model.load_weights(model_path + '/epoch-' + str(e-1) +
					'/variables/variables' )

				print("Processing observations...")
				mbatch_s, batch_l= 50, len(test_x)

				n_ite = int(np.ceil(batch_l/mbatch_s))
				start_idx, end_idx = 0, mbatch_s
				for _ in range(n_ite):
					sample_x = test_x[start_idx:end_idx]
					inp_batch = self.phoneme_infer_mbatch_gen(sample_x)

					print("Performing inference...")
					tgt_hat_batch = self.model.predict(inp_batch[0], batch_size=1, verbose=1)

					print("Saving outputs...")

					predict = tf.math.argmax(tgt_hat_batch, axis=-1)
					prev = tf.math.argmax(inp_batch[1],axis=-1)
					mask = tf.cast(inp_batch[2], tf.int64)
					predict_pure = tf.math.multiply(predict, mask)
					prev_pure = tf.math.multiply(prev, mask)
					numerator = tf.reduce_sum(tf.cast(tf.not_equal(predict_pure, prev_pure), tf.int64))

					denominator = tf.reduce_sum(mask)
					error_rate = tf.divide(numerator,denominator).numpy()
					np_predict = predict_pure.numpy()
					np_prev = prev_pure.numpy()
					array_len = tf.reduce_sum(mask, axis=-1).numpy()
					for jj in range(len(sample_x)):
						name = sample_x[jj]['file_path'].split('/')[-1]
						idx_q = np_predict[jj][:array_len[jj]]
						labels_ph = [ph_label_mapper.tkn_dict.index2Entry[i] for i in idx_q]
						with open(os.path.join(label_dir, name.split('.')[0]+ '.csv') , 'w') as f_ph:
							for i in labels_ph:
								f_ph.writelines(i + '\n')
						#np.save(name, np_predict[jj][:array_len[jj]])
					start_idx += mbatch_s
					end_idx += mbatch_s
					if end_idx > batch_l: end_idx = batch_l

	def infer_hybrid(
		self,
		test_x,
		test_x_len,
		test_x_base_names,
		test_epoch,
		model_path='model',
		out_type='y',
		gain='mmse-lsa',
		out_path='out',
		n_filters=40,
		saved_data_path=None,
		):
		"""
		Deep Xi inference. The specified 'out_type' is saved.

		Argument/s:
			test_x - noisy-speech test batch.
			test_x_len - noisy-speech test batch lengths.
			test_x_base_names - noisy-speech base names.
			test_epoch - epoch to test.
			model_path - path to model directory.
			out_type - output type (see deepxi/args.py).
			gain - gain function (see deepxi/args.py).
			out_path - path to save output files.
			saved_data_path - path to saved data necessary for enhancement.
		"""
		golden_label = ['|', '|', '|', '|', '|', '|', '|', '|', '|', '|', '|', '|', '|', '|', '|', '|',
						'|', '|', '|', '|', '|', '|', '|', '|', '|', '|', '|', '|', '|', '|', '|', '|',
						'|', '|', '|', '|', '|', '|', '|', 'j', 'i5', 'i5', 'i5', 'i5', 'i5', 'i5', 'i5',
						'i5', 'i5', 'i5', 'a3', 'a3', 'a3', 'a3', 'a3', 'a3', 'a3', 'o3', 'o3', 'o3', 'o3',
						'o3', 'u3', 'u3', 'u3', 'u3', 'u3', 'u3', 'u3', 'u3', 'u3', 'u3', 'j', 'j', 'z', 'j',
						'j', 'j', 'j', 'i5', 'i1', 'i5', 'i4', 'i4', 'i4', 'i1', 'e1', 'i5', 'e2', 'e2', 'e2',
						'e2', 'e2', 'e2', 'e1', 'e2', 'e2', 'e2', 'e1', 'e1', 'e1', 'e2', 'd', 'g', 'g', 'g',
						't', 'g', 'g', 't', 't', 't', 'o2', 'o2', 'o2', 'o2', 'ng2', 'ng2', 'ng2', 'ng2', 'ng2',
						'ng2', 'ng2', 'ng2', 'ng2', 'ng2', 'ng2', 'ng2', 'b', 'b', 'b', 'ng2', 'ng2', 'p', 'p',
						'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'i3', 'i3', 'i3', 'i3', 'i3', 'i3', 'i3', 'i3',
						'i3', 'i3', 'i3', 'i3', 'e2', 'i3', 'e2', 'e2', 'e2', 'e2', 'e2', 'e3', 'e2', 'i3', 'e2',
						'e2', 'e2', 'e2', 'e2', 'e2', 'e2', 'e2', 'e2', 'e2', 'e2', 'e2', 'e2', 'e2', '|', '|',
						'|', '|', '|', '|', '|', '|', '|', '|', '|', '|', '|', '|', '|', '|', '|', '|', '|', '|',
						'|', '|', '|', '|', '|', '|', '|', '|', '|', '|', '|', '|', '|', '|']

		ph_label_mapper = Phoneme(
			os.path.join('/home/devpath/golfbears/DeepXi/hybrid', 'initialfinal2phoneme-lexicon.txt'))
		#home_dir = '/home/devpath/datasets/hybridwav/npy'
		home_dirs = [
			'/home/ml/speech-aligner/egs/cn_phn/data/small'

		]
		for home_dir in home_dirs:

			w_f = os.path.join(home_dir, '1_n1_0db_c101.wav')
			#w_f = os.path.join(home_dir, 'gmm04_001_00.wav')
			wav_r, f_s = librosa.load(w_f, sr=16000, mono=True, dtype=np.float32)
			meg1, angle1 = self.inp_tgt.polar_analysis(wav_r)
			meg1 = np.where(meg1 == 0, np.finfo(float).eps, meg1)
			logmeg1 = np.log(meg1)
			alpha_list = np.arange(0.001, 0.02, 0.002)
			betta_list = np.arange(0.5, 0.7, 0.2)
			mean = np.expand_dims(np.mean(logmeg1[:10, :], axis=0), 0)
			stand = np.expand_dims(np.std(logmeg1[:10, :], axis=0), 0)
			means, stds, probs = phoneme_extract_gaussians()
			#means.append(np.mean(logmeg1[:25, :]))
			#stds.append(np.std(logmeg1[:25, :]))

			golden_w_f = os.path.join(home_dir, '1_n1_0db_c101.wav')
			golden_wav_r, golden_f_s = librosa.load(golden_w_f, sr=16000, mono=True, dtype=np.float32)
			golden_meg1, golden_angle1 = self.inp_tgt.polar_analysis(golden_wav_r)

			h_m_max = hybridMixMax(np.array(means), np.array(stds), mean, stand, np.array(probs), 0.8, 2.0)
			if not isinstance(test_epoch, list): test_epoch = [test_epoch]
			for e in test_epoch:
				if e < 1: raise ValueError("test_epoch must be greater than 0.")
				self.model.load_weights(model_path + '/epoch-' + str(e-1) +
					'/variables/variables' )
				tgt_hat_batch = self.model.predict(np.expand_dims(golden_meg1, 0), batch_size=1, verbose=1)
				#tgt_hat_batch = self.model.predict(np.expand_dims(meg1, 0), batch_size=1, verbose=1)
				posterior_pro = np.squeeze(tgt_hat_batch,0)
				label_predict = np.argmax(posterior_pro, axis=-1)
				labels = [ph_label_mapper.tkn_dict.index2Entry[i] for i in label_predict]
				print(labels)
				pro_er1 = posterior_pro[:,50]
				posterior_pro = np.delete(posterior_pro, 50, axis=1)

				p_predict = np.max(posterior_pro, axis=-1)
				sil_pro = posterior_pro[0]
				predict_norm_check = np.sum(posterior_pro, axis=-1)
				for a in alpha_list:
					for b in betta_list:
						h_m_max.updata_alpha_betta(a, b)
						h_m_max.update_noise(mean, stand)
						print('############################ process hyperparams: alpha.' + str(round(a,3)) + ' betta.' + str(round(b,1)))
						x_hat = np.zeros(257, float)
						for i, logf in enumerate(logmeg1):

							#new_n_p = posterior_pro[i][-1].astype(np.float64)
							#new_v_p = np.array(probs)*(1-new_n_p)
							#new_p = np.hstack((new_v_p, new_n_p))
							#new_p_check = np.sum(new_p)
							#o = h_m_max.x_estimate(posterior_pro[i,1:], logf)
							#o = h_m_max.x_estimate_mm(probs, logf)
							#if np.argmax(posterior_pro[i]) ==  0:

							o, p_mm = h_m_max.x_estimate_mixmax_nn(posterior_pro[i], logf)
							#o, p_mm = h_m_max.x_estimate_mixmax(posterior_pro[i,1:], logf)
							x_hat = np.vstack((x_hat, o))
							#print('nn label:' + str(np.argmax(posterior_pro[i,1:])) + ' vs mixmax label:' + str(np.argmax(p_mm)))
						meg = np.exp(x_hat[1:])
						wav  = self.inp_tgt.polar_synthesis(meg, angle1)
						#path2 = os.path.join(home_dir, 'gmm_gen/gmm04_001_00-SNR0DB_' +str(round(a,1))+'_'+str(round(b,1))+ '.wav')
						path2 = os.path.join(home_dir, 'pho_gen/1_n1_0db_c101_' + str(round(a, 3)) + '_' + str(round(b, 1)) + '.wav')
						sf.write(path2, wav.numpy(), 16000)
				print("Processing observations...")

	def infer_hybrid1(
		self,
		test_x,
		test_x_len,
		test_x_base_names,
		test_epoch,
		model_path='model',
		out_type='y',
		gain='mmse-lsa',
		out_path='out',
		n_filters=40,
		saved_data_path=None,
		):
		"""
		Deep Xi inference. The specified 'out_type' is saved.

		Argument/s:
			test_x - noisy-speech test batch.
			test_x_len - noisy-speech test batch lengths.
			test_x_base_names - noisy-speech base names.
			test_epoch - epoch to test.
			model_path - path to model directory.
			out_type - output type (see deepxi/args.py).
			gain - gain function (see deepxi/args.py).
			out_path - path to save output files.
			saved_data_path - path to saved data necessary for enhancement.
		"""
		ph_label_mapper = Phoneme(
			os.path.join('/home/devpath/golfbears/DeepXi/hybrid', 'initialfinal2phoneme-lexicon.txt'))
		means, stds, probs = phoneme_extract_gaussians()
		h_m_max = hybridMixMax(np.array(means), np.array(stds), np.array(means[0]), np.array(stds[0]), np.array(probs), 0.8, 2.0)
		clean_dir = '/home/devpath/datasets/new/small'
		#home_dir = '/home/devpath/datasets/new/small'#'/home/devpath/golfbears/DeepXi/set/test_noisy_speech'
		home_dirs = [
			'/home/ml/speech-aligner/egs/cn_phn/data/small'
		]
		for home_dir in home_dirs:
			w_list = os.listdir(home_dir)
			if not isinstance(test_epoch, list): test_epoch = [test_epoch]
			for e in test_epoch:
				if e < 1: raise ValueError("test_epoch must be greater than 0.")
				self.model.load_weights(model_path + '/epoch-' + str(e - 1) +
										'/variables/variables')
				for w in w_list:
					if not w.__contains__('.wav'):
						continue
					w_f = os.path.join(home_dir, w)
					wav_r, f_s = librosa.load(w_f, sr=16000, mono=True, dtype=np.float32)
					meg1, angle1 = self.inp_tgt.polar_analysis(wav_r)
					meg1 = np.where(meg1 == 0, np.finfo(float).eps, meg1)
					logmeg1 = np.log(meg1)
					fr_len = np.shape(logmeg1)[0]
					g_hists = histogram(logmeg1[0], alpha_d=0.9, alpha_s=0.9, frame_L=100, fft_len=512,
										delta=5)

					alpha_list = np.arange(0.01, 0.03, 0.02)
					betta_list = np.arange(0.5, 0.7, 0.2)
					mean = np.expand_dims(np.mean(logmeg1[:25, :], axis=0), 0)
					stand = np.expand_dims(np.std(logmeg1[:25, :], axis=0), 0)
					tgt_hat_batch = self.model.predict(np.expand_dims(meg1, 0), batch_size=1, verbose=1)
					posterior_pro = np.squeeze(tgt_hat_batch, 0)
					label_predict = np.argmax(posterior_pro, axis=-1)
					labels = [ph_label_mapper.tkn_dict.index2Entry[i] for i in label_predict]
					print(w + 'infer:' )
					print(labels)
					#pro_er1 = posterior_pro[:, 50]
					#path11 = os.path.join(clean_dir + '/infers', w.split('.')[0] + '_' + 'label')
					#np.save(path11, posterior_pro)
					#posterior_pro = np.delete(posterior_pro, 50, axis=1)
					posterior_pro = np.delete(posterior_pro, [20, 39, 42, 50], axis=1)

					p_predict = np.max(posterior_pro, axis=-1)
					sil_pro = posterior_pro[0]
					predict_norm_check = np.sum(posterior_pro, axis=-1)
					for a in alpha_list:
						for b in betta_list:

							h_m_max.updata_alpha_betta(a, b)
							h_m_max.update_noise(mean, stand)
							h_m_max.init_mixmax_win(logmeg1[0])
							print('############################ process file - ' + w)
							#print('############################ process hyperparams - ' + w + ': alpha.' + str(
							#	round(a, 3)) + ' betta.' + str(round(b, 1)))
							x_hat = np.zeros(257, float)
							rho_mix = np.zeros(257, float)
							u_hat = np.zeros(257, float)
							sigma_hat = np.zeros(257, float)
							hist_n_i = np.zeros(257, float)
							hist_n1_i = np.zeros(257, float)
							hist_n2_i = np.zeros(257, float)
							h_n_mu = np.zeros([fr_len, 257], float)
							h_n1_mu = np.zeros([fr_len, 257], float)
							h_n2_mu = np.zeros([fr_len, 257], float)
							h_n_std = np.zeros([fr_len, 257], float)
							h_n1_std = np.zeros([fr_len, 257], float)
							h_n2_std = np.zeros([fr_len, 257], float)
							for i, logf in enumerate(logmeg1):
								hists_noise, hists_noise1, hists_noise2 = g_hists.tracking_noise(logf, i)
								h_n_mu[i], h_n_std[i], h_n1_mu[i], h_n1_std[i], h_n2_mu[i], h_n2_std[
									i] = g_hists.get_mu_std()
								hist_n_i = np.vstack((hist_n_i, hists_noise))
								hist_n1_i = np.vstack((hist_n1_i, hists_noise1))
								hist_n2_i = np.vstack((hist_n2_i, hists_noise2))

								h_m_max.update_noise(np.expand_dims(h_n2_mu[i], 0), np.expand_dims(h_n2_std[i], 0))
								o, p_mm = h_m_max.x_estimate_mixmax(posterior_pro[i,1:], logf)
								x_hat = np.vstack((x_hat, o))
								rho_mix = np.vstack((rho_mix, logf))
								mu, sig = h_m_max.get_noise()
								u_hat = np.vstack((u_hat, mu))
								sigma_hat = np.vstack((sigma_hat, sig))

							meg = np.exp(x_hat[1:])
							rho_mix = rho_mix[1:]
							u_hat = u_hat[1:]
							sigma_hat = sigma_hat[1:]
							wav = self.inp_tgt.polar_synthesis(meg, angle1)
							path2 = os.path.join(clean_dir+'/cleans', w.split('.')[0] + '_' + str(round(a, 3)) + '_' + str(
								round(b, 1)) + '.wav')
							sf.write(path2, wav.numpy(), 16000)
							'''
							for bin in range(257):
								plt.figure(figsize=(20, 10))
								plt.suptitle('Tracking noise at bin: ' + str(bin))
	
								plt.subplot(1, 2, 1)
								epochs = range(1, np.shape(u_hat)[0] + 1)
								plt.plot(epochs, rho_mix[:, bin], 'b--o', label='u_hat')
								plt.title('tracking mu')
								plt.xlabel('frame Numbers')
								plt.ylabel('mu')
								plt.legend()
								plt.grid()
	
								plt.subplot(1, 2, 2)
								#epochs = range(1, np.shape(x_hat)[0] + 1)
								plt.plot(epochs, x_hat[1:, bin], 'b--o', label='sigma_hat')
								plt.title('tracking sigma')
								plt.xlabel('frame numbers')
								plt.ylabel('sigma')
								plt.legend()
								plt.grid()
								analysis_path = os.path.join(clean_dir,'analysis')
								ensures_dir(analysis_path)
								fig_title = os.path.join(analysis_path, w+'Tracking_noise_' + str(bin) + '_bin.png')
								plt.savefig(fig_title)
	
								# plt.show()
								plt.close()
							'''
				print("Processing observations...")

	def infer_hybrid_mcra(
		self,
		test_x,
		test_x_len,
		test_x_base_names,
		test_epoch,
		model_path='model',
		out_type='y',
		gain='mmse-lsa',
		out_path='out',
		n_filters=40,
		saved_data_path=None,
		):
		"""
		Deep Xi inference. The specified 'out_type' is saved.

		Argument/s:
			test_x - noisy-speech test batch.
			test_x_len - noisy-speech test batch lengths.
			test_x_base_names - noisy-speech base names.
			test_epoch - epoch to test.
			model_path - path to model directory.
			out_type - output type (see deepxi/args.py).
			gain - gain function (see deepxi/args.py).
			out_path - path to save output files.
			saved_data_path - path to saved data necessary for enhancement.
		"""
		ph_label_mapper = Phoneme(
			os.path.join('/home/devpath/golfbears/DeepXi/hybrid', 'initialfinal2phoneme-lexicon.txt'))
		means, stds, probs = phoneme_extract_gaussians()
		h_m_max = hybridMixMax(np.array(means), np.array(stds), np.array(means[0]), np.array(stds[0]), np.array(probs), 0.8, 2.0)

		clean_dir = '/home/devpath/datasets/new/small'
		home_dir = '/home/devpath/datasets/new/small'#'/home/devpath/golfbears/DeepXi/set/test_noisy_speech'
		w_list = os.listdir(home_dir)
		if not isinstance(test_epoch, list): test_epoch = [test_epoch]
		for e in test_epoch:
			if e < 1: raise ValueError("test_epoch must be greater than 0.")
			self.model.load_weights(model_path + '/epoch-' + str(e - 1) +
									'/variables/variables')
			for w in w_list:
				if not w.__contains__('.wav'):
					continue
				w_f = os.path.join(home_dir, w)
				wav_r, f_s = librosa.load(w_f, sr=16000, mono=True, dtype=np.float32)
				meg1, angle1 = self.inp_tgt.polar_analysis(wav_r)
				meg1 = np.where(meg1 == 0, np.finfo(float).eps, meg1)
				logmeg1 = np.log(meg1)
				meg_pwr = np.square(meg1)
				low_fr = np.ones(17, float) * 2
				mid_fr = np.ones(48, float) * 2.8
				hi_fr = np.ones(192, float) * 5
				delta = np.concatenate((low_fr, mid_fr, hi_fr))
				imcra_noise = imcra(alpha_d=0.89, alpha_s=0.8, alpha_p=0.2, lambda_d=meg_pwr[0], frame_L=100,
									fft_len=512,
									delta=delta, beta=1.23, b_min=1.66, gamma0=4.6, gamma1=3, zeta0=1.67)
				alpha_list = np.arange(0.01, 0.03, 0.02)
				betta_list = np.arange(0.5, 0.7, 0.2)
				mean = np.expand_dims(np.mean(logmeg1[:25, :], axis=0), 0)
				stand = np.expand_dims(np.std(logmeg1[:25, :], axis=0), 0)
				tgt_hat_batch = self.model.predict(np.expand_dims(meg1, 0), batch_size=1, verbose=1)
				posterior_pro = np.squeeze(tgt_hat_batch, 0)
				label_predict = np.argmax(posterior_pro, axis=-1)
				labels = [ph_label_mapper.tkn_dict.index2Entry[i] for i in label_predict]
				print(w + 'infer:' )
				print(labels)
				#pro_er1 = posterior_pro[:, 50]
				#path11 = os.path.join(clean_dir + '/infers', w.split('.')[0] + '_' + 'label')
				#np.save(path11, posterior_pro)
				#posterior_pro = np.delete(posterior_pro, 50, axis=1)
				posterior_pro = np.delete(posterior_pro, [20, 39, 42, 50], axis=1)

				p_predict = np.max(posterior_pro, axis=-1)
				sil_pro = posterior_pro[0]
				predict_norm_check = np.sum(posterior_pro, axis=-1)
				for a in alpha_list:
					for b in betta_list:
						h_m_max.updata_alpha_betta(a, b)
						h_m_max.update_noise(mean, stand)
						h_m_max.init_mixmax_win(logmeg1[0])
						print('############################ process file - ' + w)
						#print('############################ process hyperparams - ' + w + ': alpha.' + str(
						#	round(a, 3)) + ' betta.' + str(round(b, 1)))
						x_hat = np.zeros(257, float)
						rho_mix = np.zeros(257, float)
						u_hat = np.zeros(257, float)
						sigma_hat = np.zeros(257, float)
						m_noise_i = np.zeros(257, float)
						m_G_i = np.zeros(257, float)
						m_present_i = np.zeros(257, float)
						for i, logf in enumerate(logmeg1):
							c_noise_i, G, P = imcra_noise.tracking_noise(meg_pwr[i], i)
							#P_pre = np.minimum(P, 0.2)
							#P_pre = np.maximum(P_pre, 0.8)
							P_pre = P
							P_absent = (1-P_pre)*0.99
							P_absent_hat = P_absent*posterior_pro[i,0]
							P_pre_hat = (1-P_absent_hat)
							posterior_absent = np.minimum(P_absent, 1-posterior_pro[i,0]*0.99)
							Ratio = P_pre*0.9989/(1-posterior_pro[i,0]*0.9989)
							#o, p_mm = h_m_max.x_estimate_mixmax_bak(posterior_pro[i,1:], logf)
							o, p_mm = h_m_max.x_estimate_mixmax(posterior_pro[i,1:], logf)
							x_hat = np.vstack((x_hat, o))
							rho_mix = np.vstack((rho_mix, logf))
							mu, sig = h_m_max.get_noise()
							u_hat = np.vstack((u_hat, mu))
							sigma_hat = np.vstack((sigma_hat, sig))

						meg = np.exp(x_hat[1:])
						rho_mix = rho_mix[1:]
						u_hat = u_hat[1:]
						sigma_hat = sigma_hat[1:]
						wav = self.inp_tgt.polar_synthesis(meg, angle1)
						path2 = os.path.join(clean_dir+'/cleans', w.split('.')[0] + '_' + str(round(a, 3)) + '_' + str(
							round(b, 1)) + '.wav')
						sf.write(path2, wav.numpy(), 16000)
						'''
						for bin in range(257):
							plt.figure(figsize=(20, 10))
							plt.suptitle('Tracking noise at bin: ' + str(bin))

							plt.subplot(1, 2, 1)
							epochs = range(1, np.shape(u_hat)[0] + 1)
							plt.plot(epochs, rho_mix[:, bin], 'b--o', label='u_hat')
							plt.title('tracking mu')
							plt.xlabel('frame Numbers')
							plt.ylabel('mu')
							plt.legend()
							plt.grid()

							plt.subplot(1, 2, 2)
							#epochs = range(1, np.shape(x_hat)[0] + 1)
							plt.plot(epochs, x_hat[1:, bin], 'b--o', label='sigma_hat')
							plt.title('tracking sigma')
							plt.xlabel('frame numbers')
							plt.ylabel('sigma')
							plt.legend()
							plt.grid()
							analysis_path = os.path.join(clean_dir,'analysis')
							ensures_dir(analysis_path)
							fig_title = os.path.join(analysis_path, w+'Tracking_noise_' + str(bin) + '_bin.png')
							plt.savefig(fig_title)

							# plt.show()
							plt.close()
						'''
			print("Processing observations...")

	def infer_noisy_wav(
			self,
			test_s,
			test_x,
			test_x_len,
			test_x_base_names,
			test_epoch,
			model_path='model',
			out_type='y',
			gain='mmse-lsa',
			out_path='out',
			n_filters=40,
			saved_data_path=None,
	):
		"""
		Deep Xi inference. The specified 'out_type' is saved.

		Argument/s:
			test_x - noisy-speech test batch.
			test_x_len - noisy-speech test batch lengths.
			test_x_base_names - noisy-speech base names.
			test_epoch - epoch to test.
			model_path - path to model directory.
			out_type - output type (see deepxi/args.py).
			gain - gain function (see deepxi/args.py).
			out_path - path to save output files.
			saved_data_path - path to saved data necessary for enhancement.
		"""
		# ph_label_mapper = Phoneme(
		#	os.path.join('/home/devpath/golfbears/DeepXi/hybrid', 'initialfinal2phoneme-lexicon.txt'))
		# home_dir = '/home/devpath/datasets/noisetracking'
		home_dir = '/home/devpath/datasets/kws_test_noisy'

		process_length = np.floor(len(test_s)/8)
		w_list = np.arange(0,process_length)
		for idx, f in tqdm(enumerate(w_list), desc='test', total=process_length):
			offset = 8*idx
			s_mbatch_list = test_s[offset:offset+8]
			d_mbatch_list = random.sample(test_x, 8)
			s_mbatch, d_mbatch, s_mbatch_len, d_mbatch_len, snr_mbatch = \
				self.wav_batch(s_mbatch_list, d_mbatch_list)
			s, d, x, n_frames = \
				self.inp_tgt.mix(s_mbatch, d_mbatch, s_mbatch_len,
								 d_mbatch_len, snr_mbatch)
			ensures_dir(home_dir)
			for idx in range(8):
				w_len = s_mbatch_list[idx]['wav_len']
				filename = s_mbatch_list[idx]['file_path'].split('/')[-1]
				'''
				noisename = d_mbatch_list[idx]['file_path'].split('/')[-1]
				analysis_path = os.path.join(home_dir, filename.split('.')[0] + '_' + noisename.split('.')[0] + '_' + str(
					snr_mbatch[idx]) + 'dB.wav')
				'''
				analysis_path = os.path.join(home_dir, filename.split('.')[0] + '_' + str(snr_mbatch[idx]) + 'dB.wav')
				sf.write(analysis_path, x[idx, :w_len].numpy(), 16000)

	def infer_tracking_noise(
		self,
		test_s,
		test_x,
		test_x_len,
		test_x_base_names,
		test_epoch,
		model_path='model',
		out_type='y',
		gain='mmse-lsa',
		out_path='out',
		n_filters=40,
		saved_data_path=None,
		):
		"""
		Deep Xi inference. The specified 'out_type' is saved.

		Argument/s:
			test_x - noisy-speech test batch.
			test_x_len - noisy-speech test batch lengths.
			test_x_base_names - noisy-speech base names.
			test_epoch - epoch to test.
			model_path - path to model directory.
			out_type - output type (see deepxi/args.py).
			gain - gain function (see deepxi/args.py).
			out_path - path to save output files.
			saved_data_path - path to saved data necessary for enhancement.
		"""
		#ph_label_mapper = Phoneme(
		#	os.path.join('/home/devpath/golfbears/DeepXi/hybrid', 'initialfinal2phoneme-lexicon.txt'))
		#home_dir = '/home/devpath/datasets/noisetracking'
		home_dir = '/home/devpath/datasets/shell_train_noisy'

		means, stds, probs = phoneme_extract_gaussians()
		h_m_max = hybridMixMax(np.array(means), np.array(stds), np.array(means[0]), np.array(stds[0]), np.array(probs), 0.8, 2.0)
		h_m_max_nn = hybridMixMax(np.array(means), np.array(stds), np.array(means[0]), np.array(stds[0]), np.array(probs), 0.8, 2.0)
		h_m_max_mm = hybridMixMax(np.array(means), np.array(stds), np.array(means[0]), np.array(stds[0]), np.array(probs), 0.8, 2.0)
		s_mbatch_list = random.sample(test_s, 8)
		d_mbatch_list = random.sample(test_x, 8)
		s_mbatch, d_mbatch, s_mbatch_len, d_mbatch_len, snr_mbatch = \
			self.wav_batch(s_mbatch_list, d_mbatch_list)
		s, d, x, n_frames = \
			self.inp_tgt.mix(s_mbatch, d_mbatch, s_mbatch_len,
							 d_mbatch_len, snr_mbatch)

		inp_mbatch, angle1 = self.inp_tgt.polar_analysis(x)
		n_mbatch, _ = self.inp_tgt.polar_analysis(d)
		inp_mbatch = np.where(inp_mbatch == 0, np.finfo(float).eps, inp_mbatch)
		n_mbatch = np.where(n_mbatch == 0, np.finfo(float).eps, n_mbatch)
		inp_log_mbatch = np.log(inp_mbatch)
		n_log_mbatch = np.log(n_mbatch)
		G_min = np.ones(257, float) * 0.09
		if not isinstance(test_epoch, list): test_epoch = [test_epoch]
		for e in test_epoch:
			if e < 1: raise ValueError("test_epoch must be greater than 0.")
			self.model.load_weights(model_path + '/epoch-' + str(e - 1) +
									'/variables/variables')
			tgt_hat_batch = self.model.predict(inp_mbatch, batch_size=1, verbose=1)
			posterior_pro = np.delete(tgt_hat_batch, 50, axis=2)
			#posterior_pro = np.delete(tgt_hat_batch, [20, 39, 42, 50], axis=2)
			pho_probs = posterior_pro[:, :, 1:]
			sap_nn =  posterior_pro[:, :, 1]
			ssp_nn = 1 - sap_nn
			ssp_nn = np.where(ssp_nn == 0, np.finfo(float).eps, ssp_nn)
			normal_pho_probs = pho_probs / np.expand_dims(np.sum(pho_probs, axis=2), axis=2)
			mean = np.mean(inp_log_mbatch[:, :25, :], axis=1)
			stand = np.std(inp_log_mbatch[:, :25, :], axis=1)
			for idx in range(8):
				#mean = np.expand_dims(np.mean(inp_log_mbatch[idx, :25, :], axis=0), 0)
				#stand = np.expand_dims(np.std(inp_log_mbatch[idx, :25, :], axis=0), 0)
				#sigma = np.square(stand)

				g_vts_max = rt_vts_noise(np.array(means), np.square(np.array(stds)), np.array(probs), np.expand_dims(mean[idx],0), np.expand_dims(np.square(stand[idx]),0),  1)
				fr_len = s_mbatch_list[idx]['frame_len']
				w_len = s_mbatch_list[idx]['wav_len']
				filename = s_mbatch_list[idx]['file_path'].split('/')[-1]
				noisename = d_mbatch_list[idx]['file_path'].split('/')[-1]
				analysis_path = os.path.join(home_dir, filename.split('.')[0]+'_'+noisename.split('.')[0]+'_'+str(snr_mbatch[idx])+'dB')
				ensures_dir(analysis_path)
				sf.write(os.path.join(analysis_path, 'clean_speech.wav'), s[idx,:w_len].numpy(), 16000)
				sf.write(os.path.join(analysis_path, 'noisy_speech.wav'), x[idx,:w_len].numpy(), 16000)
				sf.write(os.path.join(analysis_path, 'noise.wav'), d[idx,:w_len].numpy(), 16000)
				alpha_list = np.arange(0.31, 0.41, 0.1)
				betta_list = np.arange(0.5, 0.7, 0.2)

				meg_pwr = np.square(inp_mbatch[idx, :fr_len])
				g_hists = histogram(inp_log_mbatch[idx, 0], alpha_d=0.9, alpha_s=0.9, frame_L=400, fft_len=512,
									delta=5)
				low_fr = np.ones(17, float) * 2
				mid_fr = np.ones(48, float) * 2.8
				hi_fr = np.ones(192, float) * 5
				delta = np.concatenate((low_fr, mid_fr, hi_fr))
				angle_tmp = angle1[idx, :fr_len]
				imcra_noise = imcra(alpha_d=0.89, alpha_s=0.8, alpha_p=0.2, lambda_d=meg_pwr[0], frame_L=100,
									fft_len=512,
									delta=delta, beta=1.23, b_min=1.66, gamma0=4.6, gamma1=3, zeta0=1.67)
				m_present = np.zeros(257, float)
				m_noise_i = np.zeros(257, float)
				m_omlsa_i = np.zeros(257, float)
				hist_n_i = np.zeros(257, float)
				hist_n1_i = np.zeros(257, float)
				hist_n2_i = np.zeros(257, float)
				h_n_mu = np.zeros([fr_len, 257], float)
				h_n1_mu = np.zeros([fr_len, 257], float)
				h_n2_mu = np.zeros([fr_len, 257], float)
				h_n_std = np.zeros([fr_len, 257], float)
				h_n1_std = np.zeros([fr_len, 257], float)
				h_n2_std = np.zeros([fr_len, 257], float)
				for i, pwr in enumerate(meg_pwr[:fr_len]):
					c_noise,G,P=imcra_noise.tracking_noise(pwr, i)
					m_present = np.vstack((m_present, P))
					m_noise_i = np.vstack((m_noise_i, c_noise))
					omlsa = np.power(G, P) * np.power(G_min, (1 - P)) * inp_mbatch[idx,i]
					m_omlsa_i = np.vstack((m_omlsa_i, omlsa))
					#m_noise_i = np.vstack((m_noise_i, c_noise))
					hists_noise, hists_noise1, hists_noise2 = g_hists.tracking_noise(inp_log_mbatch[idx, i], i)
					h_n_mu[i], h_n_std[i], h_n1_mu[i], h_n1_std[i], h_n2_mu[i], h_n2_std[i] = g_hists.get_mu_std()
					hist_n_i = np.vstack((hist_n_i, hists_noise))
					hist_n1_i = np.vstack((hist_n1_i, hists_noise1))
					hist_n2_i = np.vstack((hist_n2_i, hists_noise2))
				m_present = m_present[1:]
				m_omlsa_i = m_omlsa_i[1:]
				m_noise_i = np.log(m_noise_i[1:])
				hist_n_i = hist_n_i[1:]
				hist_n1_i = hist_n1_i[1:]
				hist_n2_i = hist_n2_i[1:]
				wav = self.inp_tgt.polar_synthesis(m_omlsa_i, angle_tmp)
				path2 = os.path.join(analysis_path, 'enhanced_omlsa.wav')
				sf.write(path2, wav.numpy(), 16000)

				nn_spp = np.expand_dims(ssp_nn[idx, :fr_len], -1)
				for a in alpha_list:
					for b in betta_list:
						h_m_max.updata_alpha_betta(a, b)
						h_m_max.update_noise(np.expand_dims(mean[idx], 0), np.expand_dims(stand[idx], 0))
						h_m_max.init_mixmax_win(inp_log_mbatch[idx][0])

						h_m_max_nn.updata_alpha_betta(a, b)
						h_m_max_nn.update_noise(np.expand_dims(mean[idx], 0), np.expand_dims(stand[idx], 0))
						h_m_max_nn.init_mixmax_win(inp_log_mbatch[idx][0])

						h_m_max_mm.updata_alpha_betta(a, b)
						h_m_max_mm.update_noise(np.expand_dims(mean[idx], 0), np.expand_dims(stand[idx], 0))
						h_m_max_mm.init_mixmax_win(inp_log_mbatch[idx][0])

						x_hat = np.zeros(257, float)
						x_hat_nn = np.zeros(257, float)
						x_hat_mm = np.zeros(257, float)
						x_hat_vts = np.zeros(257, float)
						rho_mix = np.zeros(257, float)
						rho_mix_nn = np.zeros(257, float)
						u_hat = np.zeros(257, float)
						sigma_hat = np.zeros(257, float)
						u_noise = n_log_mbatch[idx, 0]
						sigma_noise = np.zeros(257, float)
						u_hybrid = np.zeros(257, float)
						sigma_hybrid = np.zeros(257, float)
						imcra_mu = n_log_mbatch[idx, 0]
						imcra_sigma = np.zeros(257, float)
						hist_n_mu = np.zeros([fr_len,257], float)
						hist_n1_mu = np.zeros([fr_len,257], float)
						hist_n2_mu = np.zeros([fr_len,257], float)
						hist_n_std = np.zeros([fr_len,257], float)
						hist_n1_std = np.zeros([fr_len,257], float)
						hist_n2_std = np.zeros([fr_len,257], float)
						for i, logf in enumerate(inp_log_mbatch[idx,:fr_len]):
							if i != 0:
								ghost_mu = np.mean(n_log_mbatch[idx, :i], axis=0)
								ghost_std = np.std(n_log_mbatch[idx, :i], axis=0)
								ghost_std = np.where(ghost_std == 0, np.finfo(float).eps, ghost_std)
								hist_n_mu[i] = np.mean(hist_n_i[:i], axis=0)
								hist_n_std[i] = np.std(hist_n_i[:i], axis=0)
								hist_n_std[i] = np.where(hist_n_std[i] == 0, np.finfo(float).eps, hist_n_std[i])
								hist_n1_mu[i] = np.mean(hist_n1_i[:i], axis=0)
								hist_n1_std[i] = np.std(hist_n1_i[:i], axis=0)
								hist_n1_std[i] = np.where(hist_n1_std[i] == 0, np.finfo(float).eps, hist_n1_std[i])
								hist_n2_mu[i] = np.mean(hist_n2_i[:i], axis=0)
								hist_n2_std[i] = np.std(hist_n2_i[:i], axis=0)
								hist_n2_std[i] = np.where(hist_n2_std[i] == 0, np.finfo(float).eps, hist_n2_std[i])
							else:
								ghost_mu = np.mean(n_log_mbatch[idx, :20], axis=0)
								ghost_std = np.std(n_log_mbatch[idx, :20], axis=0)
								ghost_std = np.where(ghost_std == 0, np.finfo(float).eps, ghost_std)
								hist_n_mu[i] = np.mean(hist_n_i[:20], axis=0)
								hist_n_std[i] = np.std(hist_n_i[:20], axis=0)
								hist_n_std[i] = np.where(hist_n_std[i] == 0, np.finfo(float).eps, hist_n_std[i])
								hist_n1_mu[i] = np.mean(hist_n1_i[:20], axis=0)
								hist_n1_std[i] = np.std(hist_n1_i[:20], axis=0)
								hist_n1_std[i] = np.where(hist_n1_std[i] == 0, np.finfo(float).eps, hist_n1_std[i])
								hist_n2_mu[i] = np.mean(hist_n2_i[:20], axis=0)
								hist_n2_std[i] = np.std(hist_n2_i[:20], axis=0)
								hist_n2_std[i] = np.where(hist_n2_std[i] == 0, np.finfo(float).eps, hist_n2_std[i])

							h_m_max.update_noise(np.expand_dims(ghost_mu, 0), np.expand_dims(ghost_std, 0))
							o, p_mm = h_m_max.x_estimate_mixmax_nn(posterior_pro[idx, i, 1:], logf)
							#h_m_max.tracking_mu_sigma(o,p_mm)
							x_hat = np.vstack((x_hat, o))
							rho_mix = np.vstack((rho_mix, p_mm))
							mu, sig = h_m_max.get_noise()
							u_hat = np.vstack((u_hat, mu))
							sigma_hat = np.vstack((sigma_hat, sig))

							h_m_max_nn.update_noise(np.expand_dims(hist_n2_mu[i], 0), np.expand_dims(hist_n2_std[i], 0))
							o, p_mm = h_m_max_nn.x_estimate_mixmax_nn(posterior_pro[idx, i, 1:], logf)
							#h_m_max_nn.tracking_mu_sigma(o, p_mm)
							x_hat_nn = np.vstack((x_hat_nn, o))
							rho_mix_nn = np.vstack((rho_mix_nn, p_mm))
							mu_hybrid, sig_hybrid = h_m_max_nn.get_noise()
							u_hybrid = np.vstack((u_hybrid, mu_hybrid))
							sigma_hybrid = np.vstack((sigma_hybrid, sig_hybrid))

							#h_m_max_nn.update_noise(np.expand_dims(hist_n2_mu[i], 0), np.expand_dims(hist_n2_std[i], 0))
							o, p_mm = h_m_max_mm.x_estimate_mixmax_nn(posterior_pro[idx, i, 1:], logf)
							h_m_max_mm.tracking_mu_sigma(o, p_mm)
							x_hat_mm = np.vstack((x_hat_mm, o))
							#g_vts_max.update_mu_sigma(mu_hybrid,np.square(sig_hybrid))
							g_vts_max.update_mu_sigma(np.expand_dims(ghost_mu,0), np.expand_dims(np.square(ghost_std),0))
							g_vts_max.compensate_model()
							#g_vts_max.calculate_P_o_t_k_l(np.expand_dims(logf,0))
							g_vts_max.hybrid_P_o_t_k_l(np.expand_dims(logf,0), np.expand_dims(normal_pho_probs[idx,i],0))
							meg, noi = g_vts_max.update_signal_noise_spp(np.expand_dims(logf,0), np.expand_dims(nn_spp[i],0))
							#meg, noi = g_vts_max.update_signal_noise_spp(np.expand_dims(logf,0), np.expand_dims(m_present[i],0))
							x_hat_vts = np.vstack((x_hat_vts, np.squeeze(meg)))
							if i != 0:
								u_noise = np.vstack((u_noise, np.mean(n_log_mbatch[idx, :i], axis=0)))
								sigma_noise = np.vstack((sigma_noise, np.std(n_log_mbatch[idx, :i], axis=0)))
								imcra_mu = np.vstack((imcra_mu, np.mean(m_noise_i[:i], axis=0)))
								imcra_sigma = np.vstack((imcra_sigma, np.std(m_noise_i[:i], axis=0)))

						u_hybrid = u_hybrid[1:]
						sigma_hybrid = sigma_hybrid[1:]
						u_hat = u_hat[1:]
						sigma_hat = sigma_hat[1:]
						x_hat = x_hat[1:]
						x_hat_nn = x_hat_nn[1:]
						x_hat_mm = x_hat_mm[1:]
						x_hat_vts = x_hat_vts[1:]
						rho_mix = rho_mix[1:]
						rho_mix_nn = rho_mix_nn[1:]
						#imcra_mu = imcra_mu[1:]
						#imcra_sigma = imcra_sigma[1:]
						meg = np.exp(x_hat)
						wav = self.inp_tgt.polar_synthesis(meg, angle_tmp)
						path2 = os.path.join(analysis_path,  'enhanced_real_noise' + '_' + str(round(a, 3)) + '_' + str( round(b, 1)) + '.wav')
						sf.write(path2, wav.numpy(), 16000)
						meg = np.exp(x_hat_nn)
						wav = self.inp_tgt.polar_synthesis(meg, angle_tmp)
						path2 = os.path.join(analysis_path,
										 'enhanced_hist' + '_' + str(round(a, 3)) + '_' + str(round(b, 1)) + '.wav')
						sf.write(path2, wav.numpy(), 16000)
						meg = np.exp(x_hat_mm)
						wav = self.inp_tgt.polar_synthesis(meg, angle_tmp)
						path2 = os.path.join(analysis_path,
											 'enhanced_hybrid' + '_' + str(round(a, 3)) + '_' + str(round(b, 1)) + '.wav')
						sf.write(path2, wav.numpy(), 16000)
						meg = np.exp(x_hat_vts)
						wav = self.inp_tgt.polar_synthesis(meg, angle_tmp)
						path2 = os.path.join(analysis_path,
											 'enhanced_vts' + '_' + str(round(a, 3)) + '_' + str(round(b, 1)) + '.wav')
						sf.write(path2, wav.numpy(), 16000)

				for bin in range(257):

					plt.figure(figsize=(20, 10))
					plt.suptitle('Tracking noise at bin: '+ str(bin))

					plt.subplot(2, 2, 1)
					epochs = range(1, np.shape(u_hat)[0] + 1)
					plt.plot(epochs, u_noise[:,bin], 'r-', label='u_noise')
					#plt.plot(epochs, u_hat[:,bin], 'b--', label='u_hat')
					plt.plot(epochs, u_hybrid[:, bin], 'g--', label='u_hybrid')
					plt.plot(epochs, imcra_mu[:,bin], 'k-', label='u_imcra')
					plt.plot(epochs, hist_n_mu[:, bin], 'c:.', label='hist_n_mu')
					plt.plot(epochs, hist_n1_mu[:, bin], 'm:.', label='hist_n1_mu')
					plt.plot(epochs, hist_n2_mu[:, bin], 'b:.', label='hist_n2_mu')
					plt.plot(epochs, h_n_mu[:, bin], 'c:', label='h_n_mu')
					plt.plot(epochs, h_n1_mu[:, bin], 'm:', label='h_n1_mu')
					plt.plot(epochs, h_n2_mu[:, bin], 'b:', label='h_n2_mu')
					plt.title('tracking mu')
					plt.xlabel('frame Numbers')
					plt.ylabel('mu')
					plt.legend()
					plt.grid()

					plt.subplot(2, 2, 2)
					# epochs = range(1, len(f1) + 1)
					plt.plot(epochs, sigma_noise[:,bin], 'r-', label='sigma_noise')
					#plt.plot(epochs, sigma_hat[:,bin], 'b--', label='sigma_hat')
					plt.plot(epochs, sigma_hybrid[:, bin], 'g--', label='sigma_hybrid')
					plt.plot(epochs, imcra_sigma[:, bin], 'k-', label='sigma_imcra')
					plt.plot(epochs, hist_n_std[:, bin], 'c:.', label='hist_n_std')
					plt.plot(epochs, hist_n1_std[:, bin], 'm:.', label='hist_n1_std')
					plt.plot(epochs, hist_n2_std[:, bin], 'b:.', label='hist_n2_std')
					plt.plot(epochs, h_n_std[:, bin], 'c:', label='h_n_std')
					plt.plot(epochs, h_n1_std[:, bin], 'm:', label='h_n1_std')
					plt.plot(epochs, h_n2_std[:, bin], 'b:', label='h_n2_std')
					plt.title('tracking sigma')
					plt.xlabel('frame numbers')
					plt.ylabel('sigma')
					plt.legend()
					plt.grid()

					plt.subplot(2, 2, 3)
					# epochs = range(1, len(f1) + 1)
					plt.plot(epochs, n_log_mbatch[idx, :fr_len, bin], 'r-.', label='noise log magnitude')
					plt.plot(epochs, m_noise_i[:, bin], 'b--', label='molsa log magnitude')
					plt.plot(epochs, hist_n_i[:, bin], 'y--', label='hist log magnitude')
					plt.plot(epochs, hist_n1_i[:, bin], 'c--', label='hist1 log magnitude')
					plt.plot(epochs, hist_n2_i[:, bin], 'm--', label='hist2 log magnitude')

					plt.title('tracking noise log mag')
					plt.xlabel('frame numbers')
					plt.ylabel('log magnitude')
					plt.legend()
					plt.grid()

					plt.subplot(2, 2, 4)
					# epochs = range(1, len(f1) + 1)
					plt.plot(epochs, n_log_mbatch[idx, :fr_len, bin], 'r-.', label='log magnitude')
					plt.plot(epochs, x_hat[:, bin], 'b--', label='paper log magnitude')
					plt.plot(epochs, x_hat_nn[:, bin], 'g--', label='nn log magnitude')
					plt.plot(epochs, x_hat_vts[:, bin], 'y--', label='vts log magnitude')
					plt.plot(epochs, np.log(m_omlsa_i[:, bin]), 'm:.', label='omlsa log magnitude')
					plt.title('tracking clean signal')
					plt.xlabel('frame numbers')
					plt.ylabel('log magnitude')
					plt.legend()
					plt.grid()
					'''
					plt.subplot(2, 2, 3)
					# epochs = range(1, len(f1) + 1)
					plt.plot(epochs, n_log_mbatch[idx,:fr_len,bin], 'r-.', label='log magnitude')
					plt.plot(epochs, x_hat[:,bin], 'b--', label='paper log magnitude')
					plt.plot(epochs, x_hat_nn[:, bin], 'g--', label='nn log magnitude')
					plt.plot(epochs, x_hat_vts[:, bin], 'y--', label='vts log magnitude')
					plt.plot(epochs, np.log(m_omlsa_i[:, bin]), 'm:.', label='omlsa log magnitude')
					plt.title('tracking clean signal')
					plt.xlabel('frame numbers')
					plt.ylabel('log magnitude')
					plt.legend()
					plt.grid()

					plt.subplot(2, 2, 4)
					# epochs = range(1, len(f1) + 1)
					plt.plot(epochs, m_present[:,bin], 'r-.', label='omlsa_present')
					plt.plot(epochs, rho_mix[:,bin], 'b--o', label='rho_mix_paper')
					plt.plot(epochs, rho_mix_nn[:, bin], 'g--o', label='rho_mix_nn')
					plt.title('tracking present')
					plt.xlabel('frame numbers')
					plt.ylabel('present')
					plt.legend()
					plt.grid()
					'''
					fig_title = os.path.join(analysis_path,'Tracking_noise_'+ str(bin) + '_bin.png')
					plt.savefig(fig_title)
					#plt.show()
					plt.close()

			print("Processing observations...")

	def infer_tracking_noise_mcra(
		self,
		test_s,
		test_x,
		test_x_len,
		test_x_base_names,
		test_epoch,
		model_path='model',
		out_type='y',
		gain='mmse-lsa',
		out_path='out',
		n_filters=40,
		saved_data_path=None,
		):
		"""
		Deep Xi inference. The specified 'out_type' is saved.

		Argument/s:
			test_x - noisy-speech test batch.
			test_x_len - noisy-speech test batch lengths.
			test_x_base_names - noisy-speech base names.
			test_epoch - epoch to test.
			model_path - path to model directory.
			out_type - output type (see deepxi/args.py).
			gain - gain function (see deepxi/args.py).
			out_path - path to save output files.
			saved_data_path - path to saved data necessary for enhancement.
		"""

		home_dir = '/home/devpath/datasets/noisetracking'

		means, stds, probs = phoneme_extract_gaussians()
		h_m_max = hybridMixMax(np.array(means), np.array(stds), np.array(means[0]), np.array(stds[0]), np.array(probs), 0.8, 2.0)

		s_mbatch_list = random.sample(test_s, 8)
		d_mbatch_list = random.sample(test_x, 8)
		s_mbatch, d_mbatch, s_mbatch_len, d_mbatch_len, snr_mbatch = \
			self.wav_batch(s_mbatch_list, d_mbatch_list)
		s, d, x, n_frames = \
			self.inp_tgt.mix(s_mbatch, d_mbatch, s_mbatch_len,
							 d_mbatch_len, snr_mbatch)

		inp_mbatch, angle1 = self.inp_tgt.polar_analysis(x)
		n_mbatch, _ = self.inp_tgt.polar_analysis(d)
		s_mbatch, _ = self.inp_tgt.polar_analysis(s)
		inp_mbatch = np.where(inp_mbatch == 0, np.finfo(float).eps, inp_mbatch)
		n_mbatch = np.where(n_mbatch == 0, np.finfo(float).eps, n_mbatch)
		inp_log_mbatch = np.log(inp_mbatch)
		n_log_mbatch = np.log(n_mbatch)
		tgt_hat_batch = self.model.predict(inp_log_mbatch, batch_size=1, verbose=1)
		posterior_pro = np.delete(tgt_hat_batch, 50, axis=2)
		mean = np.mean(inp_log_mbatch[:, :25, :], axis=1)
		stand = np.std(inp_log_mbatch[:, :25, :], axis=1)
		G_min = np.ones(257, float)* 0.09
		if not isinstance(test_epoch, list): test_epoch = [test_epoch]
		for e in test_epoch:
			if e < 1: raise ValueError("test_epoch must be greater than 0.")
			self.model.load_weights(model_path + '/epoch-' + str(e - 1) +
									'/variables/variables')
			for idx in range(8):
				fr_len = s_mbatch_list[idx]['frame_len']
				w_len = s_mbatch_list[idx]['wav_len']
				filename = s_mbatch_list[idx]['file_path'].split('/')[-1]
				noisename = d_mbatch_list[idx]['file_path'].split('/')[-1]
				analysis_path = os.path.join(home_dir, filename.split('.')[0]+'_'+noisename.split('.')[0]+'_'+str(snr_mbatch[idx])+'dB')
				ensures_dir(analysis_path)
				sf.write(os.path.join(analysis_path, 'clean_speech.wav'), s[idx, :w_len].numpy(), 16000)
				sf.write(os.path.join(analysis_path, 'noisy_speech.wav'), x[idx, :w_len].numpy(), 16000)
				sf.write(os.path.join(analysis_path, 'noise.wav'), d[idx, :w_len].numpy(), 16000)
				alpha_list = np.arange(0.01, 0.03, 0.02)
				betta_list = np.arange(0.5, 0.7, 0.2)
				meg_pwr = np.square(inp_mbatch[idx])
				mag = inp_mbatch[idx]
				s_pwr = np.square(s_mbatch[idx])
				noise_pwr = np.square(n_mbatch[idx])
				fr_len = s_mbatch_list[idx]['frame_len']
				mcra_noise = mcra(alpha_d=0.95, alpha_s=0.8, alpha_p=0.2, lambda_d=meg_pwr[0], frame_L=100, fft_len=512,
								  delta=5)
				low_fr = np.ones(17, float) * 2
				mid_fr = np.ones(48, float) * 2.8
				hi_fr = np.ones(192, float) * 5
				delta = np.concatenate((low_fr, mid_fr, hi_fr))
				mcra_2_noise = mcra_2(alpha_d=0.89, alpha_s=0.8, alpha_p=0.2, lambda_d=meg_pwr[0], frame_L=100, fft_len=512,
								  delta=delta, gamma=0.998, beta=0.96)

				imcra_noise = imcra(alpha_d=0.89, alpha_s=0.8, alpha_p=0.2, lambda_d=meg_pwr[0], frame_L=100, fft_len=512,
								  delta=delta,beta=1.23, b_min=1.66, gamma0=4.6, gamma1=3, zeta0=1.67)
				m_noise = np.zeros(257, float)
				m_noise_2 = np.zeros(257, float)
				m_noise_i = np.zeros(257, float)
				m_G = np.zeros(257, float)
				m_G_2 = np.zeros(257, float)
				m_G_i = np.zeros(257, float)
				m_omlsa = np.zeros(257, float)
				m_omlsa_2 = np.zeros(257, float)
				m_omlsa_i = np.zeros(257, float)
				m_present = np.zeros(257, float)
				m_present_2 = np.zeros(257, float)
				m_present_i = np.zeros(257, float)

				u_noise = np.zeros(257, float)
				u_G = np.zeros(257, float)
				for i, pwr in enumerate(meg_pwr[:fr_len]):
					c_noise,G,P=mcra_noise.tracking_noise(pwr, i)
					m_noise = np.vstack((m_noise, c_noise))
					m_G = np.vstack((m_G, G))
					m_present = np.vstack((m_present, P))
					omlsa = np.power(G, P)*np.power(G_min,(1-P))* mag[i]
					m_omlsa = np.vstack((m_omlsa, omlsa))
					c_noise_2,G,P = mcra_2_noise.tracking_noise(pwr, i)
					m_noise_2 = np.vstack((m_noise_2, c_noise_2))
					m_G_2 = np.vstack((m_G_2, G))
					m_present_2 = np.vstack((m_present_2, P))
					omlsa = np.power(G, P)*np.power(G_min,(1-P))* mag[i]
					m_omlsa_2 = np.vstack((m_omlsa_2, omlsa))
					c_noise_i,G,P = imcra_noise.tracking_noise(pwr, i)
					m_noise_i = np.vstack((m_noise_i, c_noise_i))
					m_G_i = np.vstack((m_G_i, G))
					m_present_i = np.vstack((m_present_i, P))
					omlsa = np.power(G, P)*np.power(G_min,(1-P))* mag[i]
					m_omlsa_i = np.vstack((m_omlsa_i, omlsa))
					u_noise = np.vstack((u_noise, noise_pwr[i]))
					u_G = np.vstack((u_G, s_pwr[i]/(s_pwr[i]+noise_pwr[i])))

				m_noise = m_noise[1:]
				m_noise_2 = m_noise_2[1:]
				m_noise_i = m_noise_i[1:]
				u_noise = u_noise[1:]
				m_G = m_G[1:]
				m_G_2 = m_G_2[1:]
				m_G_i = m_G_i[1:]
				u_G = u_G[1:]
				u_G = np.minimum(u_G,5)
				m_omlsa = m_omlsa[1:]
				m_omlsa_2 = m_omlsa_2[1:]
				m_omlsa_i = m_omlsa_i[1:]
				m_present = m_present[1:]
				m_present_2 = m_present_2[1:]
				m_present_i = m_present_i[1:]
				angle_tmp = angle1[idx,:fr_len]
				wav = self.inp_tgt.polar_synthesis(m_omlsa, angle_tmp)
				path2 = os.path.join(analysis_path, 'enhanced_omlsa_mcra'+ '.wav')
				sf.write(path2, wav.numpy(), 16000)
				wav = self.inp_tgt.polar_synthesis(m_omlsa_2, angle_tmp)
				path2 = os.path.join(analysis_path, 'enhanced_omlsa_mcra2'+ '.wav')
				sf.write(path2, wav.numpy(), 16000)
				wav = self.inp_tgt.polar_synthesis(m_omlsa_i, angle_tmp)
				path2 = os.path.join(analysis_path, 'enhanced_omlsa_imcra'+ '.wav')
				sf.write(path2, wav.numpy(), 16000)
				for bin in range(257):
					plt.figure(figsize=(20, 10))
					plt.suptitle('Tracking noise at bin: ' + str(bin))

					plt.subplot(2, 2, 1)
					epochs = range(1, np.shape(m_noise)[0] + 1)
					plt.plot(epochs, u_noise[:,bin], 'r--o', label='real_noise')
					plt.plot(epochs, m_noise[:,bin], 'b--o', label='mcra_noise')
					plt.plot(epochs, m_noise_i[:, bin], 'c--o', label='imcra_noise')
					plt.plot(epochs, m_noise_2[:,bin], 'g--o', label='mcra_2_noise')
					plt.title('tracking noise')
					plt.xlabel('frame Numbers')
					plt.ylabel('mu')
					plt.legend()
					plt.grid()

					plt.subplot(2, 2, 3)
					plt.plot(epochs, u_G[:, bin], 'r--o', label='real_noise_signal_ratio')
					plt.plot(epochs, m_G[:,bin], 'b--o', label='mcra_G')
					plt.plot(epochs, m_G_i[:, bin], 'c--o', label='imcra_G')
					plt.plot(epochs, m_G_2[:,bin], 'g--o', label='mcra_2_G')
					plt.title('tracking G')
					plt.xlabel('frame numbers')
					plt.ylabel('sigma')
					plt.legend()
					plt.grid()

					plt.subplot(2, 2, 2)
					plt.plot(epochs, m_present[:,bin], 'b--o', label='mcra_present')
					plt.plot(epochs, m_present_i[:, bin], 'c--o', label='imcra_present')
					plt.plot(epochs, m_present_2[:,bin], 'g--o', label='mcra_2_present')
					plt.title('tracking ')
					plt.xlabel('frame numbers')
					plt.ylabel('sigma')
					plt.legend()
					plt.grid()

					plt.subplot(2, 2, 4)
					plt.plot(epochs, meg_pwr[:fr_len, bin], 'r--o', label='noisy_signal')
					plt.plot(epochs, s_pwr[:fr_len, bin], 'k--o', label='clean_signal')
					plt.plot(epochs, m_omlsa[:,bin], 'b--o', label='mcra_omlsa')
					plt.plot(epochs, m_omlsa_i[:, bin], 'c--o', label='imcra_omlsa')
					plt.plot(epochs, m_omlsa_2[:,bin], 'g--o', label='mcra_2_omlsa')
					plt.title('tracking magnitude')
					plt.xlabel('frame numbers')
					plt.ylabel('sigma')
					plt.legend()
					plt.grid()

					fig_title = os.path.join(analysis_path, 'Tracking_noise_' + str(bin) + '_bin.png')
					plt.savefig(fig_title)

					# plt.show()
					plt.close()

			print("Processing observations...")


	def infer_tracking_noise_vts(
		self,
		test_s,
		test_x,
		test_x_len,
		test_x_base_names,
		test_epoch,
		model_path='model',
		out_type='y',
		gain='mmse-lsa',
		out_path='out',
		n_filters=40,
		saved_data_path=None,
		):
		"""
		Deep Xi inference. The specified 'out_type' is saved.

		Argument/s:
			test_x - noisy-speech test batch.
			test_x_len - noisy-speech test batch lengths.
			test_x_base_names - noisy-speech base names.
			test_epoch - epoch to test.
			model_path - path to model directory.
			out_type - output type (see deepxi/args.py).
			gain - gain function (see deepxi/args.py).
			out_path - path to save output files.
			saved_data_path - path to saved data necessary for enhancement.
		"""

		home_dir = '/home/devpath/datasets/noisetracking'

		means, stds, probs = phoneme_extract_gaussians()
		s_mbatch_list = random.sample(test_s, 8)
		d_mbatch_list = random.sample(test_x, 8)
		s_mbatch, d_mbatch, s_mbatch_len, d_mbatch_len, snr_mbatch = \
			self.wav_batch(s_mbatch_list, d_mbatch_list)
		#snr_mbatch = snr_mbatch*0+3
		s, d, x, n_frames = \
			self.inp_tgt.mix(s_mbatch, d_mbatch, s_mbatch_len,
							 d_mbatch_len, snr_mbatch)

		inp_mbatch, angle1 = self.inp_tgt.polar_analysis(x)
		n_mbatch, _ = self.inp_tgt.polar_analysis(d)
		s_mbatch, _ = self.inp_tgt.polar_analysis(s)
		inp_mbatch = np.where(inp_mbatch == 0, np.finfo(float).eps, inp_mbatch)
		n_mbatch = np.where(n_mbatch == 0, np.finfo(float).eps, n_mbatch)
		s_mbatch = np.where(s_mbatch == 0, np.finfo(float).eps, s_mbatch)
		G_min = np.ones(257, float) * 0.09

		if not isinstance(test_epoch, list): test_epoch = [test_epoch]
		for e in test_epoch:
			if e < 1: raise ValueError("test_epoch must be greater than 0.")
			self.model.load_weights(model_path + '/epoch-' + str(e - 1) +
									'/variables/variables')
			tgt_hat_batch = self.model.predict(inp_mbatch, batch_size=1, verbose=1)
			posterior_pro = np.delete(tgt_hat_batch, 50, axis=2)
			pho_probs = posterior_pro[:, :, 1:]
			sap_nn =  posterior_pro[:, :, 1]
			ssp_nn = 1 - sap_nn
			ssp_nn = np.where(ssp_nn == 0, np.finfo(float).eps, ssp_nn)
			normal_pho_probs = pho_probs / np.expand_dims(np.sum(pho_probs, axis=2), axis=2)
			# posterior_pro = np.delete(tgt_hat_batch, [20, 39, 42, 50], axis=2)
			for idx in range(8):

				fr_len = s_mbatch_list[idx]['frame_len']
				w_len = s_mbatch_list[idx]['wav_len']
				filename = s_mbatch_list[idx]['file_path'].split('/')[-1]
				noisename = d_mbatch_list[idx]['file_path'].split('/')[-1]
				analysis_path = os.path.join(home_dir, filename.split('.')[0]+'_'+noisename.split('.')[0]+'_'+str(snr_mbatch[idx])+'dB')
				ensures_dir(analysis_path)
				sf.write(os.path.join(analysis_path, 'clean_speech.wav'), s[idx, :w_len].numpy(), 16000)
				sf.write(os.path.join(analysis_path, 'noisy_speech.wav'), x[idx, :w_len].numpy(), 16000)
				sf.write(os.path.join(analysis_path, 'noise.wav'), d[idx, :w_len].numpy(), 16000)
				angle_tmp = angle1[idx, :fr_len]
				nn_spp = np.expand_dims(ssp_nn[idx, :fr_len], -1)
				meg_pwr = np.square(inp_mbatch[idx, :fr_len])
				low_fr = np.ones(17, float) * 2
				mid_fr = np.ones(48, float) * 2.8
				hi_fr = np.ones(192, float) * 5
				delta = np.concatenate((low_fr, mid_fr, hi_fr))

				imcra_noise = imcra(alpha_d=0.89, alpha_s=0.8, alpha_p=0.2, lambda_d=meg_pwr[0], frame_L=100,
									fft_len=512,
									delta=delta, beta=1.23, b_min=1.66, gamma0=4.6, gamma1=3, zeta0=1.67)
				m_present = np.zeros(257, float)
				m_noise_i = np.zeros(257, float)
				m_omlsa_i = np.zeros(257, float)
				for i, pwr in enumerate(meg_pwr[:fr_len]):
					c_noise,G,P=imcra_noise.tracking_noise(pwr, i)

					m_present = np.vstack((m_present, P))
					m_noise_i = np.vstack((m_noise_i, c_noise))
					omlsa = np.power(G, P) * np.power(G_min, (1 - P)) * inp_mbatch[idx,i]
					m_omlsa_i = np.vstack((m_omlsa_i, omlsa))
				m_present = m_present[1:]
				m_omlsa_i = m_omlsa_i[1:]
				m_noise_i = m_noise_i[1:]
				wav = self.inp_tgt.polar_synthesis(m_omlsa_i, angle_tmp)
				path2 = os.path.join(analysis_path, 'enhanced_omlsa.wav')
				sf.write(path2, wav.numpy(), 16000)
				m_omlsa_log = np.log(m_omlsa_i)
				m_noise_log = np.log(np.sqrt(m_noise_i))
				#noise_pwr = (np.log(n_mbatch[idx]))*2
				noise_pwr = (np.log(n_mbatch[idx, :fr_len]))
				mean_log_noise_pwr = np.mean(noise_pwr, axis=0)
				sigma_log_noise_pwr = np.square(np.std(noise_pwr, axis=0))
				m_noise = np.zeros(257, float)
				sigma_hat = np.zeros(257, float)
				m_noise2 = np.zeros(257, float)
				sigma_hat2 = np.zeros(257, float)
				stack_signal = np.zeros([5, np.shape(noise_pwr)[0], 257], float)

				stack_noise = np.zeros([5,np.shape(noise_pwr)[0],257], float)
				stack_noise2 = np.zeros([5, np.shape(noise_pwr)[0], 257], float)
				m_noise_i = np.tile(mean_log_noise_pwr, 5).reshape([5, -1])
				m_sigma_i = np.tile(sigma_log_noise_pwr, 5).reshape([5, -1])

				#s_pwr = np.log(s_mbatch[idx])*2
				s_pwr = np.log(s_mbatch[idx, :fr_len])

				#logmeg1 = np.log(inp_mbatch[idx]) * 2
				logmeg1 = np.log(inp_mbatch[idx, :fr_len])
				mean = np.expand_dims(np.mean(logmeg1[:25, :], axis=0), 0)
				stand = np.expand_dims(np.std(logmeg1[:25, :], axis=0), 0)
				sigma = np.square(stand)
				#pwr_square = np.sum(np.square(logmeg1[:25, :]), axis=0)/25
				#Sigma = pwr_square-np.square(mean)
				#g_m_max = gmm_phoneme_noise(np.array(means) * 2, np.square(np.array(stds) * 2), np.array(probs), mean, sigma, 3)
				g_m_max = gmm_phoneme_noise(np.array(means), np.square(np.array(stds)), np.array(probs), mean,
											sigma, 1)

				for ix in range(5):
					print('############################ process loop: ' + str(ix))
					g_m_max.compensate_model()
					#g_m_max.calculate_P_o_t_k_l(logmeg1)
					g_m_max.hybrid_P_o_t_k_l(logmeg1, normal_pho_probs[idx, :fr_len])
					meg, noi = g_m_max.update_signal_noise_spp(logmeg1, nn_spp)
					#meg, noi = g_m_max.update_signal_noise(logmeg1)
					#meg, noi = g_m_max.update_signal_noise_spp(logmeg1, np.sqrt(nn_spp*m_present))
					#meg, noi = g_m_max.update_signal_noise_spp(logmeg1, (nn_spp + m_present)/2)
					sigma_hat = np.vstack((sigma_hat, np.sum(g_m_max.g_Sigma*np.expand_dims(g_m_max.w_nl,-1), axis=0)))
					m_noise = np.vstack((m_noise, np.sum(g_m_max.g_mu*np.expand_dims(g_m_max.w_nl, -1), axis=0)))
					sigma_hat2 = np.vstack((sigma_hat2, np.square(np.std(noi, axis=0))))
					m_noise2 = np.vstack((m_noise2, np.mean(noi, axis=0)))
					stack_signal[ix] = meg
					stack_noise[ix] = noi

					print("noise pr:"+str(g_m_max.w_nl))
					mag = np.exp(meg)
					wav = self.inp_tgt.polar_synthesis(mag, angle_tmp)
					path2 = os.path.join(analysis_path, 'enhanced_rnd_'+ str(ix) + '.wav')
					sf.write(path2, wav.numpy(), 16000)
				sigma_hat = sigma_hat[1:]
				m_noise = m_noise[1:]
				sigma_hat2 = sigma_hat2[1:]
				m_noise2 = m_noise2[1:]
				u_noise = logmeg1[0]
				sigma_noise = logmeg1[0]
				stack_signal2 = np.zeros([5, np.shape(noise_pwr)[0], 257], float)

				h_m_max = hybridMixMax(np.array(means), np.array(stds), np.array(means[0]), np.array(stds[0]),
									   np.array(probs),
									   0.11, 0.5)
				for ix in range(5):
					h_m_max.update_noise(np.expand_dims(m_noise[ix], 0), np.expand_dims(np.sqrt(sigma_hat[ix]), 0))
					x_hat = np.zeros(257, float)

					for i, logf in enumerate(logmeg1):
						o, p_mm = h_m_max.x_estimate_mixmax(posterior_pro[idx, i, 1:], logf)
						x_hat = np.vstack((x_hat, o))

					meg = np.exp(x_hat[1:])
					wav = self.inp_tgt.polar_synthesis(meg, angle_tmp)
					path2 = os.path.join(analysis_path, 'enhanced_mixmax_rnd_'+ str(ix) + '.wav')
					sf.write(path2, wav.numpy(), 16000)
					stack_signal2[ix] = x_hat[1:]
				h_m_max.update_noise(np.expand_dims(m_noise[3], 0), np.expand_dims(np.sqrt(sigma_hat[3]), 0))
				x_hat = np.zeros(257, float)
				omg_hat = np.zeros(257, float)
				u_hat = np.zeros(257, float)
				sig_hat = np.zeros(257, float)
				mu_noise_golden = logmeg1[0]
				sig_noise_golden = np.zeros(257, float)
				low_fr = np.ones(17, float) * 0.12
				mid_fr = np.ones(48, float) * 0.08
				hi_fr = np.ones(192, float) * 0.05
				noise_floor = np.concatenate((low_fr, mid_fr, hi_fr))

				for i, logf in enumerate(logmeg1):
					o, p_mm = h_m_max.x_estimate_mixmax(posterior_pro[idx, i, 1:], logf)
					h_m_max.update_mu_sigma(logf,p_mm)
					x_hat = np.vstack((x_hat, o))
					mu, sig = h_m_max.get_noise()
					u_hat = np.vstack((u_hat, mu))
					sig_hat = np.vstack((sig_hat, sig))

					#o, rho_mm = h_m_max.x_estimate_mixmax(normal_pho_probs[idx, i], logf)

					omg = logf*(1-m_present[i])*noise_floor + m_present[i]*o
					omg_hat = np.vstack((omg_hat, omg))
					if i != 0:
						mu_noise_golden = np.vstack((mu_noise_golden, np.mean(noise_pwr[:i], axis=0)))
						sig_noise_golden = np.vstack((sig_noise_golden, np.std(noise_pwr[:i], axis=0)))
				x_hat = x_hat[1:]
				omg_hat = omg_hat[1:]
				meg = np.exp(x_hat)
				u_hat = u_hat[1:]
				sig_hat = sig_hat[1:]
				#mu_noise_golden = np.exp(mu_noise_golden[1:])
				#sig_noise_golden = np.exp(sig_noise_golden[1:])
				wav = self.inp_tgt.polar_synthesis(meg, angle_tmp)
				path2 = os.path.join(analysis_path, 'enhanced_mixmax_adaptive_ms.wav')
				sf.write(path2, wav.numpy(), 16000)
				meg2=np.exp(omg_hat)
				wav = self.inp_tgt.polar_synthesis(meg2, angle_tmp)
				path2 = os.path.join(analysis_path, 'enhanced_mixmax_vad.wav')
				sf.write(path2, wav.numpy(), 16000)
				for bin in range(257):
					plt.figure(figsize=(20, 10))
					plt.suptitle('Tracking noise at bin: ' + str(bin))
					'''
					plt.subplot(2, 2, 1)
					epochs = range(0, 5)
					plt.plot(epochs, m_noise[:,bin], 'b--x', label='mean_noise_hat')
					plt.plot(epochs, m_noise_i[:, bin], 'r--o', label='mean_noise_real')
					plt.plot(epochs, sigma_hat2[:, bin], 'g--o', label='sigma_noise_hat')
					plt.plot(epochs, m_sigma_i[:, bin], 'c--o', label='sigma_noise_real')
					plt.title('tracking noise')
					plt.xlabel('frame Numbers')
					plt.ylabel('mu')
					plt.legend()
					plt.grid()
					'''
					plt.subplot(2, 2, 1)
					epochs = range(1, np.shape(s_pwr)[0] + 1)
					plt.plot(epochs, u_hat[:, bin], 'r-', label='u_hat')
					plt.plot(epochs, mu_noise_golden[:,bin], 'g--', label='mu_noise_golden')

					plt.plot(epochs, sig_hat[:, bin], 'm-', label='sig_hat')
					plt.plot(epochs, sig_noise_golden[:, bin], 'c--', label='sig_noise_golden')

					plt.title('tracking signal')
					plt.xlabel('frame numbers')
					plt.ylabel('sigma')
					plt.legend()
					plt.grid()

					plt.subplot(2, 2, 2)
					epochs = range(1, np.shape(s_pwr)[0] + 1)
					plt.plot(epochs, s_pwr[:, bin], 'r-', label='pure_signal')

					plt.plot(epochs, stack_signal[0, :,bin], 'g--', label='stack_signal'+str(0))
					plt.plot(epochs, stack_signal[1, :, bin], 'm--', label='stack_signal' + str(1))
					plt.plot(epochs, stack_signal[2, :, bin], 'c--', label='stack_signal' + str(2))
					plt.plot(epochs, stack_signal[3, :, bin], 'k--', label='stack_signal' + str(3))
					plt.plot(epochs, stack_signal[4, :, bin], 'b--', label='stack_signal' + str(4))

					plt.plot(epochs, m_omlsa_log[:,bin], 'b:.', label='omlsa_signal')
					plt.title('tracking signal')
					plt.xlabel('frame numbers')
					plt.ylabel('sigma')
					plt.legend()
					plt.grid()
					'''
					plt.subplot(2, 2, 3)
					epochs = range(0, 5)
					plt.plot(epochs, sigma_hat[:,bin]+0.1, 'b--x', label='sigma_noise_hat')
					plt.plot(epochs, sigma_hat2[:, bin], 'g--o', label='sigma_noise_hat')
					plt.plot(epochs, m_sigma_i[:, bin], 'r--o', label='sigma_noise_real')
					plt.title('tracking noise')
					plt.xlabel('frame Numbers')
					plt.ylabel('sigma')
					plt.legend()
					plt.grid()
					'''
					plt.subplot(2, 2, 3)
					epochs = range(1, np.shape(s_pwr)[0] + 1)
					plt.plot(epochs, s_pwr[:, bin], 'r-', label='real_noise')
					#for ids in range(5):
					#plt.plot(epochs, stack_signal2[0, :,bin], 'g--', label='mixmax_signal_'+str(0))

					plt.plot(epochs, stack_signal2[1, :, bin], 'm--', label='mixmax_signal_' + str(1))
					plt.plot(epochs, stack_signal2[2, :, bin], 'c--', label='mixmax_signal_' + str(2))
					plt.plot(epochs, stack_signal2[3, :, bin], 'k--', label='mixmax_signal_' + str(3))
					plt.plot(epochs, stack_signal2[4, :, bin], 'b--', label='mixmax_signal_' + str(4))
					plt.plot(epochs, x_hat[:, bin], 'y:.', label='mixmax_signal_adaptive')
					plt.plot(epochs, omg_hat[:, bin], 'g:.', label='mixmax_signal_vad')
					plt.title('tracking signal')
					plt.xlabel('frame numbers')
					plt.ylabel('sigma')
					plt.legend()
					plt.grid()

					plt.subplot(2, 2, 4)
					epochs = range(1, np.shape(s_pwr)[0] + 1)
					plt.plot(epochs, noise_pwr[:,bin], 'r-', label='real_noise')
					#for ids in range(5):
					plt.plot(epochs, stack_noise[0, :,bin], 'g--', label='hat_noise_'+str(0))
					plt.plot(epochs, stack_noise[1, :, bin], 'm--', label='hat_noise_' + str(1))
					plt.plot(epochs, stack_noise[2, :, bin], 'c--', label='hat_noise_' + str(2))
					plt.plot(epochs, stack_noise[3, :, bin], 'k--', label='hat_noise_' + str(3))
					plt.plot(epochs, stack_noise[4, :, bin], 'b--', label='hat_noise_' + str(4))

					plt.plot(epochs, m_noise_log[:,bin], 'y:.', label='omlsa_noise')
					plt.title('tracking signal')
					plt.xlabel('frame numbers')
					plt.ylabel('sigma')
					plt.legend()
					plt.grid()

					fig_title = os.path.join(analysis_path, 'Tracking_noise_' + str(bin) + '_bin.png')
					plt.savefig(fig_title)

					# plt.show()
					plt.close()

			print("Processing observations...")


	def test(
		self,
		test_x,
		test_x_len,
		test_x_base_names,
		test_s,
		test_s_len,
		test_s_base_names,
		test_epoch,
		log_path,
		model_path='model',
		gain='mmse-lsa',
		):
		"""
		Deep Xi testing. Objective measures are used to evaluate the performance
		of Deep Xi. Note that the 'supplementary' variable can includes other
		variables necessary for synthesis, like the noisy-speech short-time
		phase spectrum.

		Argument/s:
			test_x - noisy-speech test batch.
			test_x_len - noisy-speech test batch lengths.
			test_x_base_names - noisy-speech base names.
			test_s - clean-speech test batch.
			test_s_len - clean-speech test batch lengths.
			test_s_base_names - clean-speech base names.
			test_epoch - epoch to test.
			model_path - path to model directory.
			gain - gain function (see deepxi/args.py).
		"""
		from pesq import pesq
		from pystoi import stoi
		print("Processing observations...")
		inp_batch, supplementary_batch, n_frames = self.observation_batch(test_x, test_x_len)
		if not isinstance(test_epoch, list): test_epoch = [test_epoch]
		if not isinstance(gain, list): gain = [gain]
		for e in test_epoch:
			for g in gain:

				if e < 1: raise ValueError("test_epoch must be greater than 0.")

				self.model.load_weights(model_path + '/epoch-' + str(e-1) +
					'/variables/variables' )

				print("Performing inference...")
				tgt_hat_batch = self.model.predict(inp_batch, batch_size=1, verbose=1)

				print("Performing synthesis and objective scoring...")
				results = {}
				batch_size = len(test_x_len)
				for i in tqdm(range(batch_size)):
					base_name = test_x_base_names[i]
					inp = inp_batch[i,:n_frames[i],:]
					supplementary = supplementary_batch[i,:n_frames[i],:]
					tgt_hat = tgt_hat_batch[i,:n_frames[i],:]

					y = self.inp_tgt.enhanced_speech(inp, supplementary, tgt_hat, g).numpy()

					for (j, basename) in enumerate(test_s_base_names):
						if basename in test_x_base_names[i]: ref_idx = j

					s = self.inp_tgt.normalise(test_s[ref_idx,
						0:test_s_len[ref_idx]]).numpy() # from int16 to float.
					y = y[0:len(s)]

					try: noise_src = test_x_base_names[i].split("_")[-2]
					except IndexError: noise_src = "Null"
					if noise_src == "Null": snr_level = 0
					else: snr_level = int(test_x_base_names[i].split("_")[-1][:-2])

					results = self.add_score(results, (noise_src, snr_level, 'STOI'),
						100*stoi(s, y, self.inp_tgt.f_s, extended=False))
					results = self.add_score(results, (noise_src, snr_level, 'eSTOI'),
						100*stoi(s, y, self.inp_tgt.f_s, extended=True))
					results = self.add_score(results, (noise_src, snr_level, 'PESQ'),
						pesq(self.inp_tgt.f_s, s, y, 'nb'))
					results = self.add_score(results, (noise_src, snr_level, 'MOS-LQO'),
						pesq(self.inp_tgt.f_s, s, y, 'wb'))

				noise_srcs, snr_levels, metrics = set(), set(), set()
				for key, value in results.items():
					noise_srcs.add(key[0])
					snr_levels.add(key[1])
					metrics.add(key[2])

				if not os.path.exists(log_path + "/results"): os.makedirs(log_path + "/results")

				with open(log_path + "/results/" + self.ver + "_e" + str(e) + '_' + g + ".csv", "w") as f:
					f.write("noise,snr_db")
					for k in sorted(metrics): f.write(',' + k)
					f.write('\n')
					for i in sorted(noise_srcs):
						for j in sorted(snr_levels):
							f.write("{},{}".format(i, j))
							for k in sorted(metrics):
								if (i, j, k) in results.keys():
									f.write(",{:.2f}".format(np.mean(results[(i,j,k)])))
							f.write('\n')

				avg_results = {}
				for i in sorted(noise_srcs):
					for j in sorted(snr_levels):
						if (j >= self.min_snr) and (j <= self.max_snr):
							for k in sorted(metrics):
								if (i, j, k) in results.keys():
									avg_results = self.add_score(avg_results, k, results[(i,j,k)])

				if not os.path.exists(log_path + "/results/average.csv"):
					with open(log_path + "/results/average.csv", "w") as f:
						f.write("ver")
						for i in sorted(metrics): f.write("," + i)
						f.write('\n')

				with open(log_path + "/results/average.csv", "a") as f:
					f.write(self.ver + "_e" + str(e) + '_' + g)
					for i in sorted(metrics):
						if i in avg_results.keys():
							f.write(",{:.2f}".format(np.mean(avg_results[i])))
					f.write('\n')

	def sample(
		self,
		sample_size,
		sample_dir='data',
		):
		"""
		Gathers a sample of the training set. The sample can be used to compute
		statistics for mapping functions.

		Argument/s:
			sample_size - number of training examples included in the sample.
			sample_dir - path to the saved sample.
		"""
		sample_path = sample_dir + '/sample'
		if os.path.exists(sample_path + '.npz'):
			print('Loading sample...')
			with np.load(sample_path + '.npz') as sample:
				s_sample = sample['s_sample']
				d_sample = sample['d_sample']
				x_sample = sample['x_sample']
				wav_len = sample['wav_len']
		elif self.train_s_list == None:
			raise ValueError('No sample.npz file exists.')
		else:
			if sample_size == None: raise ValueError("sample_size is not set.")
			print('Gathering a sample of the training set...')
			s_sample_list = random.sample(self.train_s_list, sample_size)
			d_sample_list = random.sample(self.train_d_list, sample_size)
			s_sample_int, d_sample_int, s_sample_len, d_sample_len, snr_sample = self.wav_batch(s_sample_list,
				d_sample_list)
			s_sample = np.zeros_like(s_sample_int, np.float32)
			d_sample = np.zeros_like(s_sample_int, np.float32)
			x_sample = np.zeros_like(s_sample_int, np.float32)
			for i in tqdm(range(s_sample.shape[0])):
				s, d, x, _ = self.inp_tgt.mix(s_sample_int[i:i+1], d_sample_int[i:i+1],
					s_sample_len[i:i+1], d_sample_len[i:i+1], snr_sample[i:i+1])
				s_sample[i, 0:s_sample_len[i]] = s
				d_sample[i, 0:s_sample_len[i]] = d
				x_sample[i, 0:s_sample_len[i]] = x
			wav_len = s_sample_len
			if not os.path.exists(sample_dir): os.makedirs(sample_dir)
			np.savez(sample_path + '.npz', s_sample=s_sample,
				d_sample=d_sample, x_sample=x_sample, wav_len=wav_len)
			sample = {'s_sample': s_sample, 'd_sample': d_sample,
				'x_sample': x_sample, 'wav_len': wav_len}
			save_mat(sample_path + '.mat', sample, 'stats')
			print('Sample of the training set saved.')
		return s_sample, d_sample, x_sample, wav_len

	def dataset(self, n_epochs, buffer_size=16):
		"""
		Used to create a tf.data.Dataset for training.

		Argument/s:
			n_epochs - number of epochs to generate.
			buffer_size - number of mini-batches to keep in buffer.

		Returns:
			dataset - tf.data.Dataset
		"""
		if(self.inp_tgt_type == "MagPhonme"):
			dataset = tf.data.Dataset.from_generator(
				self.phoneme_mbatch_gen_aug,
				(tf.float32, tf.float32, tf.float32),
				(tf.TensorShape([None, None, self.inp_tgt.n_feat]),
				 tf.TensorShape([None, None, self.inp_tgt.n_outp]),
				 tf.TensorShape([None, None])),
				[tf.constant(n_epochs)]
			)
		else:
			dataset = tf.data.Dataset.from_generator(
				self.mbatch_gen,
				(tf.float32, tf.float32, tf.float32),
				(tf.TensorShape([None, None, self.inp_tgt.n_feat]),
					tf.TensorShape([None, None, self.inp_tgt.n_outp]),
					tf.TensorShape([None, None])),
				[tf.constant(n_epochs)]
				)
		dataset = dataset.prefetch(buffer_size)
		return dataset

	def mbatch_gen(self, n_epochs):
		"""
		A generator that yields a mini-batch of training examples.

		Argument/s:
			n_epochs - number of epochs to generate.

		Returns:
			inp_mbatch - mini-batch of observations (input to network).
			xi_bar_mbatch - mini-batch of targets (mapped a priori SNR).
			seq_mask_mbatch - mini-batch of sequence masks.
		"""
		for _ in range(n_epochs):
			random.shuffle(self.train_s_list)
			start_idx, end_idx = 0, self.mbatch_size
			for _ in range(self.n_iter):
				s_mbatch_list = self.train_s_list[start_idx:end_idx]
				d_mbatch_list = random.sample(self.train_d_list, end_idx-start_idx)
				s_mbatch, d_mbatch, s_mbatch_len, d_mbatch_len, snr_mbatch = \
					self.wav_batch(s_mbatch_list, d_mbatch_list)
				inp_mbatch, xi_bar_mbatch, n_frames_mbatch = \
					self.inp_tgt.example(s_mbatch, d_mbatch, s_mbatch_len,
					d_mbatch_len, snr_mbatch)
				seq_mask_mbatch = tf.cast(tf.sequence_mask(n_frames_mbatch), tf.float32)
				start_idx += self.mbatch_size; end_idx += self.mbatch_size
				if end_idx > self.n_examples: end_idx = self.n_examples
				#yield inp_mbatch, xi_bar_mbatch, seq_mask_mbatch
		return inp_mbatch, xi_bar_mbatch, seq_mask_mbatch

	def phoneme_mbatch_gen(self, n_epochs):
		"""
		A generator that yields a mini-batch of training examples.

		Argument/s:
			n_epochs - number of epochs to generate.

		Returns:
			inp_mbatch - mini-batch of observations (input to network).
			xi_bar_mbatch - mini-batch of targets (mapped a priori SNR).
			seq_mask_mbatch - mini-batch of sequence masks.
		"""
		for _ in range(n_epochs):
			random.shuffle(self.train_s_list)
			start_idx, end_idx = 0, self.mbatch_size
			for _ in range(self.n_iter):
				s_mbatch_list = self.train_s_list[start_idx:end_idx]
				s_mbatch, s_mbatch_len, l_mbatch,  l_mbatch_len = \
					self.phoneme_wav_batch(s_mbatch_list)
				inp_mbatch, _ = self.inp_tgt.polar_analysis(self.inp_tgt.normalise(s_mbatch))
				if np.shape(l_mbatch)[1] > tf.shape(inp_mbatch)[1]:
					max_len = tf.shape(inp_mbatch)[1]
					l_mbatch = l_mbatch[:,:max_len]
				elif np.shape(l_mbatch)[1] <  tf.shape(inp_mbatch)[1]:
					tail_batch = np.zeros([l_mbatch.shape[0], tf.shape(inp_mbatch)[1]-np.shape(l_mbatch)[1]], np.int32)
					l_mbatch = np.hstack((l_mbatch, tail_batch))
				seq_mask_mbatch = tf.cast(tf.sequence_mask(l_mbatch_len), tf.float32)
				if np.shape(seq_mask_mbatch)[1] > tf.shape(inp_mbatch)[1]:
					max_len = tf.shape(inp_mbatch)[1]
					seq_mask_mbatch = seq_mask_mbatch[:,:max_len]
				elif np.shape(seq_mask_mbatch)[1] <  tf.shape(inp_mbatch)[1]:
					tail_batch = np.zeros([seq_mask_mbatch.shape[0], tf.shape(inp_mbatch)[1]-np.shape(seq_mask_mbatch)[1]], np.float32)
					seq_mask_mbatch = np.hstack((seq_mask_mbatch, tail_batch))
				mask1 = tf.sequence_mask(l_mbatch, self.inp_tgt.n_outp)
				l_match_plus1 = l_mbatch + np.ones(l_mbatch.shape)
				mask2 = tf.sequence_mask(l_match_plus1, self.inp_tgt.n_outp)
				tgt_mbatch = tf.cast(tf.math.logical_xor(mask1, mask2), tf.float32)
				start_idx += self.mbatch_size; end_idx += self.mbatch_size
				if end_idx > self.n_examples: end_idx = self.n_examples
				if tf.shape(inp_mbatch)[1] != tf.shape(tgt_mbatch)[1] or\
					tf.shape(inp_mbatch)[1] != tf.shape(seq_mask_mbatch)[1] or\
					tf.shape(seq_mask_mbatch)[1] != tf.shape(tgt_mbatch)[1]:
					print('not match shape - inp_mbatch:'+ str(tf.shape(inp_mbatch)[1].numpy()) +  ' tgt_mbatch:'+ str(tf.shape(tgt_mbatch)[1].numpy()) +  ' seq_mask_mbatch:'+ str(tf.shape(seq_mask_mbatch)[1].numpy()))
					continue
				yield inp_mbatch, tgt_mbatch, seq_mask_mbatch
		#return inp_mbatch, tgt_mbatch, seq_mask_mbatch

	def phoneme_mbatch_gen_aug(self, n_epochs):
		"""
		A generator that yields a mini-batch of training examples.

		Argument/s:
			n_epochs - number of epochs to generate.

		Returns:
			inp_mbatch - mini-batch of observations (input to network).
			xi_bar_mbatch - mini-batch of targets (mapped a priori SNR).
			seq_mask_mbatch - mini-batch of sequence masks.
		"""
		for _ in range(n_epochs):
			random.shuffle(self.train_s_list)
			start_idx, end_idx = 0, self.mbatch_size
			for _ in range(self.n_iter):
				s_mbatch_list = self.train_s_list[start_idx:end_idx]
				d_mbatch_list = random.sample(self.train_d_list, end_idx-start_idx)
				s_mbatch, d_mbatch, s_mbatch_len, d_mbatch_len, snr_mbatch = \
					self.wav_batch(s_mbatch_list, d_mbatch_list)
				s, d, x, n_frames = \
					self.inp_tgt.mix(s_mbatch, d_mbatch, s_mbatch_len,
					d_mbatch_len, snr_mbatch)
				inp_mbatch, _ = self.inp_tgt.polar_analysis(x)
				l_mbatch,  l_mbatch_len = \
					self.phoneme_wav_batch_aug(s_mbatch_list)
				if np.shape(l_mbatch)[1] > tf.shape(inp_mbatch)[1]:
					max_len = tf.shape(inp_mbatch)[1]
					l_mbatch = l_mbatch[:,:max_len]
				elif np.shape(l_mbatch)[1] <  tf.shape(inp_mbatch)[1]:
					tail_batch = np.zeros([l_mbatch.shape[0], tf.shape(inp_mbatch)[1]-np.shape(l_mbatch)[1]], np.int32)
					l_mbatch = np.hstack((l_mbatch, tail_batch))
				seq_mask_mbatch = tf.cast(tf.sequence_mask(l_mbatch_len), tf.float32)
				if np.shape(seq_mask_mbatch)[1] > tf.shape(inp_mbatch)[1]:
					max_len = tf.shape(inp_mbatch)[1]
					seq_mask_mbatch = seq_mask_mbatch[:,:max_len]
				elif np.shape(seq_mask_mbatch)[1] <  tf.shape(inp_mbatch)[1]:
					tail_batch = np.zeros([seq_mask_mbatch.shape[0], tf.shape(inp_mbatch)[1]-np.shape(seq_mask_mbatch)[1]], np.float32)
					seq_mask_mbatch = np.hstack((seq_mask_mbatch, tail_batch))
				mask1 = tf.sequence_mask(l_mbatch, self.inp_tgt.n_outp)
				l_match_plus1 = l_mbatch + np.ones(l_mbatch.shape)
				mask2 = tf.sequence_mask(l_match_plus1, self.inp_tgt.n_outp)
				tgt_mbatch = tf.cast(tf.math.logical_xor(mask1, mask2), tf.float32)
				start_idx += self.mbatch_size; end_idx += self.mbatch_size
				if end_idx > self.n_examples: end_idx = self.n_examples
				if tf.shape(inp_mbatch)[1] != tf.shape(tgt_mbatch)[1] or\
					tf.shape(inp_mbatch)[1] != tf.shape(seq_mask_mbatch)[1] or\
					tf.shape(seq_mask_mbatch)[1] != tf.shape(tgt_mbatch)[1]:
					print('not match shape - inp_mbatch:'+ str(tf.shape(inp_mbatch)[1].numpy()) +  ' tgt_mbatch:'+ str(tf.shape(tgt_mbatch)[1].numpy()) +  ' seq_mask_mbatch:'+ str(tf.shape(seq_mask_mbatch)[1].numpy()))
					continue
				yield inp_mbatch, tgt_mbatch, seq_mask_mbatch
		#return inp_mbatch, tgt_mbatch, seq_mask_mbatch
	def phoneme_val_mbatch_gen(self, val_s_list):
		"""
		A generator that yields a mini-batch of training examples.

		Argument/s:
			n_epochs - number of epochs to generate.

		Returns:
			inp_mbatch - mini-batch of observations (input to network).
			xi_bar_mbatch - mini-batch of targets (mapped a priori SNR).
			seq_mask_mbatch - mini-batch of sequence masks.
		"""

		s_mbatch_list = random.sample(val_s_list, 300)
		s_mbatch, s_mbatch_len, l_mbatch,  l_mbatch_len = \
			self.phoneme_wav_batch(s_mbatch_list)
		inp_mbatch, _ = self.inp_tgt.polar_analysis(self.inp_tgt.normalise(s_mbatch))
		if np.shape(l_mbatch)[1] > tf.shape(inp_mbatch)[1]:
			max_len = tf.shape(inp_mbatch)[1]
			l_mbatch = l_mbatch[:, :max_len]
		elif np.shape(l_mbatch)[1] < tf.shape(inp_mbatch)[1]:
			tail_batch = np.zeros([l_mbatch.shape[0], tf.shape(inp_mbatch)[1] - np.shape(l_mbatch)[1]], np.int32)
			l_mbatch = np.hstack((l_mbatch, tail_batch))
		seq_mask_mbatch = tf.cast(tf.sequence_mask(l_mbatch_len), tf.float32)
		if np.shape(seq_mask_mbatch)[1] > tf.shape(inp_mbatch)[1]:
			max_len = tf.shape(inp_mbatch)[1]
			seq_mask_mbatch = seq_mask_mbatch[:, :max_len]
		elif np.shape(seq_mask_mbatch)[1] < tf.shape(inp_mbatch)[1]:
			tail_batch = np.zeros([seq_mask_mbatch.shape[0], tf.shape(inp_mbatch)[1] - np.shape(seq_mask_mbatch)[1]],
								  np.float32)
			seq_mask_mbatch = np.hstack((seq_mask_mbatch, tail_batch))
		mask1 = tf.sequence_mask(l_mbatch.astype(np.int32), self.inp_tgt.n_outp)
		l_match_plus1 = l_mbatch + np.ones(l_mbatch.shape)
		mask2 = tf.sequence_mask(l_match_plus1.astype(np.int32), self.inp_tgt.n_outp)
		tgt_mbatch = tf.cast(tf.math.logical_xor(mask1, mask2), tf.float32)

		return inp_mbatch, tgt_mbatch, seq_mask_mbatch

	def phoneme_infer_mbatch_gen(self, s_mbatch_list):
		"""
		A generator that yields a mini-batch of training examples.

		Argument/s:
			n_epochs - number of epochs to generate.

		Returns:
			inp_mbatch - mini-batch of observations (input to network).
			xi_bar_mbatch - mini-batch of targets (mapped a priori SNR).
			seq_mask_mbatch - mini-batch of sequence masks.
		"""

		#s_mbatch_list = random.sample(val_s_list, 300)
		s_mbatch, s_mbatch_len, l_mbatch,  l_mbatch_len = \
			self.phoneme_wav_batch(s_mbatch_list)
		inp_mbatch, _ = self.inp_tgt.polar_analysis(self.inp_tgt.normalise(s_mbatch))
		if np.shape(l_mbatch)[1] > tf.shape(inp_mbatch)[1]:
			max_len = tf.shape(inp_mbatch)[1]
			l_mbatch = l_mbatch[:, :max_len]
		elif np.shape(l_mbatch)[1] < tf.shape(inp_mbatch)[1]:
			tail_batch = np.zeros([l_mbatch.shape[0], tf.shape(inp_mbatch)[1] - np.shape(l_mbatch)[1]], np.int32)
			l_mbatch = np.hstack((l_mbatch, tail_batch))
		seq_mask_mbatch = tf.cast(tf.sequence_mask(l_mbatch_len), tf.float32)
		if np.shape(seq_mask_mbatch)[1] > tf.shape(inp_mbatch)[1]:
			max_len = tf.shape(inp_mbatch)[1]
			seq_mask_mbatch = seq_mask_mbatch[:, :max_len]
		elif np.shape(seq_mask_mbatch)[1] < tf.shape(inp_mbatch)[1]:
			tail_batch = np.zeros([seq_mask_mbatch.shape[0], tf.shape(inp_mbatch)[1] - np.shape(seq_mask_mbatch)[1]],
								  np.float32)
			seq_mask_mbatch = np.hstack((seq_mask_mbatch, tail_batch))
		mask1 = tf.sequence_mask(l_mbatch.astype(np.int32), self.inp_tgt.n_outp)
		l_match_plus1 = l_mbatch + np.ones(l_mbatch.shape)
		mask2 = tf.sequence_mask(l_match_plus1.astype(np.int32), self.inp_tgt.n_outp)
		tgt_mbatch = tf.cast(tf.math.logical_xor(mask1, mask2), tf.float32)

		return inp_mbatch, tgt_mbatch, seq_mask_mbatch


	def val_batch(
		self,
		save_path,
		val_s,
		val_d,
		val_s_len,
		val_d_len,
		val_snr
		):
		"""
		Creates and saves the examples for the validation set. If
		already saved, the function will load the batch of examples.

		Argument/s:
			save_path - path to save the validation batch.
			val_s - validation clean speech waveforms.
			val_d - validation noise waveforms.
			val_s_len - validation clean speech waveform lengths.
			val_d_len - validation noise waveform lengths.
			val_snr - validation SNR levels.

		Returns:
			inp_batch - batch of observations (input to network).
			tgt_batch - batch of targets (mapped a priori SNR).
			seq_mask_batch - batch of sequence masks.
		"""
		print('Processing validation batch...')
		batch_size = len(val_s)
		max_n_frames = self.inp_tgt.n_frames(max(val_s_len))
		inp_batch = np.zeros([batch_size, max_n_frames, self.inp_tgt.n_feat], np.float32)
		tgt_batch = np.zeros([batch_size, max_n_frames, self.inp_tgt.n_outp], np.float32)
		seq_mask_batch = np.zeros([batch_size, max_n_frames], np.float32)
		for i in tqdm(range(batch_size)):
			inp, tgt, _ = self.inp_tgt.example(val_s[i:i+1], val_d[i:i+1],
				val_s_len[i:i+1], val_d_len[i:i+1], val_snr[i:i+1])
			n_frames = self.inp_tgt.n_frames(val_s_len[i])
			inp_batch[i,:n_frames,:] = inp.numpy()
			if tf.is_tensor(tgt): tgt_batch[i,:n_frames,:] = tgt.numpy()
			else: tgt_batch[i,:n_frames,:] = tgt
			seq_mask_batch[i,:n_frames] = tf.cast(tf.sequence_mask(n_frames), tf.float32)
		return inp_batch, tgt_batch, seq_mask_batch

	def observation_batch(self, x_batch, x_batch_len):
		"""
		Computes observations (inp) from noisy speech recordings.

		Argument/s:
			x_batch - noisy-speech batch.
			x_batch_len - noisy-speech batch lengths.

		Returns:
			inp_batch - batch of observations (input to network).
			supplementary_batch - batch of noisy-speech short-time phase spectrums.
			n_frames_batch - number of frames in each observation.
		"""
		batch_size = len(x_batch)
		max_n_frames = self.inp_tgt.n_frames(max(x_batch_len))
		inp_batch = np.zeros([batch_size, max_n_frames, self.inp_tgt.n_feat], np.float32)
		supplementary_batch = np.zeros([batch_size, max_n_frames, self.inp_tgt.n_feat], np.float32)
		n_frames_batch = [self.inp_tgt.n_frames(i) for i in x_batch_len]
		for i in tqdm(range(batch_size)):
			inp, supplementary = self.inp_tgt.observation(x_batch[i,:x_batch_len[i]])
			inp_batch[i,:n_frames_batch[i],:] = inp
			supplementary_batch[i,:n_frames_batch[i],:] = supplementary
		return inp_batch, supplementary_batch, n_frames_batch

	def wav_batch(self, s_list, d_list):
		"""
		Loads .wav files into batches.

		Argument/s:
			s_list - clean-speech list.
			d_list - noise list.

		Returns:
			s_batch - batch of clean speech.
			d_batch - batch of noise.
			s_batch_len - sequence length of each clean speech waveform.
			d_batch_len - sequence length of each noise waveform.
			snr_batch - batch of SNR levels.
		"""
		batch_size = len(s_list)
		max_len = max([dic['wav_len'] for dic in s_list])
		s_batch = np.zeros([batch_size, max_len], np.int16)
		d_batch = np.zeros([batch_size, max_len], np.int16)
		s_batch_len = np.zeros(batch_size, np.int32)
		for i in range(batch_size):
			(wav, _) = read_wav(s_list[i]['file_path'])
			s_batch[i,:s_list[i]['wav_len']] = wav
			s_batch_len[i] = s_list[i]['wav_len']
			flag = True
			while flag:
				if d_list[i]['wav_len'] < s_batch_len[i]: d_list[i] = random.choice(self.train_d_list)
				else: flag = False
			(wav, _) = read_wav(d_list[i]['file_path'])
			rand_idx = np.random.randint(0, 1+d_list[i]['wav_len']-s_batch_len[i])
			d_batch[i,:s_batch_len[i]] = wav[rand_idx:rand_idx+s_batch_len[i]]
		d_batch_len = s_batch_len
		# snr_batch = np.random.randint(self.min_snr, self.max_snr+1, batch_size)
		snr_batch = np.array(random.choices(self.snr_levels, k=batch_size))
		return s_batch, d_batch, s_batch_len, d_batch_len, snr_batch

	def phoneme_wav_batch(self, s_list):
		"""
		Loads .wav files into batches.

		Argument/s:
			s_list - clean-speech list.
			d_list - noise list.

		Returns:
			s_batch - batch of clean speech.
			d_batch - batch of noise.
			s_batch_len - sequence length of each clean speech waveform.
			d_batch_len - sequence length of each noise waveform.
			snr_batch - batch of SNR levels.
		"""
		batch_size = len(s_list)
		max_len = max([dic['wav_len'] for dic in s_list])
		s_batch = np.zeros([batch_size, max_len], np.int16)
		max_frame_len = max([dic['frame_len'] for dic in s_list])
		max_label_len = max([len(dic['label']) for dic in s_list])
		
		l_batch = np.zeros([batch_size, max(max_frame_len, max_label_len)], np.int32)
		s_batch_len = np.zeros(batch_size, np.int32)
		l_batch_len = np.zeros(batch_size, np.int32)
		for i in range(batch_size):
			(wav, _) = read_wav(s_list[i]['file_path'])

			s_batch_len[i] = s_list[i]['wav_len']
			l_batch_len[i] = min(len(s_list[i]['label']), s_list[i]['frame_len'])
			s_batch[i, :s_batch_len[i]] = wav
			l_batch[i,:l_batch_len[i]] = s_list[i]['label'][:l_batch_len[i]]

		return s_batch, s_batch_len, l_batch, l_batch_len

	def phoneme_wav_batch_aug(self, s_list):
		"""
		Loads .wav files into batches.

		Argument/s:
			s_list - clean-speech list.
			d_list - noise list.

		Returns:
			s_batch - batch of clean speech.
			d_batch - batch of noise.
			s_batch_len - sequence length of each clean speech waveform.
			d_batch_len - sequence length of each noise waveform.
			snr_batch - batch of SNR levels.
		"""


		batch_size = len(s_list)
		max_len = max([dic['wav_len'] for dic in s_list])
		max_frame_len = max([dic['frame_len'] for dic in s_list])
		max_label_len = max([len(dic['label']) for dic in s_list])
		l_batch = np.zeros([batch_size, max(max_frame_len, max_label_len)], np.int32)
		l_batch_len = np.zeros(batch_size, np.int32)
		for i in range(batch_size):

			l_batch_len[i] = min(len(s_list[i]['label']), s_list[i]['frame_len'])
			l_batch[i, :l_batch_len[i]] = s_list[i]['label'][:l_batch_len[i]]

		return l_batch, l_batch_len

	def add_score(self, dict, key, score):
		"""
		Adds score/s to the list for the given key.

		Argument/s:
			dict - dictionary with condition as keys and a list of objective
				scores as values.
			key - noisy-speech conditions.
			score - objective score.

		Returns:
			dict - updated dictionary.
		"""
		if isinstance(score, list):
			if key in dict.keys(): dict[key].extend(score)
			else: dict[key] = score
		else:
			if key in dict.keys(): dict[key].append(score)
			else: dict[key] = [score]
		return dict

class SaveWeights(Callback):
	def __init__(self, model_path):
		super(SaveWeights, self).__init__()
		self.model_path = model_path

	def on_epoch_end(self, epoch, logs=None):
		self.model.save(self.model_path + "/epoch-" + str(epoch))

class TransformerSchedular(LearningRateSchedule):
	def __init__(self, d_model, warmup_steps):
		super(TransformerSchedular, self).__init__()
		self.d_model = float(d_model)
		self.warmup_steps = warmup_steps

	def __call__(self, step):
		arg1 = tf.math.rsqrt(step)
		arg2 = step * (self.warmup_steps ** -1.5)
		return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

	def get_config(self):
		config = {'d_model': self.d_model, 'warmup_steps': self.warmup_steps}
		return config
