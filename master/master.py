import os
import re


pwd = '/home/ar773/'
wrk = 'MLSALT2/practical2/master/'

#generate the DNNSTRUCTURE line for HTE.dnntrain
def gen_layer_str(i, ctxt, fdim,nhid):
	s = str(fdim*(ctxt*2 + 1))
	while i > 0:
		s = s + 'X'+str(nhid)
		i = i -1
	return s

#generate the CONTEXTSHIFT line for HTE.dnntrain
def gen_str(i):
	if i == 0:
		return '0'
	else:
		return str(i) + ',-' + str(i) + ',' + gen_str(i-1)

#modify the various parameters of DNN
def modify_file(fname,ctxt,fdim,layer,nhid,ptl2,ftl2):
	s = open(fname).read()
	context = gen_str(ctxt)
	lstr = gen_layer_str(layer,ctxt,fdim,nhid)
	r = re.sub('set CONTEXTSHIFT.*#', 'set CONTEXTSHIFT=' + context + ' #', s)
	r = re.sub('set DNNSTRUCTURE.*#', 'set DNNSTRUCTURE=' + lstr+'X3000 #', r)
	r = re.sub('set FTWEIGHTDECAY.*#', 'set FTWEIGHTDECAY=' + str(ptl2) + ' #', r)
	r = re.sub('set PTWEIGHTDECAY.*#', 'set PTWEIGHTDECAY=' + str(ftl2) + ' #', r)
	f = open(fname, 'w')
	f.write(r)
	f.close()

#Train a dnn model for a paramater set
def dnn_train(addr,env,ph,layer):
	
	
	temp = wrk + '../tools/steps/step-dnntrain -DNNTRAINHTE ' +pwd+wrk +'HTE.dnntrain '+ wrk +'../convert/mfc13d/env/environment'+env+' '+pwd+wrk+'../gmm-models/align-'+ph+'/align/timit_train.mlf '+pwd+wrk+'../gmm-models/'+ph+'/hmm84/MMF ' + pwd+wrk+'../gmm-models/'+ph+'/hmms.mlist ' +wrk+'result_'+addr
	os.system(temp)

	#copy and rename CV Training accuracy file	
	temp = 'cp '+wrk+'result_'+addr+'/dnn'+str(layer+2)+'.finetune/LOG'+' '+wrk+'super-result/result_'+addr
	os.system(temp)
	temp = 'mv -f '+wrk+'super-result/result_'+addr+'/LOG'+' '+wrk+'super-result/result_'+addr+'/LOG_CV_TR'
	os.system(temp)

	


#Decode on a combination of a trained DNN-HMM acoustic model and a phone loop language model
def dnn_ploop_decode(addr,layer,ins,train):

	if train == 'full':
		temp = wrk + '../tools/steps/step-decode -INSWORD '+str(ins)+' '+ pwd + wrk + 'result_'+addr+ ' dnn'+str(layer+2)+'.finetune '+wrk+'result_'+addr+ '/decode_full_pl'
		os.system(temp)
		
		#copy and rename decoding accuracy file
		temp = 'cp '+wrk+'result_'+addr+ '/decode_full_pl/test/LOG'+' '+wrk+'super-result/result_'+addr
		os.system(temp)
		temp = 'mv -f '+wrk+'super-result/result_'+addr+'/LOG'+' '+wrk+'super-result/result_'+addr+'/LOG_decode_full_pl_i'+str(ins)
		os.system(temp)

	elif train == 'subtrain':
		temp = wrk + '../tools/steps/step-decode -SUBTRAIN -INSWORD '+str(ins)+' '+ pwd + wrk + 'result_'+addr+ ' dnn'+str(layer+2)+'.finetune '+wrk+'result_'+addr+ '/decode_subtrain_pl'
		os.system(temp)

		#copy and rename decoding accuracy file
		temp = 'cp '+wrk+'result_'+addr+ '/decode_subtrain_pl/test/LOG'+' '+wrk+'super-result/result_'+addr
		os.system(temp)
		temp = 'mv -f '+wrk+'super-result/result_'+addr+'/LOG'+' '+wrk+'super-result/result_'+addr+'/LOG_decode_subtrain_pl_i'+str(ins)
		os.system(temp)


	else:
		temp = wrk + '../tools/steps/step-decode -CORETEST -INSWORD '+str(ins)+' '+ pwd + wrk + 'result_'+addr+ ' dnn'+str(layer+2)+'.finetune '+wrk+'result_'+addr+ '/decode_coretest_pl'
		os.system(temp)

		#copy and rename decoding accuracy file
		temp = 'cp '+wrk+'result_'+addr+ '/decode_coretest_pl/test/LOG'+' '+wrk+'super-result/result_'+addr
		os.system(temp)
		temp = 'mv -f '+wrk+'super-result/result_'+addr+'/LOG'+' '+wrk+'super-result/result_'+addr+'/LOG_decode_coretest_pl_i'+str(ins)
		os.system(temp)

	

#Decode on a combination of a trained DNN-HMM acoustic model and a Bigarm language model
def dnn_bigram_decode(addr,layer,ins,gscale,train):

	if train == 'full':
		temp = wrk + '../tools/steps/step-decode -DECODEHTE '+ pwd + wrk + '../bigram/HTE.phoneloop -GRAMMARSCALE '+str(gscale)+' -INSWORD '+str(ins)+' '+ pwd + wrk + 'result_'+addr+ ' dnn'+str(layer+2)+'.finetune '+wrk+'result_'+addr+ '/decode_full_bi'
		os.system(temp)

		#copy and rename decoding accuracy file
		temp = 'cp '+wrk+'result_'+addr+ '/decode_full_bi/test/LOG'+' '+wrk+'super-result/result_'+addr
		os.system(temp)
		temp = 'mv -f '+wrk+'super-result/result_'+addr+'/LOG'+' '+wrk+'super-result/result_'+addr+'/LOG_decode_full_bi_i'+str(ins)+'_g'+str(gscale)
		os.system(temp)

	elif train == 'subtrain':
		temp = wrk + '../tools/steps/step-decode -SUBTRAIN -DECODEHTE '+ pwd + wrk + '../bigram/HTE.phoneloop -GRAMMARSCALE '+str(gscale)+' -INSWORD '+str(ins)+' '+ pwd + wrk + 'result_'+addr+ ' dnn'+str(layer+2)+'.finetune '+wrk+'result_'+addr+ '/decode_subtrain_bi'
		os.system(temp)

		#copy and rename decoding accuracy file
		temp = 'cp '+wrk+'result_'+addr+ '/decode_subtrain_bi/test/LOG'+' '+wrk+'super-result/result_'+addr
		os.system(temp)
		temp = 'mv -f '+wrk+'super-result/result_'+addr+'/LOG'+' '+wrk+'super-result/result_'+addr+'/LOG_decode_subtrain_bi_i'+str(ins)+'_g'+str(gscale)
		os.system(temp)

	else:
		temp = wrk + '../tools/steps/step-decode -CORETEST -DECODEHTE '+ pwd + wrk + '../bigram/HTE.phoneloop -GRAMMARSCALE '+str(gscale)+' -INSWORD '+str(ins)+' '+ pwd + wrk + 'result_'+addr+ ' dnn'+str(layer+2)+'.finetune '+wrk+'result_'+addr+ '/decode_coretest_bi'
		os.system(temp)

		#copy and rename decoding accuracy file
		temp = 'cp '+wrk+'result_'+addr+ '/decode_coretest_bi/test/LOG'+' '+wrk+'super-result/result_'+addr
		os.system(temp)
		temp = 'mv -f '+wrk+'super-result/result_'+addr+'/LOG'+' '+wrk+'super-result/result_'+addr+'/LOG_decode_coretest_bi_i'+str(ins)+'_g'+str(gscale)
		os.system(temp)
	
	



	

def master():
	
	#front-end parameters	
	env = '_E_D_A_Z'
	dim = 39
	ctxt = 6	
	
	#acoustic model parameters
	layers = [1]
	hidunits = [250]
	ptdecay = [0.001]
	ftdecay = [0.001]
	phone = ['mono','xwbil','xwbir','xwtri']
	
	
	#language model parameters
	insword_pl = [0.0]
	insword_bi = [0.0]
	grammarscale = [0.0]
	
	for ph in phone:	
		for hid in hidunits:
			for layer in layers:
				for ptl2 in ptdecay:
					for ftl2 in ftdecay:
						#modify HTE.dnntrain
						fname = wrk +'HTE.dnntrain'
						modify_file(fname, ctxt, dim, layer,hid,ptl2, ftl2)

						#string encapsulating everythin						
						addr = ph+'_h'+str(hid)+'_l' + str(layer)+'_pt'+str(ptl2)+'_ft'+str(ftl2)
						print addr
						os.system('mkdir -p '+wrk+'super-result/result_'+addr)
			
						#DNN_TRAIN
						dnn_train(addr,env,ph,layer)

						#PHONE LOOP DNN_DECODE
						for ins in insword_pl:
							dnn_ploop_decode(addr,layer,ins,'coretest')
							#dnn_ploop_decode(addr,layer,ins,'full')
							#dnn_ploop_decode(addr,layer,ins,'subtrain')
						
						#PHONE LOOP DNN_DECODE
						for ins in insword_bi:
							for gscale in grammarscale:
								dnn_bigram_decode(addr,layer,ins,gscale,'coretest')
								#dnn_bigram_decode(addr,layer,ins,gscale,'full')
								#dnn_bigram_decode(addr,layer,ins,gscale,'subtrain')
												
						#REMOVE FILES
						os.system('rm -r -f ' +wrk + 'result*')

		
		
if __name__ == "__main__":
	master()
	







