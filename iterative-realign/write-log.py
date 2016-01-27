
import copy

pwd = '/home/ar773/'
wrk = 'MLSALT2/practical2/iterative-realign/'




def update_cv_tr(addr,write_str):
	try:
		fn = pwd+wrk+'super-result/result_'+addr+'/LOG_CV_TR'		
		with open(fn) as f:
		    content = f.readlines()
		temp = ''
		for line in content:
			if "Validation Accuracy" in line:
				temp = line[24:29]
				
		if temp=='':
			write_str = write_str+',ERROR'
		else:
			write_str = write_str+','+temp
		temp = ''
		for line in content:
			if "Train Accuracy" in line:
				temp = line[19:24]
				
		
		if temp=='':
			write_str = write_str+',ERROR'
		else:
			write_str = write_str+','+temp
		return write_str

	except (OSError,IOError) as e:
		return ''

def update_bi(addr,ins,gscale,write_str,de):
	try:
		fn = pwd+wrk+'super-result/result_'+addr+'/LOG_decode'+de+'_bi_i'+str(ins)+'_g'+str(gscale)		
		with open(fn) as f:
		    content = f.readlines()
		temp = ''
		for line in content:
			if "WORD: %Corr" in line:
				temp = line[23:28]
				
		if temp=='':
			write_str = write_str+',ERROR'
		else:
			write_str = write_str+','+temp
		return write_str

	except (OSError,IOError) as e:
		return ''
	

def update_pl(addr,ins,write_str,de):
	try:
		fn = pwd+wrk+'super-result/result_'+addr+'/LOG_decode'+de+'_pl_i'+str(ins)		
		with open(fn) as f:
		    content = f.readlines()
		temp = ''
		for line in content:
			if "WORD: %Corr" in line:
				temp = line[23:28]
				
		if temp=='':
			write_str = write_str+',ERROR'
		else:
			write_str = write_str+','+temp
		return write_str

	except (OSError,IOError) as e:
		return ''

def write_log(de):

	
	
	env = '_E_D_A_Z'
	dim = 39
	ctxt_ls = [6]	
	
	#acoustic model parameters
	layers = [6]
	hidunits = [500]
	ptdecay = ['0.001']
	ftdecay = ['0.001']
	phone = ['mono','xwtri','xwbil','xwbir']
	
	
	#language model parameters
	insword_pl = [-4.0]
	insword_bi = [0.0]
	grammarscale = [0.0]
	
	

	#Update CV TR Accuracy for DNN-HMM acoustic model
	fname = pwd+wrk+'cv_tr.csv'
	f_cv_tr = open(fname,'w')
	f_cv_tr.write('iter,ph_model,nhid,ctxt,layers,ptdecay,ftdecay,cv_acc,tr_acc\n')
	write_str = ''
	
	for ctxt in ctxt_ls:
		for ph in phone:	
			for hid in hidunits:
				for layer in layers:
					for ptl2 in ptdecay:
						for ftl2 in ftdecay:
						
							for i in range(0,10):
								write_str = str(i)+','+ ph+','+str(hid)+',' +str(ctxt)+','+ str(layer)+','+str(ptl2)+','+str(ftl2)

								#string encapsulating everything						
								addr = str(i)+ph+'_h'+str(hid)+'_c'+str(ctxt)+'_l' + str(layer)+'_pt'+str(ptl2)+'_ft'+str(ftl2)
							

								write_str = update_cv_tr(addr,write_str)
	
								if write_str is not '':
									f_cv_tr.write(write_str+'\n')

	f_cv_tr.close()				

	#Update Decoding Accuracy file
	fname = pwd+wrk+'decode'+de+'_pl.csv'
	f_pl = open(fname,'w')
	f_pl.write('iter,ph_model,nhid,ctxt,layers,ptdecay,ftdecay,ins_pl,decode_acc\n')
	write_str = ''
	for ctxt in ctxt_ls:
		for ph in phone:	
			for hid in hidunits:
				for layer in layers:
					for ptl2 in ptdecay:
						for ftl2 in ftdecay:
							for ins in insword_pl:
								for i in range(0,10):
									write_str =str(i)+','+ ph+','+str(hid)+',' +str(ctxt)+',' + str(layer)+','+str(ptl2)+','+str(ftl2)+','+str(ins)
									#string encapsulating everything						
									addr = str(i)+ph+'_h'+str(hid)+'_c'+str(ctxt)+'_l' + str(layer)+'_pt'+str(ptl2)+'_ft'+str(ftl2)
								

									write_str = update_pl(addr,ins,write_str,de)

									if write_str is not '':
										f_pl.write(write_str+'\n')

	f_pl.close()

	#Update Decoding Accuracy file
	fname = pwd+wrk+'decode'+de+'_bi.csv'
	f_bi = open(fname,'w')
	f_bi.write('iter,ph_model,nhid,ctxt,layers,ptdecay,ftdecay,ins,gscale,decode_acc\n')
	write_str = ''
	for ctxt in ctxt_ls:
		for ph in phone:	
			for hid in hidunits:
				for layer in layers:
					for ptl2 in ptdecay:
						for ftl2 in ftdecay:
							for ins in insword_bi:
								for gscale in grammarscale:
									write_str = ph+','+str(hid)+',' +str(ctxt)+','+ str(layer)+','+str(ptl2)+','+str(ftl2)+','+str(ins)+','+str(gscale)

									#string encapsulating everything						
									addr = ph+'_h'+str(hid)+'_c'+str(ctxt)+'_l' + str(layer)+'_pt'+str(ptl2)+'_ft'+str(ftl2)
								

									write_str = update_bi(addr,ins,gscale,write_str,de)

									if write_str is not '':
										f_bi.write(write_str+'\n')
	
	f_bi.close()		
	

if __name__ == "__main__":

	de = '_coretest'
	write_log(de)

	de = '_full'
	write_log(de)

	de = '_subtrain'
	write_log(de)
	
