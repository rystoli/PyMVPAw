import glob
import random
#GOAL
# needs to write 1d per condition per condition per run per subject (4x12xN)


############################################
# LOAD FILES
############################################
subdir = "prts_new/"
paths = glob.glob(subdir+"*.prt")
files = [paths[i].split('/')[-1] for i,path in enumerate(paths)]

#load prts per run per subj into dict prts_Ss
prts_raw = {}
for i,fpath in enumerate(paths):
    with open(('%s' % (str(fpath),))) as f:
        order = f.readlines()
        prts_raw[fpath.split('/')[1]] = order

#make dict per prt, where keys are conditions, values a list of onsets
prts_Cond = {}
for prt_name,prt in prts_raw.iteritems():
    prts_Cond[prt_name] = {}
    for li,l in enumerate(prt):
        if l[0].isalpha() == True:  #checks for text lines saying condition names
            cond = l.split('\r')[0] #saves cond name for key
            prts_Cond[prt_name][cond] = [str(int(ol.split('\t')[0])*2) for ol in prt[li+2:li+2+(int(prt[li+1].split('\r')[0]))]]

#rename to generic conditions:
# black,male,blue = A ; white,female,red = B, typ = 1, atyp = 2

prtsG = {}
for prt_name,prt in prts_Cond.iteritems():
    prtsG[prt_name] = {}
    for cond,onsets in prt.iteritems():
        if len(onsets) > 0:
            if cond in ['black_typ','male_typ','blue_typ']: prtsG[prt_name]['A1'] = onsets
            elif cond in ['black_atyp','male_atyp','blue_atyp']: prtsG[prt_name]['A2'] = onsets
            elif cond in ['white_typ','female_typ','red_typ']: prtsG[prt_name]['B1'] = onsets
            elif cond in ['white_atyp','female_atyp','red_atyp']: prtsG[prt_name]['B2'] = onsets
            elif cond == 'error': prtsG[prt_name]['err'] = onsets
    if 'err' not in prtsG[prt_name].keys(): prtsG[prt_name]['err'] = ['*']          
for prt_name,prt in prtsG.iteritems():
    for cond,onsets in prt.iteritems():
        if len(prt_name) == 9:
            with open(prt_name[:4]+'0'+prt_name[4]+'_'+cond+'.1D','w') as f:
                f.write(' '.join(onsets))
        else:
            with open(prt_name.split('.')[0]+'_'+cond+'.1D','w') as f:
                f.write(' '.join(onsets))

#Find out task order per Ss
taskorder = {'424': {},'425': {},'426': {},'427': {},'428': {},'429': {},'430': {},'431': {},'432': {},'433': {},'434': {},'435': {},'436': {},'438': {},'439': {},'440': {}}
for prt_name,prt in prts_Cond.iteritems():
    for cond,onsets in prt.iteritems():
        if (len(onsets) > 0) and (cond != 'error'): taskorder[prt_name.split('_')[0]][int(prt_name.split('_')[1].split('.')[0])] = cond

for s,order in taskorder.iteritems():
    for run,task in order.iteritems():
        if task.split('_')[0] in ['blue','red']: taskorder[s][run]='color'
        elif task.split('_')[0] in ['male','female']: taskorder[s][run]='sex'
        elif task.split('_')[0] in ['black','white']: taskorder[s][run]='race'

h5save('taskorderSs.hdf5',taskorder,compression=9)



