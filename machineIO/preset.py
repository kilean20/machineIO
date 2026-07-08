try:
    from epics import caget, caget_many
except:
    pass

from typing import List, Dict, Union, Optional, Callable
import numpy as np


def get_limits(PV_CSETs: List[str]):
    '''
    Automatically retrive limit for PV put
    PV_CSETs: list of CSET-PVs 
    '''
    lo_lim = []
    hi_lim = []
    failedPV = []
    for pv in PV_CSETs:
        failedPV = []
        if ':V_CSET' in pv:
#             tmp = [pv.replace(':V_CSET',':V_CSET.LOPR'), pv.replace(':V_CSET',':V_CSET.HOPR')]
            tmp = [pv.replace(':V_CSET',':V_CSET.DRVL'), pv.replace(':V_CSET',':V_CSET.DRVH')]
            try:
                tmp = caget_many(tmp)
            except:
                failedPV.append(pv)
            lo_lim.append(tmp[0])
            hi_lim.append(tmp[1])
        elif ':I_CSET' in pv:
#             tmp = [pv.replace(':I_CSET',':I_CSET.LOPR'), pv.replace(':I_CSET',':I_CSET.HOPR')]
            tmp = [pv.replace(':I_CSET',':I_CSET.DRVL'), pv.replace(':I_CSET',':I_CSET.DRVH')]
            try:
                tmp = caget_many(tmp)
            except:
                failedPV.append(pv)
            lo_lim.append(tmp[0])
            hi_lim.append(tmp[1])
        else:
            lo_lim.append(-np.inf)
            hi_lim.append( np.inf)
    lo_lim = np.array(lo_lim)
    hi_lim = np.array(hi_lim)
    if len(failedPV)>0:
        raise RuntimeError(f'failed to find operation limit for {failedPV}. Manually ensure the control limit')
    assert np.all(lo_lim < hi_lim)
    return lo_lim, hi_lim
    
    
def get_RDs(PV_CSETs: List[str]):
    PV_RDs = []
    failedPV = []
    for pv in PV_CSETs:
        if '_CSET' in pv:
            PV_RDs.append(pv.replace('_CSET','_RD'))
        elif '_MTR.VAL' in pv:
            PV_RDs.append(pv.replace('_MTR.VAL','_MTR.RBV'))
        else:
            failedPV.append(pv)
    if len(failedPV)>0:
        raise RuntimeError(f'failed to find operation limit for {failedPV}. Manually ensure the control limit')
    return PV_RDs


def get_MEBT_objective_goal_from_BPMoverview(fname):
    try:
        with open(fname,'r') as f:
            lines = f.readlines()
    except:
        path = '/files/shared/phyapps-operations/data/bpm_overview/snapshots/'
        with open(path+fname,'r') as f:
            lines = f.readlines() 
    lines = lines[340:]
    for i,line in enumerate(lines):
        if 'BPM DATA' in line:
            break
            
    def float_vec(list_of_str):
        return [float(s) for s in list_of_str]
        
    MEBT_BPM_vals = [float_vec(line.split()) for line in lines[i+3:i+6]]
    
    return { 
            'FE_MEBT:BPM_D1056:XPOS_RD' : MEBT_BPM_vals[0][3],
            'FE_MEBT:BPM_D1056:YPOS_RD' : MEBT_BPM_vals[0][4],
            'FE_MEBT:BPM_D1056:PHASE_RD': MEBT_BPM_vals[0][1],
            'FE_MEBT:BPM_D1072:XPOS_RD' : MEBT_BPM_vals[1][3],
            'FE_MEBT:BPM_D1072:YPOS_RD' : MEBT_BPM_vals[1][4],
            'FE_MEBT:BPM_D1072:PHASE_RD': MEBT_BPM_vals[1][1],
            'FE_MEBT:BPM_D1094:XPOS_RD' : MEBT_BPM_vals[2][3],
            'FE_MEBT:BPM_D1094:YPOS_RD' : MEBT_BPM_vals[2][4],
            'FE_MEBT:BPM_D1094:PHASE_RD': MEBT_BPM_vals[2][1],
           } 


FRIB_BPMs = [
 'FE_MEBT:BPM_D1056',
 'FE_MEBT:BPM_D1072',
 'FE_MEBT:BPM_D1094',
 'FE_MEBT:BPM_D1111',
 'LS1_CA01:BPM_D1129',
 'LS1_CA01:BPM_D1144',
 'LS1_WA01:BPM_D1155',
 'LS1_CA02:BPM_D1163',
 'LS1_CA02:BPM_D1177',
 'LS1_WA02:BPM_D1188',
 'LS1_CA03:BPM_D1196',
 'LS1_CA03:BPM_D1211',
 'LS1_WA03:BPM_D1222',
 'LS1_CB01:BPM_D1231',
 'LS1_CB01:BPM_D1251',
 'LS1_CB01:BPM_D1271',
 'LS1_WB01:BPM_D1286',
 'LS1_CB02:BPM_D1295',
 'LS1_CB02:BPM_D1315',
 'LS1_CB02:BPM_D1335',
 'LS1_WB02:BPM_D1350',
 'LS1_CB03:BPM_D1359',
 'LS1_CB03:BPM_D1379',
 'LS1_CB03:BPM_D1399',
 'LS1_WB03:BPM_D1413',
 'LS1_CB04:BPM_D1423',
 'LS1_CB04:BPM_D1442',
 'LS1_CB04:BPM_D1462',
 'LS1_WB04:BPM_D1477',
 'LS1_CB05:BPM_D1486',
 'LS1_CB05:BPM_D1506',
 'LS1_CB05:BPM_D1526',
 'LS1_WB05:BPM_D1541',
 'LS1_CB06:BPM_D1550',
 'LS1_CB06:BPM_D1570',
 'LS1_CB06:BPM_D1590',
 'LS1_WB06:BPM_D1604',
 'LS1_CB07:BPM_D1614',
 'LS1_CB07:BPM_D1634',
 'LS1_CB07:BPM_D1654',
 'LS1_WB07:BPM_D1668',
 'LS1_CB08:BPM_D1677',
 'LS1_CB08:BPM_D1697',
 'LS1_CB08:BPM_D1717',
 'LS1_WB08:BPM_D1732',
 'LS1_CB09:BPM_D1741',
 'LS1_CB09:BPM_D1761',
 'LS1_CB09:BPM_D1781',
 'LS1_WB09:BPM_D1796',
 'LS1_CB10:BPM_D1805',
 'LS1_CB10:BPM_D1825',
 'LS1_CB10:BPM_D1845',
 'LS1_WB10:BPM_D1859',
 'LS1_CB11:BPM_D1869',
 'LS1_CB11:BPM_D1889',
 'LS1_CB11:BPM_D1909',
 'LS1_WB11:BPM_D1923',
 'LS1_BTS:BPM_D1967',
 'LS1_BTS:BPM_D1980',
 'LS1_BTS:BPM_D2027',
 'LS1_BTS:BPM_D2054',
 'LS1_BTS:BPM_D2116',
 'LS1_BTS:BPM_D2130',
 'FS1_CSS:BPM_D2212',
 'FS1_CSS:BPM_D2223',
 'FS1_CSS:BPM_D2248',
 'FS1_CSS:BPM_D2278',
 'FS1_CSS:BPM_D2313',
 'FS1_CSS:BPM_D2369',
 'FS1_CSS:BPM_D2383',
 'FS1_BBS:BPM_D2421',
 'FS1_BTS:BPM_D2424',
 'FS1_SEE:BPM_D2449',
 'FS1_BBS:BPM_D2466',
 'FS1_BTS:BPM_D2467',
 'FS1_BTS:BPM_D2486',
 'FS1_SEE:BPM_D2487',
 'FS1_BMS:BPM_D2502',
 'FS1_BMS:BPM_D2537',
 'FS1_BMS:BPM_D2587',
 'FS1_BMS:BPM_D2600',
 'FS1_BMS:BPM_D2665',
 'FS1_BMS:BPM_D2690',
 'FS1_BMS:BPM_D2702',
 'LS2_WC01:BPM_D2742',
 'LS2_WC02:BPM_D2782',
 'LS2_WC03:BPM_D2821',
 'LS2_WC04:BPM_D2861',
 'LS2_WC05:BPM_D2901',
 'LS2_WC06:BPM_D2941',
 'LS2_WC07:BPM_D2981',
 'LS2_WC08:BPM_D3020',
 'LS2_WC09:BPM_D3060',
 'LS2_WC10:BPM_D3100',
 'LS2_WC11:BPM_D3140',
 'LS2_WC12:BPM_D3180',
 'LS2_WD01:BPM_D3242',
 'LS2_WD02:BPM_D3304',
 'LS2_WD03:BPM_D3366',
 'LS2_WD04:BPM_D3428',
 'LS2_WD05:BPM_D3490',
 'LS2_WD06:BPM_D3552',
 'LS2_WD07:BPM_D3614',
 'LS2_WD08:BPM_D3676',
 'LS2_WD09:BPM_D3738',
 'LS2_WD10:BPM_D3800',
 'LS2_WD11:BPM_D3862',
 'LS2_WD12:BPM_D3924',
 'FS2_BTS:BPM_D3943',
 'FS2_BTS:BPM_D3958',
 'FS2_BTS:BPM_D4006',
 'FS2_BBS:BPM_D4019',
 'FS2_BBS:BPM_D4054',
 'FS2_BBS:BPM_D4087',
 'FS2_BMS:BPM_D4142',
 'FS2_BMS:BPM_D4164',
 'FS2_BMS:BPM_D4177',
 'FS2_BMS:BPM_D4216',
 'FS2_BMS:BPM_D4283',
 'FS2_BMS:BPM_D4326',
 'LS3_WD01:BPM_D4389',
 'LS3_WD02:BPM_D4451',
 'LS3_WD03:BPM_D4513',
 'LS3_WD04:BPM_D4575',
 'LS3_WD05:BPM_D4637',
 'LS3_WD06:BPM_D4699',
 'LS3_BTS:BPM_D4753',
 'LS3_BTS:BPM_D4769',
 'LS3_BTS:BPM_D4843',
 'LS3_BTS:BPM_D4886',
 'LS3_BTS:BPM_D4968',
 'LS3_BTS:BPM_D5010',
 'LS3_BTS:BPM_D5092',
 'LS3_BTS:BPM_D5134',
 'LS3_BTS:BPM_D5216',
 'LS3_BTS:BPM_D5259',
 'LS3_BTS:BPM_D5340',
 'LS3_BTS:BPM_D5381',
 'LS3_BTS:BPM_D5430',
 'LS3_BTS:BPM_D5445',
 'BDS_BTS:BPM_D5499',
 'BDS_BTS:BPM_D5513',
 'BDS_BTS:BPM_D5565',
 'BDS_BBS:BPM_D5625',
 'BDS_BTS:BPM_D5649',
 'BDS_BBS:BPM_D5653',
 'BDS_BBS:BPM_D5680',
 'BDS_FFS:BPM_D5742',
 'BDS_FFS:BPM_D5772',
 'BDS_FFS:BPM_D5790',
 'BDS_FFS:BPM_D5803',
 'BDS_FFS:BPM_D5818']