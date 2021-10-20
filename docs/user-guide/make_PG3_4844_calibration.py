from mantid.simpleapi import Load, LoadDiffCal
import scippneutron as scn

ws = Load('PG3_4844_event.nxs')
ws = LoadDiffCal('PG3_golden.cal', InputWorkspace='ws', WorkspaceName='ws')
cal = scn.from_mantid(ws[0]).rename_dims({'row': 'spectrum'})
cal['tzero'].unit = 'us'
cal['difc'].unit = 'us/angstrom'
cal['difa'].unit = 'us/(angstrom*angstrom)'
cal.to_hdf5('PG3_4844_calibration.h5')
