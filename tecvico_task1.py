
# Import libraries
import numpy as np
import scipy as scipy
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt
import pydicom as  dicom
import nrrd as nrrd
import nibabel as nib
import dicom2nifti as dicom2



baseAdress = r"C:\\Users\\Admin\\Desktop\\Data_Hosseinzadehh\\"

# Get dcm files
Adress2 = "10_ct-20220324T071328Z-001\\10_ct\\1-"
Adress3 = '10_pet\\1-'  
list_Adress_dcm = []
m = baseAdress+Adress2
m2 = baseAdress+Adress3


x = 1
while x < 92:
    if x < 10:
        list_Adress_dcm.append(m + '0' + str(x) + '.dcm')
    else:
        list_Adress_dcm.append(m + str(x) + '.dcm')
    x += 1    
  

x = 1
while x < 92:
    if x < 10:
        list_Adress_dcm.append(m2 + '0' + str(x) + '.dcm')
    else:
        list_Adress_dcm.append(m2 + str(x) + '.dcm')
    x += 1


list_Adress_dcm.append(baseAdress + 'CHGJ007_ct.dcm') 


list_dcm = []
for item in list_Adress_dcm:
    header_dicm = dicom.read_file(item)
    list_dcm.append(header_dicm)


# Get nii.gz files 
Adress4 = 'Ct_test\\'
list_Adress_nii1 = []
m4 = baseAdress+Adress4
list_Adress_nii1.append(m4 + 'CHGJ007.nii.gz')
list_Adress_nii1.append(m4 + 'CHGJ008.nii.gz')
list_Adress_nii1.append(m4 + 'CHGJ010.nii.gz')    
    
Adress5 = 'Gtv_test\\'
m5 = baseAdress+Adress5
list_Adress_nii1.append(m5 + 'CHGJ007.nii.gz')
list_Adress_nii1.append(m5 + 'CHGJ008.nii.gz')    
list_Adress_nii1.append(m5 + 'CHGJ010.nii.gz')
    

Adress6 = 'CHGJ009_ct.nii.gz'
list_Adress_nii1.append(baseAdress+Adress6)  
    

list_nii1 = []
for item in list_Adress_nii1:
    header_nii1 = sitk.ReadImage(item)
    list_nii1.append(header_nii1)
    
    
    
# Get nrrd files
list_Adress_nrrd = []
list_Adress_nrrd.append(baseAdress + '10_ct.nrrd')
list_Adress_nrrd.append(baseAdress + '10_pet.nrrd')
list_Adress_nrrd.append(baseAdress + 'CHGJ0010_ct.nrrd')

list_nrrd = []
for item in list_Adress_nrrd:
    header_nrrd = nrrd.read_header(item)
    list_nrrd.append(header_nrrd)
  
    
# Get nii files
list_Adress_nii2 = []
list_Adress_nii2.append(baseAdress+'CHGJ007.nii')
list_Adress_nii2.append(baseAdress+'CHGJ008_ct.nii')
list_Adress_nii2.append(baseAdress+'DWI.nii')
list_Adress_nii2.append(baseAdress+'T2W.nii')   
    
list_nii2 = []
for item in list_Adress_nii2:
    header_nii2 = nib.load(item)
    list_nii2.append(header_nii2)
  



dataset_dcm = pd.DataFrame(list_dcm)
dataset_nii1 = pd.DataFrame(list_nii1)
dataset_nii2 = pd.DataFrame(list_nii2)
dataset_nrrd = pd.DataFrame(list_nrrd)
    