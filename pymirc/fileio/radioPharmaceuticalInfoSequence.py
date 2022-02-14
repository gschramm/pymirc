from pydicom.dataset import Dataset

import warnings

def GE_PET_nuclide_code_dict():
  d = {
  '11C':'C-105A1',
  '13N':'C-107A1',
  '14O':'C-1018C',
  '15O':'C-B1038',
  '18F':'C-111A1',
  '22Na':'C-155A1',
  '38K':'C-135A4',
  '43Sc':'126605',
  '44Sc':'126600',
  '45Ti':'C-166A2',
  '51Mn':'126601',
  '52Fe':'C-130A1',
  '52Mn':'C-149A1',
  '52mMn':'126607',
  '60Cu':'C-127A4',
  '61Cu':'C-127A1',
  '62Cu':'C-127A5',
  '62Zn':'C-141A1',
  '64Cu':'C-127A2',
  '66Ga':'C-131A1',
  '68Ga':'C-131A3',
  '68Ge':'C-128A2',
  '70As': '126602',
  '72As':'C-115A2',
  '73Se':'C-116A2',
  '75Br':'C-113A1',
  '76Br':'C-113A2',
  '77Br':'C-113A3',
  '82Rb':'C-159A2',
  '86Y':'C-162A3',
  '89Zr':'C-168A4',
  '90Nb':'126603',
  '90Y':'C-162A7',
  '94mTc':'C-163AA',
  '124I':'C-114A5',
  '152Tb':'126606'}

  return d

def radioPharmaceuticalDict():
  rpd = {
   '126752'   : ['28H1 ^89^Zr'                                      ,'DCM'],
   '126713'   : ['2FA F^18^'                                        ,'DCM'],
   '126751'   : ['7D12 ^89^Zr'                                      ,'DCM'],
   '126750'   : ['7E11 ^89^Zr'                                      ,'DCM'],
   'C-B1043'  : ['Acetate C^11^'                                    ,'SRT'],
   '126729'   : ['AGN-150998 ^89^Zr'                                ,'DCM'],
   'C-B103C'  : ['Ammonia N^13^'                                    ,'SRT'],
   '126754'   : ['Anti-B220 ^89^Zr'                                 ,'DCM'],
   '126700'   : ['ATSM Cu^60^'                                      ,'DCM'],
   '126701'   : ['ATSM Cu^61^'                                      ,'DCM'],
   '126702'   : ['ATSM Cu^62^'                                      ,'DCM'],
   'C-B07DB'  : ['ATSM Cu^64^'                                      ,'SRT'],
   '126722'   : ['Benralizumab ^89^Zr'                              ,'DCM'],
   '126516'   : ['Bevacizumab ^89^Zr'                               ,'DCM'],
   '126727'   : ['Blinatumomab ^89^Zr'                              ,'DCM'],
   '126735'   : ['Brentuximab ^89^Zr'                               ,'DCM'],
   'C-B07DC'  : ['Butanol O^15^'                                    ,'SRT'],
   'C-B103B'  : ['Carbon dioxide O^15^'                             ,'SRT'],
   'C-B1045'  : ['Carbon monoxide C^11^'                            ,'SRT'],
   'C-B103A'  : ['Carbon monoxide O^15^'                            ,'SRT'],
   'C-B103F'  : ['Carfentanil C^11^'                                ,'SRT'],
   '126513'   : ['Cetuximab ^89^Zr'                                 ,'DCM'],
   '126517'   : ['cG250-F(ab)(2) ^89^Zr'                            ,'DCM'],
   '126703'   : ['Choline C^11^'                                    ,'DCM'],
   '126715'   : ['CLR1404 I^124^'                                   ,'DCM'],
   '126716'   : ['CLR1404 I^131^'                                   ,'DCM'],
   '126746'   : ['cMAb U36 ^89^Zr'                                  ,'DCM'],
   '126515'   : ['cU36 ^89^Zr'                                      ,'DCM'],
   '126762'   : ['Df-[FK](2) ^89^Zr'                                ,'DCM'],
   '126763'   : ['Df-[FK](2)-3PEG(4) ^89^Zr'                        ,'DCM'],
   '126520'   : ['Df-CD45 ^89^Zr'                                   ,'DCM'],
   '126760'   : ['Df-FK ^89^Zr'                                     ,'DCM'],
   '126761'   : ['Df-FK-PEG(3) ^89^Zr'                              ,'DCM'],
   '126747'   : ['DN30 ^89^Zr'                                      ,'DCM'],
   '126519'   : ['E4G10 ^89^Zr'                                     ,'DCM'],
   '126732'   : ['Ecromeximab ^89^Zr'                               ,'DCM'],
   'C2713594' : ['Edotreotide Ga^68^'                               ,'UMLS'],
   'C-B07DD'  : ['EDTA Ga^68^'                                      ,'SRT'],
   '126704'   : ['Fallypride C^11^'                                 ,'DCM'],
   '126705'   : ['Fallypride F^18^'                                 ,'DCM'],
   '126706'   : ['FLB 457 C^11^'                                    ,'DCM'],
   '126501'   : ['Florbetaben F^18^'                                ,'DCM'],
   'C-E0269'  : ['Florbetapir F^18^'                                ,'SRT'],
   '126503'   : ['Flubatine F^18^'                                  ,'DCM'],
   '126712'   : ['Flubatine F^18^'                                  ,'DCM'],
   'C-E0265'  : ['Fluciclatide F^18^'                               ,'SRT'],
   'C-E026A'  : ['Fluciclovine F^18^'                               ,'SRT'],
   'C-B07DE'  : ['Flumazenil C^11^'                                 ,'SRT'],
   'C-B07DF'  : ['Flumazenil F^18^'                                 ,'SRT'],
   'C-B07E0'  : ['Fluorethyltyrosin F^18^'                          ,'SRT'],
   'C-B07E4'  : ['Fluorobenzothiazole F^18^'                        ,'SRT'],
   'C-E0273'  : ['Fluorocholine F^18^'                              ,'SRT'],
   'C-B1031'  : ['Fluorodeoxyglucose F^18^'                         ,'SRT'],
   'C1831937' : ['Fluoroestradiol (FES) F^18^'                      ,'UMLS'],
   'C1541539' : ['Fluoroetanidazole F^18^'                          ,'UMLS'],
   'C-B1034'  : ['Fluoro-L-dopa F^18^'                              ,'SRT'],
   'C-B07E2'  : ['Fluoromethane F^18^'                              ,'SRT'],
   'C-B07E1'  : ['Fluoromisonidazole F^18^'                         ,'SRT'],
   'C2934038' : ['Fluoropropyl-dihydrotetrabenazine (DTBZ) F^18^'   ,'UMLS'],
   '126707'   : ['Fluorotriopride F^18^'                            ,'DCM'],
   'C-B07E3'  : ['Fluorouracil F^18^'                               ,'SRT'],
   'C-E0267'  : ['Flutemetamol F^18^'                               ,'SRT'],
   '126748'   : ['Fresolimumab ^89^Zr'                              ,'DCM'],
   '126731'   : ['GA201 ^89^Zr'                                     ,'DCM'],
   'C-B1046'  : ['Germanium Ge^68^'                                 ,'SRT'],
   '126724'   : ['Glembatumumab vedotin ^89^Zr'                     ,'DCM'],
   'C-B103D'  : ['Glutamate N^13^'                                  ,'SRT'],
   '126709'   : ['Glutamine C^11^'                                  ,'DCM'],
   '126710'   : ['Glutamine C^14^'                                  ,'DCM'],
   '126711'   : ['Glutamine F^18^'                                  ,'DCM'],
   'C2981788' : ['ISO-1 F^18^'                                      ,'UMLS'],
   '126514'   : ['J591 ^89^Zr'                                      ,'DCM'],
   '126740'   : ['Margetuximab ^89^Zr'                              ,'DCM'],
   '126730'   : ['MEDI-551 ^89^Zr'                                  ,'DCM'],
   'C-B07E5'  : ['Mespiperone C^11^'                                ,'SRT'],
   'C-B103E'  : ['Methionine C^11^'                                 ,'SRT'],
   '126738'   : ['Mogamulizumab ^89^Zr'                             ,'DCM'],
   '126510'   : ['Monoclonal Antibody (mAb) ^64^Cu'                 ,'DCM'],
   '126511'   : ['Monoclonal Antibody (mAb) ^89^Zr'                 ,'DCM'],
   'C-B07E6'  : ['Monoclonal antibody I^124^'                       ,'SRT'],
   '126753'   : ['Nanocolloidal albumin ^89^Zr'                     ,'DCM'],
   '126714'   : ['Nifene F^18^'                                     ,'DCM'],
   '126721'   : ['Obinituzimab ^89^Zr'                              ,'DCM'],
   '126723'   : ['Ocaratuzumab ^89^Zr'                              ,'DCM'],
   'C-B1038'  : ['Oxygen O^15^'                                     ,'SRT'],
   'C-B1039'  : ['Oxygen-water O^15^'                               ,'SRT'],
   'C-B1044'  : ['Palmitate C^11^'                                  ,'SRT'],
   '126736'   : ['Panitumumab ^89^Zr'                               ,'DCM'],
   '126728'   : ['Pegdinetanib ^89^Zr'                              ,'DCM'],
   '126725'   : ['Pinatuzumab vedotin ^89^Zr'                       ,'DCM'],
   '126500'   : ['Pittsburgh compound B C^11^'                      ,'DCM'],
   '126726'   : ['Polatuzumab vedotin ^89^Zr'                       ,'DCM'],
   'C-B07E7'  : ['PTSM Cu^62^'                                      ,'SRT'],
   '126518'   : ['R1507 ^89^Zr'                                     ,'DCM'],
   'C-B1042'  : ['Raclopride C^11^'                                 ,'SRT'],
   '126742'   : ['Ranibizumab ^89^Zr'                               ,'DCM'],
   '126737'   : ['Rituximab ^89^Zr'                                 ,'DCM'],
   '126755'   : ['RO5323441 ^89^Zr'                                 ,'DCM'],
   '126756'   : ['RO542908 ^89^Zr'                                  ,'DCM'],
   '126733'   : ['Roledumab ^89^Zr'                                 ,'DCM'],
   'C-B1037'  : ['Rubidium chloride Rb^82^'                         ,'SRT'],
   '126741'   : ['SAR3419 ^89^Zr'                                   ,'DCM'],
   'C-B1032'  : ['Sodium fluoride F^18^'                            ,'SRT'],
   'C-B07E8'  : ['Sodium iodide I^124^'                             ,'SRT'],
   'C-B1047'  : ['Sodium Na^22^'                                    ,'SRT'],
   'C-B1033'  : ['Spiperone F^18^'                                  ,'SRT'],
   '126502'   : ['T807 F^18^'                                       ,'DCM'],
   'C-B1036'  : ['Thymidine (FLT) F^18^'                            ,'SRT'],
   '126512'   : ['Trastuzumab ^89^Zr'                               ,'DCM'],
   '126749'   : ['TRC105 ^89^Zr'                                    ,'DCM'],
   'C1742831' : ['tyrosine-3-octreotate Ga^68^'                     ,'UMLS'],
   '126739'   : ['Ublituximab ^89^Zr'                               ,'DCM'],
   '126734'   : ['XmAb5574 ^89^Zr'                                  ,'DCM']}


  return rpd

#-----------------------------------------------------------------------------------

def radioNuclideDict():
  isd = {
  'C-105A1' : ['^11^Carbon', 'SRT'],
  'C-107A1' : ['^13^Nitrogen', 'SRT'],
  'C-1018C' : ['^14^Oxygen', 'SRT'],
  'C-B1038' : ['^15^Oxygen', 'SRT'],
  'C-111A1' : ['^18^Fluorine', 'SRT'],
  'C-155A1' : ['^22^Sodium', 'SRT'],
  'C-135A4' : ['^38^Potassium', 'SRT'],
  '126605'  : ['^43^Scandium', 'DCM'],
  '126600'  : ['^44^Scandium', 'DCM'],
  'C-166A2' : ['^45^Titanium', 'SRT'],
  '126601'  : ['^51^Manganese', 'DCM'],
  'C-130A1' : ['^52^Iron', 'SRT'],
  'C-149A1' : ['^52^Manganese', 'SRT'],
  '126607'  : ['^52m^Manganese', 'DCM'],
  'C-127A4' : ['^60^Copper', 'SRT'],
  'C-127A1' : ['^61^Copper', 'SRT'],
  'C-127A5' : ['^62^Copper', 'SRT'],
  'C-141A1' : ['^62^Zinc', 'SRT'],
  'C-127A2' : ['^64^Copper', 'SRT'],
  'C-131A1' : ['^66^Gallium', 'SRT'],
  'C-131A3' : ['^68^Gallium', 'SRT'],
  'C-128A2' : ['^68^Germanium', 'SRT'],
  '126602'  : ['^70^Arsenic', 'DCM'],
  'C-115A2' : ['^72^Arsenic', 'SRT'],
  'C-116A2' : ['^73^Selenium', 'SRT'],
  'C-113A1' : ['^75^Bromine', 'SRT'],
  'C-113A2' : ['^76^Bromine', 'SRT'],
  'C-113A3' : ['^77^Bromine', 'SRT'],
  'C-159A2' : ['^82^Rubidium', 'SRT'],
  'C-162A3' : ['^86^Yttrium', 'SRT'],
  'C-168A4' : ['^89^Zirconium', 'SRT'],
  '126603'  : ['^90^Niobium', 'DCM'],
  'C-162A7' : ['^90^Yttrium', 'SRT'],
  'C-163AA' : ['^94m^Technetium', 'SRT'],
  'C-114A5' : ['^124^Iodine', 'SRT'],
  '126606'  : ['^152^Terbium', 'DCM']}

  return isd


#-----------------------------------------------------------------------------------

def radioNuclideCodeSequence(codeval = 'C-111A1'):

  rnd = radioNuclideDict()
  
  ds = Dataset()
  
  if codeval in rnd.keys():
    ds.CodeValue              = codeval
    val                       = rnd[codeval]
    ds.CodeMeaning            = val[0]
    ds.CodingSchemeDesignator = val[1]
  else:
    warnings.warn('code value: ' + codeval + ' not supported') 

  return ds


#-----------------------------------------------------------------------------------

def radioPharmaceuticalCodeSequence(codeval = 'C-B1031'):

  radiopharmadict = radioPharmaceuticalDict()
  
  ds = Dataset()
  
  if codeval in radiopharmadict.keys():
    ds.CodeValue              = codeval
    val                       = radiopharmadict[codeval]
    ds.CodeMeaning            = val[0]
    ds.CodingSchemeDesignator = val[1]
  else:
    warnings.warn('code value: ' + codeval + ' not supported') 

  return ds

#-----------------------------------------------------------------------------------

def radioPharmaceuticalInfoSequence(nuclidecode          = None, 
                                    pharmacode           = None,
                                    StartDateTime        = None,
                                    TotalDose            = None, # Dose in Bq
                                    HalfLife             = None, # half life in s
                                    PositronFraction     = None  # positron fraction
                                   ):
  ds = Dataset()

  if nuclidecode != None: 
    rns = radioNuclideCodeSequence(codeval = nuclidecode)
    ds.RadionuclideCodeSequence = [rns]
  if pharmacode  != None: 
    rps = radioPharmaceuticalCodeSequence(codeval = pharmacode)
    ds.RadiopharmaceuticalCodeSequence = [rps]
    ds.Radiopharmaceutical             = rps.CodeMeaning
 
  # some settings for known isotopes
  if HalfLife == None:
    if   nuclidecode == 'C-111A1': HalfLife = 6586.2  #18-F
    elif nuclidecode == 'C-105A1': HalfLife = 1223.4  #11-C
    elif nuclidecode == 'C-107A1': HalfLife = 597.9   #13-N
    elif nuclidecode == 'C-B1038': HalfLife = 122.24  #15-O
    elif nuclidecode == 'C-131A3': HalfLife = 4057.74 #68-Ga
  if HalfLife != None: ds.RadionuclideHalfLife = HalfLife

  if PositronFraction == None:
    if   nuclidecode == 'C-111A1': PositronFraction = 0.967  #18-F
    elif nuclidecode == 'C-105A1': PositronFraction = 0.998  #11-C
    elif nuclidecode == 'C-107A1': PositronFraction = 0.998  #13-N
    elif nuclidecode == 'C-B1038': PositronFraction = 0.999  #15-O
    elif nuclidecode == 'C-131A3': PositronFraction = 0.889  #68-Ga
  if PositronFraction != None: ds.RadionuclidePositronFraction = PositronFraction

  if TotalDose     != None: ds.RadionuclideTotalDose            = TotalDose
  if StartDateTime != None: ds.RadiopharmaceuticalStartDateTime = StartDateTime
 
  return ds
