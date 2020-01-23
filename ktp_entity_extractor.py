import sys
import os
import re
import math
import copy
import pandas as pd
import numpy as np
import bisect
from datetime import datetime
import kyc_config as cfg

def levenshtein(source, target):
    if len(source) < len(target):
        return levenshtein(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]

def correct2numbers(words):
    words = words.replace(' ','')
    if isNumber(words):
        result = ''
        for cc in words:
            if cc in ['T','I']:
                result+='1'
            elif cc>='0' and cc<='9' :
                result+=cc
            else :
                result+='0'
        words = result
    return words

def calDegBox(box,x,y,w):
    ls_cal_abs = [np.abs(nx-x)+np.abs(ny-y) for nx,ny in box]
    index = np.argmin(ls_cal_abs)

    ls_cal_abs2 = [np.abs(nx-x-w)+np.abs(ny-y) for nx,ny in box]
    index2 = np.argmin(ls_cal_abs2)

    x1,y1 = box[index]
    x2,y2 = box[index2]
    myradians = math.atan2(y1-y2, x1-x2)
    mydegrees = math.degrees(myradians)
    mydegrees = mydegrees if mydegrees >= 0 else 360+mydegrees
    return mydegrees

def calDeg(x1,y1,x2,y2):
    myradians = math.atan2(y1-y2, x1-x2)
    mydegrees = math.degrees(myradians)
    mydegrees = mydegrees if mydegrees >= 0 else 360+mydegrees
    return mydegrees

def convert_format(text_response):
    ls_word = []
    if ('textAnnotations' in text_response):
        for text in text_response['textAnnotations']:
            boxes = {}
            boxes['label'] = text['description']

            boxes['x1'] = text['boundingPoly']['vertices'][0].get('x',0)
            boxes['y1'] = text['boundingPoly']['vertices'][0].get('y',0)
            boxes['x2'] = text['boundingPoly']['vertices'][1].get('x',0)
            boxes['y2'] = text['boundingPoly']['vertices'][1].get('y',0)
            boxes['x3'] = text['boundingPoly']['vertices'][2].get('x',0)
            boxes['y3'] = text['boundingPoly']['vertices'][2].get('y',0)
            boxes['x4'] = text['boundingPoly']['vertices'][3].get('x',0)
            boxes['y4'] = text['boundingPoly']['vertices'][3].get('y',0)

            boxes['w'] = boxes['x3'] - boxes['x1']
            boxes['h'] = boxes['y3'] - boxes['y1']

            #print(boxes)
            ls_word.append(boxes)
    return ls_word

def get_attribute_ktp(ls_word,field_name,field_keywords,typo_tolerance, debug_mode=False):
    if(len(ls_word)==0):
        return None

    if(field_name == 'nama'):
        ls_word = np.asarray([word for word in ls_word if word['label'].lower() not in ['jawa','nusa'] ])

    new_ls_word = np.asarray([word['label'].lower() for word in ls_word])

    ls_dist = [levenshtein(field_keywords, word.lower()) for word in new_ls_word]
    if np.min(ls_dist) > typo_tolerance:

        if(field_name == 'kota' and field_keywords!='kota'):
            return get_attribute_ktp(ls_word,field_name,'kota',1,debug_mode)

        return None
    index = np.argmin(ls_dist)
    x,y = ls_word[index]['x1'], ls_word[index]['y1']
    w = ls_word[index]['w']
    degree = calDeg(ls_word[index]['x1'],ls_word[index]['y1'],ls_word[index]['x2'],ls_word[index]['y2'])

    ls_y = np.asarray([np.abs(y-word['y1'])<300 for word in ls_word])

    value_words = [ww for ww, val in zip(ls_word,ls_y) if (val and np.abs(calDeg(x,y,ww['x1'],ww['y1'])-degree)<3)]

    if debug_mode:
        print(value_words)

    # handling special attributes

    value_words = [val for val in value_words if len(val['label'].replace(' ','').replace(':',''))>0]

    d = [levenshtein('gol.', str(val['label']).lower()) for val in value_words]
    if(len(d)>0 and min(d) <= 1):
        idx = np.argmin(d)
        value_words.pop(idx)

    d = [levenshtein('darah', str(val['label']).lower()) for val in value_words]
    if(len(d)>0 and min(d) <= 1):
        idx = np.argmin(d)
        value_words.pop(idx)

    if(field_name == 'nik'):
        if(len(value_words)>0):
            global max_x
            max_x = max([val['x2'] for val in value_words])

    if(field_name == 'kota'):
        field_value = ""
        for val in value_words:
            field_value = field_value + ' '+ str(val['label'])
        field_value = field_value.lstrip()

        if(field_keywords == 'kabupaten'):
            return 'KABUPATEN '+field_value
        else:
            return 'KOTA '+field_value

    if(field_name == 'ttl'):
            d = [levenshtein('lahir', str(val['label']).lower()) for val in value_words]
            if(len(d)>0 and min(d) <= 2):
                idx = np.argmin(d)
                value_words.pop(idx)
    elif(field_name == 'jenis_kelamin'):
            score_laki, score_wanita = 999,999
            d = [levenshtein('laki-laki', str(val['label']).lower()) for val in value_words]
            if(len(d)>0 and min(d) <= 2):
                return 'LAKI-LAKI'

            d = [levenshtein('laki', str(val['label']).lower()) for val in value_words]
            if(len(d)>0 and min(d) <= 1):
                return 'LAKI-LAKI'

            d = [levenshtein('wanita', str(val['label']).lower()) for val in value_words]
            if(len(d)>0 and min(d) <= 2):
                return 'WANITA'

            d = [levenshtein('perempuan', str(val['label']).lower()) for val in value_words]
            if(len(d)>0 and min(d) <= 2):
                return 'PEREMPUAN'

            return None

    elif(field_name == 'gol_darah'):
            vals = [val['label'] for val in value_words if len(val['label']) <= 3]
            if(len(vals)>0):
                return vals[0]
            else:
                return None


    elif(field_name == 'pekerjaan'):
            d = [levenshtein('kartu', str(val['label']).lower()) for val in value_words]
            if(len(d)>0 and min(d) <= 2):
                idx = np.argmin(d)
                value_words.pop(idx)

            value_words = [val for val in value_words if val['x1'] <= max_x]

    elif(field_name == 'kewarganegaraan'):
            d = [levenshtein('wni', str(val['label']).lower()) for val in value_words]
            if(len(d)>0):
                return 'WNI'

            xlocs = [val['x1'] for val in value_words]
            if(len(xlocs)>0):
                idx = np.argmin(xlocs)
                return value_words[idx]['label']
            else:
                return None


    elif(field_name == 'status_perkawinan'):
            xlocs = [val['x1'] for val in value_words]
            if(len(xlocs)>0):
                idx = np.argmin(xlocs)
                field_value = value_words[idx]['label']

                if(levenshtein('belum',field_value.lower()) <= 1):
                    return 'BELUM KAWIN'
                else:
                    return field_value
            else:
                return None


    elif(field_name == 'berlaku_hingga'):
            d = [levenshtein('hingga', str(val['label']).lower()) for val in value_words]
            if(len(d)>0 and min(d) <= 2):
                idx = np.argmin(d)
                value_words.pop(idx)

            xlocs = [val['x1'] for val in value_words]
            if(len(xlocs)>0):
                idx = np.argmin(xlocs)
                field_value = value_words[idx]['label']
                if(levenshtein('seumur',field_value.lower()) <= 2):
                    return 'SEUMUR HIDUP'
                else:
                    return field_value
            else:
                return None


    field_value = ""
    for val in value_words:
        field_value = field_value + ' '+ str(val['label'])
    field_value = field_value.lstrip()

    return field_value

def get_gender(ls_word):
    new_ls_word = np.asarray([word['label'].lower() for word in ls_word])

    d = [levenshtein('laki-laki', word.lower()) for word in new_ls_word]
    if(len(d)>0 and min(d) <= 3):
            return 'male'

    d = [levenshtein('wanita', word.lower()) for word in new_ls_word]
    if(len(d)>0 and min(d) <= 2):
            return 'female'

    d = [levenshtein('perempuan', word.lower()) for word in new_ls_word]
    if(len(d)>0 and min(d) <= 2):
            return 'female'

    d = [levenshtein('pria', word.lower()) for word in new_ls_word]
    if(len(d)>0 and min(d) <= 1):
            return 'male'

    d = [levenshtein('laki', word.lower()) for word in new_ls_word]
    if(len(d)>0 and min(d) <= 1):
            return 'male'

    return None

def extract_date(date_string):
    if(date_string == None):
        return None

    date = None
    try:
        regex = re.compile(r'(\d{1,2}-\d{1,2}-\d{1,4})')
        tgl = re.findall(regex, date_string)
        if(len(tgl)>0):
            date = datetime.strptime(tgl[0], '%d-%m-%Y')
        else:
            tgl = ''.join([n for n in date_string if n.isdigit()])
            if(len(tgl)==8):
                date = datetime.strptime(tgl[0:2]+'-'+tgl[2:4]+'-'+tgl[4:], '%d-%m-%Y')
    except ValueError:
        return None

    if(date==None):
        return None

    if((date.year < 1910) or (date.year > 2100)):
        return None

    return date

def find_occupation(occ):
    if(occ==None):
        return None

    result = occ
    if(levenshtein('mengurus rumah tangga',occ.lower()) <= 6):
            result = 'Mengurus Rumah Tangga'
    if(levenshtein('buruh harian lepas',occ.lower()) <= 6):
            result = 'Buruh Harian Lepas'
    if(levenshtein('pegawai negeri sipil',occ.lower()) <= 5):
            result = 'Pegawai Negeri Sipil'
    if(levenshtein('pelajar/mahasiswa',occ.lower()) <= 4):
            result = 'Pelajar/Mahasiswa'
    if(levenshtein('pelajar/mhs',occ.lower()) <= 3):
            result = 'Pelajar/Mahasiswa'
    if(levenshtein('belum/tidak bekerja',occ.lower()) <= 5):
            result = 'Belum/Tidak Bekerja'
    if(levenshtein('karyawan swasta',occ.lower()) <= 4):
            result = 'Karyawan Swasta'
    if(levenshtein('pegawai negeri',occ.lower()) <= 4):
            result = 'Pegawai Negeri'
    if(levenshtein('wiraswasta',occ[0:10].lower()) <= 3):
            result = 'Wiraswasta'
    if(levenshtein('peg negeri',occ.lower()) <= 3):
            result = 'Pegawai Negeri'
    if(levenshtein('peg swasta',occ.lower()) <= 3):
            result = 'Pegawai Swasta'

    return result


fields_ktp = [
    {'field_name': 'provinsi', 'keywords': 'provinsi', 'typo_tolerance': 2},
    {'field_name': 'kota', 'keywords': 'kabupaten', 'typo_tolerance': 2},
    {'field_name': 'nik', 'keywords': 'nik', 'typo_tolerance': 1},
    {'field_name': 'nama', 'keywords': 'nama', 'typo_tolerance': 2},
    {'field_name': 'ttl', 'keywords': 'tempat/tgl', 'typo_tolerance': 5},
    {'field_name': 'jenis_kelamin', 'keywords': 'kelamin', 'typo_tolerance': 3},
    {'field_name': 'gol_darah', 'keywords': 'darah', 'typo_tolerance': 3},
    {'field_name': 'alamat', 'keywords': 'alamat', 'typo_tolerance': 2},
    {'field_name': 'rt_rw', 'keywords': 'rt/rw', 'typo_tolerance': 2},
    {'field_name': 'kel_desa', 'keywords': 'kel/desa', 'typo_tolerance': 3},
    {'field_name': 'kecamatan', 'keywords': 'kecamatan', 'typo_tolerance': 3},
    {'field_name': 'agama', 'keywords': 'agama', 'typo_tolerance': 3},
    {'field_name': 'status_perkawinan', 'keywords': 'perkawinan', 'typo_tolerance': 4},
    {'field_name': 'pekerjaan', 'keywords': 'pekerjaan', 'typo_tolerance': 4},
    {'field_name': 'kewarganegaraan', 'keywords': 'kewarganegaraan', 'typo_tolerance': 4},
    {'field_name': 'berlaku_hingga', 'keywords': 'berlaku', 'typo_tolerance': 4}
]

def extract_ktp_data(text_response,debug_mode=False):

    ktp_extract = pd.DataFrame(columns=['province','city','identity_number','fullname','birth_place','birth_date','nationality','occupation','gender','marital_status',
                                        'blood_type','address','rt_rw','kel_desa','kecamatan','religion','expired_date','state'])

    attributes = {}

    ls_word = convert_format(text_response)

    if(len(ls_word)==0):
        attributes['state'] = "rejected"
        ktp_extract = ktp_extract.append(attributes,ignore_index=True)
        return ktp_extract

    global max_x
    max_x = 9999

    raw_result = {}

    for field in fields_ktp:
        field_value = get_attribute_ktp(ls_word,field['field_name'],field['keywords'],field['typo_tolerance'],debug_mode)
        if(field_value != None):
            field_value = str(field_value).replace(': ','').replace(':','')
        #print(field['field_name'] +': '+str(field_value) )
        raw_result[field['field_name']] = field_value


    attributes['state'] = 'ok'

    attributes['identity_number'] = raw_result['nik']
    if(attributes['identity_number'] != None):
        attributes['identity_number'] = ''.join([i for i in raw_result['nik'] if i.isdigit()])

    if(attributes['identity_number'] == None):
        attributes['state'] = "rejected"
        ktp_extract = ktp_extract.append(attributes,ignore_index=True)
        return ktp_extract

    attributes['fullname'] = raw_result['nama']
    if(raw_result['nama'] != None):
        attributes['fullname'] = ''.join([i for i in raw_result['nama'] if not i.isdigit()]).replace('-','').strip()


    if(raw_result['jenis_kelamin'] == 'LAKI-LAKI'):
        attributes['gender'] = 'male'
    elif(raw_result['jenis_kelamin'] in ['WANITA','PEREMPUAN']):
        attributes['gender'] = 'female'
    else:
        attributes['gender'] = get_gender(ls_word)

    attributes['birth_place'] = None
    attributes['birth_date'] = None

    if(raw_result['ttl'] != None):
        ttls = raw_result['ttl'].split(', ')
        if(len(ttls)>=2):
            attributes['birth_place'] = ttls[0]
            attributes['birth_date'] = extract_date(ttls[1])

        elif(len(ttls)==1):
            attributes['birth_place'] = ttls[0]

        if(attributes['birth_date'] == None):
            attributes['birth_date'] = extract_date(raw_result['ttl'])

    if(attributes['birth_place'] != None):
        attributes['birth_place'] = ''.join([i for i in attributes['birth_place'] if not i.isdigit()]).replace('-','').replace('.','').strip()

    attributes['nationality'] = raw_result['kewarganegaraan']

    if(attributes['nationality'] == "WNI"):
        attributes['nationality'] = "INDONESIA"

    attributes['marital_status'] = raw_result['status_perkawinan']
    if(attributes['marital_status'] != None):
        if(levenshtein('belum kawin',attributes['marital_status'].lower()) <= 2):
            attributes['marital_status'] = 'single'
        elif(levenshtein('tidak kawin',attributes['marital_status'].lower()) <= 2):
            attributes['marital_status'] = 'single'
        elif(levenshtein('kawin',attributes['marital_status'].lower()) <= 1):
            attributes['marital_status'] = 'married'
        elif(levenshtein('janda',attributes['marital_status'].lower()) <= 2):
            attributes['marital_status'] = 'widowed'
        elif(levenshtein('duda',attributes['marital_status'].lower()) <= 2):
            attributes['marital_status'] = 'widowed'
        elif(levenshtein('cerai',attributes['marital_status'].lower()) <= 2):
            attributes['marital_status'] = 'widowed'
        else:
            attributes['marital_status'] = None

    attributes['occupation'] = find_occupation(raw_result['pekerjaan'])


    #bonus
    if(raw_result['gol_darah'] != None):
        attributes['blood_type'] = ''.join([i for i in raw_result['gol_darah'] if not i.isdigit()]).strip()
        if(attributes['blood_type'].lower() not in ['a','b','ab','o']):
            attributes['blood_type'] = None
    else:
        attributes['blood_type'] = None

    attributes['province'] = raw_result['provinsi']
    attributes['city'] = raw_result['kota']
    attributes['address'] = raw_result['alamat']
    attributes['rt_rw'] = raw_result['rt_rw']
    attributes['kel_desa'] = raw_result['kel_desa']
    attributes['kecamatan'] = raw_result['kecamatan']
    attributes['religion'] = raw_result['agama']
    attributes['expired_date'] = raw_result['berlaku_hingga']

    ktp_extract = ktp_extract.append(attributes,ignore_index=True)

    return ktp_extract

def process_extract_entities(ocr_path):
        try:
            text_response = np.load(ocr_path).item()
        except:
            print(ocr_path+' cannot be loaded')

        ktp_extract = extract_ktp_data(text_response)
        print(ktp_extract.iloc[0])

        ocr_name = ocr_path.split('/')[-1].split('.')[0]
        output_name = cfg.output_loc+'data_'+ocr_name+'.csv'
        ktp_extract.to_csv(output_name,index=False)

if __name__ == '__main__':
    if(len(sys.argv) > 1):
        # input: ocr file path
        ocr_path = sys.argv[1]
        print('Extracting data from '+ocr_path)
        process_extract_entities(ocr_path)
    else:
        print('argument is missing: ocr output file path')
