import sys
import kyc_config as cfg
import ocr_text_extractor as ocr
import ktp_entity_extractor as extractor

if __name__ == '__main__':
    if(len(sys.argv) > 1):
        # input: image path
        img_path = sys.argv[1]
        print('OCR processing '+img_path)
        ocr.process_ocr(img_path)

        img_name = img_path.split('/')[-1].split('.')[0]
        ocr_path = cfg.json_loc+'ocr_'+img_name+'.npy'
        print('Extracting data from '+ocr_path)
        extractor.process_extract_entities(ocr_path)
    else:
        print('argument is missing: image path')
