# KTPextractor

This is a service to extract data from KTP image. This is a part of open source project by Data Scientists of Bukalapak. Other open source projects: https://github.com/bukalapak?q=data

### Config File
Please fill in the configuration in file `kyc_config.py`
`gcv_api_key_path`: path location of the GCV API Key. To get an API, check https://cloud.google.com/vision/docs/setup
`json_loc` = path location to save the OCR output from GCV
`output_loc` = path location to save the extracted KTP data

### OCR Text Extractor
To extract texts from an image (OCR), use the following command:
```
python ocr_text_extractor.py <image_path>
```
The OCR output file will be saved in the `json_loc` (check config file)

### KTP Entity Extractor
To extract attributes from the KTP based on the OCR output, use the following command:
```
python ktp_entity_extractor.py <path of ocr output file>
```
The extracted KTP data will be saved in csv format in the `output_loc` (check config file)

### KTP Data Extractor
To extract KTP data directly from KTP image, use the following command:
```
python KTPextractor_main.py <image_path>
```
The extracted KTP data will be saved in csv format in the `output_loc` (check config file)
