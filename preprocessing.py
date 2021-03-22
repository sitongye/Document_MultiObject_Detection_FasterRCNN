# Author: Sitong Ye sitongye94@outlook.com
# Date: 05.02.2020

import yaml
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
import json
import pandas as pd
import random
from shutil import copyfile
from optparse import OptionParser

class Datagenerator:

    def __init__(self, configfile):

        with open(configfile) as file:
            self.dataconfig = yaml.load(file, Loader=yaml.FullLoader)
            print(self.dataconfig)
        # parameters to be parsed: trainsetsplit, pathconfig, classes, annotation_name

        self.traintestsplit= self.dataconfig["TRAIN_TEST_SPLIT"]
        self.classes = self.dataconfig["DETECT_CLASSES"]
        self.dataframe= pd.DataFrame()
        self.annotationname = self.dataconfig["ANNOTATION_NAME"]

    def __create_dataframe(self):
        self.dataframe = pd.DataFrame(columns=[
            'file_name', 'xmin', 'ymin',
            'xmax', 'ymax', 'label'])
        data_path = os.path.join(self.dataconfig['DATA_PATH'])
        image_folder = self.dataconfig['IMAGE_FOLDER']
        sig_config_path = os.path.join(data_path, self.dataconfig["JSON_CONFIG_FOLDER"])
        with open(os.path.join(sig_config_path, self.dataconfig['JSON_CONFIG_FILE'])) as json_cfg_file:
            cfg = json.load(json_cfg_file)

# read configuration file into dataframe
        for image in cfg:
            filename = image['filename'].split('/')[-1]
            #fileclass = image['class']
            annotations = image['annotations']  # this is a list...
            for obj in annotations:  # every obj is a dictionary...
                cls = obj['class']
                xmin = obj['x']
                ymin = obj['y']
                width = obj['width']
                height = obj['height']
                if 'id' in obj:
                    iden = obj['id']
                else:
                    iden = None
                typ = obj['type']
                # ready to assign to dataframe
                self.dataframe = self.dataframe.append({'file_name': filename,
                                              'xmin': int(xmin),
                                              'ymin': int(ymin),
                                              'xmax': int(xmin + width),
                                              'ymax': int(ymin + height),
                                              'label': cls}, ignore_index=True)
        self.dataframe = self.dataframe[self.dataframe['label'].isin(self.classes)]
        print('number of image:', len(self.dataframe['file_name'].unique()))
        print('all classes of labels', list(self.dataframe['label'].unique()))
        return self.dataframe

    def __train_test_split(self):
        random.seed(21102019)
        all_image = list(self.dataframe['file_name'].unique())
        test_image = random.sample(all_image,
                                   int(len(all_image)*self.traintestsplit))
        train_image = list(set(all_image) - set(test_image))

        # create train and test directory
        train_path = os.path.join('.','data', self.dataconfig["OUTPUT_TRAIN_FOLDER_NAME"])
        if os.path.exists(train_path) is False:
            os.mkdir(train_path)
        test_path = os.path.join('.','data', self.dataconfig["OUTPUT_TEST_FOLDER_NAME"])
        if os.path.exists(test_path) is False:
            os.mkdir(test_path)

        train_df = pd.DataFrame(columns=self.dataframe.columns)
        for img in train_image:
            train_df = train_df.append(self.dataframe.loc[self.dataframe['file_name']==img,:],ignore_index=True)
        test_df = pd.DataFrame(columns=self.dataframe.columns)
        for img in test_image:
            test_df = test_df.append(self.dataframe.loc[self.dataframe['file_name']==img,:],ignore_index=True)

        # separate train and test into train_test folder
        for roots, dirs, files in os.walk(os.path.join(self.dataconfig["DATA_PATH"], self.dataconfig["IMAGE_FOLDER"])):
            for file in files:
                old_path = os.path.join(os.path.join(self.dataconfig["DATA_PATH"], self.dataconfig["IMAGE_FOLDER"]), file)
                # print(file)
                if str(file) in train_df['file_name'].values:
                    new_path = os.path.join(train_path, file)
                elif str(file) in test_df['file_name'].values:
                    new_path = os.path.join(test_path, file)
                copyfile(old_path, new_path)

        # generate txt_annotation_file

        f = open(os.path.join(self.dataconfig["OUTPUT_DATA_PATH"], "train_"+self.annotationname), "w+")
        for idx, row in train_df.iterrows():
            img = cv2.imread(os.path.join(self.dataconfig["OUTPUT_DATA_PATH"], self.dataconfig["OUTPUT_TRAIN_FOLDER_NAME"], row['file_name']))
            height, width = img.shape[:2]
            x1 = int(row['xmin'])
            x2 = int(row['xmax'])
            y1 = int(row['ymin'])
            y2 = int(row['ymax'])

            file_path = os.path.join(self.dataconfig["OUTPUT_DATA_PATH"], self.dataconfig["OUTPUT_TRAIN_FOLDER_NAME"])
            fileName = os.path.join(file_path, row['file_name'])
            className = row['label']
            f.write(fileName + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + className + '\n')
        f.close()

        f = open(os.path.join(self.dataconfig["OUTPUT_DATA_PATH"], "test_"+self.annotationname), "w+")
        for idx, row in test_df.iterrows():
            img = cv2.imread(os.path.join(self.dataconfig["OUTPUT_DATA_PATH"], self.dataconfig["OUTPUT_TEST_FOLDER_NAME"],
                                          row['file_name']))
            #height, width = img.shape[:2]
            x1 = int(row['xmin'])
            x2 = int(row['xmax'])
            y1 = int(row['ymin'])
            y2 = int(row['ymax'])

            file_path = os.path.join(self.dataconfig["OUTPUT_DATA_PATH"], self.dataconfig["OUTPUT_TEST_FOLDER_NAME"])
            fileName = os.path.join(file_path, row['file_name'])
            className = row['label']
            f.write(fileName + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + className + '\n')
        f.close()

    def generate(self):
        self.__create_dataframe()
        self.__train_test_split()

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-c", "--config", help="path to pass config yaml file")
    (options, args) = parser.parse_args()
    cfg = options.config
    print(options)
    Datagenerator(cfg).generate()
