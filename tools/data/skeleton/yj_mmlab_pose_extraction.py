#폴더 안에 있는 모든 클립 파일들에 대해서 포즈를 추출해서 지정한 경로로 ".pkl"파일을 만들어주는 코드
import subprocess
import argparse
import os

parser = argparse.ArgumentParser(description='Pose Extraction Test')

#--source_folder 읽어올 한가지 클래스에 대한 클립들만 들어있는 source 경로
parser.add_argument('--source_folder', required=True, help = 'One Class Clip Video Source Folder')
parser.add_argument('--out_folder', required=True, help='Output(.pkl) folder')

args = parser.parse_args()



dir = str(args.source_folder)
files = os.listdir(dir)

for file in files:
    input_vid = dir+file
    subprocess.call("python " + "tools/data/skeleton/ntu_pose_extraction.py "+input_vid+" "+str(args.out_folder) +file.split('.')[0]+'.pkl', shell=True)
