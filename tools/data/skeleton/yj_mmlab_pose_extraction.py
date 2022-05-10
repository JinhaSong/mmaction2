#폴더 안에 있는 모든 클립 파일들에 대해서 포즈를 추출해서 지정한 경로로 ".pkl"파일을 만들어주는 코드
import subprocess
import argparse
#이제 여기에 클립 담겨 있는 경로랑 파일들 이름 읽어와서 반복문으로 넘기면 됨.
#그치그치 그 전에 argument를 어떻게 넘길지부터 정하자.(이거 정해서 하는게 맞아.)
#ntu ~ 저거엔 필요한 argument 인자가 총 2개인데, 하나는 input 비디오, 하나는 output 비디오(경로로 주는게 좋을듯)
parser = argparse.ArgumentParser(description='Pose Extraction Test')

#--source_folder 읽어올 한가지 클래스에 대한 클립들만 들어있는 source 경로
parser.add_argument('--source_folder', required=True, help = 'One Class Clip Video Source Folder')
parser.add_argument('--out_folder', required=True, help='Output(.pkl) folder')

args = parser.parse_args()
# parser에 넘겨주는 건 끝났고 그럼 이제 parser에서 받은 인자들로 가지고 폴더 안에 파일 탐색하면서
# 포즈 추출해서 out_folder에 저장하는거까지하면 끝.

#TODO : 반복문으로 폴더 안에 파일들 탐색하면서 아래 subprocess 콜하기

#====

#subprocess.call("python "+"tools/data/skeleton/ntu_pose_extraction.py",shell=True)